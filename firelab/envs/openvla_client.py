import threading
import time
from queue import Empty, Full, Queue
from typing import Any, Dict

from firelab.envs.openvla.http_policy import OpenVLAHTTPPolicy
from firelab.envs.vla_chunk import to_vla_chunk


class AsyncOpenVLAInferenceClient:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 8778,
        chunk_size: int = 8,
        timeout: float | None = 120.0,
        max_batch_size: int | None = None,
    ):
        self.host = host
        self.port = port
        self.chunk_size = chunk_size
        self.timeout = timeout
        self.max_batch_size = max_batch_size

        self._request_id = 0
        self._abandoned_workers = 0
        self._start_worker()
        print(f"Async OpenVLA worker thread started. Connecting to {OpenVLAHTTPPolicy._make_base_url(host, port)}...")

    def _connect(self) -> OpenVLAHTTPPolicy:
        return OpenVLAHTTPPolicy(host=self.host, port=self.port, timeout=self.timeout, max_batch_size=self.max_batch_size)

    def _start_worker(self):
        self.request_queue = Queue(maxsize=1)
        self.result_queue = Queue()
        self.worker_thread = threading.Thread(
            target=self._worker_loop, args=(self.request_queue, self.result_queue), daemon=True
        )
        self.worker_thread.start()

    def reset_worker(self):
        try:
            self.request_queue.put_nowait(None)
        except Full:
            pass
        self._abandoned_workers += 1
        print(f"[OpenVLA] async worker reset (#{self._abandoned_workers})")
        self._start_worker()

    def _worker_loop(self, request_queue: Queue, result_queue: Queue):
        with self._connect() as policy:
            while True:
                item = request_queue.get()
                if item is None:
                    break
                request_id, observations = item
                try:
                    result_queue.put((request_id, to_vla_chunk(policy.infer(observations)["actions"], self.chunk_size)))
                except Exception as e:
                    print(f"[OpenVLA Worker Thread Error] {e}")
                    result_queue.put((request_id, {"error": str(e)}))
        print("Async OpenVLA worker thread stopped.")

    def request_action(self, observations: Dict[str, Any]):
        self._request_id += 1
        try:
            self.request_queue.put((self._request_id, observations), timeout=5.0)
        except Full:
            self.reset_worker()
            self.request_queue.put((self._request_id, observations))

    def get_result(self, timeout: float | None = None) -> Dict[str, Any]:
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            try:
                request_id, result = self.result_queue.get(timeout=remaining)
            except Empty:
                raise TimeoutError(f"No OpenVLA result for request {self._request_id} within {timeout} s")
            if request_id == self._request_id:
                break
        if "error" in result:
            raise RuntimeError(f"Received error from OpenVLA server or worker: {result['error']}")
        return result

    def get_action_sync(self, observations: Dict[str, Any], retries: int = 3) -> Dict[str, Any]:
        last_error = None
        for attempt in range(1, retries + 1):
            try:
                with self._connect() as policy:
                    return to_vla_chunk(policy.infer(observations)["actions"], self.chunk_size)
            except Exception as e:
                last_error = e
                print(f"[OpenVLA] sync request attempt {attempt}/{retries} failed: {e}")
        raise RuntimeError(f"OpenVLA server did not answer after {retries} attempts ({last_error})")

    def close(self):
        if self.worker_thread.is_alive():
            try:
                self.request_queue.put(None, timeout=2)
            except Exception:
                pass
            self.worker_thread.join(timeout=2)

    def __del__(self):
        self.close()
