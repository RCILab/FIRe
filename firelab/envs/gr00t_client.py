import threading
import time
from queue import Empty, Full, Queue
from typing import Any, Dict

import zmq
from firelab.envs.gr00t_base import BaseInferenceClient, TorchSerializer


class AsyncGr00tInferenceClient(BaseInferenceClient):
    def __init__(self, host: str = "localhost", port: int = 5555, api_token: str = None):
        self.context = zmq.Context.instance()
        self.host = host
        self.port = port
        self.api_token = api_token
        self.socket_address = f"tcp://{self.host}:{self.port}"

        self._request_id = 0
        self._abandoned_workers = 0
        self._start_worker()
        print(f"Async worker thread started. Connecting to {self.socket_address}...")

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
        print(f"[Gr00t] async worker reset (#{self._abandoned_workers})")
        self._start_worker()

    def _worker_loop(self, request_queue: Queue, result_queue: Queue):
        socket = self.context.socket(zmq.REQ)
        socket.setsockopt(zmq.LINGER, 0)
        socket.connect(self.socket_address)

        while True:
            item = request_queue.get()
            if item is None:
                break
            request_id, observations = item

            try:
                request = {"endpoint": "get_action", "data": observations}
                if self.api_token:
                    request["api_token"] = self.api_token

                socket.send(TorchSerializer.to_bytes(request))

                message = socket.recv()
                response = TorchSerializer.from_bytes(message)

                if "error" in response:
                    raise RuntimeError(f"Server error: {response['error']}")

                result_queue.put((request_id, response))

            except Exception as e:
                print(f"[Worker Thread Error] {e}")
                result_queue.put((request_id, {"error": str(e)}))

        socket.close()
        print("Async worker thread stopped.")

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
                raise TimeoutError(f"No Gr00t result for request {self._request_id} within {timeout} s")
            if request_id == self._request_id:
                break
        if "error" in result:
             raise RuntimeError(f"Received error from server or worker: {result['error']}")
        return result

    def close(self):
        if self.worker_thread.is_alive():
            try:
                self.request_queue.put(None, timeout=2)
            except Exception:
                pass
            self.worker_thread.join(timeout=2)
        if self._abandoned_workers == 0 and not self.worker_thread.is_alive():
            self.context.term()

    def __del__(self):
        self.close()

    def get_action_sync(
        self, observations: Dict[str, Any], timeout: float | None = 120.0, retries: int = 3
    ) -> Dict[str, Any]:
        request = {"endpoint": "get_action", "data": observations}
        if self.api_token:
            request["api_token"] = self.api_token
        payload = TorchSerializer.to_bytes(request)

        last_error = None
        for attempt in range(1, retries + 1):
            socket = self.context.socket(zmq.REQ)
            socket.setsockopt(zmq.LINGER, 0)
            if timeout is not None:
                socket.setsockopt(zmq.RCVTIMEO, int(timeout * 1000))
                socket.setsockopt(zmq.SNDTIMEO, int(timeout * 1000))
            socket.connect(self.socket_address)
            try:
                socket.send(payload)
                response = TorchSerializer.from_bytes(socket.recv())
                if "error" in response:
                    raise RuntimeError(f"Server error: {response['error']}")
                return response
            except zmq.Again:
                last_error = f"no reply within {timeout} s"
                print(f"[Gr00t] sync request attempt {attempt}/{retries} failed: {last_error}")
            finally:
                socket.close()

        raise RuntimeError(f"Gr00t server did not answer after {retries} attempts ({last_error})")
