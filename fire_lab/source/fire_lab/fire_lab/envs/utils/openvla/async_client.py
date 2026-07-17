from __future__ import annotations

import threading
from queue import Queue
from typing import Any, Dict

from .http_policy import OpenVLAHTTPPolicy


class AsyncOpenVLAInferenceClient:
    """
    Async wrapper for the OpenVLA-OFT HTTP batch server.

    Compatible with the GR00T/Pi05 client surface:
      - request_action()
      - get_result()
      - get_action_sync()
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8778,
        timeout: float | None = 120.0,
        max_batch_size: int | None = None,
    ):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.max_batch_size = max_batch_size
        self.base_url = OpenVLAHTTPPolicy._make_base_url(host, port)

        self._request_queue: Queue = Queue(maxsize=1)
        self._result_queue: Queue = Queue(maxsize=1)
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

        print(f"[AsyncOpenVLA] Worker thread started ({self.base_url})")

    def _worker_loop(self):
        with OpenVLAHTTPPolicy(
            host=self.host,
            port=self.port,
            timeout=self.timeout,
            max_batch_size=self.max_batch_size,
        ) as policy:
            while True:
                observations = self._request_queue.get()
                if observations is None:
                    break

                try:
                    result = policy.infer(observations)
                    self._put_latest(self._result_queue, result)
                except Exception as e:
                    self._put_latest(self._result_queue, {"error": str(e)})

    @staticmethod
    def _put_latest(queue: Queue, item: Any):
        while not queue.empty():
            try:
                queue.get_nowait()
            except Exception:
                pass
        queue.put(item)

    def request_action(self, observations: Dict[str, Any]):
        self._put_latest(self._request_queue, observations)

    def get_result(self) -> Dict[str, Any]:
        result = self._result_queue.get()
        if "error" in result:
            raise RuntimeError(f"OpenVLA inference error: {result['error']}")
        return result

    def get_action_sync(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        with OpenVLAHTTPPolicy(
            host=self.host,
            port=self.port,
            timeout=self.timeout,
            max_batch_size=self.max_batch_size,
        ) as policy:
            return policy.infer(observations)

    def close(self):
        if self._worker.is_alive():
            self._request_queue.put(None)
            self._worker.join(timeout=2)

    def __del__(self):
        self.close()
