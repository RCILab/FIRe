import threading
from queue import Queue
from typing import Dict, Any, Optional

from .pi05.websocket_client_policy import WebsocketClientPolicy


class AsyncPi05InferenceClient:
    """
    Async wrapper around synchronous WebsocketClientPolicy.
    Designed to be drop-in compatible with gr00t async usage:
      - request_action()
      - get_result()
      - get_action_sync()
    """

    def __init__(
        self,
        # host: str = "127.0.0.1",
        host: str = "163.180.160.225",
        port: int = 8000,
        api_key: Optional[str] = None,
        batched: bool = True,
    ):
        self.host = host
        self.port = port
        self.api_key = api_key
        self.batched = batched

        # Single-slot queues to enforce pipeline discipline
        self._request_queue: Queue = Queue(maxsize=1)
        self._result_queue: Queue = Queue(maxsize=1)

        self._worker = threading.Thread(
            target=self._worker_loop,
            daemon=True,
        )
        self._worker.start()

        print(f"[AsyncPi05] Worker thread started (ws://{host}:{port})")

    # ------------------------------------------------------------------
    # Worker thread
    # ------------------------------------------------------------------

    def _worker_loop(self):
        """
        The worker owns the WebSocket connection.
        infer() is blocking, but this thread isolates it.
        """
        policy = WebsocketClientPolicy(
            host=self.host,
            port=self.port,
            api_key=self.api_key,
        )

        while True:
            obs = self._request_queue.get()
            if obs is None:
                break

            try:
                result = policy.infer(self._prepare_observations(obs))
                self._put_latest(self._result_queue, result)
            except Exception as e:
                self._put_latest(self._result_queue, {"error": str(e)})

    @staticmethod
    def _put_latest(queue: Queue, item: Any):
        """
        Ensure queue contains only the latest item.
        """
        while not queue.empty():
            try:
                queue.get_nowait()
            except Exception:
                pass
        queue.put(item)

    def _prepare_observations(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        observations = dict(observations)
        observations.setdefault("_batched", self.batched)
        return observations

    # ------------------------------------------------------------------
    # Public async API
    # ------------------------------------------------------------------

    def request_action(self, observations: Dict[str, Any]):
        """
        Non-blocking.
        Sends observations to worker thread.
        """
        self._put_latest(self._request_queue, observations)

    def get_result(self) -> Dict[str, Any]:
        """
        Blocking.
        Waits for worker to finish inference.
        """
        result = self._result_queue.get()
        if "error" in result:
            raise RuntimeError(f"Pi05 inference error: {result['error']}")
        return result

    # ------------------------------------------------------------------
    # Public sync API (for reset)
    # ------------------------------------------------------------------

    def get_action_sync(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """
        Synchronous one-shot inference.
        Used only at env.reset().
        """
        policy = WebsocketClientPolicy(
            host=self.host,
            port=self.port,
            api_key=self.api_key,
        )
        return policy.infer(self._prepare_observations(observations))

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self):
        if self._worker.is_alive():
            self._request_queue.put(None)
            self._worker.join(timeout=2)

    def __del__(self):
        self.close()
