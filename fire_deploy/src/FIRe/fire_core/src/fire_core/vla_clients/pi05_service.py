import threading
from queue import Queue
from typing import Dict, Any, Optional

import numpy as np

from .pi05.websocket_client_policy import WebsocketClientPolicy


class AsyncPi05InferenceClient:
    """
    Async wrapper around the synchronous openpi ``WebsocketClientPolicy``.

    Drop-in compatible with the gr00t client usage:
      - request_action()
      - get_result()
      - get_action_sync()

    The rest of FIRe speaks the gr00t-canonical observation/action format
    (``video.*`` / ``state.*`` inputs, ``action.eef_position_delta`` outputs).
    This client translates that to/from the pi05 (openpi) server format
    (``left_image`` / ``state`` inputs, ``actions`` outputs) so pi05 behaves
    exactly like gr00t to every caller.
    """

    def __init__(
        self,
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
                result = policy.infer(self._to_pi05_observation(obs))
                self._put_latest(self._result_queue, self._to_canonical_action(result))
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

    # ------------------------------------------------------------------
    # Format conversion (gr00t-canonical <-> pi05/openpi server)
    # ------------------------------------------------------------------

    @staticmethod
    def _as_image(value: Any) -> np.ndarray:
        # FIRe emits (B, T, H, W, C); the pi05 server wants a single (H, W, C).
        img = np.asarray(value, dtype=np.uint8)
        return img.reshape(img.shape[-3:])

    def _to_pi05_observation(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """gr00t-canonical observation -> pi05 (openpi) server observation."""
        left = self._as_image(observations["video.left_view"])
        right = self._as_image(observations["video.right_view"])
        wrist = self._as_image(observations["video.wrist_view"])

        eef_pos = np.asarray(observations["state.eef_position"], dtype=np.float32).reshape(-1)[:3]
        eef_quat = np.asarray(observations["state.eef_quaternion"], dtype=np.float32).reshape(-1)[:4]
        gripper = np.asarray(observations["state.gripper_qpos"], dtype=np.float32).reshape(-1)
        state = np.concatenate([eef_pos, eef_quat, gripper], axis=-1).astype(np.float32)

        pi05_obs: Dict[str, Any] = {
            "left_image": left,
            "right_image": right,
            "wrist_image": wrist,
            "state": state,
        }
        if self.batched:
            pi05_obs = {k: v[None, ...] for k, v in pi05_obs.items()}
        pi05_obs["_batched"] = self.batched
        return pi05_obs

    @staticmethod
    def _to_canonical_action(result: Dict[str, Any]) -> Dict[str, Any]:
        """pi05 (openpi) server action -> gr00t-canonical action chunk."""
        if "error" in result:
            return result
        if "actions" not in result:
            raise RuntimeError(f"pi05 server response missing 'actions' key: {list(result)}")

        actions = np.asarray(result["actions"], dtype=np.float32)
        if actions.ndim == 3:  # (B, H, D) -> single robot
            actions = actions[0]
        if actions.ndim != 2:
            raise ValueError(f"Expected pi05 actions with shape (H, D), got {actions.shape}")

        H, D = actions.shape
        pos = actions[:, 0:3]
        rot = actions[:, 3:6] if D >= 6 else np.zeros((H, 3), dtype=np.float32)
        canonical: Dict[str, Any] = {
            "action.eef_position_delta": pos,
            "action.eef_rotation_delta": rot,
        }
        if D > 6:
            canonical["gripper_actions"] = actions[:, 6:7]
        return canonical

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
        return self._to_canonical_action(policy.infer(self._to_pi05_observation(observations)))

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self):
        worker = getattr(self, "_worker", None)
        if worker is not None and worker.is_alive():
            self._request_queue.put(None)
            worker.join(timeout=2)

    def __del__(self):
        self.close()
