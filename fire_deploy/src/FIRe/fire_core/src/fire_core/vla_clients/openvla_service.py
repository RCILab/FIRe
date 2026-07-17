from __future__ import annotations

import threading
from queue import Queue
from typing import Any, Dict, Optional

import numpy as np

from .openvla.http_policy import OpenVLAHTTPPolicy


# Default per-task language instruction (mirrors the vla_lab openvla env).
_TASK_INSTRUCTIONS = {
    "peg_insert": "insert the peg into the hole",
    "gear_mesh": "mesh the gears together",
    "nut_thread": "thread the nut onto the bolt",
}


def _instruction_for_task(task_name: Optional[str]) -> str:
    if not task_name:
        return "complete the manipulation task"
    # task names may be prefixed, e.g. "forge-gear_mesh".
    key = task_name.split("-")[-1]
    return _TASK_INSTRUCTIONS.get(key, "complete the manipulation task")


class AsyncOpenVLAInferenceClient:
    """
    Async wrapper for the OpenVLA-OFT HTTP batch server
    (openvla-oft/vla-scripts/deploy_batch.py).

    Drop-in compatible with the gr00t / pi05 client surface:
      - request_action()
      - get_result()
      - get_action_sync()

    Callers pass FIRe's gr00t-canonical observation (``video.*`` / ``state.*``)
    and receive a gr00t-canonical action chunk (``action.eef_position_delta``).
    This client translates to/from the OpenVLA ``/act_batch`` format
    (``full_image`` / ``wrist_image_*`` / ``state`` inputs, ``actions`` output).
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8778,
        timeout: float | None = 120.0,
        max_batch_size: int | None = None,
        instruction: Optional[str] = None,
        task_name: Optional[str] = None,
    ):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.max_batch_size = max_batch_size
        self.instruction = instruction or _instruction_for_task(task_name)
        self.base_url = OpenVLAHTTPPolicy._make_base_url(host, port)

        self._request_queue: Queue = Queue(maxsize=1)
        self._result_queue: Queue = Queue(maxsize=1)
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

        print(f"[AsyncOpenVLA] Worker thread started ({self.base_url}) | task='{self.instruction}'")

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
                    result = policy.infer(self._to_openvla_payload(observations))
                    self._put_latest(self._result_queue, self._to_canonical_action(result))
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

    # ------------------------------------------------------------------
    # Format conversion (gr00t-canonical <-> OpenVLA-OFT server)
    # ------------------------------------------------------------------

    @staticmethod
    def _as_image(value: Any) -> np.ndarray:
        # FIRe emits (B, T, H, W, C); the OpenVLA server wants a single (H, W, C).
        img = np.asarray(value, dtype=np.uint8)
        return img.reshape(img.shape[-3:])

    def _to_openvla_payload(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """gr00t-canonical observation -> OpenVLA ``/act_batch`` payload."""
        eef_pos = np.asarray(observations["state.eef_position"], dtype=np.float32).reshape(-1)[:3]
        eef_quat = np.asarray(observations["state.eef_quaternion"], dtype=np.float32).reshape(-1)[:4]
        gripper = np.asarray(observations["state.gripper_qpos"], dtype=np.float32).reshape(-1)
        state = np.concatenate([eef_pos, eef_quat, gripper], axis=-1).astype(np.float32)

        observation = {
            "full_image": self._as_image(observations["video.left_view"]),
            "wrist_image_left": self._as_image(observations["video.wrist_view"]),
            "wrist_image_right": self._as_image(observations["video.right_view"]),
            "state": state,
        }
        return {"observations": [observation], "instruction": self.instruction}

    @staticmethod
    def _to_canonical_action(result: Dict[str, Any]) -> Dict[str, Any]:
        """OpenVLA ``actions`` -> gr00t-canonical action chunk."""
        if "error" in result:
            return result
        if "actions" not in result:
            raise RuntimeError(f"OpenVLA server response missing 'actions' key: {list(result)}")

        actions = np.asarray(result["actions"], dtype=np.float32)
        if actions.ndim == 3:  # (B, H, D) -> single robot
            actions = actions[0]
        if actions.ndim != 2:
            raise ValueError(f"Expected OpenVLA actions with shape (H, D), got {actions.shape}")

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
    # Public API
    # ------------------------------------------------------------------

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
            return self._to_canonical_action(policy.infer(self._to_openvla_payload(observations)))

    def close(self):
        worker = getattr(self, "_worker", None)
        if worker is not None and worker.is_alive():
            self._request_queue.put(None)
            worker.join(timeout=2)

    def __del__(self):
        self.close()
