from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from multiprocessing import RawValue
from typing import Any, List, Optional

import numpy as np
import pandas as pd
import torch

from fire_core.utils import Features, feature_slice, write_obs_shm
from fire_core.vla_observation import (
    VLAObservationNotReady,
    get_ready_vla_observation,
    wait_for_ready_vla_observation,
)


# ──────────────────────────────────────────────────────────────────────────────
# Data class
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class StepResult:
    obs_dict: Optional[dict]
    action_dict: dict
    policy_action: Optional[np.ndarray] = None
    obs_array: Optional[np.ndarray] = None
    model_obs_array: Optional[np.ndarray] = None
    metadata: Optional[dict] = None
    processed_action: Optional[dict] = None


# ──────────────────────────────────────────────────────────────────────────────
# Abstract base
# ──────────────────────────────────────────────────────────────────────────────

class ControlStrategy(ABC):
    """Strategy implementing the same action-first step order as Isaac Lab's DirectRLEnv."""

    # Strategies with this set to True emit arm_actions as absolute task-space
    # poses (pos+quat) rather than relative deltas. run_control_loop lets such
    # strategies skip process_action and send them to the robot via the same
    # absolute-pose path used by teleop.
    sends_task_space_pose: bool = False

    @abstractmethod
    def reset(self) -> dict: ...

    @abstractmethod
    def step(self, step_idx: int) -> Optional[StepResult]: ...

    def after_action_sent(self, step_idx: int, result: StepResult) -> StepResult:
        return result


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

GRIPPER_ACTION_KEYS: tuple[str, ...] = (
    "gripper_actions",
    "gripper_action",
    "action.gripper_close",
    "action.gripper",
)


def _zero_action(action_dim: int) -> dict[str, np.ndarray]:
    return {"arm_actions": np.zeros(action_dim, dtype=np.float32)}


def _action_features(robot: Any) -> Features:
    task = getattr(robot, "task", None)
    return getattr(task, "action_features", {})


def _model_action_to_action_dict(
    model_action: np.ndarray,
    action_features: Features,
) -> dict[str, np.ndarray]:
    action = np.asarray(model_action, dtype=np.float32).reshape(-1)
    gripper_slice = feature_slice(action_features, "gripper_actions")
    if gripper_slice is None:
        return {"arm_actions": action}

    gripper_action = action[gripper_slice].copy()
    arm_chunks: list[np.ndarray] = []
    offset = 0
    for key, shape in action_features.items():
        dim = int(np.prod(shape))
        chunk = action[offset : offset + dim]
        if key != "gripper_actions":
            arm_chunks.append(chunk)
        offset += dim

    arm_action = (
        np.concatenate(arm_chunks, axis=0).astype(np.float32, copy=False)
        if arm_chunks
        else np.zeros((0,), dtype=np.float32)
    )
    return {
        "arm_actions": arm_action,
        "gripper_actions": gripper_action,
    }


def _normalize_gripper_chunk(values: np.ndarray) -> np.ndarray:
    chunk = np.asarray(values, dtype=np.float32)
    if chunk.ndim == 3 and chunk.shape[0] == 1:
        chunk = chunk.squeeze(0)
    if chunk.ndim == 0:
        chunk = chunk.reshape(1, 1)
    elif chunk.ndim == 1:
        chunk = chunk.reshape(-1, 1)
    else:
        chunk = chunk.reshape(chunk.shape[0], -1)
    return chunk


def _extract_gripper_chunk(raw_actions: dict[str, Any]) -> Optional[np.ndarray]:
    for key in GRIPPER_ACTION_KEYS:
        if key in raw_actions:
            return _normalize_gripper_chunk(raw_actions[key])
    return None


def _chunk_row(chunk: np.ndarray, chunk_idx: int) -> np.ndarray:
    row_idx = min(chunk_idx, chunk.shape[0] - 1)
    return np.asarray(chunk[row_idx], dtype=np.float32).reshape(-1)


def _spin_wait_inference(action_flag: RawValue, action_shm: torch.Tensor, action_dim: int) -> np.ndarray:
    """Pure spin-wait until action_flag becomes 1.

    NOTE: Never use time.sleep().
          Linux's sleep resolution (1-4ms) adds several milliseconds of extra
          waiting even when inference has already finished, breaking the
          control cycle.
    """
    while action_flag.value == 0:
        pass
    action_flag.value = 0
    return action_shm.numpy().flatten()[:action_dim].copy()


# ──────────────────────────────────────────────────────────────────────────────
# Live inference
# ──────────────────────────────────────────────────────────────────────────────

class LiveInferenceStrategy(ControlStrategy):
    """Real-time: apply action -> collect obs -> inference (Isaac Lab order)."""

    def __init__(
        self,
        robot,
        obs_buf: np.ndarray,
        obs_features: Features,
        obs_flag: RawValue,
        action_flag: RawValue,
        action_shm: torch.Tensor,
        action_dim: int,
    ):
        self._robot = robot
        self._obs_buf = obs_buf
        self._obs_features = obs_features
        self._obs_flag = obs_flag
        self._action_flag = action_flag
        self._action_shm = action_shm
        self._action_dim = action_dim
        self._action_features = _action_features(robot)
        self._buffered_action: dict = _zero_action(action_dim)
        self._pending_obs_dict: Optional[dict] = None

    def _wait_inference(self) -> np.ndarray:
        return _spin_wait_inference(self._action_flag, self._action_shm, self._action_dim)

    def _trigger_inference(self, obs_dict: dict) -> None:
        write_obs_shm(obs_dict, self._obs_buf, self._obs_features)
        self._obs_flag.value = 1

    def reset(self) -> dict:
        self._buffered_action = _zero_action(self._action_dim)
        obs_dict = self._robot.get_observation()
        self._pending_obs_dict = obs_dict
        self._trigger_inference(obs_dict)
        return obs_dict

    def step(self, step_idx: int) -> Optional[StepResult]:
        arm_action = self._wait_inference()
        self._buffered_action = _model_action_to_action_dict(
            arm_action,
            self._action_features,
        )
        return StepResult(
            obs_dict=self._pending_obs_dict,
            action_dict=self._buffered_action,
            policy_action=arm_action,
        )

    def after_action_sent(self, step_idx: int, result: StepResult) -> StepResult:
        obs_dict = self._robot.get_observation()
        self._pending_obs_dict = obs_dict
        self._trigger_inference(obs_dict)
        return result


# ──────────────────────────────────────────────────────────────────────────────
# Live inference + VLA
# ──────────────────────────────────────────────────────────────────────────────

class LiveInferenceWithVLAStrategy(LiveInferenceStrategy):
    """Strategy that adds the VLA chunk on top of the RL action."""

    def __init__(
        self,
        robot,
        obs_buf: np.ndarray,
        obs_features: Features,
        obs_flag: RawValue,
        action_flag: RawValue,
        action_shm: torch.Tensor,
        action_dim: int,
        *,
        vla_policy,
        vla_chunk_size: int = 16,
    ):
        super().__init__(robot, obs_buf, obs_features,
                         obs_flag, action_flag, action_shm, action_dim)
        self._vla = vla_policy
        self._chunk_size = int(vla_chunk_size)
        self._vla_chunk: Optional[np.ndarray] = None
        self._vla_gripper_chunk: Optional[np.ndarray] = None
        self._vla_requested = False

    def _build_vla_chunk(self, raw_actions: dict) -> np.ndarray:
        pos = np.asarray(raw_actions["action.eef_position_delta"], dtype=np.float32)
        rot = np.asarray(raw_actions["action.eef_rotation_delta"], dtype=np.float32)
        if pos.ndim == 3: pos = pos.squeeze(0)
        if rot.ndim == 3: rot = rot.squeeze(0)
        chunk = np.concatenate([pos, rot], axis=-1)
        if chunk.shape[-1] < self._action_dim:
            pad = np.zeros((chunk.shape[0], self._action_dim - chunk.shape[-1]), dtype=np.float32)
            chunk = np.concatenate([chunk, pad], axis=-1)
        return chunk

    def _set_vla_chunk(self, raw_actions: dict) -> None:
        chunk = self._build_vla_chunk(raw_actions)
        # Align the replan cadence to the VLA's actual action horizon so we never
        # index past the returned chunk. Different backends return different
        # horizons (gr00t=16, pi05=10, ...); cap by the user-requested size.
        horizon = int(chunk.shape[0])
        effective = min(self._chunk_size, horizon) if horizon > 0 else self._chunk_size
        self._chunk_size = max(1, effective)
        self._vla_chunk = chunk[: self._chunk_size]
        self._vla_gripper_chunk = _extract_gripper_chunk(raw_actions)

    def reset(self) -> dict:
        obs_dict = super().reset()
        vla_obs = wait_for_ready_vla_observation(self._robot)
        raw = self._vla.get_action_sync(vla_obs)
        self._set_vla_chunk(raw)
        self._vla_requested = False
        return obs_dict

    def step(self, step_idx: int) -> Optional[StepResult]:
        chunk_idx = step_idx % self._chunk_size

        if chunk_idx == 0 and step_idx != 0 and self._vla_requested:
            try:
                self._set_vla_chunk(self._vla.get_result())
            except Exception as e:
                print(f"[WARN] VLA get_result failed: {e}, reusing previous chunk.")
            self._vla_requested = False

        if chunk_idx == self._chunk_size // 2 and not self._vla_requested:
            try:
                self._vla.request_action(get_ready_vla_observation(self._robot))
                self._vla_requested = True
            except VLAObservationNotReady as e:
                print(f"[WARN] VLA request skipped: {e}")

        rl_action = self._wait_inference()
        combined = (rl_action + self._vla_chunk[chunk_idx]).astype(np.float32)
        # combined = self._vla_chunk[chunk_idx]
        # combined = rl_action

        self._buffered_action = _model_action_to_action_dict(
            combined,
            self._action_features,
        )
        if self._vla_gripper_chunk is not None:
            self._buffered_action["gripper_actions"] = _chunk_row(
                self._vla_gripper_chunk,
                chunk_idx,
            )
        return StepResult(
            obs_dict=self._pending_obs_dict,
            action_dict=self._buffered_action,
            policy_action=combined,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Live inference — VLA only (no RL policy)
# ──────────────────────────────────────────────────────────────────────────────

class VLAOnlyStrategy(ControlStrategy):
    """Action driven purely by a VLA server chunk — no RL policy.

    Supports both action conventions, auto-detected from the server's returned
    keys (``sends_task_space_pose`` is set accordingly in _build_vla_chunk):

      - Absolute EE pose (gr00t pick_place): ``action.eef_position`` +
        ``action.eef_quaternion`` → 7-dim ``[pos(3), quat(4)]`` sent through the
        absolute task-space path, exactly as Inverse3 teleop commanded during
        recording (process_action() bypassed — no delta scaling/EMA).
      - Relative EE delta (pi05 / openvla): ``action.eef_position_delta`` +
        ``action.eef_rotation_delta`` → 6-dim sent through process_action(),
        which turns it into an absolute EE pose target.
    """

    def __init__(
        self,
        robot,
        action_features: Features,
        *,
        vla_policy,
        vla_chunk_size: int = 16,
    ):
        self._robot = robot
        self._action_features = action_features
        self._vla = vla_policy
        self._chunk_size = int(vla_chunk_size)
        self._vla_chunk: Optional[np.ndarray] = None
        self._vla_gripper_chunk: Optional[np.ndarray] = None
        self._vla_requested = False
        self._buffered_action: dict = {"arm_actions": np.zeros(7, dtype=np.float32)}

    def _build_vla_chunk(self, raw_actions: dict) -> np.ndarray:
        if "action.eef_position" in raw_actions:
            # Absolute EE pose (gr00t pick_place): [pos(3), quat(4)] per row,
            # sent through the absolute task-space path (no process_action).
            self.sends_task_space_pose = True
            pos = np.asarray(raw_actions["action.eef_position"], dtype=np.float32)
            quat = np.asarray(raw_actions["action.eef_quaternion"], dtype=np.float32)
            if pos.ndim == 3: pos = pos.squeeze(0)
            if quat.ndim == 3: quat = quat.squeeze(0)
            # Model output is not guaranteed unit-norm; renormalize per row so
            # the controller receives valid quaternions.
            norms = np.clip(np.linalg.norm(quat, axis=-1, keepdims=True), 1e-8, None)
            quat = quat / norms
            return np.concatenate([pos, quat], axis=-1)
        # Relative EE delta (pi05 / openvla): [pos_delta(3), rot_delta(3)] per
        # row, turned into an absolute pose by task.process_action().
        self.sends_task_space_pose = False
        pos = np.asarray(raw_actions["action.eef_position_delta"], dtype=np.float32)
        rot = np.asarray(raw_actions["action.eef_rotation_delta"], dtype=np.float32)
        if pos.ndim == 3: pos = pos.squeeze(0)
        if rot.ndim == 3: rot = rot.squeeze(0)
        return np.concatenate([pos, rot], axis=-1)

    def _set_vla_chunk(self, raw_actions: dict) -> None:
        chunk = self._build_vla_chunk(raw_actions)
        # Align the replan cadence to the VLA's actual action horizon so we never
        # index past the returned chunk. Different backends return different
        # horizons (gr00t=16, pi05=10, ...); cap by the user-requested size.
        horizon = int(chunk.shape[0])
        effective = min(self._chunk_size, horizon) if horizon > 0 else self._chunk_size
        self._chunk_size = max(1, effective)
        self._vla_chunk = chunk[: self._chunk_size]
        self._vla_gripper_chunk = _extract_gripper_chunk(raw_actions)

    def reset(self) -> dict:
        obs_dict = self._robot.get_observation()
        vla_obs = wait_for_ready_vla_observation(self._robot)
        raw = self._vla.get_action_sync(vla_obs)
        self._set_vla_chunk(raw)
        self._vla_requested = False
        return obs_dict

    def step(self, step_idx: int) -> Optional[StepResult]:
        chunk_idx = step_idx % self._chunk_size

        if chunk_idx == 0 and step_idx != 0 and self._vla_requested:
            try:
                self._set_vla_chunk(self._vla.get_result())
            except Exception as e:
                print(f"[WARN] VLA get_result failed: {e}, reusing previous chunk.")
            self._vla_requested = False

        if chunk_idx == self._chunk_size // 2 and not self._vla_requested:
            try:
                self._vla.request_action(get_ready_vla_observation(self._robot))
                self._vla_requested = True
            except VLAObservationNotReady as e:
                print(f"[WARN] VLA request skipped: {e}")

        action = self._vla_chunk[chunk_idx].astype(np.float32)
        self._buffered_action = {"arm_actions": action}
        if self._vla_gripper_chunk is not None:
            grip = _chunk_row(self._vla_gripper_chunk, chunk_idx)
            self._buffered_action["gripper_actions"] = grip
        
        return StepResult(
            obs_dict=self._robot.get_observation(),
            action_dict=self._buffered_action,
            policy_action=action,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Replay — raw (CSV obs → inference)
# ──────────────────────────────────────────────────────────────────────────────

class ReplayRawStrategy(ControlStrategy):
    """Replay --raw: CSV obs -> inference -> action, logging live robot obs."""

    def __init__(
        self,
        replay_data: pd.DataFrame,
        obs_buf: np.ndarray,
        obs_flag: RawValue,
        action_flag: RawValue,
        action_shm: torch.Tensor,
        obs_dim: int,
        action_dim: int,
        robot=None,
        obs_features: Optional[Features] = None,
    ):
        self._data = replay_data
        self._obs_buf = obs_buf
        self._obs_flag = obs_flag
        self._action_flag = action_flag
        self._action_shm = action_shm
        self._obs_dim = obs_dim
        self._action_dim = action_dim
        self._buffered_action: dict = _zero_action(action_dim)
        self._robot = robot
        self._obs_features = obs_features
        self._action_features = _action_features(robot) if robot is not None else {}
        self._pending_obs_np: Optional[np.ndarray] = None

    def _load_obs_row(self, step_idx: int) -> np.ndarray:
        row = self._data.iloc[step_idx]
        return row[[f"obs_{i}" for i in range(self._obs_dim)]].to_numpy(dtype=np.float32)

    def _row_metadata(self, step_idx: int) -> dict:
        row = self._data.iloc[step_idx]
        metadata = {}
        for col in ("episode", "step"):
            if col in self._data.columns:
                value = row[col]
                metadata[col] = int(value) if float(value).is_integer() else float(value)
        return metadata

    def _trigger_inference(self, obs_np: np.ndarray) -> None:
        self._obs_buf[:] = obs_np
        self._obs_flag.value = 1

    def _wait_inference(self) -> np.ndarray:
        return _spin_wait_inference(self._action_flag, self._action_shm, self._action_dim)

    def _sync_initial_action_from_obs(self, obs_np: np.ndarray) -> None:
        if self._robot is None or self._obs_features is None:
            return
        prev_slice = feature_slice(self._obs_features, "prev_actions")
        if prev_slice is None:
            print("[WARN] Replay raw initial action sync skipped: prev_actions not in obs_features.")
            return
        task = getattr(self._robot, "task", None)
        if task is None:
            print("[WARN] Replay raw initial action sync skipped: robot has no task.")
            return
        prev_actions = obs_np[prev_slice].astype(np.float32, copy=True)
        synced = False
        for action_name, prev_name in (("action", "prev_action"), ("actions", "prev_actions")):
            if not hasattr(task, action_name):
                continue
            action_arr = getattr(task, action_name)
            dim = min(np.asarray(action_arr).size, prev_actions.size)
            next_action = np.asarray(action_arr, dtype=np.float32).copy()
            next_action.reshape(-1)[:dim] = prev_actions[:dim]
            setattr(task, action_name, next_action.reshape(np.asarray(action_arr).shape))
            if hasattr(task, prev_name):
                prev_arr = getattr(task, prev_name)
                next_prev = np.asarray(prev_arr, dtype=np.float32).copy()
                prev_dim = min(next_prev.size, prev_actions.size)
                next_prev.reshape(-1)[:prev_dim] = prev_actions[:prev_dim]
                setattr(task, prev_name, next_prev.reshape(np.asarray(prev_arr).shape))
            synced = True
        if synced:
            print("[INFO] Replay raw initial EMA state synced from obs_0.prev_actions.")
        else:
            print("[WARN] Replay raw initial action sync skipped: task has no action buffer.")

    def reset(self) -> dict:
        self._buffered_action = _zero_action(self._action_dim)
        self._pending_obs_np = self._load_obs_row(0)
        self._sync_initial_action_from_obs(self._pending_obs_np)
        self._trigger_inference(self._pending_obs_np)
        return {}

    def step(self, step_idx: int) -> Optional[StepResult]:
        if step_idx >= len(self._data):
            print("\n[INFO] Replay CSV finished.")
            return None
        arm_action = self._wait_inference()
        self._buffered_action = _model_action_to_action_dict(
            arm_action,
            self._action_features,
        )
        obs_dict = self._robot.get_observation() if self._robot is not None else None
        print(f"\r[REPLAY-RAW] {step_idx + 1}/{len(self._data)}", end="")
        return StepResult(
            obs_dict=obs_dict,
            action_dict=self._buffered_action,
            policy_action=arm_action,
            model_obs_array=self._pending_obs_np,
            metadata=self._row_metadata(step_idx),
        )

    def after_action_sent(self, step_idx: int, result: StepResult) -> StepResult:
        next_idx = step_idx + 1
        if next_idx >= len(self._data):
            return result
        self._pending_obs_np = self._load_obs_row(next_idx)
        self._trigger_inference(self._pending_obs_np)
        return result


# ──────────────────────────────────────────────────────────────────────────────
# Replay — pose (CSV target pose → direct send)
# ──────────────────────────────────────────────────────────────────────────────

class ReplayPoseStrategy(ControlStrategy):
    """Replay --pose: send CSV target pose directly as the action (no inference)."""

    sends_task_space_pose = True

    def __init__(self, replay_data: pd.DataFrame, pose_cols: List[str], action_dim: int):
        self._data = replay_data
        self._pose_cols = pose_cols
        self._action_dim = action_dim
        self._buffered_action: dict = _zero_action(action_dim)

    def reset(self) -> dict:
        self._buffered_action = _zero_action(self._action_dim)
        return {}

    def step(self, step_idx: int) -> Optional[StepResult]:
        if step_idx >= len(self._data):
            print("\n[INFO] Replay CSV finished.")
            return None
        sent_action = self._buffered_action.copy()
        next_idx = step_idx + 1
        if next_idx < len(self._data):
            row = self._data.iloc[next_idx]
            target_pose = row[self._pose_cols].to_numpy(dtype=np.float32)
            self._buffered_action = {"arm_actions": target_pose}
        else:
            self._buffered_action = _zero_action(self._action_dim)
        print(f"\r[REPLAY-POSE] {step_idx + 1}/{len(self._data)}", end="")
        return StepResult(obs_dict=None, action_dict=sent_action)
