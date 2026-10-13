from __future__ import annotations

import datetime
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from fire_core.utils import Features, flatten, obs_to_indexed, flat_array_to_indexed
from fire_core.strategies import StepResult


def _flatten_extra_log(data: Dict[str, Any]) -> Dict[str, float]:
    """Scalars keep their key as the column name; arrays become key_0..key_N."""
    out: Dict[str, float] = {}
    for key, value in data.items():
        arr = np.asarray(value, dtype=np.float32)
        if arr.ndim == 0:
            out[key] = float(arr)
        else:
            out.update(flat_array_to_indexed(key, arr))
    return out


class StepLogger:
    """Record the StepResult of every step and save it to CSV."""

    def __init__(self, robot, obs_features: Features, log_features: Features):
        self._robot = robot
        self._obs_features = obs_features
        self._log_features = log_features
        self._buffer: List[dict] = []

    def record(
        self,
        result: StepResult,
        step_idx: Optional[int] = None,
        t_wall: Optional[float] = None,
    ) -> None:
        record: dict = dict(result.metadata or {})

        if result.model_obs_array is not None:
            record.update(flat_array_to_indexed("model_obs", result.model_obs_array))

        if result.obs_array is not None:
            record.update(flat_array_to_indexed("obs", result.obs_array))
        else:
            obs = result.obs_dict if result.obs_dict is not None else self._robot.get_observation()
            record.update(obs_to_indexed(obs, self._obs_features))

        policy_action = result.policy_action
        if policy_action is None and result.action_dict:
            policy_action = result.action_dict.get("arm_actions")
        if policy_action is not None:
            record.update(flat_array_to_indexed("policy_action", policy_action))
            record.update(flat_array_to_indexed("normalized_action", policy_action))

        task_action = getattr(self._robot.task, "action", None)
        if task_action is not None:
            record.update(flat_array_to_indexed("ema_action", task_action))
            record.update(flat_array_to_indexed("raw_action", task_action))

        if result.processed_action is not None:
            processed_arm = result.processed_action.get("processed_arm_action")
            processed_gripper = result.processed_action.get("processed_gripper_action")
            if processed_arm is not None:
                record.update(flat_array_to_indexed("processed_arm_action", processed_arm))
            if processed_gripper is not None:
                record.update(flat_array_to_indexed("processed_gripper_action", processed_gripper))

        record.update(flatten(self._robot.task.get_log(), self._log_features))

        # Additive columns (appended after the legacy ones so the existing column
        # layout is unchanged). Task extras are kept out of log_features so the
        # recorder / plot schemas built from log_features do not change.
        get_extra_log = getattr(self._robot.task, "get_extra_log", None)
        if get_extra_log is not None:
            record.update(_flatten_extra_log(get_extra_log()))
        if result.residual_action is not None:
            record.update(flat_array_to_indexed("residual_action", result.residual_action))
        if result.vla_action is not None:
            record.update(flat_array_to_indexed("vla_action", result.vla_action))
        if result.vla_chunk_idx is not None:
            record["vla_chunk_idx"] = int(result.vla_chunk_idx)
        # Replay metadata may already carry the source CSV's "step"; keep it.
        record.setdefault("step", int(step_idx) if step_idx is not None else len(self._buffer))
        # float64 on purpose: float32 cannot resolve sub-second epoch time.
        record["t_wall"] = float(t_wall) if t_wall is not None else time.time()
        self._buffer.append(record)

    def save(self, save_dir: str) -> Optional[str]:
        """Write the buffered steps to CSV and return its path (None if empty)."""
        if not self._buffer:
            return None
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(save_dir, f"collected_data_{ts}.csv")
        pd.DataFrame(self._buffer).to_csv(path, index=False)
        print(f"\n[INFO] Saved {len(self._buffer)} steps → {path}")
        return path