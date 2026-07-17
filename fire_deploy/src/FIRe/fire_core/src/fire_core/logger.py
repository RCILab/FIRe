from __future__ import annotations

import datetime
import os
from typing import List, Optional

import numpy as np
import pandas as pd

from fire_core.utils import Features, flatten, obs_to_indexed, flat_array_to_indexed
from fire_core.strategies import StepResult


class StepLogger:
    """Record the StepResult of every step and save it to CSV."""

    def __init__(self, robot, obs_features: Features, log_features: Features):
        self._robot = robot
        self._obs_features = obs_features
        self._log_features = log_features
        self._buffer: List[dict] = []

    def record(self, result: StepResult) -> None:
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
        self._buffer.append(record)

    def save(self, save_dir: str) -> None:
        if not self._buffer:
            return
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(save_dir, f"collected_data_{ts}.csv")
        pd.DataFrame(self._buffer).to_csv(path, index=False)
        print(f"\n[INFO] Saved {len(self._buffer)} steps → {path}")