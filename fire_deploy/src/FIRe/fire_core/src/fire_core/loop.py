from __future__ import annotations

import time
from typing import Optional
import numpy as np

from fire_core.strategies import ControlStrategy
from fire_core.logger import StepLogger
# Adjust the import to the exact path where BaseRecorder is located.
from fire_core.recorders.base_recorder import BaseRecorder


def run_control_loop(
    robot,
    strategy: ControlStrategy,
    *,
    control_hz: float,
    max_steps: int,
    logger: Optional[StepLogger] = None,
    recorder: Optional[BaseRecorder] = None
) -> None:
    dt = 1.0 / control_hz
    print(f"\n[INFO] Control loop @ {control_hz} Hz  (Ctrl+C to stop)")

    strategy.reset()

    step_idx = 0
    try:
        while True:
            if max_steps > 0 and step_idx >= max_steps:
                print(f"\n[INFO] Reached max steps ({max_steps}). Auto-stopping.")
                break

            t_start = time.time()

            # 1. Compute the next action via the Strategy
            result = strategy.step(step_idx)
            if result is None:
                break

            # 2. Send the action to the robot
            #    Strategies that emit absolute task-space poses (teleop replay,
            #    VLA-only) skip process_action (relative delta conversion) and
            #    send via the same absolute-pose path used by teleop.
            if getattr(strategy, "sends_task_space_pose", False):
                if result.action_dict:
                    result.processed_action = robot.send_processed_action(result.action_dict)
            else:
                result.processed_action = robot.send_action(result.action_dict)

            # 3. Record via the existing Logger (e.g. CSV)
            if logger is not None:
                logger.record(result)

            # 4. Record the LeRobot/GR00T/Pi0 dataset (core part)
            if recorder is not None:                
                arm_action = result.policy_action

                # Extract the gripper action (using the robot task interface)
                gripper_action = robot.task.get_gripper_action(result.action_dict)

                recorder.record(
                    arm_action=arm_action,
                    gripper_action=gripper_action,
                )

            # 5. Post-processing
            result = strategy.after_action_sent(step_idx, result)

            # 6. Frequency (Hz) synchronization
            sleep_t = dt - (time.time() - t_start)
            if sleep_t > 0:
                time.sleep(sleep_t)

            step_idx += 1

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")