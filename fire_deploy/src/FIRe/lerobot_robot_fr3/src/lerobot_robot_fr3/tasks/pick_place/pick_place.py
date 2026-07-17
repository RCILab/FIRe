import numpy as np
from typing import Any, Dict, Optional, Tuple

from lerobot_robot_fr3.tasks.base_task import Task
from .pick_place_cfg import PickPlaceTaskCfg
from lerobot_robot_fr3.utils.rotation_utils import quat_from_angle_axis, quat_mul


class PickPlace(Task):
    """Real-robot pick_place task: pick up the green cube and place it on top
    of the white gear.

    VLA/teleop-only — no RL policy is deployed for this task. Task-space (EE
    pose) control, matching Forge/Factory and the VLA canonical action
    (state.eef_position / state.eef_quaternion). Inverse3 teleop drives the
    robot with absolute task-space poses via get_arm_action()/
    get_gripper_action() and bypasses process_action() entirely (see
    FR3Robot.send_teleop_action); process_action() only applies if this task
    is later driven by a relative task-space policy.
    """

    def __init__(self, name: str) -> None:
        super().__init__(name)

        self.create_config()
        self.create_buffer()

    # ── config / buffers ──────────────────────────────────────────────────────
    def create_config(self) -> None:
        if self.name == "pick_place":
            self.env_cfg = PickPlaceTaskCfg()
        else:
            raise ValueError(f"Unknown task name: {self.name}")

        self.ctrl_cfg = self.env_cfg.ctrl
        self.task_cfg = self.env_cfg.task

        self.ema_factor = self.ctrl_cfg.ema_factor

    def create_buffer(self) -> None:
        self.action = np.zeros(self.env_cfg.action_space, dtype=np.float32)

        self.pos_threshold = np.array(self.ctrl_cfg.pos_action_threshold, dtype=np.float32)
        self.rot_threshold = np.array(self.ctrl_cfg.rot_action_threshold, dtype=np.float32)

        self.ctrl_target_pos = np.zeros(3, dtype=np.float32)
        self.ctrl_target_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    def reset(self) -> None:
        self.action[:] = 0.0
        self.prev_action = np.zeros_like(self.action)
        self.ctrl_target_pos = self.robot.ee_pos.copy()
        self.ctrl_target_quat = self.robot.ee_quat.copy()

    # ── observation ───────────────────────────────────────────────────────────
    def get_observation(self) -> Dict[str, np.ndarray]:
        self.prev_action = self.action.copy()
        return {
            "fingertip_pos": self.robot.ee_pos,
            "fingertip_quat": self.robot.ee_quat,
            "ee_linvel": self.robot.ee_linvel,
            "ee_angvel": self.robot.ee_angvel,
            "prev_actions": self.prev_action,
        }

    def get_vla_observation(self) -> Dict[str, np.ndarray]:
        if self.camera_sensor is not None:
            camera_data = self.camera_sensor.data
        else:
            camera_data = {
                "left": np.zeros((256, 256, 3), dtype=np.uint8),
                "right": np.zeros((256, 256, 3), dtype=np.uint8),
                "wrist": np.zeros((256, 256, 3), dtype=np.uint8),
            }
        return {
            "video.left_view": np.expand_dims(camera_data["left"].astype(np.uint8), axis=(0, 1)),
            "video.right_view": np.expand_dims(camera_data["right"].astype(np.uint8), axis=(0, 1)),
            "video.wrist_view": np.expand_dims(camera_data["wrist"].astype(np.uint8), axis=(0, 1)),
            "state.eef_position": np.expand_dims(self.robot.ee_pos.astype(np.float64), axis=(0, 1)),
            "state.eef_quaternion": np.expand_dims(self.robot.ee_quat.astype(np.float64), axis=(0, 1)),
            "state.gripper_qpos": np.expand_dims(self.robot.gripper_qpos.astype(np.float64), axis=(0, 1)),
        }

    # ── action ────────────────────────────────────────────────────────────────
    def get_arm_action(self, action: Dict[str, np.ndarray]) -> np.ndarray:
        return np.asarray(action["arm_actions"], dtype=np.float32).reshape(-1)

    def get_gripper_action(self, action: Dict[str, np.ndarray]) -> np.ndarray:
        if "gripper_actions" in action:
            return np.asarray(action["gripper_actions"], dtype=np.float32).reshape(-1)
        return np.array([self.task_cfg.gripper_close], dtype=np.float32)

    def process_action(
        self,
        arm_action: np.ndarray,
        gripper_action: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Relative task-space delta (pos + axis-angle rot) -> absolute EE pose.

        Not exercised by Inverse3 teleop (see class docstring); kept for a
        future relative-action policy driving this task.
        """
        arm_action = np.asarray(arm_action, dtype=np.float32).reshape(-1)[:6]
        self.action = self.ema_factor * arm_action + (1 - self.ema_factor) * self.action

        pos_action = self.action[0:3] * self.pos_threshold
        rot_action = self.action[3:6] * self.rot_threshold

        self.ctrl_target_pos = self.robot.ee_pos + pos_action

        angle = np.linalg.norm(rot_action)
        if angle > 1e-6:
            rot_quat = quat_from_angle_axis(angle, rot_action / angle)
        else:
            rot_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.ctrl_target_quat = quat_mul(rot_quat, self.robot.ee_quat)

        target_pose = np.concatenate([self.ctrl_target_pos, self.ctrl_target_quat]).astype(np.float32)
        return target_pose, gripper_action

    # ── logging ───────────────────────────────────────────────────────────────
    def get_log(self) -> Dict[str, np.ndarray]:
        return {
            "ee_pos": self.robot.ee_pos,
            "ee_quat": self.robot.ee_quat,
            "target_pos": self.ctrl_target_pos,
            "target_quat": self.ctrl_target_quat,
        }

    @property
    def vla_action_spec(self) -> Dict[str, Any]:
        """Inverse3 teleop records the absolute EE pose (position + quaternion)
        that was sent to the robot — the same representation as
        observation.state — not a relative delta. Teleop bypasses
        process_action() (see class docstring), so the recorded action is
        exactly ``[eef_position(3), eef_quaternion(4), gripper(1)]``."""
        return {
            "arm_dim": 7,
            "names": [f"arm_action_{i}" for i in range(7)] + ["gripper_action"],
            "info_names": ["x", "y", "z", "qw", "qx", "qy", "qz", "gripper_close"],
            "modality": {
                "eef_position": {"start": 0, "end": 3},
                "eef_quaternion": {"start": 3, "end": 7, "rotation_type": "quaternion"},
                "gripper_close": {"start": 7, "end": 8},
            },
        }

    # ── feature specs ─────────────────────────────────────────────────────────
    @property
    def observation_features(self) -> Dict[str, Tuple[int, ...]]:
        return {
            "fingertip_pos": (3,),
            "fingertip_quat": (4,),
            "ee_linvel": (3,),
            "ee_angvel": (3,),
            "prev_actions": (6,),
        }

    @property
    def action_features(self) -> Dict[str, Tuple[int, ...]]:
        return {
            "arm_actions": (6,),
            "gripper_actions": (1,),
        }

    @property
    def log_features(self) -> Dict[str, Tuple[int, ...]]:
        return {
            "ee_pos": (3,),
            "ee_quat": (4,),
            "target_pos": (3,),
            "target_quat": (4,),
        }
