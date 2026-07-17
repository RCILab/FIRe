import os
import numpy as np
from typing import Dict, Optional, Tuple
from lerobot_robot_fr3.tasks.base_task import Task
from .factory_cfg import (
    FactoryTaskPegInsertCfg,
    FactoryTaskGearMeshCfg,
    FactoryTaskNutThreadCfg,
)
from lerobot_robot_fr3.utils.rotation_utils import quat_from_angle_axis, quat_mul, get_euler_xyz, quat_from_euler_xyz


class Factory(Task):
    def __init__(self, name: str) -> None:
        super().__init__(name)

        self.create_config()
        self.create_buffer()

        # Position of the fixed asset (bolt/hole tip base) expressed in the robot
        # base frame. In sim this is the (randomized) asset pose; on the real robot
        # it MUST match where the physical asset actually sits, otherwise both the
        # `fingertip_pos_rel_fixed` observation and the action anchor
        # (`fixed_pos_action_frame`) are offset by the same error. Resolved from cfg
        # and overridable at runtime via FIRE_FIXED_ASSET_POS="x,y,z".
        self.fixed_pos = self._resolve_fixed_pos()
        self.fixed_quat = np.array([1.0, 0.0, 0.0, 0.0])

    def _resolve_fixed_pos(self) -> np.ndarray:
        """Resolve the fixed-asset position (robot base frame).

        Priority: FIRE_FIXED_ASSET_POS env var ("x,y,z") > cfg.fixed_asset_pos >
        the [0.6, 0.0, 0.05] fallback. On the real robot this must be set to the
        measured asset location for the policy's reference frame to be correct.
        """
        default = getattr(self.env_cfg, "fixed_asset_pos", [0.6, 0.0, 0.05])
        env_val = os.environ.get("FIRE_FIXED_ASSET_POS")
        if env_val:
            try:
                parsed = [float(x) for x in env_val.replace(",", " ").split()]
            except ValueError:
                parsed = []
            if len(parsed) == 3:
                return np.array(parsed, dtype=np.float32)
            print(
                f"[Factory] Ignoring malformed FIRE_FIXED_ASSET_POS={env_val!r}; "
                "expected 'x,y,z'."
            )
        return np.array(default, dtype=np.float32)

    def create_config(self) -> None:
        if self.name == "peg_insert":
            self.env_cfg = FactoryTaskPegInsertCfg()
        elif self.name == "gear_mesh":
            self.env_cfg = FactoryTaskGearMeshCfg()
        elif self.name == "nut_thread":
            self.env_cfg = FactoryTaskNutThreadCfg()
        else:
            raise ValueError(f"Unknown task name: {self.name}")

        self.ctrl_cfg = self.env_cfg.ctrl
        self.task_cfg = self.env_cfg.task

        self.ema_factor = self.ctrl_cfg.ema_factor

    def create_buffer(self) -> None:
        self.action = np.zeros(self.env_cfg.action_space, dtype=np.float32)
        
        self.pos_threshold = np.array(self.ctrl_cfg.pos_action_threshold, dtype=np.float32)
        self.rot_threshold = np.array(self.ctrl_cfg.rot_action_threshold, dtype=np.float32)

        self.ctrl_target_fingertip_midpoint_pos = np.zeros(3, dtype=np.float32)
        self.ctrl_target_fingertip_midpoint_quat = np.array([1, 0, 0, 0], dtype=np.float32)
    
    def reset(self) -> None:
        fixed_tip_pos_local = np.zeros(3, dtype=np.float32)
        fixed_tip_pos_local[2] += self.task_cfg.fixed_asset_cfg.height
        fixed_tip_pos_local[2] += self.task_cfg.fixed_asset_cfg.base_height
        if self.task_cfg.name == "gear_mesh":
            fixed_tip_pos_local[0] = self.task_cfg.fixed_asset_cfg.medium_gear_base_offset[0]

        fixed_tip_pos = self.fixed_pos + fixed_tip_pos_local
        self.fixed_pos_obs_frame = fixed_tip_pos
        
        self.prev_action = np.zeros_like(self.action)
    
    def get_observation(self) -> Dict[str, np.ndarray]:
        self.prev_action = self.action.copy()
        
        obs_dict = {
            "fingertip_pos_rel_fixed": self.robot.ee_pos - self.fixed_pos_obs_frame,
            "fingertip_quat": self.robot.ee_quat,
            "ee_linvel": self.robot.ee_linvel,
            "ee_angvel": self.robot.ee_angvel,
            "prev_actions": self.prev_action,
        }

        return obs_dict
    
    def get_vla_observation(self) -> Dict[str, np.ndarray]:

        if self.camera_sensor is not None:
            camera_data = self.camera_sensor.data
        else:
            camera_data = {
                "left": np.zeros((256, 256, 3), dtype=np.uint8),
                "right": np.zeros((256, 256, 3), dtype=np.uint8),
                "wrist": np.zeros((256, 256, 3), dtype=np.uint8),
            }

        left_camera_data = camera_data["left"]
        right_camera_data = camera_data["right"]
        wrist_camera_data = camera_data["wrist"]
        
        vla_obs = {
            "video.left_view": np.expand_dims(left_camera_data.astype(np.uint8), axis=(0, 1)),
            "video.right_view": np.expand_dims(right_camera_data.astype(np.uint8), axis=(0, 1)),
            "video.wrist_view": np.expand_dims(wrist_camera_data.astype(np.uint8), axis=(0, 1)),
            "state.eef_position": np.expand_dims(self.robot.ee_pos.astype(np.float64), axis=(0, 1)),
            "state.eef_quaternion": np.expand_dims(self.robot.ee_quat.astype(np.float64), axis=(0, 1)),
            "state.gripper_qpos": np.expand_dims(self.robot.gripper_qpos.astype(np.float64), axis=(0, 1)),
        }
        return vla_obs

    def get_arm_action(self, action: Dict[str, np.ndarray]) -> np.ndarray:
        return action["arm_actions"]
    
    @property
    def controls_gripper(self) -> bool:
        return False

    def get_gripper_action(self, action: Dict[str, np.ndarray]) -> np.ndarray:
        if "gripper_actions" in action:
            return np.asarray(action["gripper_actions"], dtype=np.float32).reshape(-1)
        return np.array([-1.0], dtype=np.float32)
    
    def get_log(self) -> Dict[str, np.ndarray]:
        return {
            "ee_pos": self.robot.ee_pos,
            "ee_quat": self.robot.ee_quat,
            "target_pos": self.ctrl_target_fingertip_midpoint_pos,
            "target_quat": self.ctrl_target_fingertip_midpoint_quat,
        }
    
    @property
    def observation_features(self) -> Dict[str, Tuple[int, ...]]:
        return {
            "fingertip_pos_rel_fixed": (3,),
            "fingertip_quat": (4,),
            "ee_linvel": (3,),
            "ee_angvel": (3,),
            "prev_actions": (6,),
        }

    @property
    def action_features(self) -> Dict[str, Tuple[int, ...]]:
        return {
            "arm_actions": (self.env_cfg.action_space,),
        }
    
    @property
    def log_features(self) -> Dict[str, Tuple[int, ...]]:
        return {
            "ee_pos" : (3,),
            "ee_quat": (4,),
            "target_pos": (3,),
            "target_quat": (4,),
        }
    
    def process_action(
        self,
        arm_action: np.ndarray,
        gripper_action: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        # apply EMA smoothing to the input action to ensure smoother control
        self.action = self.ema_factor * arm_action.copy() + (1 - self.ema_factor) * self.action

        pos_action = self.action[0:3] * self.pos_threshold
        rot_action = self.action[3:6]

        if self.task_cfg.unidirectional_rot:
            rot_action[2] = -(rot_action[2] + 1.0) * 0.5  # [-1, 0]
        rot_action = rot_action * self.rot_threshold

        self.ctrl_target_fingertip_midpoint_pos = self.robot.ee_pos + pos_action
        # To speed up learning, never allow the policy to move more than 5cm away from the base.
        fixed_pos_action_frame = self.fixed_pos_obs_frame
        delta_pos = self.ctrl_target_fingertip_midpoint_pos - fixed_pos_action_frame
        pos_error_clipped = np.clip(
            delta_pos, -self.ctrl_cfg.pos_action_bounds[0], self.ctrl_cfg.pos_action_bounds[1]
        )
        self.ctrl_target_fingertip_midpoint_pos = fixed_pos_action_frame + pos_error_clipped

        # Convert to quat and set rot target
        angle = np.linalg.norm(rot_action)
        if angle > 1e-6:
            axis = rot_action / angle
            rot_action_quat = quat_from_angle_axis(angle, axis)
        else:
            rot_action_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.ctrl_target_fingertip_midpoint_quat = quat_mul(rot_action_quat, self.robot.ee_quat)

        target_euler_xyz = np.stack(get_euler_xyz(self.ctrl_target_fingertip_midpoint_quat))
        target_euler_xyz[0] = 3.14159  # Restrict actions to be upright.
        target_euler_xyz[1] = 0.0

        self.ctrl_target_fingertip_midpoint_quat = quat_from_euler_xyz(
            roll=target_euler_xyz[0], pitch=target_euler_xyz[1], yaw=target_euler_xyz[2]
        )
        ctrl_target_fingertip_midpoint_pose = np.concatenate([self.ctrl_target_fingertip_midpoint_pos, self.ctrl_target_fingertip_midpoint_quat])

        return ctrl_target_fingertip_midpoint_pose, gripper_action
