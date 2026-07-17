import numpy as np
from typing import Dict, Optional, Tuple
from ..factory import Factory
from .forge_cfg import (
    ForgeTaskPegInsertCfg,
    ForgeTaskGearMeshCfg,
    ForgeTaskNutThreadCfg,
)
from lerobot_robot_fr3.utils.transformation_utils import tf_inverse, tf_combine
from lerobot_robot_fr3.utils.rotation_utils import quat_apply, quat_from_euler_xyz, quat_mul, get_euler_xyz


class Forge(Factory):
    def __init__(self, name: str):
        super().__init__(name)
    
    def create_config(self) -> None:
        if self.name == "peg_insert":
            self.env_cfg = ForgeTaskPegInsertCfg()
        elif self.name == "gear_mesh":
            self.env_cfg = ForgeTaskGearMeshCfg()
        elif self.name == "nut_thread":
            self.env_cfg = ForgeTaskNutThreadCfg()
        else:
            raise ValueError(f"Unknown task name: {self.name}")
        
        self.ctrl_cfg = self.env_cfg.ctrl
        self.task_cfg = self.env_cfg.task

        self.alpha = self.env_cfg.ft_smoothing_factor

    def create_buffer(self) -> None:
        super().create_buffer()

        self.force_sensor_world_smooth = np.zeros(6, dtype=np.float32)
        self.force_sensor_world = np.zeros(6, dtype=np.float32)

        # /ee_state/pose is fr3_hand_tcp. The FT topic is expressed in
        # bota_ft_sensor_wrench. URDF: wrench -> hand_tcp has yaw -pi/4,
        # so wrench vectors are rotated into TCP with the inverse yaw +pi/4.
        self._q_tcp_from_ft = quat_from_euler_xyz(
            roll=0.0,
            pitch=0.0,
            yaw=np.pi / 4,
        )

    def reset(self) -> None:
        super().reset()

        # Compute initial action for correct EMA computation.
        fixed_pos_action_frame = self.fixed_pos_obs_frame
        pos_action = self.robot.ee_pos - fixed_pos_action_frame
        pos_action_bounds = np.array(self.ctrl_cfg.pos_action_bounds, dtype=np.float32)
        pos_action = pos_action * (1.0 / pos_action_bounds)
        self.action[0:3] = self.prev_action[0:3] = pos_action

        # Relative yaw to bolt.
        unrot_180_euler = np.array([-np.pi, 0.0, 0.0], dtype=np.float32)
        unrot_quat = quat_from_euler_xyz(
            roll=unrot_180_euler[0], pitch=unrot_180_euler[1], yaw=unrot_180_euler[2]
        )

        fingertip_quat_rel_bolt = quat_mul(unrot_quat, self.robot.ee_quat)
        fingertip_yaw_bolt = get_euler_xyz(fingertip_quat_rel_bolt)[2]
        fingertip_yaw_bolt = np.where(
            fingertip_yaw_bolt > np.pi / 2, fingertip_yaw_bolt - 2 * np.pi, fingertip_yaw_bolt
        )
        fingertip_yaw_bolt = np.where(
            fingertip_yaw_bolt < -np.pi, fingertip_yaw_bolt + 2 * np.pi, fingertip_yaw_bolt
        )

        yaw_action = (fingertip_yaw_bolt + np.deg2rad(180.0)) / np.deg2rad(270.0) * 2.0 - 1.0
        self.action[5] = self.prev_action[5] = yaw_action
        self.action[6] = self.prev_action[6] = -1.0

        # ema_rand = np.random.rand()
        # ema_lower, ema_upper = self.ctrl_cfg.ema_factor_range
        # self.ema_factor = ema_lower + ema_rand * (ema_upper - ema_lower)
        # elimate randomness of ema_factor
        self.ema_factor = self.ctrl_cfg.ema_factor

        contact_rand = np.random.rand()
        contact_lower, contact_upper = self.task_cfg.contact_penalty_threshold_range
        self.contact_penalty_thresholds = np.array(
            [contact_lower + contact_rand * (contact_upper - contact_lower)],
            dtype=np.float32
        )

        self.force_sensor_world_smooth[:] = 0.0

    def get_observation(self) -> Dict[str, np.ndarray]:
        obs_dict = super().get_observation()

        prev_action = self.action.copy()
        prev_action[3:5] = 0.0

        ft_force = self.compute_ft_force()

        # Match the obs pre-processing done during training in
        # ForgeEnv._compute_intermediate_values:
        #   (A) fingertip_quat: the w and z components are forced to 0. The policy
        #       was trained on the [0, x, y, 0] manifold (yaw is not observed through
        #       the quaternion), so feeding a raw quaternion is out-of-distribution.
        #       No renormalization is applied (training does not renormalize either),
        #       and the sign flip (flip_quats) is a domain randomization that the
        #       policy is invariant to, so it is omitted here.
        #   (B) ee_angvel: only the z component is kept (x, y forced to 0).
        fingertip_quat = self.robot.ee_quat.copy()
        fingertip_quat[[0, 3]] = 0.0

        ee_angvel = self.robot.ee_angvel.copy()
        ee_angvel[0:2] = 0.0

        obs_dict.update({
            "fingertip_pos_rel_fixed": self.robot.ee_pos - self.fixed_pos_obs_frame,
            "fingertip_quat": fingertip_quat,
            "ee_angvel": ee_angvel,
            "force_threshold": self.contact_penalty_thresholds,
            "ft_force": ft_force,
            "prev_actions": prev_action,
        })

        return obs_dict
    
    def get_gr00t_observation(self) -> Dict[str, np.ndarray]:
        return super().get_gr00t_observation()

    def get_arm_action(self, action: Dict[str, np.ndarray]) -> np.ndarray:
        return super().get_arm_action(action)

    def get_gripper_action(self, action: Dict[str, np.ndarray]) -> np.ndarray:
        return super().get_gripper_action(action)
    
    def get_log(self):
        log = super().get_log()
        log["ft_force"] = self.force
        return log
    
    @property
    def observation_features(self) -> Dict[str, Tuple[int, ...]]:
        return {
            "fingertip_pos_rel_fixed": (3,),
            "fingertip_quat": (4,),
            "ee_linvel": (3,),
            "ee_angvel": (3,),
            "ft_force": (3,),
            "force_threshold": (1,),
            "prev_actions": (7,),
        }

    @property
    def action_features(self) -> Dict[str, Tuple[int, ...]]:
        return {
            "arm_actions": (6,),
            "success_pred": (1,),
        }

    @property
    def log_features(self) -> Dict[str, Tuple[int, ...]]:
        return {**super().log_features, "ft_force": (3,)}

    def process_action(
        self,
        arm_action: np.ndarray,
        gripper_action: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        # apply EMA smoothing to the input action to ensure smoother control
        self.action = self.ema_factor * arm_action.copy() + (1 - self.ema_factor) * self.action

        pos_action = self.action[0:3] * self.ctrl_cfg.pos_action_bounds
        rot_action = self.action[3:6] * self.ctrl_cfg.rot_action_bounds

        fixed_pos_action_frame = self.fixed_pos_obs_frame
        ctrl_target_fingertip_preclipped_pos = fixed_pos_action_frame + pos_action
        rot_action[0:2] = 0.0

        rot_action[2] = np.deg2rad(-180.0) + np.deg2rad(270.0) * (rot_action[2] + 1.0) / 2.0
        bolt_frame_quat = quat_from_euler_xyz(
            roll=rot_action[0], pitch=rot_action[1], yaw=rot_action[2]
        )

        rot_180_euler = np.array([np.pi, 0.0, 0.0])
        quat_bolt_to_ee = quat_from_euler_xyz(
            roll=rot_180_euler[0], pitch=rot_180_euler[1], yaw=rot_180_euler[2]
        )

        ctrl_target_fingertip_preclipped_quat = quat_mul(quat_bolt_to_ee, bolt_frame_quat)

        self.delta_pos = ctrl_target_fingertip_preclipped_pos  - self.robot.ee_pos
        pos_error_clipped = np.clip(self.delta_pos, -self.pos_threshold, self.pos_threshold)
        self.ctrl_target_fingertip_midpoint_pos = self.robot.ee_pos + pos_error_clipped

        curr_roll, curr_pitch, curr_yaw = get_euler_xyz(self.robot.ee_quat)
        desired_roll, desired_pitch, desired_yaw = get_euler_xyz(ctrl_target_fingertip_preclipped_quat)
        desired_xyz = np.stack([desired_roll, desired_pitch, desired_yaw])

        curr_yaw = self.wrap_yaw(curr_yaw)
        desired_yaw = self.wrap_yaw(desired_yaw)

        self.delta_yaw = desired_yaw - curr_yaw
        clipped_yaw = np.clip(self.delta_yaw, -self.rot_threshold[2], self.rot_threshold[2])
        desired_xyz[2] = curr_yaw + clipped_yaw

        desired_roll = np.where(desired_roll < 0.0, desired_roll + 2 * np.pi, desired_roll)
        desired_pitch = np.where(desired_pitch < 0.0, desired_pitch + 2 * np.pi, desired_pitch)

        delta_roll = desired_roll - curr_roll
        clipped_roll = np.clip(delta_roll, -self.rot_threshold[0], self.rot_threshold[0])
        desired_xyz[0] = curr_roll + clipped_roll

        curr_pitch = np.where(curr_pitch > np.pi, curr_pitch - 2 * np.pi, curr_pitch)
        desired_pitch = np.where(desired_pitch > np.pi, desired_pitch - 2 * np.pi, desired_pitch)

        delta_pitch = desired_pitch - curr_pitch
        clipped_pitch = np.clip(delta_pitch, -self.rot_threshold[1], self.rot_threshold[1])
        desired_xyz[1] = curr_pitch + clipped_pitch

        self.ctrl_target_fingertip_midpoint_quat = quat_from_euler_xyz(
            roll=desired_xyz[0], pitch=desired_xyz[1], yaw=desired_xyz[2]
        )

        ctrl_target_fingertip_midpoint_pose = np.concatenate([self.ctrl_target_fingertip_midpoint_pos, self.ctrl_target_fingertip_midpoint_quat])

        return ctrl_target_fingertip_midpoint_pose, gripper_action

    def compute_ft_force(self) -> np.ndarray:
        if self.ft_sensor is not None:
            force_local = np.array(self.ft_sensor.force, dtype=np.float32)
            torque_local = np.array(self.ft_sensor.torque, dtype=np.float32)
        else:
            force_local = np.zeros(3, dtype=np.float32)
            torque_local = np.zeros(3, dtype=np.float32)

        q_world_from_ft = quat_mul(self.robot.ee_quat, self._q_tcp_from_ft)
        self.force_sensor_world = np.concatenate([
            quat_apply(q_world_from_ft, force_local),
            quat_apply(q_world_from_ft, torque_local),
        ]).astype(np.float32)

        self.force_sensor_world_smooth = self.alpha * self.force_sensor_world + (1 - self.alpha) * self.force_sensor_world_smooth
        self.force_sensor_smooth = self.force_sensor_world_smooth.copy()
        self.force = self.force_sensor_smooth[0:3]

        return self.force

    def change_FT_frame(
        self, source_F: np.ndarray, source_T: np.ndarray,
        source_frame: Tuple[np.ndarray, np.ndarray], target_frame: Tuple[np.ndarray, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Modern Robotics eq. 3.95
        source_frame_inv = tf_inverse(source_frame[0], source_frame[1])
        target_T_source_quat, target_T_source_pos = tf_combine(
            source_frame_inv[0], source_frame_inv[1], target_frame[0], target_frame[1]
        )
        target_F = quat_apply(target_T_source_quat, source_F)
        target_T = quat_apply(
            target_T_source_quat, (source_T + np.cross(target_T_source_pos, source_F))
        )
        return target_F, target_T

    def wrap_yaw(self, angle) -> np.ndarray:
        return np.where(angle > np.deg2rad(235), angle - 2 * np.pi, angle)
