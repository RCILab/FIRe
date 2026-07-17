# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import torch

import isaaclab.sim as sim_utils
import isaacsim.core.utils.torch as torch_utils

from isaaclab.assets import Articulation
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.sensors import TiledCamera

from fire_lab.tasks.direct.base_line.factory import factory_utils

from fire_lab.tasks.direct.vla.pi05.factory.factory_pi05_env_cfg import FactoryPi05EnvCfg
from fire_lab.tasks.direct.vla.common.factory_env import FactoryEnv
from fire_lab.tasks.direct.base_line.forge import forge_utils
import torch.nn.functional as F


class FactoryPi05Env(FactoryEnv):
    cfg: FactoryPi05EnvCfg

    def __init__(self, cfg: FactoryPi05EnvCfg, render_mode: str | None = None, **kwargs):
        # Update number of obs/states
        super().__init__(cfg, render_mode, **kwargs)
        self.cfg_task = cfg.task

        # Force sensor information.
        self.force_sensor_body_idx = self._robot.body_names.index("force_sensor")
        self.force_sensor_smooth = torch.zeros((self.num_envs, 6), device=self.device)
        self.force_sensor_world_smooth = torch.zeros((self.num_envs, 6), device=self.device)
        self._prev_phi_f = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

    def _setup_scene(self):
        """Initialize simulation scene."""
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(), translation=(0.0, 0.0, -1.05))

        # spawn a usd file of a table into the scene
        cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd")
        cfg.func(
            "/World/envs/env_.*/Table", cfg, translation=(0.55, 0.0, 0.0), orientation=(0.70711, 0.0, 0.0, 0.70711)
        )

        self._robot = Articulation(self.cfg.robot)
        self._fixed_asset = Articulation(self.cfg_task.fixed_asset)
        self._held_asset = Articulation(self.cfg_task.held_asset)
        if self.cfg_task.name == "gear_mesh":
            self._small_gear_asset = Articulation(self.cfg_task.small_gear_cfg)
            self._large_gear_asset = Articulation(self.cfg_task.large_gear_cfg)

        self._wrist_camera = TiledCamera(self.cfg.wrist_camera)
        self._left_camera = TiledCamera(self.cfg.left_camera)
        self._right_camera = TiledCamera(self.cfg.right_camera)

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            # we need to explicitly filter collisions for CPU simulation
            self.scene.filter_collisions()

        self.scene.articulations["robot"] = self._robot
        self.scene.articulations["fixed_asset"] = self._fixed_asset
        self.scene.articulations["held_asset"] = self._held_asset
        if self.cfg_task.name == "gear_mesh":
            self.scene.articulations["small_gear"] = self._small_gear_asset
            self.scene.articulations["large_gear"] = self._large_gear_asset

        self.scene.sensors["wrist_camera"] = self._wrist_camera
        self.scene.sensors["left_camera"] = self._left_camera
        self.scene.sensors["right_camera"] = self._right_camera

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _reset_idx(self, env_ids):
        """We assume all envs will always be reset at the same time."""
        super()._reset_idx(env_ids)
        if hasattr(self, "_prev_phi_f"):
            self._prev_phi_f[env_ids] = 0.0

        contact_rand = torch.rand((self.num_envs,), dtype=torch.float32, device=self.device)
        contact_lower, contact_upper = [5.0, 10.0]
        self.contact_penalty_thresholds = contact_lower + contact_rand * (contact_upper - contact_lower)

    def _apply_action(self):
        super()._apply_action()

        # Step (0): Scale actions to allowed range.
        pos_actions = self.actions[:, 0:3]
        pos_actions = pos_actions @ torch.diag(torch.tensor(self.cfg.ctrl.pos_action_bounds, device=self.device))

        rot_actions = self.actions[:, 3:6]
        rot_actions = rot_actions @ torch.diag(torch.tensor(self.cfg.ctrl.rot_action_bounds, device=self.device))

        # Step (1): Compute desired pose targets in EE frame.
        # (1.a) Position. Action frame is assumed to be the top of the bolt (noisy estimate).
        fixed_pos_action_frame = self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise
        ctrl_target_fingertip_preclipped_pos = fixed_pos_action_frame + pos_actions
        # (1.b) Enforce rotation action constraints.
        rot_actions[:, 0:2] = 0.0

        # Assumes joint limit is in (+x, -y)-quadrant of world frame.
        rot_actions[:, 2] = np.deg2rad(-180.0) + np.deg2rad(270.0) * (rot_actions[:, 2] + 1.0) / 2.0  # Joint limit.
        # (1.c) Get desired orientation target.
        bolt_frame_quat = torch_utils.quat_from_euler_xyz(
            roll=rot_actions[:, 0], pitch=rot_actions[:, 1], yaw=rot_actions[:, 2]
        )

        rot_180_euler = torch.tensor([np.pi, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        quat_bolt_to_ee = torch_utils.quat_from_euler_xyz(
            roll=rot_180_euler[:, 0], pitch=rot_180_euler[:, 1], yaw=rot_180_euler[:, 2]
        )

        ctrl_target_fingertip_preclipped_quat = torch_utils.quat_mul(quat_bolt_to_ee, bolt_frame_quat)

        # Step (2): Clip targets if they are too far from current EE pose.
        # (2.a): Clip position targets.
        self.delta_pos = ctrl_target_fingertip_preclipped_pos - self.fingertip_midpoint_pos  # Used for action_penalty.
        pos_error_clipped = torch.clip(self.delta_pos, -self.pos_threshold, self.pos_threshold)
        ctrl_target_fingertip_midpoint_pos = self.fingertip_midpoint_pos + pos_error_clipped

        # (2.b) Clip orientation targets. Use Euler angles. We assume we are near upright, so
        # clipping yaw will effectively cause slow motions. When we clip, we also need to make
        # sure we avoid the joint limit.

        # (2.b.i) Get current and desired Euler angles.
        curr_roll, curr_pitch, curr_yaw = torch_utils.get_euler_xyz(self.fingertip_midpoint_quat)
        desired_roll, desired_pitch, desired_yaw = torch_utils.get_euler_xyz(ctrl_target_fingertip_preclipped_quat)

        # (2.b.ii) Correct the direction of motion to avoid joint limit.
        # Map yaws between [-125, 235] degrees (so that angles appear on a continuous span uninterrupted by the joint limit).
        curr_yaw = factory_utils.wrap_yaw(curr_yaw)
        desired_yaw = factory_utils.wrap_yaw(desired_yaw)

        # (2.b.iii) Clip motion in the correct direction.
        self.delta_yaw = desired_yaw - curr_yaw  # Used later for action_penalty.



    def _get_observations(self):
        """Get actor/critic inputs using asymmetric critic."""
        obs_dict, state_dict = self._get_factory_obs_state_dict()

        prev_actions = self.actions.clone()
        prev_actions[:, 3:5] = 0.0

        noisy_force, force_sensor_smooth = self.get_ft_force()

        obs_dict.update({
            "force_threshold": self.contact_penalty_thresholds[:, None],
            "ft_force": noisy_force,
            "prev_actions": prev_actions,
        })

        state_dict.update({
            "ft_force": force_sensor_smooth[:, 0:3],
            "force_threshold": self.contact_penalty_thresholds[:, None],
            "prev_actions": prev_actions,
        })

        obs_tensors = factory_utils.collapse_obs_dict(obs_dict, self.cfg.obs_order + ["prev_actions"])
        state_tensors = factory_utils.collapse_obs_dict(state_dict, self.cfg.state_order + ["prev_actions"])
        return {"policy": obs_tensors, "critic": state_tensors}
    
    def _get_rewards(self):
        """FORGE reward includes a contact penalty and success prediction error."""
        rew_buf = self.get_sparse_rewards()

        return rew_buf

    def get_ft_force(self):

        self.force_sensor_world = self._robot.root_physx_view.get_link_incoming_joint_force()[
            :, self._robot.body_names.index("force_sensor")
        ]
        alpha = 0.25
        self.force_sensor_world_smooth = alpha * self.force_sensor_world + (1 - alpha) * self.force_sensor_world_smooth

        force_sensor_smooth = torch.zeros_like(self.force_sensor_world)
        identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        force_sensor_smooth[:, :3], force_sensor_smooth[:, 3:6] = forge_utils.change_FT_frame(
            self.force_sensor_world_smooth[:, 0:3],
            self.force_sensor_world_smooth[:, 3:6],
            (identity_quat, torch.zeros((self.num_envs, 3), device=self.device)),
            (identity_quat, self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise),
        )

        # Compute noisy force values.
        force_noise = torch.randn((self.num_envs, 3), dtype=torch.float32, device=self.device)
        force_noise *= 1.0
        noisy_force = force_sensor_smooth[:, 0:3] + force_noise

        return noisy_force, force_sensor_smooth[:, 0:3]
    
    def get_sparse_rewards(self) -> torch.Tensor:
        """
        Reward structure (paper-aligned):
            r = r_s + r_penalty + r_shaping
            r_s        : sparse success reward
            r_penalty  : -alpha * max(0, ||F|| - F_limit)
            r_shaping  : lambda_f * (gamma * phi(s_{t+1}) - phi(s_t))
            phi(s)     : -log(c_f * max(0, ||F|| - F_limit) + 1)
            gamma      : fixed to 0.995
        """
        # ------------------------------------------------------------
        # Initialize reward buffer
        # ------------------------------------------------------------
        rew_buf = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        # ============================================================
        # (1) Success reward: r_s
        # ============================================================
        check_rot = self.cfg_task.name == "nut_thread"
        curr_successes = self._get_curr_successes(
            success_threshold=self.cfg_task.success_threshold,
            check_rot=check_rot,
        )
        r_s = curr_successes.float()
        # ============================================================
        # (2) Contact force penalty: r_penalty
        #     r_penalty = -alpha * max(0, ||F|| - F_limit)
        # ============================================================
        _, force_sensor_smooth = self.get_ft_force()
        contact_force = torch.norm(
            force_sensor_smooth[:, 0:3], p=2, dim=-1
        )

        F_limit = self.contact_penalty_thresholds
        contact_over = F.relu(contact_force - F_limit)
        alpha = getattr(self.cfg_task, "contact_penalty_alpha", None)
        if alpha is None:
            alpha = getattr(self.cfg_task, "contact_penalty_scale", 1.0)
        r_penalty = -alpha * contact_over
        # ============================================================
        # (3) Potential-based shaping: r_shaping
        #     r_shaping = lambda_f * (gamma * phi(s_{t+1}) - phi(s_t))
        # ============================================================
        gamma = 0.995  # FIXED as requested
        lambda_f = getattr(self.cfg_task, "force_shaping_lambda", 0.0)
        c_f = getattr(self.cfg_task, "force_potential_c", 1.0)
        # phi(s_{t+1})
        phi_curr = -torch.log(c_f * contact_over + 1.0)
        # shaping reward
        r_shaping = lambda_f * (gamma * phi_curr - self._prev_phi_f)
        # update buffer for next step
        self._prev_phi_f = phi_curr.detach()
        # ============================================================
        # (4) Total reward
        # ============================================================
        rew_buf += r_s
        rew_buf += r_penalty
        rew_buf += r_shaping
        return rew_buf
