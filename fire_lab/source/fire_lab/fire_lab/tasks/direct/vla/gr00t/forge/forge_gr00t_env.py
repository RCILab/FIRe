# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
import torch

from fire_lab.tasks.direct.base_line.factory import factory_utils
from fire_lab.tasks.direct.base_line.forge import forge_utils
from fire_lab.tasks.direct.vla.common.forge_env import ForgeEnv
from fire_lab.tasks.direct.vla.gr00t.forge.forge_gr00t_env_cfg import ForgeGr00tEnvCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab.sensors import TiledCamera
import torch.nn.functional as F
import wandb


class ForgeGr00tEnv(ForgeEnv):
    cfg: ForgeGr00tEnvCfg

    def __init__(self, cfg: ForgeGr00tEnvCfg, render_mode: str | None = None, **kwargs):
        """Initialize additional randomization and logging tensors."""
        super().__init__(cfg, render_mode, **kwargs)

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
        lambda_f = getattr(self.cfg_task, "force_shaping_lambda", 0.01)
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

        wandb.log({
            "success rate": curr_successes.float().mean() * 100,
            "r_penalty": r_penalty.float().mean(),
            "r_shaping": r_shaping.float().mean(),
        })
        # print(f"======= {self.episode_length_buf[0]} / {self.max_episode_length} =======")
        # print(f"success rate: {curr_successes.float().mean()*100}%")

        return rew_buf

    def _get_observations(self):
        """Add additional FORGE observations."""
        obs_dict, state_dict = self._get_factory_obs_state_dict()

        noisy_fixed_pos = self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise
        prev_actions = self.actions.clone()
        prev_actions[:, 3:5] = 0.0

        noisy_force, force_sensor_smooth = self.get_ft_force()

        obs_dict.update({
            "fingertip_pos": self.noisy_fingertip_pos,
            "fingertip_pos_rel_fixed": self.noisy_fingertip_pos - noisy_fixed_pos,
            "fingertip_quat": self.noisy_fingertip_quat,
            "force_threshold": self.contact_penalty_thresholds[:, None],
            "ft_force": noisy_force,
            "prev_actions": prev_actions,
        })

        state_dict.update({
            "ema_factor": self.ema_factor,
            "ft_force": force_sensor_smooth[:, 0:3],
            "force_threshold": self.contact_penalty_thresholds[:, None],
            "prev_actions": prev_actions,
        })

        obs_tensors = factory_utils.collapse_obs_dict(obs_dict, self.cfg.obs_order + ["prev_actions"])
        state_tensors = factory_utils.collapse_obs_dict(state_dict, self.cfg.state_order + ["prev_actions"])

        return {"policy": obs_tensors, "critic": state_tensors}
