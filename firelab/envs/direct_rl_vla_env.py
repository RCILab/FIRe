# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import torch
from abc import abstractmethod
from typing import Any, Dict


from isaacsim.core.simulation_manager import SimulationManager

from isaaclab.envs.common import VecEnvObs, VecEnvStepReturn
from isaaclab.envs.direct_rl_env_cfg import DirectRLEnvCfg
from isaaclab.envs.direct_rl_env import DirectRLEnv

from firelab.envs.gr00t_client import AsyncGr00tInferenceClient
from firelab.envs.openvla_client import AsyncOpenVLAInferenceClient
from firelab.envs.pi05_client import AsyncPi05InferenceClient


class DirectRLVLAEnv(DirectRLEnv):
    _VLA_DEFAULTS = {
        "gr00t": {"chunk_size": 16, "port": 8888},
        "pi05": {"chunk_size": 8, "port": 8000},
        "openvla": {"chunk_size": 8, "port": 8778},
    }

    def __init__(self, cfg: DirectRLEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)

        self.vla_model = str(getattr(cfg, "vla_model", "gr00t")).lower()
        if self.vla_model not in self._VLA_DEFAULTS:
            raise ValueError(f"Unknown vla_model '{self.vla_model}' (expected one of {tuple(self._VLA_DEFAULTS)})")
        defaults = self._VLA_DEFAULTS[self.vla_model]
        self.vla_chunk_size = int(getattr(cfg, "vla_chunk_size", 0) or defaults["chunk_size"])
        vla_host = getattr(cfg, "vla_host", "localhost")
        vla_port = int(getattr(cfg, "vla_port", 0) or os.environ.get("VLA_PORT", defaults["port"]))
        self.vla_result_timeout = 60.0
        self.residual_action_scale = (
            torch.as_tensor(getattr(cfg, "residual_action_scale", 1.0), dtype=torch.float32, device=self.device)
            .expand(6)
            .clone()
        )
        self._scale_residual = bool((self.residual_action_scale != 1.0).any())

        self.record_actions = False
        self.recorded_actions: Dict[str, torch.Tensor] = {}
        if self.vla_model == "gr00t":
            self.vla_policy = AsyncGr00tInferenceClient(host=vla_host, port=vla_port)
        elif self.vla_model == "pi05":
            self.vla_policy = AsyncPi05InferenceClient(host=vla_host, port=vla_port, chunk_size=self.vla_chunk_size)
        else:
            self.vla_policy = AsyncOpenVLAInferenceClient(host=vla_host, port=vla_port, chunk_size=self.vla_chunk_size)
        print(f"Initialize {self.vla_model} client ({vla_host}:{vla_port}, chunk {self.vla_chunk_size})")

        self.vla_chunk: Dict[str, Any] | None = None
        self.vla_chunk_actions: torch.Tensor | None = None

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[VecEnvObs, dict]:
        if seed is not None:
            self.seed(seed)

        indices = torch.arange(self.num_envs, dtype=torch.int64, device=self.device)
        self._reset_idx(indices)

        self.scene.write_data_to_sim()
        self.sim.forward()

        if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
            self.sim.render()

        if self.cfg.wait_for_textures and self.sim.has_rtx_sensors():
            while SimulationManager.assets_loading():
                self.sim.render()

        self.vla_chunk = self.vla_policy.get_action_sync(self._get_vla_observations())

        return self._get_observations(), self.extras

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        if self.episode_length_buf[0] % self.vla_chunk_size == 0:
            if self.episode_length_buf[0].item() != 0:
                try:
                    self.vla_chunk = self.vla_policy.get_result(timeout=self.vla_result_timeout)
                except Exception as e:
                    print(f"[VLA] async result failed ({e}); re-requesting synchronously")
                    self.vla_policy.reset_worker()
                    self.vla_chunk = self.vla_policy.get_action_sync(self._get_vla_observations())

            chunk_pos = torch.tensor(self.vla_chunk["action.eef_position_delta"], dtype=torch.float32, device=self.device)
            chunk_rot = torch.tensor(self.vla_chunk["action.eef_rotation_delta"], dtype=torch.float32, device=self.device)
            chunk_pose = torch.cat(
                (chunk_pos, chunk_rot), dim=-1
            )

            self.vla_chunk_actions = chunk_pose

            if self.action_space.shape[-1] != self.vla_chunk_actions.shape[-1]:
                self.vla_chunk_actions = torch.cat(
                    (self.vla_chunk_actions, torch.zeros((self.num_envs, self.vla_chunk_size, 1), device=self.device)), dim=-1
                )

        elif self.episode_length_buf[0] % self.vla_chunk_size == self.vla_chunk_size // 2:
            self.vla_policy.request_action(self._get_vla_observations())


        action = action.to(self.device)
        if self.cfg.action_noise_model:
            action = self._action_noise_model(action)

        residual_action = action
        vla_action = self.vla_chunk_actions[:, self.episode_length_buf[0] % self.vla_chunk_size, :]
        if self._scale_residual:
            action = action.clone()
            action[:, :6] *= self.residual_action_scale
        final_action = action + vla_action
        if self.record_actions:
            self.recorded_actions = {
                "ep_step": self.episode_length_buf.clone(),
                "vla": vla_action.clone(),
                "residual": residual_action.clone(),
                "final": final_action.clone(),
            }
        self._pre_physics_step(final_action)
        if self.record_actions:
            self.recorded_actions["applied"] = self.actions.clone()

        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            self._apply_action()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            self.scene.update(dt=self.physics_dt)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        self.reset_terminated[:], self.reset_time_outs[:] = self._get_dones()
        self.reset_buf = self.reset_terminated | self.reset_time_outs
        self.reward_buf = self._get_rewards()
        if self.record_actions:
            self.recorded_actions["reward"] = self.reward_buf.clone()
            if hasattr(self, "ep_had_success"):
                self.recorded_actions["ep_had_success"] = self.ep_had_success.clone()
            if hasattr(self, "force_sensor_smooth"):
                self.recorded_actions["force"] = self.force_sensor_smooth[:, :3].clone()
            if hasattr(self, "_record_extra_state"):
                self.recorded_actions.update({k: v.clone() for k, v in self._record_extra_state().items()})
            if hasattr(self, "_get_curr_successes") and hasattr(self, "cfg_task"):
                self.recorded_actions["curr_success"] = self._get_curr_successes(
                    success_threshold=self.cfg_task.success_threshold, check_rot=self.cfg_task.name == "nut_thread"
                )

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self._reset_idx(reset_env_ids)
            self.scene.write_data_to_sim()
            self.sim.forward()
            if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
                self.sim.render()

        if self.cfg.events:
            if "interval" in self.event_manager.available_modes:
                self.event_manager.apply(mode="interval", dt=self.step_dt)

        self.obs_buf = self._get_observations()

        if self.cfg.observation_noise_model:
            self.obs_buf["policy"] = self._observation_noise_model(self.obs_buf["policy"])


        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def _get_vla_observations(self) -> Dict[str, Any]:
        if self.vla_model == "pi05":
            return self._get_pi05_observations()
        if self.vla_model == "openvla":
            return self._get_openvla_observations()
        return self._get_gr00t_observations()

    def _get_pi05_observations(self) -> Dict[str, Any]:
        raise NotImplementedError(f"Please implement the '_get_pi05_observations' method for {self.__class__.__name__}.")

    def _get_openvla_observations(self) -> Dict[str, Any]:
        raise NotImplementedError(f"Please implement the '_get_openvla_observations' method for {self.__class__.__name__}.")

    @abstractmethod
    def _get_gr00t_observations(self) -> Dict[str, Any]:
        raise NotImplementedError(f"Please implement the '_get_gr00t_observations' method for {self.__class__.__name__}.")
