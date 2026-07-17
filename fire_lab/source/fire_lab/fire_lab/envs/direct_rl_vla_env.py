# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch

from isaaclab.envs.common import VecEnvObs, VecEnvStepReturn
from isaaclab.envs.direct_rl_env import DirectRLEnv
from isaaclab.envs.direct_rl_env_cfg import DirectRLEnvCfg
from isaacsim.core.simulation_manager import SimulationManager

from .utils import AsyncGr00tInferenceClient, AsyncOpenVLAInferenceClient, AsyncPi05InferenceClient


class DirectRLVlaEnv(DirectRLEnv):
    """Direct RL environment mixin that adds chunked VLA inference."""

    _DEFAULTS = {
        "gr00t": {
            "chunk_size": 16,
            "host": "localhost",
            "port": 5555,
            "client_cls": AsyncGr00tInferenceClient,
        },
        "pi05": {
            "chunk_size": 8,
            "host": "127.0.0.1",
            "port": 8000,
            "client_cls": AsyncPi05InferenceClient,
        },
        "openvla": {
            "chunk_size": 8,
            "host": "127.0.0.1",
            "port": 8778,
            "client_cls": AsyncOpenVLAInferenceClient,
        },
    }

    default_vla_model = "gr00t"

    def __init__(self, cfg: DirectRLEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)

        self.vla_model = getattr(cfg, "vla_model", self.default_vla_model).lower()
        if self.vla_model not in self._DEFAULTS:
            raise ValueError(f"Unsupported vla_model '{self.vla_model}'. Expected one of {tuple(self._DEFAULTS)}.")

        defaults = self._DEFAULTS[self.vla_model]
        self.vla_chunk_size = getattr(cfg, "vla_chunk_size", defaults["chunk_size"])
        client_kwargs = {
            "host": getattr(cfg, "vla_host", defaults["host"]),
            "port": getattr(cfg, "vla_port", defaults["port"]),
        }
        self.vla_policy = defaults["client_cls"](**client_kwargs)
        print(f"Initialize {self.vla_model} Client Node")

        self.vla_actions: Dict[str, Any] | None = None
        self.processed_vla_actions: torch.Tensor | None = None

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

        self.vla_actions = self.vla_policy.get_action_sync(self._get_vla_observations())
        self.processed_vla_actions = self._parse_vla_actions(self.vla_actions)

        return self._get_observations(), self.extras

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        t = int(self.episode_length_buf[0].item())
        chunk_idx = t % self.vla_chunk_size

        if chunk_idx == 0:
            if t != 0:
                try:
                    self.vla_actions = self.vla_policy.get_result()
                except Exception as e:
                    print(f"Error getting {self.vla_model} action: {e}")
                    self.vla_actions = None

            self.processed_vla_actions = self._parse_vla_actions(self.vla_actions)
            print(f"vla action shape: {self.processed_vla_actions.shape}, chunk_idx: {chunk_idx}")
            print(f"rl action shape: {action.shape}, chunk_idx: {chunk_idx}")

        elif chunk_idx == (self.vla_chunk_size // 2):
            self.vla_policy.request_action(self._get_vla_observations())

        action = action.to(self.device)
        if self.cfg.action_noise_model:
            action = self._action_noise_model(action)

        chunk_action = self.processed_vla_actions[:, chunk_idx, :]
        self._pre_physics_step(action + chunk_action)

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

    def _parse_vla_actions(self, out: Dict[str, Any] | None) -> torch.Tensor:
        if self.vla_model == "gr00t":
            return self._parse_gr00t_actions(out)
        if self.vla_model == "pi05":
            return self._parse_pi05_actions(out)
        if self.vla_model == "openvla":
            return self._parse_openvla_actions(out)
        raise ValueError(f"Unsupported vla_model '{self.vla_model}'.")

    def _parse_gr00t_actions(self, out: Dict[str, Any] | None) -> torch.Tensor:
        if out is None or "action.eef_position_delta" not in out or "action.eef_rotation_delta" not in out:
            return self._zero_vla_actions()

        actions_pos = torch.as_tensor(out["action.eef_position_delta"], dtype=torch.float32, device=self.device)
        actions_rot = torch.as_tensor(out["action.eef_rotation_delta"], dtype=torch.float32, device=self.device)
        actions = torch.cat((actions_pos, actions_rot), dim=-1)
        return self._fit_vla_actions(actions)

    def _parse_pi05_actions(self, out: Dict[str, Any] | None) -> torch.Tensor:
        if out is None or "actions" not in out:
            return self._zero_vla_actions()

        actions = torch.as_tensor(np.array(out["actions"], copy=True), dtype=torch.float32, device=self.device)
        if actions.ndim == 2:
            actions = actions.unsqueeze(0)
        if actions.ndim != 3:
            raise ValueError(f"Expected pi05 actions with shape (B, H, D), got {tuple(actions.shape)}")
        return self._fit_vla_actions(actions)

    def _parse_openvla_actions(self, out: Dict[str, Any] | None) -> torch.Tensor:
        if out is None or "actions" not in out:
            return self._zero_vla_actions()

        actions = torch.as_tensor(np.array(out["actions"], copy=True), dtype=torch.float32, device=self.device)
        if actions.ndim == 2:
            actions = actions.unsqueeze(0)
        if actions.ndim != 3:
            raise ValueError(f"Expected openvla actions with shape (B, H, D), got {tuple(actions.shape)}")
        return self._fit_vla_actions(actions)

    def _fit_vla_actions(self, actions: torch.Tensor) -> torch.Tensor:
        B, H, D = actions.shape

        if H < self.vla_chunk_size:
            pad = torch.zeros((B, self.vla_chunk_size - H, D), device=self.device)
            actions = torch.cat([actions, pad], dim=1)
        elif H > self.vla_chunk_size:
            actions = actions[:, : self.vla_chunk_size, :]

        if B == 1:
            actions = actions.expand(self.num_envs, -1, -1)
        elif B < self.num_envs:
            repeats = (self.num_envs + B - 1) // B
            actions = actions.repeat(repeats, 1, 1)[: self.num_envs]
        elif B > self.num_envs:
            actions = actions[: self.num_envs]

        act_dim_env = self.action_space.shape[-1]
        if actions.shape[-1] < act_dim_env:
            pad = torch.zeros(
                (self.num_envs, self.vla_chunk_size, act_dim_env - actions.shape[-1]),
                device=self.device,
            )
            actions = torch.cat([actions, pad], dim=-1)
        elif actions.shape[-1] > act_dim_env:
            actions = actions[..., :act_dim_env]

        return actions

    def _zero_vla_actions(self) -> torch.Tensor:
        return torch.zeros((self.num_envs, self.vla_chunk_size, self.action_space.shape[-1]), device=self.device)

    def _get_vla_observations(self) -> Dict[str, Any]:
        if self.vla_model == "gr00t":
            return self._get_gr00t_observations()
        if self.vla_model == "pi05":
            return self._get_pi05_observations()
        if self.vla_model == "openvla":
            return self._get_openvla_observations()
        raise ValueError(f"Unsupported vla_model '{self.vla_model}'.")

    def _get_gr00t_observations(self) -> Dict[str, Any]:
        left_view, right_view, wrist_view = self._get_vla_images()
        eef_pos, eef_quat, gripper_qpos = self._get_vla_robot_state(np.float64)

        return {
            "video.left_view": np.expand_dims(left_view, axis=1),
            "video.right_view": np.expand_dims(right_view, axis=1),
            "video.wrist_view": np.expand_dims(wrist_view, axis=1),
            "state.eef_position": np.expand_dims(eef_pos, axis=1),
            "state.eef_quaternion": np.expand_dims(eef_quat, axis=1),
            "state.gripper_qpos": np.expand_dims(gripper_qpos, axis=1),
        }

    def _get_pi05_observations(self) -> Dict[str, Any]:
        left_view, right_view, wrist_view = self._get_vla_images()
        eef_pos, eef_quat, gripper_qpos = self._get_vla_robot_state(np.float32)
        state = np.concatenate([eef_pos, eef_quat, gripper_qpos], axis=-1)

        return {
            "left_image": left_view,
            "right_image": right_view,
            "wrist_image": wrist_view,
            "state": state,
        }

    def _get_openvla_observations(self) -> Dict[str, Any]:
        left_view, right_view, wrist_view = self._get_vla_images()
        eef_pos, eef_quat, gripper_qpos = self._get_vla_robot_state(np.float32)
        state = np.concatenate([eef_pos, eef_quat, gripper_qpos], axis=-1)

        observations = []
        for env_idx in range(self.num_envs):
            observations.append(
                {
                    "full_image": left_view[env_idx],
                    "wrist_image_left": wrist_view[env_idx],
                    "wrist_image_right": right_view[env_idx],
                    "state": state[env_idx],
                }
            )

        return {
            "observations": observations,
            "instruction": self._get_openvla_instruction(),
        }

    def _get_openvla_instruction(self) -> str:
        instruction = getattr(self.cfg, "vla_instruction", "")
        if instruction:
            return instruction

        task_name = getattr(self.cfg, "task_name", "")
        return {
            "peg_insert": "insert the peg into the hole",
            "gear_mesh": "mesh the gears together",
            "nut_thread": "thread the nut onto the bolt",
        }.get(task_name, "complete the manipulation task")

    def _get_vla_images(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            self._left_camera.data.output["rgb"].cpu().numpy().astype(np.uint8),
            self._right_camera.data.output["rgb"].cpu().numpy().astype(np.uint8),
            self._wrist_camera.data.output["rgb"].cpu().numpy().astype(np.uint8),
        )

    def _get_vla_robot_state(self, dtype: np.dtype) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            self.fingertip_midpoint_pos.cpu().numpy().astype(dtype),
            self.fingertip_midpoint_quat.cpu().numpy().astype(dtype),
            self.joint_pos[:, 7:9].cpu().numpy().astype(dtype),
        )
