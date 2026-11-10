# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, Dict

import numpy as np
import torch

import isaacsim.core.utils.torch as torch_utils

from isaaclab.utils.math import axis_angle_from_quat

from firelab.tasks.factory import factory_utils
from firelab.tasks.fire.vla_env import FIReVLAEnv

from .eef_frame_utils import EEF_POS_SCALE, YAW_UNIT, cont_yaw, euler, quat_from_rotvec, to_frame, wrap_pi
from .fire_env_cfg import FIRePegInsertCfg


class EefFrameVLAClient:
    def __init__(self, client, env: "FIReEnv"):
        self._client = client
        self._env = env
        self._pending_eef = None

    @staticmethod
    def _eef_from_obs(obs: Dict[str, Any]):
        if "observations" in obs:
            state = np.stack([np.asarray(o["state"], dtype=np.float64) for o in obs["observations"]])
            return state[:, 0:3].copy(), state[:, 3:7].copy()
        if "observation/state" in obs:
            state = np.asarray(obs["observation/state"], dtype=np.float64)
            return state[:, 0:3].copy(), state[:, 3:7].copy()
        return (np.asarray(obs["state.eef_position"])[:, 0].copy(), np.asarray(obs["state.eef_quaternion"])[:, 0].copy())

    def request_action(self, obs):
        self._pending_eef = self._eef_from_obs(obs)
        return self._client.request_action(obs)

    def get_result(self, *args, **kwargs):
        return self._env._receive_chunk(self._client.get_result(*args, **kwargs), self._pending_eef)

    def get_action_sync(self, obs, *args, **kwargs):
        eef = self._eef_from_obs(obs)
        return self._env._receive_chunk(self._client.get_action_sync(obs, *args, **kwargs), eef)

    def __getattr__(self, name):
        return getattr(self._client, name)


class FIReEnv(FIReVLAEnv):
    cfg: FIRePegInsertCfg

    def __init__(self, cfg: FIRePegInsertCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        assert all(abs(b - EEF_POS_SCALE) < 1e-12 for b in self.cfg.ctrl.pos_action_bounds)
        self.vla_clip = self.cfg.vla_clip_m / EEF_POS_SCALE if self.cfg.vla_clip_m > 0 else None
        self.quat_bolt_to_ee = torch_utils.quat_from_euler_xyz(
            roll=torch.full((self.num_envs,), np.pi, device=self.device),
            pitch=torch.zeros(self.num_envs, device=self.device),
            yaw=torch.zeros(self.num_envs, device=self.device),
        )
        self.force_sensor_idx = self._robot.body_names.index("force_sensor")
        self.vla_policy = EefFrameVLAClient(self.vla_policy, self)
        print(f"[FIRe] 6-DoF EEF-frame {self.vla_model} VLA (translation clipped to "
              f"+-{self.cfg.vla_clip_m} m) + residual (scale {self.cfg.residual_action_scale})")
        if self.cfg.zero_residual:
            print("[FIRe] residual action replaced by 0 (VLA only)")

    def _receive_chunk(self, result: Dict[str, Any], eef) -> Dict[str, Any]:
        out = dict(result)
        dpos = np.asarray(result["action.eef_position_delta"])
        if self.vla_clip is not None:
            dpos = np.clip(dpos, -self.vla_clip, self.vla_clip)
        out["action.eef_position_delta"] = dpos
        out["action.eef_rotation_delta"] = np.asarray(result["action.eef_rotation_delta"]).copy()
        ref_quat = torch.as_tensor(eef[1], dtype=torch.float32, device=self.device)
        out["ref_pos"] = torch.as_tensor(eef[0], dtype=torch.float32, device=self.device)
        out["ref_quat"] = ref_quat
        out["ref_yaw"] = cont_yaw(ref_quat)
        out["ref_frame"] = (self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise).clone()
        return out

    def _bolt_yaw(self, quat: torch.Tensor) -> torch.Tensor:
        return euler(torch_utils.quat_mul(torch_utils.quat_conjugate(self.quat_bolt_to_ee), quat))[:, 2]

    def _vla_target(self, vla: torch.Tensor):
        g = self.vla_chunk
        q_ref = g["ref_quat"]
        pos = g["ref_pos"] + EEF_POS_SCALE * torch_utils.quat_apply(q_ref, vla[:, 0:3])
        quat = torch_utils.quat_mul(q_ref, torch_utils.quat_from_euler_xyz(roll=vla[:, 3], pitch=vla[:, 4], yaw=vla[:, 5]))
        return pos, quat, self._unwrap_yaw(quat, g["ref_yaw"] + vla[:, 5])

    def _unwrap_yaw(self, quat: torch.Tensor, yaw_ref: torch.Tensor) -> torch.Tensor:
        return yaw_ref + wrap_pi(self._bolt_yaw(quat) - yaw_ref)

    def _rel_rot(self, q_t: torch.Tensor, quat: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
        rpy = euler(torch_utils.quat_mul(torch_utils.quat_conjugate(q_t), quat))
        dyaw = yaw - cont_yaw(q_t)
        return torch.cat([rpy[:, 0:2], (dyaw + wrap_pi(rpy[:, 2] - dyaw)).unsqueeze(-1)], dim=-1)

    def _to_fixed_frame_action(self, pos, quat, yaw_ref, extra):
        g = self.vla_chunk
        rpy = euler(torch_utils.quat_mul(torch_utils.quat_conjugate(self.quat_bolt_to_ee), quat))
        yaw = yaw_ref + wrap_pi(rpy[:, 2] - yaw_ref)
        return torch.cat(
            [
                (pos - g["ref_frame"]) / EEF_POS_SCALE,
                rpy[:, 0:2],
                ((yaw + np.pi) / YAW_UNIT - 1.0).unsqueeze(-1),
                extra,
            ],
            dim=-1,
        )

    @staticmethod
    def _res_quat(r: torch.Tensor) -> torch.Tensor:
        tilt = quat_from_rotvec(torch.cat([r[:, 0:2], torch.zeros_like(r[:, 2:3])], dim=-1))
        zero = torch.zeros_like(r[:, 2])
        return torch_utils.quat_mul(tilt, torch_utils.quat_from_euler_xyz(roll=zero, pitch=zero, yaw=r[:, 2]))

    def _compose(self, vla: torch.Tensor, res: torch.Tensor):
        p_t = self.fingertip_midpoint_pos
        q_t = self.fingertip_midpoint_quat
        q_t_inv = torch_utils.quat_conjugate(q_t)

        p_v, q_v, yaw_v = self._vla_target(vla)
        a_vla = torch.cat([to_frame(q_t, p_v - p_t), self._rel_rot(q_t, q_v, yaw_v)], dim=-1)
        a_res = torch.cat([EEF_POS_SCALE * res[:, 0:3], res[:, 3:5], res[:, 5:6] * YAW_UNIT], dim=-1)

        p_r = p_t + torch_utils.quat_apply(q_t, a_vla[:, 0:3] + a_res[:, 0:3])
        q_r = torch_utils.quat_mul(
            torch_utils.quat_mul(q_t, self._res_quat(a_res[:, 3:6])), torch_utils.quat_mul(q_t_inv, q_v)
        )
        yaw_r = self._unwrap_yaw(q_r, yaw_v + a_res[:, 5])
        a = torch.cat([a_vla[:, 0:3] + a_res[:, 0:3], self._rel_rot(q_t, q_r, yaw_r)], dim=-1)

        fixed = self._to_fixed_frame_action(p_r, q_r, yaw_r, vla[:, 6:] + res[:, 6:])
        return fixed, (a_vla, a_res, a)

    def _current_vla(self) -> torch.Tensor:
        return self.vla_chunk_actions[:, self.episode_length_buf[0] % self.vla_chunk_size, :]

    def _set_chunk_actions(self):
        g = self.vla_chunk
        chunk = torch.cat(
            (
                torch.tensor(g["action.eef_position_delta"], dtype=torch.float32, device=self.device),
                torch.tensor(g["action.eef_rotation_delta"], dtype=torch.float32, device=self.device),
            ),
            dim=-1,
        )
        if self.action_space.shape[-1] != chunk.shape[-1]:
            chunk = torch.cat((chunk, torch.zeros((self.num_envs, self.vla_chunk_size, 1), device=self.device)), dim=-1)
        self.vla_chunk_actions = chunk

    def step(self, action: torch.Tensor):
        if self.cfg.zero_residual:
            action = torch.zeros_like(action)
        k = self.episode_length_buf[0].item()
        if k in (0, 1):
            eef = (
                self.fingertip_midpoint_pos.cpu().numpy().astype(np.float64),
                self.fingertip_midpoint_quat.cpu().numpy().astype(np.float64),
            )
            if k == 0:
                hold = np.zeros((self.num_envs, self.vla_chunk_size, 3))
                self.vla_chunk = self._receive_chunk(
                    {"action.eef_position_delta": hold, "action.eef_rotation_delta": hold.copy()}, eef
                )
            else:
                self.vla_chunk = self.vla_policy.get_action_sync(self._get_vla_observations())
                self._set_chunk_actions()
        return super().step(action)

    def _pre_physics_step(self, action):
        vla = self._current_vla()
        fixed, (a_vla, a_res, a) = self._compose(vla, action - vla)
        if self.record_actions and "vla" in self.recorded_actions:
            p_v, q_v, yaw_v = self._vla_target(vla)
            self.recorded_actions["vla_fixed"] = self._to_fixed_frame_action(p_v, q_v, yaw_v, vla[:, 6:])
            self.recorded_actions["final_fixed"] = fixed.clone()
            self.recorded_actions["vla_eef"] = a_vla.clone()
            self.recorded_actions["res_eef"] = a_res.clone()
            self.recorded_actions["final_eef"] = a.clone()
        return super()._pre_physics_step(fixed)

    def _apply_action(self):
        if self.last_update_timestamp < self._robot._data._sim_timestamp:
            self._compute_intermediate_values(dt=self.physics_dt)

        pos_actions = self.actions[:, 0:3] @ torch.diag(torch.tensor(self.cfg.ctrl.pos_action_bounds, device=self.device))
        rot_actions = self.actions[:, 3:6] @ torch.diag(torch.tensor(self.cfg.ctrl.rot_action_bounds, device=self.device))

        fixed_pos_action_frame = self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise
        ctrl_target_fingertip_preclipped_pos = fixed_pos_action_frame + pos_actions
        rot_actions[:, 2] = np.deg2rad(-180.0) + np.deg2rad(270.0) * (rot_actions[:, 2] + 1.0) / 2.0
        bolt_frame_quat = torch_utils.quat_from_euler_xyz(roll=rot_actions[:, 0], pitch=rot_actions[:, 1], yaw=rot_actions[:, 2])
        rot_180 = torch.tensor([np.pi, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        quat_bolt_to_ee = torch_utils.quat_from_euler_xyz(roll=rot_180[:, 0], pitch=rot_180[:, 1], yaw=rot_180[:, 2])
        ctrl_target_fingertip_preclipped_quat = torch_utils.quat_mul(quat_bolt_to_ee, bolt_frame_quat)

        self.delta_pos = ctrl_target_fingertip_preclipped_pos - self.fingertip_midpoint_pos
        pos_error_clipped = torch.clip(self.delta_pos, -self.pos_threshold, self.pos_threshold)
        ctrl_target_fingertip_midpoint_pos = self.fingertip_midpoint_pos + pos_error_clipped

        curr_roll, curr_pitch, curr_yaw = torch_utils.get_euler_xyz(self.fingertip_midpoint_quat)
        desired_roll, desired_pitch, desired_yaw = torch_utils.get_euler_xyz(ctrl_target_fingertip_preclipped_quat)
        desired_xyz = torch.stack([desired_roll, desired_pitch, desired_yaw], dim=1)

        curr_yaw = factory_utils.wrap_yaw(curr_yaw)
        desired_yaw = factory_utils.wrap_yaw(desired_yaw)
        self.delta_yaw = desired_yaw - curr_yaw
        clipped_yaw = torch.clip(self.delta_yaw, -self.rot_threshold[:, 2], self.rot_threshold[:, 2])
        desired_xyz[:, 2] = curr_yaw + clipped_yaw

        desired_roll = torch.where(desired_roll < 0.0, desired_roll + 2 * torch.pi, desired_roll)
        desired_pitch = torch.where(desired_pitch < 0.0, desired_pitch + 2 * torch.pi, desired_pitch)
        delta_roll = desired_roll - curr_roll
        clipped_roll = torch.clip(delta_roll, -self.rot_threshold[:, 0], self.rot_threshold[:, 0])
        desired_xyz[:, 0] = curr_roll + clipped_roll
        curr_pitch = torch.where(curr_pitch > torch.pi, curr_pitch - 2 * torch.pi, curr_pitch)
        desired_pitch = torch.where(desired_pitch > torch.pi, desired_pitch - 2 * torch.pi, desired_pitch)
        delta_pitch = desired_pitch - curr_pitch
        clipped_pitch = torch.clip(delta_pitch, -self.rot_threshold[:, 1], self.rot_threshold[:, 1])
        desired_xyz[:, 1] = curr_pitch + clipped_pitch

        ctrl_target_fingertip_midpoint_quat = torch_utils.quat_from_euler_xyz(
            roll=desired_xyz[:, 0], pitch=desired_xyz[:, 1], yaw=desired_xyz[:, 2]
        )
        self.generate_ctrl_signals(
            ctrl_target_fingertip_midpoint_pos=ctrl_target_fingertip_midpoint_pos,
            ctrl_target_fingertip_midpoint_quat=ctrl_target_fingertip_midpoint_quat,
            ctrl_target_gripper_dof_pos=0.0,
        )

    def _compute_intermediate_values(self, dt):
        super()._compute_intermediate_values(dt)
        noise = torch_utils.quat_from_angle_axis(
            torch.randn((self.num_envs,), device=self.device) * np.deg2rad(self.cfg.obs_rand.fingertip_rot_deg),
            torch.nn.functional.normalize(torch.randn((self.num_envs, 3), device=self.device), dim=-1),
        )
        q = torch_utils.quat_mul(self.fingertip_midpoint_quat, noise)
        self.noisy_fingertip_quat = q * torch.sign(q[:, 0:1] + 1e-9)

    def randomize_initial_state(self, env_ids):
        super().randomize_initial_state(env_ids)
        self.prev_obs_fingertip_quat = None

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        q_flip = torch.tensor([0.0, 1.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        rpy = euler(torch_utils.quat_mul(torch_utils.quat_conjugate(q_flip), self.fingertip_midpoint_quat))
        rot_bounds = torch.tensor(self.cfg.ctrl.rot_action_bounds[0:2], device=self.device)
        self.actions[env_ids, 3:5] = rpy[env_ids, 0:2] / rot_bounds

    def _policy_rate_angvel(self):
        q = self.noisy_fingertip_quat
        prev = getattr(self, "prev_obs_fingertip_quat", None)
        self.prev_obs_fingertip_quat = q.clone()
        if prev is None:
            return torch.zeros((self.num_envs, 3), device=self.device)
        rot_diff = torch_utils.quat_mul(q, torch_utils.quat_conjugate(prev))
        rot_diff = rot_diff * torch.sign(rot_diff[:, 0:1] + 1e-9)
        return axis_angle_from_quat(rot_diff) / self.step_dt

    def _prev_target_rel(self) -> torch.Tensor:
        pos = self.actions[:, 0:3] @ torch.diag(torch.tensor(self.cfg.ctrl.pos_action_bounds, device=self.device))
        rot = self.actions[:, 3:6] @ torch.diag(torch.tensor(self.cfg.ctrl.rot_action_bounds, device=self.device))
        p_target = self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise + pos
        yaw = -np.pi + YAW_UNIT * (rot[:, 2] + 1.0)
        q_target = torch_utils.quat_mul(
            self.quat_bolt_to_ee, torch_utils.quat_from_euler_xyz(roll=rot[:, 0], pitch=rot[:, 1], yaw=yaw)
        )
        q = self.noisy_fingertip_quat
        return torch.cat(
            [to_frame(q, p_target - self.noisy_fingertip_pos) / EEF_POS_SCALE, self._rel_rot(q, q_target, yaw)], dim=-1
        )

    def _get_observations(self):
        if self.cfg.full_ee_angvel:
            self.ee_angvel_fd = self._policy_rate_angvel()
        out = super()._get_observations()
        obs = out["policy"]
        assert obs.shape[1] == 24
        q = obs[:, 3:7]
        obs[:, 0:3] = to_frame(q, obs[:, 0:3])
        obs[:, 7:10] = to_frame(q, obs[:, 7:10])
        obs[:, 10:13] = to_frame(q, obs[:, 10:13])
        force_w = torch_utils.quat_apply(self._robot.data.body_quat_w[:, self.force_sensor_idx], obs[:, 13:16])
        obs[:, 13:16] = to_frame(self.fingertip_midpoint_quat, force_w)
        prev = torch.cat([self._prev_target_rel(), self.actions[:, 6:]], dim=-1)
        obs[:, -prev.shape[1]:] = prev
        out["critic"][:, -prev.shape[1]:] = prev
        return out
