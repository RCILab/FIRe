# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import torch

import isaacsim.core.utils.torch as torch_utils

EEF_POS_SCALE = 0.05
YAW_UNIT = np.deg2rad(135.0)


def wrap_pi(x: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(x), torch.cos(x))


def euler(quat: torch.Tensor) -> torch.Tensor:
    roll, pitch, yaw = torch_utils.get_euler_xyz(quat)
    return wrap_pi(torch.stack([roll, pitch, yaw], dim=-1))


def quat_to_yaw_action(quat: torch.Tensor) -> torch.Tensor:
    r = torch.tensor([-np.pi, 0.0, 0.0], device=quat.device).repeat(quat.shape[0], 1)
    unrot_180 = torch_utils.quat_from_euler_xyz(roll=r[:, 0], pitch=r[:, 1], yaw=r[:, 2])
    yaw = torch_utils.get_euler_xyz(torch_utils.quat_mul(unrot_180, quat))[-1]
    yaw = wrap_pi(yaw)
    yaw = torch.where(yaw > torch.pi / 2, yaw - 2 * torch.pi, yaw)
    return (yaw + np.deg2rad(180.0)) / np.deg2rad(270.0) * 2.0 - 1.0


def cont_yaw(quat: torch.Tensor) -> torch.Tensor:
    return -np.pi + YAW_UNIT * (quat_to_yaw_action(quat) + 1.0)


def quat_from_rotvec(v: torch.Tensor) -> torch.Tensor:
    angle = torch.linalg.norm(v, dim=-1, keepdim=True)
    half = 0.5 * angle
    k = torch.where(angle > 1e-6, torch.sin(half) / angle.clamp_min(1e-6), 0.5 - angle**2 / 48.0)
    return torch.cat([torch.cos(half), v * k], dim=-1)


def to_frame(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return torch_utils.quat_apply(torch_utils.quat_conjugate(q), v)
