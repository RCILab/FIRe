# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch

from isaaclab.utils import configclass
from isaaclab.utils.noise import NoiseCfg, NoiseModelCfg

from firelab.tasks.fire.vla_env_cfg import (
    FIReVLAGearMeshCfg,
    FIReVLANutThreadCfg,
    FIReVLAPegInsertCfg,
)

RESIDUAL_ACTION_SCALE = [1.0, 1.0, 1.0, 0.0873, 0.0873, 1.0]


def per_dim_gaussian_noise(data: torch.Tensor, cfg: "PerDimGaussianNoiseCfg") -> torch.Tensor:
    std = torch.as_tensor(cfg.std, dtype=data.dtype, device=data.device)
    return data + std * torch.randn_like(data)


@configclass
class PerDimGaussianNoiseCfg(NoiseCfg):
    func = per_dim_gaussian_noise
    std: list = [0.0]


def combine_noise() -> NoiseModelCfg:
    return NoiseModelCfg(noise_cfg=PerDimGaussianNoiseCfg(std=[0.02, 0.02, 0.02, 0.0, 0.0, 0.0, 0.0]))


@configclass
class FIRePegInsertCfg(FIReVLAPegInsertCfg):
    vla_model: str = "gr00t"
    vla_host: str = "localhost"
    vla_port: int = 0
    vla_chunk_size: int = 0
    vla_prompt: str = "Insert peg into the socket"
    vla_clip_m: float = 0.10
    residual_action_scale: list = RESIDUAL_ACTION_SCALE
    action_noise_model: NoiseModelCfg = combine_noise()
    zero_residual: bool = False
    full_ee_angvel: bool = True


@configclass
class FIReGearMeshCfg(FIReVLAGearMeshCfg):
    vla_model: str = "gr00t"
    vla_host: str = "localhost"
    vla_port: int = 0
    vla_chunk_size: int = 0
    vla_prompt: str = "Insert and mesh gear into the base with other gears"
    vla_clip_m: float = 0.10
    residual_action_scale: list = RESIDUAL_ACTION_SCALE
    action_noise_model: NoiseModelCfg = combine_noise()
    zero_residual: bool = False
    full_ee_angvel: bool = True


@configclass
class FIReNutThreadCfg(FIReVLANutThreadCfg):
    vla_model: str = "gr00t"
    vla_host: str = "localhost"
    vla_port: int = 0
    vla_chunk_size: int = 0
    vla_prompt: str = "Thread the nut onto the bolt"
    vla_clip_m: float = 0.10
    residual_action_scale: list = RESIDUAL_ACTION_SCALE
    action_noise_model: NoiseModelCfg = combine_noise()
    zero_residual: bool = False
    full_ee_angvel: bool = True


@configclass
class FIRePegInsertVLAOnlyCfg(FIRePegInsertCfg):
    zero_residual: bool = True
    action_noise_model: NoiseModelCfg | None = None


@configclass
class FIReGearMeshVLAOnlyCfg(FIReGearMeshCfg):
    zero_residual: bool = True
    action_noise_model: NoiseModelCfg | None = None


@configclass
class FIReNutThreadVLAOnlyCfg(FIReNutThreadCfg):
    zero_residual: bool = True
    action_noise_model: NoiseModelCfg | None = None
