# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
from dataclasses import dataclass, field

from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.sensors import TiledCameraCfg
import isaaclab.sim as sim_utils

# from fire_lab.tasks.direct.vla.pi05.forge.forge_pi05_tasks_cfg import ForgePi05Task, ForgePi05GearMesh, ForgePi05NutThread, ForgePi05PegInsert
from fire_lab.tasks.direct.base_line.factory.factory_env_cfg import OBS_DIM_CFG, STATE_DIM_CFG
from fire_lab.tasks.direct.base_line.forge.forge_env_cfg import ForgeEnvCfg
from fire_lab.tasks.direct.vla.pi05.forge.forge_pi05_tasks_cfg import ForgePi05Task, ForgePi05GearMesh, ForgePi05NutThread, ForgePi05PegInsert


OBS_DIM_CFG.update({"force_threshold": 1, "ft_force": 3})

STATE_DIM_CFG.update({"force_threshold": 1, "ft_force": 3})


@configclass
class ForgePi05EnvCfg(ForgeEnvCfg):
    vla_model: str = "pi05"
    vla_host: str = "127.0.0.1"
    vla_port: int = 8000
    vla_chunk_size: int = 8

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=2.0)
    task: ForgePi05Task = ForgePi05Task()

    # Camera Config
    left_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/left_camera",  # attach to the wrist camera
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.6, 0.6, 0.4), rot=(0.0, 0.0, -0.46175, -0.88701), convention="opengl"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
        width=256,
        height=256,
    )
    right_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/right_camera", # attach to the wrist camera
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.6, -0.6, 0.4), rot=(0.88701, 0.46175, 0.0, 0.0), convention="opengl"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
        width=256,
        height=256,
    )
    wrist_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/panda_hand/wrist_camera", # attach to the wrist camera
        offset=TiledCameraCfg.OffsetCfg(pos=(0.05, 0.0, 0.0), rot=(0.7032332, -0.0739128, -0.0739128, 0.7032332), convention="ros"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
        width=256,
        height=256,
    )

    is_demo_save: bool = False


################## GR00T Env ##################
@configclass
class ForgeTaskPegInsertPi05Cfg(ForgePi05EnvCfg):
    task_name = "peg_insert"
    task = ForgePi05PegInsert()
    episode_length_s = 10.0


@configclass
class ForgeTaskGearMeshPi05Cfg(ForgePi05EnvCfg):
    task_name = "gear_mesh"
    task = ForgePi05GearMesh()
    episode_length_s = 20.0


@configclass
class ForgeTaskNutThreadPi05Cfg(ForgePi05EnvCfg):
    task_name = "nut_thread"
    task = ForgePi05NutThread()
    episode_length_s = 30.0
