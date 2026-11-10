# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.sensors import TiledCameraCfg
import isaaclab.sim as sim_utils

from firelab.tasks.fire.tasks_cfg import FIReTask, FIReGearMeshTask, FIReNutThreadTask, FIRePegInsertTask
from firelab.tasks.factory.factory_env_cfg import OBS_DIM_CFG, STATE_DIM_CFG
from firelab.tasks.fire.base_env_cfg import FIReBaseEnvCfg


OBS_DIM_CFG.update({"force_threshold": 1, "ft_force": 3})

STATE_DIM_CFG.update({"force_threshold": 1, "ft_force": 3})


@configclass
class FIReVLAEnvCfg(FIReBaseEnvCfg):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=2.0)
    task: FIReTask = FIReTask()

    left_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/left_camera",
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
        prim_path="/World/envs/env_.*/right_camera",
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
        prim_path="/World/envs/env_.*/Robot/panda_hand/wrist_camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.05, 0.0, 0.0), rot=(0.7032332, -0.0739128, -0.0739128, 0.7032332), convention="ros"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
        width=256,
        height=256,
    )


@configclass
class FIReVLAPegInsertCfg(FIReVLAEnvCfg):
    task_name = "peg_insert"
    task = FIRePegInsertTask()
    episode_length_s = 10.0

@configclass
class FIReVLAGearMeshCfg(FIReVLAEnvCfg):
    task_name = "gear_mesh"
    task = FIReGearMeshTask()
    episode_length_s = 20.0


@configclass
class FIReVLANutThreadCfg(FIReVLAEnvCfg):
    task_name = "nut_thread"
    task = FIReNutThreadTask()
    episode_length_s = 30.0
