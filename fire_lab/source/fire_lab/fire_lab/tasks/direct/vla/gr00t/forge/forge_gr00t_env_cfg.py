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

from fire_lab.tasks.direct.vla.gr00t.forge.forge_gr00t_tasks_cfg import ForgeGr00tTask, ForgeGr00tGearMesh, ForgeGr00tNutThread, ForgeGr00tPegInsert
from fire_lab.tasks.direct.base_line.factory.factory_env_cfg import OBS_DIM_CFG, STATE_DIM_CFG
from fire_lab.tasks.direct.base_line.forge.forge_env_cfg import ForgeEnvCfg


OBS_DIM_CFG.update({"force_threshold": 1, "ft_force": 3})

STATE_DIM_CFG.update({"force_threshold": 1, "ft_force": 3})


@dataclass
class DemoSaveCfg:

    dataset_path: str = "/home/hyunho_RCI/datasets/gr00t-rl/forge/peg_insert"
    chunk_id: str = "chunk-000"

    video_dir: dict = field(init=False)
    data_dir: str = field(init=False)

    def __post_init__(self):
        self.video_dir = {
            "left_camera": os.path.join(self.dataset_path, "videos", self.chunk_id, "observation.images.left_view"),
            "right_camera": os.path.join(self.dataset_path, "videos", self.chunk_id, "observation.images.right_view"),
            "wrist_camera": os.path.join(self.dataset_path, "videos", self.chunk_id, "observation.images.wrist_view"),
        }
        self.data_dir = os.path.join(self.dataset_path, "data", self.chunk_id)

@dataclass
class DemoSavePegInsert(DemoSaveCfg):
    dataset_path: str = "/home/hyunho_RCI/datasets/gr00t-rl/forge/peg_insert"

@dataclass
class DemoSaveGearMesh(DemoSaveCfg):
    dataset_path: str = "/home/hyunho_RCI/datasets/gr00t-rl/forge/gear_mesh"

@dataclass
class DemoSaveNutThread(DemoSaveCfg):
    dataset_path: str = "/home/hyunho_RCI/datasets/gr00t-rl/forge/nut_thread"


@configclass
class ForgeGr00tEnvCfg(ForgeEnvCfg):
    vla_model: str = "gr00t"
    vla_host: str = "localhost"
    vla_port: int = 5555
    vla_chunk_size: int = 16

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=2.0)
    task: ForgeGr00tTask = ForgeGr00tTask()

    # Camera Config
    left_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/left_camera", # attach to the wrist camera
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
    demo_save_cfg: DemoSaveCfg = DemoSaveCfg()


################## GR00T Env ##################
@configclass
class ForgeTaskPegInsertGr00tCfg(ForgeGr00tEnvCfg):
    task_name = "peg_insert"
    task = ForgeGr00tPegInsert()
    episode_length_s = 10.0

@configclass
class ForgeTaskGearMeshGr00tCfg(ForgeGr00tEnvCfg):
    task_name = "gear_mesh"
    task = ForgeGr00tGearMesh()
    episode_length_s = 20.0
    

@configclass
class ForgeTaskNutThreadGr00tCfg(ForgeGr00tEnvCfg):
    task_name = "nut_thread"
    task = ForgeGr00tNutThread()
    episode_length_s = 30.0
    


################# GR00T Demo Save Env ##################
@configclass
class ForgeTaskPegInsertGr00tDemoSaveCfg(ForgeTaskPegInsertGr00tCfg):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=128, env_spacing=2.0)
    episode_length_s = 6.0
    is_demo_save: bool = True
    demo_save_cfg: DemoSavePegInsert = DemoSavePegInsert()


@configclass
class ForgeTaskGearMeshGr00tDemoSaveCfg(ForgeTaskGearMeshGr00tCfg):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=128, env_spacing=2.0)
    episode_length_s = 10.0
    is_demo_save: bool = True
    demo_save_cfg: DemoSaveGearMesh = DemoSaveGearMesh()


@configclass
class ForgeTaskNutThreadGr00tDemoSaveCfg(ForgeTaskNutThreadGr00tCfg):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=128, env_spacing=2.0)
    episode_length_s = 10.0
    is_demo_save: bool = True
    demo_save_cfg: DemoSaveNutThread = DemoSaveNutThread()
