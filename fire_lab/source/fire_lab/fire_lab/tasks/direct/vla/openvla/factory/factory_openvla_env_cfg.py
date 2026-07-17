# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from fire_lab.tasks.direct.vla.pi05.factory.factory_pi05_env_cfg import (
    FactoryPi05EnvCfg,
)
from fire_lab.tasks.direct.vla.pi05.factory.factory_pi05_tasks_cfg import (
    FactoryPi05GearMesh,
    FactoryPi05NutThread,
    FactoryPi05PegInsert,
)


@configclass
class FactoryOpenVLAEnvCfg(FactoryPi05EnvCfg):
    vla_model: str = "openvla"
    vla_host: str = "127.0.0.1"
    vla_port: int = 8778
    vla_chunk_size: int = 8
    vla_instruction: str = ""


@configclass
class FactoryTaskPegInsertOpenVLACfg(FactoryOpenVLAEnvCfg):
    vla_instruction: str = "insert the peg into the hole"
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=32, env_spacing=2.0)
    task_name = "peg_insert"
    task = FactoryPi05PegInsert()
    episode_length_s = 10.0


@configclass
class FactoryTaskGearMeshOpenVLACfg(FactoryOpenVLAEnvCfg):
    vla_instruction: str = "mesh the gears together"
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=32, env_spacing=2.0)
    task_name = "gear_mesh"
    task = FactoryPi05GearMesh()
    episode_length_s = 20.0


@configclass
class FactoryTaskNutThreadOpenVLACfg(FactoryOpenVLAEnvCfg):
    vla_instruction: str = "thread the nut onto the bolt"
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=32, env_spacing=2.0)
    task_name = "nut_thread"
    task = FactoryPi05NutThread()
    episode_length_s = 30.0
