# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from fire_lab.tasks.direct.vla.pi05.factory import agents

##
# Register Gym environments.
##

################## OpenVLA Env ##################
gym.register(
    id="FireLab-VLA-OpenVLA-Factory-PegInsert-Direct-v1",
    entry_point=f"{__name__}.factory_openvla_env:FactoryOpenVLAEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.factory_openvla_env_cfg:FactoryTaskPegInsertOpenVLACfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg_peg_insert.yaml",
    },
)
gym.register(
    id="FireLab-VLA-OpenVLA-Factory-GearMesh-Direct-v1",
    entry_point=f"{__name__}.factory_openvla_env:FactoryOpenVLAEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.factory_openvla_env_cfg:FactoryTaskGearMeshOpenVLACfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg_gear_mesh.yaml",
    },
)
gym.register(
    id="FireLab-VLA-OpenVLA-Factory-NutThread-Direct-v1",
    entry_point=f"{__name__}.factory_openvla_env:FactoryOpenVLAEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.factory_openvla_env_cfg:FactoryTaskNutThreadOpenVLACfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg_nut_thread.yaml",
    },
)
