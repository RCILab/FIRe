# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents
from .factory_env import FactoryEnv

##
# Register Gym environments.
##

gym.register(
    id="FireLab-BaseLine-Factory-PegInsert-Direct-v0",
    entry_point=f"{__name__}.factory_env:FactoryEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.factory_env_cfg:FactoryTaskPegInsertCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg_peg_insert.yaml",
    },
)

gym.register(
    id="FireLab-BaseLine-Factory-GearMesh-Direct-v0",
    entry_point=f"{__name__}.factory_env:FactoryEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.factory_env_cfg:FactoryTaskGearMeshCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg_gear_mesh.yaml",
    },
)

gym.register(
    id="FireLab-BaseLine-Factory-NutThread-Direct-v0",
    entry_point=f"{__name__}.factory_env:FactoryEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.factory_env_cfg:FactoryTaskNutThreadCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg_nut_thread.yaml",
    },
)
