# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

################## GR00T Env ##################
gym.register(
    id="FireLab-VLA-Gr00t-Factory-PegInsert-Direct-v1",
    entry_point=f"{__name__}.factory_gr00t_env:FactoryGr00tEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.factory_gr00t_env_cfg:FactoryTaskPegInsertGr00tCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg_peg_insert.yaml",
    },
)
gym.register(
    id="FireLab-VLA-Gr00t-Factory-GearMesh-Direct-v1",
    entry_point=f"{__name__}.factory_gr00t_env:FactoryGr00tEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.factory_gr00t_env_cfg:FactoryTaskGearMeshGr00tCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg_gear_mesh.yaml",
    },
)
gym.register(
    id="FireLab-VLA-Gr00t-Factory-NutThread-Direct-v1",
    entry_point=f"{__name__}.factory_gr00t_env:FactoryGr00tEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.factory_gr00t_env_cfg:FactoryTaskNutThreadGr00tCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg_nut_thread.yaml",
    },
)

################## GR00T Demo Save Env ##################
gym.register(
    id="FireLab-VLA-Gr00t-Factory-PegInsert-Demo-Save-Direct-v0",
    entry_point=f"{__name__}.factory_gr00t_demo_save_env:FactoryGr00tDemoSaveEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.factory_gr00t_env_cfg:FactoryTaskPegInsertGr00tDemoSaveCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg_peg_insert.yaml",
    },
)
gym.register(
    id="FireLab-VLA-Gr00t-Factory-GearMesh-Demo-Save-Direct-v0",
    entry_point=f"{__name__}.factory_gr00t_demo_save_env:FactoryGr00tDemoSaveEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.factory_gr00t_env_cfg:FactoryTaskGearMeshGr00tDemoSaveCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg_gear_mesh.yaml",
    },
)
gym.register(
    id="FireLab-VLA-Gr00t-Factory-NutThread-Demo-Save-Direct-v0",
    entry_point=f"{__name__}.factory_gr00t_demo_save_env:FactoryGr00tDemoSaveEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.factory_gr00t_env_cfg:FactoryTaskNutThreadGr00tDemoSaveCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg_nut_thread.yaml",
    },
)
