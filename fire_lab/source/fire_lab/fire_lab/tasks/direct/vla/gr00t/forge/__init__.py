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
    id="FireLab-VLA-Gr00t-Forge-PegInsert-Direct-v1",
    entry_point=f"{__name__}.forge_gr00t_env:ForgeGr00tEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.forge_gr00t_env_cfg:ForgeTaskPegInsertGr00tCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg_peg_insert.yaml",
    },
)
gym.register(
    id="FireLab-VLA-Gr00t-Forge-GearMesh-Direct-v1",
    entry_point=f"{__name__}.forge_gr00t_env:ForgeGr00tEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.forge_gr00t_env_cfg:ForgeTaskGearMeshGr00tCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg_gear_mesh.yaml",
    },
)
gym.register(
    id="FireLab-VLA-Gr00t-Forge-NutThread-Direct-v1",
    entry_point=f"{__name__}.forge_gr00t_env:ForgeGr00tEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.forge_gr00t_env_cfg:ForgeTaskNutThreadGr00tCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg_nut_thread.yaml",
    },
)

################## GR00T Demo Save Env ##################
gym.register(
    id="FireLab-VLA-Gr00t-Forge-PegInsert-Demo-Save-Direct-v0",
    entry_point=f"{__name__}.forge_gr00t_demo_save_env:ForgeGr00tDemoSaveEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.forge_gr00t_env_cfg:ForgeTaskPegInsertGr00tDemoSaveCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg_peg_insert.yaml",
    },
)
gym.register(
    id="FireLab-VLA-Gr00t-Forge-GearMesh-Demo-Save-Direct-v0",
    entry_point=f"{__name__}.forge_gr00t_demo_save_env:ForgeGr00tDemoSaveEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.forge_gr00t_env_cfg:ForgeTaskGearMeshGr00tDemoSaveCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg_gear_mesh.yaml",
    },
)
gym.register(
    id="FireLab-VLA-Gr00t-Forge-NutThread-Demo-Save-Direct-v0",
    entry_point=f"{__name__}.forge_gr00t_demo_save_env:ForgeGr00tDemoSaveEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.forge_gr00t_env_cfg:ForgeTaskNutThreadGr00tDemoSaveCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg_nut_thread.yaml",
    },
)
