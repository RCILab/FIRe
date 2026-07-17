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
    id="FireLab-VLA-Pi05-Forge-PegInsert-Direct-v1",
    entry_point=f"{__name__}.forge_pi05_env:ForgePi05Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.forge_pi05_env_cfg:ForgeTaskPegInsertPi05Cfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg_peg_insert.yaml",
    },
)
gym.register(
    id="FireLab-VLA-Pi05-Forge-GearMesh-Direct-v1",
    entry_point=f"{__name__}.forge_pi05_env:ForgePi05Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.forge_pi05_env_cfg:ForgeTaskGearMeshPi05Cfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg_gear_mesh.yaml",
    },
)
gym.register(
    id="FireLab-VLA-Pi05-Forge-NutThread-Direct-v1",
    entry_point=f"{__name__}.forge_pi05_env:ForgePi05Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.forge_pi05_env_cfg:ForgeTaskNutThreadPi05Cfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg_nut_thread.yaml",
    },
)
