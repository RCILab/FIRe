# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

################## Pi05 Env ##################
gym.register(
    id="FireLab-VLA-Pi05-Factory-PegInsert-Direct-v1",
    entry_point=f"{__name__}.factory_pi05_env:FactoryPi05Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.factory_pi05_env_cfg:FactoryTaskPegInsertPi05Cfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg_peg_insert.yaml",
    },
)
gym.register(
    id="FireLab-VLA-Pi05-Factory-GearMesh-Direct-v1",
    entry_point=f"{__name__}.factory_pi05_env:FactoryPi05Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.factory_pi05_env_cfg:FactoryTaskGearMeshPi05Cfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg_gear_mesh.yaml",
    },
)
gym.register(
    id="FireLab-VLA-Pi05-Factory-NutThread-Direct-v1",
    entry_point=f"{__name__}.factory_pi05_env:FactoryPi05Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.factory_pi05_env_cfg:FactoryTaskNutThreadPi05Cfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg_nut_thread.yaml",
    },
)
