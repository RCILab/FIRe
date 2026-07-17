# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from fire_lab.tasks.direct.base_line.factory.factory_tasks_cfg import FactoryTask, GearMesh, NutThread, PegInsert


@configclass
class FactoryPi05Task(FactoryTask):
    action_penalty_ee_scale: float = 0.0
    action_penalty_asset_scale: float = 0.001
    action_grad_penalty_scale: float = 0.1
    contact_penalty_scale: float = 0.05
    delay_until_ratio: float = 0.25
    contact_penalty_threshold_range = [5.0, 10.0]


@configclass
class FactoryPi05PegInsert(PegInsert, FactoryPi05Task):
    contact_penalty_scale: float = 0.01


@configclass
class FactoryPi05GearMesh(GearMesh, FactoryPi05Task):
    contact_penalty_scale: float = 0.0025


@configclass
class FactoryPi05NutThread(NutThread, FactoryPi05Task):
    contact_penalty_scale: float = 0.0025
 