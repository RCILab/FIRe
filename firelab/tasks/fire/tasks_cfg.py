# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from firelab.tasks.factory.factory_tasks_cfg import FactoryTask, GearMesh, NutThread, PegInsert


@configclass
class FIReTask(FactoryTask):
    contact_penalty_scale: float = 0.0025
    force_shaping_lambda: float = 0.01
    force_potential_c: float = 1.0
    contact_penalty_threshold_range = [5.0, 10.0]


@configclass
class FIRePegInsertTask(PegInsert, FIReTask):
    pass


@configclass
class FIReGearMeshTask(GearMesh, FIReTask):
    pass


@configclass
class FIReNutThreadTask(NutThread, FIReTask):
    pass
