# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from firelab.tasks.factory.factory_tasks_cfg import FactoryTask


@configclass
class FIReBaseTask(FactoryTask):
    contact_penalty_threshold_range = [5.0, 10.0]
