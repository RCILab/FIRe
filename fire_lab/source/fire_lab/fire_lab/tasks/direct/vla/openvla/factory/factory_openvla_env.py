# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from fire_lab.tasks.direct.vla.pi05.factory.factory_pi05_env import FactoryPi05Env

from .factory_openvla_env_cfg import FactoryOpenVLAEnvCfg


class FactoryOpenVLAEnv(FactoryPi05Env):
    cfg: FactoryOpenVLAEnvCfg
