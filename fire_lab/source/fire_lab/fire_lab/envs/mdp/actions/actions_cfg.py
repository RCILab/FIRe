# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.controllers import DifferentialIKControllerCfg, OperationalSpaceControllerCfg
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from . import (
    chunked_actions,
)

from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.envs.mdp.actions.actions_cfg import BinaryJointPositionActionCfg

##
# Task-space Actions.
##


@configclass
class DifferentialInverseKinematicsChunkedActionCfg(DifferentialInverseKinematicsActionCfg):
    """
    Configuration for inverse differential kinematics action term.
    """

    class_type: type[ActionTerm] = chunked_actions.DifferentialInverseKinematicsChunkedAction

    """The configuration for the differential IK controller."""
    vla_model_name: str = "gr00t"  # gr00t or pi05
    chunk_size: int = 16
    vla_server_port: int = 5555
    vla_only: bool = False


@configclass
class BinaryJointPositionChunkedActionCfg(BinaryJointPositionActionCfg):
    """
    Configuration for the binary joint position action term.
    """

    class_type: type[ActionTerm] = chunked_actions.BinaryJointPositionChunkedAction
