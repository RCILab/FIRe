# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Baseline and ablation variants of the GR00T FIRe environment.

All variants share the FIRe environment (observation processing, FORGE action space,
domain randomization, number of environments) and differ only in the listed item:

    NoVla        VLA base action replaced by zeros, cameras removed ("RL FIRe-rew")
    NoForce      force removed from actor and critic inputs and from the reward ("FIRe w/o force")
    ObsOnly      force kept in the inputs, force-aware reward removed ("FIRe obs-only")
    RewOnly      force removed from the actor input only, force-aware reward kept
    PenaltyOnly  shaping term r_sh removed (force_shaping_lambda = 0)
    ShapingOnly  direct penalty r_p removed (contact_penalty_scale = 0)
"""

from isaaclab.utils import configclass

from fire_lab.tasks.direct.vla.gr00t.forge.forge_gr00t_env_cfg import (
    ForgeTaskGearMeshGr00tCfg,
    ForgeTaskNutThreadGr00tCfg,
    ForgeTaskPegInsertGr00tCfg,
)
from fire_lab.tasks.direct.vla.gr00t.forge.forge_gr00t_tasks_cfg import (
    ForgeGr00tGearMesh,
    ForgeGr00tNutThread,
    ForgeGr00tPegInsert,
)

ACTOR_OBS_NO_FORCE = [
    "fingertip_pos_rel_fixed",
    "fingertip_quat",
    "ee_linvel",
    "ee_angvel",
]

CRITIC_STATE_NO_FORCE = [
    "fingertip_pos",
    "fingertip_quat",
    "ee_linvel",
    "ee_angvel",
    "joint_pos",
    "held_pos",
    "held_pos_rel_fixed",
    "held_quat",
    "fixed_pos",
    "fixed_quat",
    "task_prop_gains",
    "ema_factor",
    "pos_threshold",
    "rot_threshold",
]


##
# Task (reward) variants.
##


@configclass
class ForgeGr00tPegInsertNoForceReward(ForgeGr00tPegInsert):
    contact_penalty_scale: float = 0.0
    force_shaping_lambda: float = 0.0


@configclass
class ForgeGr00tGearMeshNoForceReward(ForgeGr00tGearMesh):
    contact_penalty_scale: float = 0.0
    force_shaping_lambda: float = 0.0


@configclass
class ForgeGr00tNutThreadNoForceReward(ForgeGr00tNutThread):
    contact_penalty_scale: float = 0.0
    force_shaping_lambda: float = 0.0


@configclass
class ForgeGr00tPegInsertNoShaping(ForgeGr00tPegInsert):
    force_shaping_lambda: float = 0.0


@configclass
class ForgeGr00tGearMeshNoShaping(ForgeGr00tGearMesh):
    force_shaping_lambda: float = 0.0


@configclass
class ForgeGr00tNutThreadNoShaping(ForgeGr00tNutThread):
    force_shaping_lambda: float = 0.0


@configclass
class ForgeGr00tPegInsertNoPenalty(ForgeGr00tPegInsert):
    contact_penalty_scale: float = 0.0


@configclass
class ForgeGr00tGearMeshNoPenalty(ForgeGr00tGearMesh):
    contact_penalty_scale: float = 0.0


@configclass
class ForgeGr00tNutThreadNoPenalty(ForgeGr00tNutThread):
    contact_penalty_scale: float = 0.0


##
# NoVla: from-scratch RL with the FIRe reward.
##


@configclass
class ForgeTaskPegInsertGr00tNoVlaCfg(ForgeTaskPegInsertGr00tCfg):
    use_vla: bool = False


@configclass
class ForgeTaskGearMeshGr00tNoVlaCfg(ForgeTaskGearMeshGr00tCfg):
    use_vla: bool = False


@configclass
class ForgeTaskNutThreadGr00tNoVlaCfg(ForgeTaskNutThreadGr00tCfg):
    use_vla: bool = False


##
# NoForce: no force in the inputs or the reward.
##


@configclass
class ForgeTaskPegInsertGr00tNoForceCfg(ForgeTaskPegInsertGr00tCfg):
    task = ForgeGr00tPegInsertNoForceReward()
    obs_order: list = ACTOR_OBS_NO_FORCE
    state_order: list = CRITIC_STATE_NO_FORCE


@configclass
class ForgeTaskGearMeshGr00tNoForceCfg(ForgeTaskGearMeshGr00tCfg):
    task = ForgeGr00tGearMeshNoForceReward()
    obs_order: list = ACTOR_OBS_NO_FORCE
    state_order: list = CRITIC_STATE_NO_FORCE


@configclass
class ForgeTaskNutThreadGr00tNoForceCfg(ForgeTaskNutThreadGr00tCfg):
    task = ForgeGr00tNutThreadNoForceReward()
    obs_order: list = ACTOR_OBS_NO_FORCE
    state_order: list = CRITIC_STATE_NO_FORCE


##
# ObsOnly: force in the inputs, no force-aware reward.
##


@configclass
class ForgeTaskPegInsertGr00tObsOnlyCfg(ForgeTaskPegInsertGr00tCfg):
    task = ForgeGr00tPegInsertNoForceReward()


@configclass
class ForgeTaskGearMeshGr00tObsOnlyCfg(ForgeTaskGearMeshGr00tCfg):
    task = ForgeGr00tGearMeshNoForceReward()


@configclass
class ForgeTaskNutThreadGr00tObsOnlyCfg(ForgeTaskNutThreadGr00tCfg):
    task = ForgeGr00tNutThreadNoForceReward()


##
# RewOnly: force-aware reward, no force in the actor input (the critic keeps it).
##


@configclass
class ForgeTaskPegInsertGr00tRewOnlyCfg(ForgeTaskPegInsertGr00tCfg):
    obs_order: list = ACTOR_OBS_NO_FORCE


@configclass
class ForgeTaskGearMeshGr00tRewOnlyCfg(ForgeTaskGearMeshGr00tCfg):
    obs_order: list = ACTOR_OBS_NO_FORCE


@configclass
class ForgeTaskNutThreadGr00tRewOnlyCfg(ForgeTaskNutThreadGr00tCfg):
    obs_order: list = ACTOR_OBS_NO_FORCE


##
# PenaltyOnly / ShapingOnly: one of the two force-aware reward terms.
##


@configclass
class ForgeTaskPegInsertGr00tPenaltyOnlyCfg(ForgeTaskPegInsertGr00tCfg):
    task = ForgeGr00tPegInsertNoShaping()


@configclass
class ForgeTaskGearMeshGr00tPenaltyOnlyCfg(ForgeTaskGearMeshGr00tCfg):
    task = ForgeGr00tGearMeshNoShaping()


@configclass
class ForgeTaskNutThreadGr00tPenaltyOnlyCfg(ForgeTaskNutThreadGr00tCfg):
    task = ForgeGr00tNutThreadNoShaping()


@configclass
class ForgeTaskPegInsertGr00tShapingOnlyCfg(ForgeTaskPegInsertGr00tCfg):
    task = ForgeGr00tPegInsertNoPenalty()


@configclass
class ForgeTaskGearMeshGr00tShapingOnlyCfg(ForgeTaskGearMeshGr00tCfg):
    task = ForgeGr00tGearMeshNoPenalty()


@configclass
class ForgeTaskNutThreadGr00tShapingOnlyCfg(ForgeTaskNutThreadGr00tCfg):
    task = ForgeGr00tNutThreadNoPenalty()
