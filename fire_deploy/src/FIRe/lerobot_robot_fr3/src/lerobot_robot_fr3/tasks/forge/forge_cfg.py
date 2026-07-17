from ..factory.factory_cfg import (
    CtrlCfg, FactoryTask, FactoryEnvCfg,
    PegInsert, GearMesh, NutThread
)


class ForgeCtrlCfg(CtrlCfg):
    ema_factor_range = [0.025, 0.1]
    ema_factor = 0.0625
    default_task_prop_gains = [565.0, 565.0, 565.0, 28.0, 28.0, 28.0]
    task_prop_gains_noise_level = [0.41, 0.41, 0.41, 0.41, 0.41, 0.41]
    pos_threshold_noise_level = [0.25, 0.25, 0.25]
    rot_threshold_noise_level = [0.29, 0.29, 0.29]
    default_dead_zone = [5.0, 5.0, 5.0, 1.0, 1.0, 1.0]


class ForgeTask(FactoryTask):
    action_penalty_ee_scale: float = 0.0
    action_penalty_asset_scale: float = 0.001
    action_grad_penalty_scale: float = 0.1
    contact_penalty_scale: float = 0.05
    delay_until_ratio: float = 0.25
    contact_penalty_threshold_range = [5.0, 10.0]


class ForgePegInsert(PegInsert, ForgeTask):
    contact_penalty_scale: float = 0.2


class ForgeGearMesh(GearMesh, ForgeTask):
    contact_penalty_scale: float = 0.05


class ForgeNutThread(NutThread, ForgeTask):
    contact_penalty_scale: float = 0.05



class ForgeEnvCfg(FactoryEnvCfg):
    action_space: int = 7
    ctrl: ForgeCtrlCfg = ForgeCtrlCfg()
    task: ForgeTask = ForgeTask()

    ft_smoothing_factor: float = 0.25

    obs_order: list = [
        "fingertip_pos_rel_fixed",
        "fingertip_quat",
        "ee_linvel",
        "ee_angvel",
        "ft_force",
        "force_threshold",
        "prev_actions"
    ]


class ForgeTaskPegInsertCfg(ForgeEnvCfg):
    task_name = "peg_insert"
    task = ForgePegInsert()


class ForgeTaskGearMeshCfg(ForgeEnvCfg):
    task_name = "gear_mesh"
    task = ForgeGearMesh()


class ForgeTaskNutThreadCfg(ForgeEnvCfg):
    task_name = "nut_thread"
    task = ForgeNutThread()