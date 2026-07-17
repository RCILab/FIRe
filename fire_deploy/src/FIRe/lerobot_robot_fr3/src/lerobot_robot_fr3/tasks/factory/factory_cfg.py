class CtrlCfg:
    ema_factor = 0.2
    
    pos_action_bounds = [0.05, 0.05, 0.05]
    rot_action_bounds = [1.0, 1.0, 1.0]

    pos_action_threshold = [0.02, 0.02, 0.02]
    rot_action_threshold = [0.097, 0.097, 0.097]


class FixedAssetCfg:
    usd_path: str = ""
    diameter: float = 0.0
    height: float = 0.0
    base_height: float = 0.0  # Used to compute held asset CoM.
    friction: float = 0.75
    mass: float = 0.05


class HeldAssetCfg:
    usd_path: str = ""
    diameter: float = 0.0  # Used for gripper width.
    height: float = 0.0
    friction: float = 0.75
    mass: float = 0.05


class RobotCfg:
    robot_usd: str = ""
    franka_fingerpad_length: float = 0.01  # 0.017608
    friction: float = 0.75


class FactoryTask:
    robot_cfg: RobotCfg = RobotCfg()
    name: str = ""
    duration_s = 5.0

    fixed_asset_cfg: FixedAssetCfg = FixedAssetCfg()
    held_asset_cfg: HeldAssetCfg = HeldAssetCfg()
    asset_size: float = 0.0

    # Robot
    hand_init_pos: list = [0.0, 0.0, 0.015]  # Relative to fixed asset tip.
    hand_init_pos_noise: list = [0.0, 0.0, 0.0]  # [0.02, 0.02, 0.01]
    hand_init_orn: list = [3.1416, 0, 2.356]
    hand_init_orn_noise: list = [0.0, 0.0, 0.0]  # [0.0, 0.0, 1.57]

    # Action
    unidirectional_rot: bool = False

    # Fixed Asset (applies to all tasks)
    fixed_asset_init_pos_noise: list = [0.0, 0.0, 0.0]  # [0.05, 0.05, 0.05]
    fixed_asset_init_orn_deg: float = 0.0
    fixed_asset_init_orn_range_deg: float = 0.0  # 360.0

    # Held Asset (applies to all tasks)
    held_asset_pos_noise: list = [0.0, 0.0, 0.0]  # [0.0, 0.006, 0.003]  # noise level of the held asset in gripper
    held_asset_rot_init: float = -90.0


class Peg8mm(HeldAssetCfg):
    diameter = 0.007986
    height = 0.050
    mass = 0.019


class Hole8mm(FixedAssetCfg):
    diameter = 0.0081
    height = 0.025
    base_height = 0.0


class PegInsert(FactoryTask):
    name = "peg_insert"
    fixed_asset_cfg = Hole8mm()
    held_asset_cfg = Peg8mm()
    asset_size = 8.0
    duration_s = 10.0

    # Robot
    hand_init_pos: list = [0.0, 0.0, 0.047]  # Relative to fixed asset tip.
    hand_init_pos_noise: list = [0.0, 0.0, 0.0]  # [0.02, 0.02, 0.01]
    hand_init_orn: list = [3.1416, 0.0, 0.0]
    hand_init_orn_noise: list = [0.0, 0.0, 0.0]  # [0.0, 0.0, 0.785]

    # Fixed Asset (applies to all tasks)
    fixed_asset_init_pos_noise: list = [0.0, 0.0, 0.0]  # [0.05, 0.05, 0.05]
    fixed_asset_init_orn_deg: float = 0.0
    fixed_asset_init_orn_range_deg: float = 360.0

    # Held Asset (applies to all tasks)
    held_asset_pos_noise: list = [0.0, 0.0, 0.0]  # [0.003, 0.0, 0.003]  # noise level of the held asset in gripper
    held_asset_rot_init: float = 0.0

    # Rewards
    keypoint_coef_baseline: list = [5, 4]
    keypoint_coef_coarse: list = [50, 2]
    keypoint_coef_fine: list = [100, 0]
    # Fraction of socket height.
    success_threshold: float = 0.04
    engage_threshold: float = 0.9


class GearBase(FixedAssetCfg):
    height = 0.02
    base_height = 0.005
    small_gear_base_offset = [5.075e-2, 0.0, 0.0]
    medium_gear_base_offset = [2.025e-2, 0.0, 0.0]
    large_gear_base_offset = [-3.025e-2, 0.0, 0.0]


class MediumGear(HeldAssetCfg):
    diameter = 0.03  # Used for gripper width.
    height: float = 0.03
    mass = 0.012


class GearMesh(FactoryTask):
    name = "gear_mesh"
    fixed_asset_cfg = GearBase()
    held_asset_cfg = MediumGear()
    duration_s = 20.0

    # Gears Asset
    add_flanking_gears = True
    add_flanking_gears_prob = 1.0

    # Robot
    hand_init_pos: list = [0.0, 0.0, 0.035]  # Relative to fixed asset tip.
    hand_init_pos_noise: list = [0.0, 0.0, 0.0]  # [0.02, 0.02, 0.01]
    hand_init_orn: list = [3.1416, 0, 0.0]
    hand_init_orn_noise: list = [0.0, 0.0, 0.0]  # [0.0, 0.0, 0.785]

    # Fixed Asset (applies to all tasks)
    fixed_asset_init_pos_noise: list = [0.0, 0.0, 0.0]  # [0.05, 0.05, 0.05]
    fixed_asset_init_orn_deg: float = 0.0
    fixed_asset_init_orn_range_deg: float = 15.0

    # Held Asset (applies to all tasks)
    held_asset_pos_noise: list = [0.0, 0.0, 0.0]  # [0.003, 0.0, 0.003]  # noise level of the held asset in gripper
    held_asset_rot_init: float = -90.0

    keypoint_coef_baseline: list = [5, 4]
    keypoint_coef_coarse: list = [50, 2]
    keypoint_coef_fine: list = [100, 0]
    # Fraction of gear peg height.
    success_threshold: float = 0.05
    engage_threshold: float = 0.9


class NutM16(HeldAssetCfg):
    diameter = 0.024
    height = 0.01
    mass = 0.03
    friction = 0.01  # Additive with the nut means friction is (-0.25 + 0.75)/2 = 0.25


class BoltM16(FixedAssetCfg):
    diameter = 0.024
    height = 0.025
    base_height = 0.01
    thread_pitch = 0.002


class NutThread(FactoryTask):
    name = "nut_thread"
    fixed_asset_cfg = BoltM16()
    held_asset_cfg = NutM16()
    asset_size = 16.0
    duration_s = 30.0

    # Robot
    hand_init_pos: list = [0.0, 0.0, 0.015]  # Relative to fixed asset tip.
    hand_init_pos_noise: list = [0.02, 0.02, 0.01]
    hand_init_orn: list = [3.1416, 0.0, 1.83]
    hand_init_orn_noise: list = [0.0, 0.0, 0.26]

    # Action
    unidirectional_rot: bool = True

    # Fixed Asset (applies to all tasks)
    fixed_asset_init_pos_noise: list = [0.05, 0.05, 0.05]
    fixed_asset_init_orn_deg: float = 120.0
    fixed_asset_init_orn_range_deg: float = 30.0

    # Held Asset (applies to all tasks)
    held_asset_pos_noise: list = [0.0, 0.003, 0.003]  # noise level of the held asset in gripper
    held_asset_rot_init: float = -90.0


class FactoryEnvCfg:
    action_space = 6
    observation_space = 19
    # Fixed-asset position (robot base frame). Override per real-world setup, or at
    # runtime via the FIRE_FIXED_ASSET_POS env var. See Factory._resolve_fixed_pos.
    fixed_asset_pos: list = [0.6, 0.0, 0.05]
    obs_order: list = [
        "fingertip_pos_rel_fixed",
        "fingertip_quat",
        "ee_linvel",
        "ee_angvel",
        "prev_actions"
    ]

    task_name: str = "peg_insert"  # peg_insert, gear_mesh, nut_thread
    task: FactoryTask = FactoryTask()
    ctrl: CtrlCfg = CtrlCfg()


class FactoryTaskPegInsertCfg(FactoryEnvCfg):
    task_name = "peg_insert"
    task = PegInsert()


class FactoryTaskGearMeshCfg(FactoryEnvCfg):
    task_name = "gear_mesh"
    task = GearMesh()


class FactoryTaskNutThreadCfg(FactoryEnvCfg):
    task_name = "nut_thread"
    task = NutThread()
