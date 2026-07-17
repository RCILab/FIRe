"""Config for the pick_place task.

VLA/teleop-only: no RL policy is deployed for this task. Task-space (EE pose)
control, matching the VLA canonical action space (state.eef_position /
state.eef_quaternion) used by Forge/Factory. Scene is described only for
documentation — no fixed object/target pose constants are wired since this
task is driven by Inverse3 teleop + GR00T recording, not a trained RL policy
observation.
"""


class PickPlaceCtrlCfg:
    ema_factor = 0.2

    pos_action_bounds = [0.05, 0.05, 0.05]
    rot_action_bounds = [1.0, 1.0, 1.0]

    pos_action_threshold = [0.02, 0.02, 0.02]
    rot_action_threshold = [0.097, 0.097, 0.097]


class PickPlaceTask:
    name: str = "pick_place"
    duration_s: float = 15.0

    # Descriptive only (pick up the green cube, place it on top of the white gear).
    object_name: str = "green_cube"
    target_name: str = "white_gear"

    gripper_open: float = 1.0
    gripper_close: float = -1.0


class PickPlaceEnvCfg:
    action_space: int = 6  # task-space pos delta (3) + rot delta (3); gripper is separate

    obs_order: list = [
        "fingertip_pos",
        "fingertip_quat",
        "ee_linvel",
        "ee_angvel",
        "prev_actions",
    ]

    task_name: str = "pick_place"
    task: PickPlaceTask = PickPlaceTask()
    ctrl: PickPlaceCtrlCfg = PickPlaceCtrlCfg()


class PickPlaceTaskCfg(PickPlaceEnvCfg):
    task_name = "pick_place"
    task = PickPlaceTask()
