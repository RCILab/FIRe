# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import os
from dataclasses import dataclass, field

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass
from isaaclab.sensors import TiledCameraCfg

from fire_lab.tasks.direct.base_line.factory.factory_env_cfg import OBS_DIM_CFG, STATE_DIM_CFG, ObsRandCfg, ObsRandCfg, CtrlCfg
from fire_lab.tasks.direct.base_line.factory.factory_tasks_cfg import ASSET_DIR
from fire_lab.tasks.direct.vla.gr00t.factory.factory_gr00t_tasks_cfg import FactoryGr00tTask, FactoryGr00tGearMesh, FactoryGr00tNutThread, FactoryGr00tPegInsert

OBS_DIM_CFG.update({"force_threshold": 1, "ft_force": 3})

STATE_DIM_CFG.update({"force_threshold": 1, "ft_force": 3})


@dataclass
class DemoSaveCfg:

    dataset_path: str = "/home/hyunho_RCI/datasets/gr00t-rl/factory"
    chunk_id: str = "chunk-000"

    video_dir: dict = field(init=False)
    data_dir: str = field(init=False)

    def __post_init__(self):
        self.video_dir = {
            "left_camera": os.path.join(self.dataset_path, "videos", self.chunk_id, "observation.images.left_view"),
            "right_camera": os.path.join(self.dataset_path, "videos", self.chunk_id, "observation.images.right_view"),
            "wrist_camera": os.path.join(self.dataset_path, "videos", self.chunk_id, "observation.images.wrist_view"),
        }
        self.data_dir = os.path.join(self.dataset_path, "data", self.chunk_id)

@dataclass
class DemoSavePegInsert(DemoSaveCfg):
    dataset_path: str = "/home/hyunho_RCI/datasets/gr00t-rl/factory/peg_insert"

@dataclass
class DemoSaveGearMesh(DemoSaveCfg):
    dataset_path: str = "/home/hyunho_RCI/datasets/gr00t-rl/factory/gear_mesh"

@dataclass
class DemoSaveNutThread(DemoSaveCfg):
    dataset_path: str = "/home/hyunho_RCI/datasets/gr00t-rl/factory/nut_thread"

@configclass
class FactoryGr00tEnvCfg(DirectRLEnvCfg):
    vla_model: str = "gr00t"
    vla_host: str = "localhost"
    vla_port: int = 5555
    vla_chunk_size: int = 16

    decimation = 8
    action_space = 6
    # num_*: will be overwritten to correspond to obs_order, state_order.
    observation_space = 21
    state_space = 72
    obs_order: list = [
        "fingertip_pos_rel_fixed",
        "fingertip_quat",
        "ee_linvel",
        "ee_angvel",
        "force_threshold",
        "ft_force",
    ]
    state_order: list = [
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
        "force_threshold",
        "ft_force",
    ]

    task_name: str = "peg_insert"  # peg_insert, gear_mesh, nut_thread
    task: FactoryGr00tTask = FactoryGr00tTask()
    obs_rand: ObsRandCfg = ObsRandCfg()
    ctrl: CtrlCfg = CtrlCfg()

    episode_length_s = 10.0  # Probably need to override.
    sim: SimulationCfg = SimulationCfg(
        device="cuda:0",
        dt=1 / 120,
        gravity=(0.0, 0.0, -9.81),
        physx=PhysxCfg(
            solver_type=1,
            max_position_iteration_count=192,  # Important to avoid interpenetration.
            max_velocity_iteration_count=1,
            bounce_threshold_velocity=0.2,
            friction_offset_threshold=0.01,
            friction_correlation_distance=0.00625,
            gpu_max_rigid_contact_count=2**23,
            gpu_max_rigid_patch_count=2**23,
            gpu_collision_stack_size=2**28,
            gpu_max_num_partitions=1,  # Important for stable simulation.
        ),
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=128, env_spacing=2.0)

    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ASSET_DIR}/franka_mimic.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=3666.0,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
                max_contact_impulse=1e32,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": 0.00871,
                "panda_joint2": -0.10368,
                "panda_joint3": -0.00794,
                "panda_joint4": -1.49139,
                "panda_joint5": -0.00083,
                "panda_joint6": 1.38774,
                "panda_joint7": 0.0,
                "panda_finger_joint2": 0.04,
            },
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        actuators={
            "panda_arm1": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-4]"],
                stiffness=0.0,
                damping=0.0,
                friction=0.0,
                armature=0.0,
                effort_limit_sim=87,
                velocity_limit_sim=124.6,
            ),
            "panda_arm2": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],
                stiffness=0.0,
                damping=0.0,
                friction=0.0,
                armature=0.0,
                effort_limit_sim=12,
                velocity_limit_sim=149.5,
            ),
            "panda_hand": ImplicitActuatorCfg(
                joint_names_expr=["panda_finger_joint[1-2]"],
                effort_limit_sim=40.0,
                velocity_limit_sim=0.04,
                stiffness=7500.0,
                damping=173.0,
                friction=0.1,
                armature=0.0,
            ),
        },
    )

    # Camera Config
    left_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/left_camera", # attach to the wrist camera
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.6, 0.6, 0.4), rot=(0.0, 0.0, -0.46175, -0.88701), convention="opengl"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
        width=256,
        height=256,
    )
    right_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/right_camera", # attach to the wrist camera
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.6, -0.6, 0.4), rot=(0.88701, 0.46175, 0.0, 0.0), convention="opengl"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
        width=256,
        height=256,
    )
    wrist_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/panda_hand/wrist_camera", # attach to the wrist camera
        offset=TiledCameraCfg.OffsetCfg(pos=(0.05, 0.0, 0.0), rot=(0.7032332, -0.0739128, -0.0739128, 0.7032332), convention="ros"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
        width=256,
        height=256,
    )

    is_demo_save: bool = False
    demo_save_cfg: DemoSaveCfg = DemoSaveCfg()


################## GR00T Env ##################
@configclass
class FactoryTaskPegInsertGr00tCfg(FactoryGr00tEnvCfg):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=2.0)
    task_name = "peg_insert"
    task = FactoryGr00tPegInsert()
    episode_length_s = 10.0

@configclass
class FactoryTaskGearMeshGr00tCfg(FactoryGr00tEnvCfg):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=2.0)
    task_name = "gear_mesh"
    task = FactoryGr00tGearMesh()
    episode_length_s = 20.0
    

@configclass
class FactoryTaskNutThreadGr00tCfg(FactoryGr00tEnvCfg):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=2.0)
    task_name = "nut_thread"
    task = FactoryGr00tNutThread()
    episode_length_s = 30.0
    


################# GR00T Demo Save Env ##################
@configclass
class FactoryTaskPegInsertGr00tDemoSaveCfg(FactoryTaskPegInsertGr00tCfg):
    obs_order: list = [
        "fingertip_pos_rel_fixed",
        "fingertip_quat",
        "ee_linvel",
        "ee_angvel",
    ]
    state_order: list = [
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
    ]
    episode_length_s = 6.0
    is_demo_save: bool = True
    demo_save_cfg: DemoSavePegInsert = DemoSavePegInsert()


@configclass
class FactoryTaskGearMeshGr00tDemoSaveCfg(FactoryTaskPegInsertGr00tDemoSaveCfg):
    episode_length_s = 10.0
    is_demo_save: bool = True
    demo_save_cfg: DemoSaveGearMesh = DemoSaveGearMesh()


@configclass
class FactoryTaskNutThreadGr00tDemoSaveCfg(FactoryTaskPegInsertGr00tDemoSaveCfg):
    episode_length_s = 25.0
    is_demo_save: bool = True
    demo_save_cfg: DemoSaveNutThread = DemoSaveNutThread()
