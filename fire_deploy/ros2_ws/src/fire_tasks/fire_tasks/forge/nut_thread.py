# Copyright 2026 Hyunho Cho
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from fire_tasks.forge.common import build_forge_tree

# base orientation [0.6099, 0.7927, 0, 0] (qx, qy, qz, qw) = hand_init_orn [pi, 0, 1.83]
NUT_THREAD_FRANKA_BASE_ORIENTATION = [0.6099, 0.7927, 0.0, 0.0]
# yaw random noise: rotate about the end-effector's local Z axis, applied in the EE frame.
# nut_thread uses a narrower range than gear_mesh: hand_init_orn_noise yaw = 0.26 rad.
NUT_THREAD_FRANKA_YAW_RANGE = [-0.13, 0.13]

NUT_THREAD_FRANKA_GRASP_PARAMS = dict(
    width=0.024,            # nut across-flats width [m]
    speed=0.03,              # closing speed [m/s]
    force=100.0,             # firm hold to resist threading torque [N]
    epsilon_inner=0.005,     # grasp-success tolerance [m]
    epsilon_outer=0.005,
)


def _nut_thread_approach_position(x_offset, y_offset):
    # nut_thread: x = 0.6 (center axis, no offset)
    #             z = fixed_root(0.05) + tip_offset(height 0.025 + base 0.01 = 0.035) + hand_z(0.015)
    return [0.6 + x_offset, 0.0 + y_offset, 0.05 + 0.035 + 0.015 + 0.01]


def create_franka_nut_thread_tree(robot_config=None):
    return build_forge_tree(
        robot_config=robot_config,
        task_label="Nut_Thread",
        approach_position_fn=_nut_thread_approach_position,
        base_orientation=NUT_THREAD_FRANKA_BASE_ORIENTATION,
        yaw_range=NUT_THREAD_FRANKA_YAW_RANGE,
        grasp_params=NUT_THREAD_FRANKA_GRASP_PARAMS,
    )
