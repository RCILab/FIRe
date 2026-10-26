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

# base orientation [1, 0, 0, 0] (qx, qy, qz, qw) = 180 deg about X -> EE faces down
GEAR_MESH_FRANKA_BASE_ORIENTATION = [1.0, 0.0, 0.0, 0.0]
# yaw random noise: rotate about the end-effector's local Z axis (the down-pointing
# axis when the EE faces down), so the noise is applied in the EE frame.
GEAR_MESH_FRANKA_YAW_RANGE = [-0.785, 0.785]

GEAR_MESH_FRANKA_GRASP_PARAMS = dict(
    width=0.028,           # gear grip width [m]
    speed=0.05,             # closing speed [m/s]
    force=100.0,            # firm hold for a heavier gear [N]
    epsilon_inner=0.005,    # grasp-success tolerance [m]
    epsilon_outer=0.005,
)


def _gear_mesh_approach_position(x_offset, y_offset):
    # gear_mesh: x = 0.6 + medium_gear_base_offset(0.02025)
    #            z = fixed_root(0.05) + tip_offset(height 0.02 + base 0.005 = 0.025) + hand_z(0.035)
    return [0.6 + 0.02025 + x_offset, 0.0 + y_offset, 0.05 + 0.025 + 0.035]


def create_franka_gear_mesh_tree(robot_config=None):
    return build_forge_tree(
        robot_config=robot_config,
        task_label="Gear_Mesh",
        approach_position_fn=_gear_mesh_approach_position,
        base_orientation=GEAR_MESH_FRANKA_BASE_ORIENTATION,
        yaw_range=GEAR_MESH_FRANKA_YAW_RANGE,
        grasp_params=GEAR_MESH_FRANKA_GRASP_PARAMS,
    )
