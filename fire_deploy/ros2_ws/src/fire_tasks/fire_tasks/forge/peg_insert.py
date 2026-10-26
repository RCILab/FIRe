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

PEG_INSERT_FRANKA_BASE_ORIENTATION = [1.0, 0.0, 0.0, 0.0]
# peg_insert intentionally has no yaw noise: the peg is keyed and cannot rotate freely.
PEG_INSERT_FRANKA_YAW_RANGE = [0.0, 0.0]

PEG_INSERT_FRANKA_GRASP_PARAMS = dict(
    width=0.005,          # peg diameter [m]
    speed=0.05,            # closing speed [m/s]
    force=100.0,           # firm hold for a small, light peg [N]
    epsilon_inner=0.005,   # grasp-success tolerance [m]
    epsilon_outer=0.005,
)


def _peg_insert_approach_position(x_offset, y_offset):
    return [0.6 + x_offset, 0.0 + y_offset, 0.05 + 0.025 + 0.047]


def create_franka_peg_insert_tree(robot_config=None):
    return build_forge_tree(
        robot_config=robot_config,
        task_label="Peg_Insert",
        approach_position_fn=_peg_insert_approach_position,
        base_orientation=PEG_INSERT_FRANKA_BASE_ORIENTATION,
        yaw_range=PEG_INSERT_FRANKA_YAW_RANGE,
        grasp_params=PEG_INSERT_FRANKA_GRASP_PARAMS,
        # The peg is light and this task never consumes FT data, so skip the tare
        # (and its settle wait). gear_mesh/nut_thread keep it.
        tare_ft_sensor=False,
    )
