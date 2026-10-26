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

"""The FR3 with the Bota SensONE in MuJoCo: cho_bringup_franka's MuJoCo bringup, on fire_description's robot.

    ros2 launch fire_bringup bringup_mujoco_robot.launch.py control_mode:=torque

Every argument of the core's bringup_mujoco_robot.launch.py is passed through as
given; this sets xacro_file to fire_description's one-file URDF, whose ros2_control
block names its MuJoCo model (the sensor's bodies included), and payload_file to
the hand with the sensor. MuJoCo publishes no wrench here.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

CORE_LAUNCH = os.path.join(
    get_package_share_directory('cho_bringup_franka'), 'launch', 'bringup_mujoco_robot.launch.py')


def generate_launch_description():
    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(CORE_LAUNCH),
            launch_arguments={
                'xacro_file': os.path.join(get_package_share_directory('fire_description'), 'urdf',
                                           'fr3_with_ft_sensor', 'fr3_franka_hand.urdf'),
                'payload_file': os.path.join(
                    get_package_share_directory('fire_bringup'), 'config', 'payload_ft_sensor.yaml'),
            }.items()),
    ])
