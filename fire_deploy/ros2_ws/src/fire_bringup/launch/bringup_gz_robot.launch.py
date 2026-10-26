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

"""The FR3 with the Bota SensONE in Gazebo: cho_bringup_franka's Gazebo bringup, the sensor mounted.

    ros2 launch fire_bringup bringup_gz_robot.launch.py control_mode:=torque

Every argument of the core's bringup_gz_robot.launch.py is passed through as given;
this sets the sensor's ee_connection() xacro, the link the hand hangs off and the
payload, as bringup_real_robot.launch.py does, and adds this description's share
to Gazebo's resource path, which resolves the sensor's mesh.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import AppendEnvironmentVariable, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

CORE_LAUNCH = os.path.join(
    get_package_share_directory('cho_bringup_franka'), 'launch', 'bringup_gz_robot.launch.py')


def generate_launch_description():
    description_share = get_package_share_directory('fire_description')
    return LaunchDescription([
        AppendEnvironmentVariable('GZ_SIM_RESOURCE_PATH', os.path.dirname(description_share)),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(CORE_LAUNCH),
            launch_arguments={
                'ee_connection_xacro': os.path.join(description_share, 'xacro', 'bota_ft_sensor.xacro'),
                'special_connection': 'bota_ft_sensor_wrench',
                'payload_file': os.path.join(
                    get_package_share_directory('fire_bringup'), 'config', 'payload_ft_sensor.yaml'),
            }.items()),
    ])
