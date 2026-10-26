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

"""The FR3 with the Bota SensONE in Isaac Sim: cho_bringup_franka's Isaac bringup, on fire_description's robot.

    ros2 launch fire_bringup bringup_isaac_robot.launch.py control_mode:=torque action_chunk:=true

Every argument of the core's bringup_isaac_robot.launch.py is passed through as
given; this sets xacro_file and robot_usd to fire_description's robot (its USD is
built once, fire_description usd/README.md), payload_file to the hand with the
sensor, and ft_link to bota_ft_sensor_wrench, so Isaac publishes the load on the
sensor's wrench frame on /isaac/ft_raw. scripts/isaac_ft_sensor.py turns that into
the real driver's /bota_ft_sensor/wrench and serves /bota_ft_sensor/tare, so a task
runs the same here as on the robot.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

CORE_LAUNCH = os.path.join(
    get_package_share_directory('cho_bringup_franka'), 'launch', 'bringup_isaac_robot.launch.py')


def generate_launch_description():
    description_share = get_package_share_directory('fire_description')
    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(CORE_LAUNCH),
            launch_arguments={
                'xacro_file': os.path.join(description_share, 'urdf', 'fr3_with_ft_sensor',
                                           'fr3_franka_hand.urdf'),
                'robot_usd': os.path.join(description_share, 'usd', 'fr3_with_ft_sensor',
                                          'fr3_franka_hand', 'fr3_franka_hand.usda'),
                'payload_file': os.path.join(
                    get_package_share_directory('fire_bringup'), 'config', 'payload_ft_sensor.yaml'),
                'ft_link': 'bota_ft_sensor_wrench',
            }.items()),
        Node(
            package='fire_bringup',
            executable='isaac_ft_sensor.py',
            output='screen',
            parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time', default='true')}],
        ),
    ])
