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

"""The FR3 with the Bota SensONE on the real robot: cho_bringup_franka's real bringup, the sensor mounted.

    ros2 launch fire_bringup bringup_real_robot.launch.py control_mode:=torque \
        controller_name:=joint_space_impedance_controller
    ros2 launch fire_bringup bota_ft_sensor.launch.py      # the sensor's driver

Every argument of the core's bringup_real_robot.launch.py is passed through as given
(`ros2 launch cho_bringup_franka bringup_real_robot.launch.py --show-args`); this
sets the three that mount the sensor: fire_description's ee_connection() xacro,
the link the hand then hangs off, and the payload of the hand with the sensor
(config/payload_ft_sensor.yaml, 0.97 kg). Desk's end-effector load has to match
that payload.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

CORE_LAUNCH = os.path.join(
    get_package_share_directory('cho_bringup_franka'), 'launch', 'bringup_real_robot.launch.py')


def mount_arguments():
    """What the core bringup is given to put the sensor between the flange and the hand."""
    return {
        'ee_connection_xacro': os.path.join(
            get_package_share_directory('fire_description'), 'xacro', 'bota_ft_sensor.xacro'),
        'special_connection': 'bota_ft_sensor_wrench',
        'payload_file': os.path.join(
            get_package_share_directory('fire_bringup'), 'config', 'payload_ft_sensor.yaml'),
    }


def generate_launch_description():
    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(CORE_LAUNCH),
            launch_arguments=mount_arguments().items()),
    ])
