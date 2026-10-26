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

"""Run the Bota FT sensor driver for a sensor mounted on one of the arms here.

Built on Bota's own packages rather than copies of them: the driver node comes
from bota_driver (../../extern/bota_driver_ros2) and its default configuration
from bota_driver_example (../../extern/bota_driver_ros2_example).

What differs from bota_driver_example's launch of the same name: only the
driver starts. The arm's robot description already carries the sensor's links
(fire_description, through this package's bringups), so a second
robot_state_publisher and RViz would conflict with the bringup. For Bota's standalone views use their
launches directly, e.g. `ros2 launch bota_driver_example plotjuggler.launch.py`.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    arguments = [
        DeclareLaunchArgument(
            'bota_ft_sensor_link_name', default_value='bota_ft_sensor',
            description='Sensor link prefix; also the driver node name'),
        DeclareLaunchArgument(
            'config_file',
            default_value=PathJoinSubstitution([
                FindPackageShare('bota_driver_example'), 'bota_config', 'bota_binary.json']),
            description='Bota driver JSON configuration (default: Bota\'s serial example)'),
        DeclareLaunchArgument(
            'output_rate', default_value='500.0',
            description='Driver output rate [Hz]'),
    ]
    driver = Node(
        package='bota_driver',
        executable='bota_driver_node',
        output='screen',
        parameters=[{
            'node_name': LaunchConfiguration('bota_ft_sensor_link_name'),
            'config_file': LaunchConfiguration('config_file'),
            'output_rate': ParameterValue(LaunchConfiguration('output_rate'), value_type=float),
        }],
    )
    return LaunchDescription(arguments + [driver])
