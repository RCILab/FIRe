#!/usr/bin/env python3
# Copyright (c) 2026 Hyunho Cho
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Present Isaac Sim's simulated joint reaction as the Bota FT sensor.

Isaac publishes the raw 6D load carried by bota_ft_sensor_wrench's incoming
joint as an unstamped geometry_msgs/Wrench on /isaac/ft_raw (the core Isaac
bringup's ft_link:=; it cannot build a stamped message without rclpy -- see
cho_simulation_isaac's run_isaac_sim.py). This node does the ROS-side half:

  * stamps it and republishes as geometry_msgs/WrenchStamped on the same topic
    the real driver uses, `<sensor_name>/wrench`
    (bota_driver_ros2's src/bota_driver_node.cpp:84),
  * serves `<sensor_name>/tare`, the std_srvs/Trigger that fire_tasks'
    TareFTSensorServiceBehavior calls at the start of the forge tasks.
    Without it those trees fail on an unavailable service.

Tare latches the current reading as the zero offset, which is what the real
sensor's tare does. The real one averages for ~3 s; here the value is noise-free,
so a single sample is equivalent.
"""

import rclpy
from geometry_msgs.msg import Wrench, WrenchStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_srvs.srv import Trigger


class IsaacFTSensor(Node):

    def __init__(self):
        super().__init__('isaac_ft_sensor')
        self.declare_parameter('raw_topic', '/isaac/ft_raw')
        self.declare_parameter('sensor_name', 'bota_ft_sensor')
        self.declare_parameter('frame_id', 'bota_ft_sensor_wrench')

        raw_topic = self.get_parameter('raw_topic').get_parameter_value().string_value
        sensor_name = self.get_parameter('sensor_name').get_parameter_value().string_value
        self._frame_id = self.get_parameter('frame_id').get_parameter_value().string_value

        self._offset = [0.0] * 6
        self._last = [0.0] * 6
        self._seen = False

        wrench_topic = f'{sensor_name}/wrench'
        tare_service = f'/{sensor_name}/tare'

        self._publisher = self.create_publisher(WrenchStamped, wrench_topic, 10)
        self.create_subscription(Wrench, raw_topic, self._on_raw, 10)
        self.create_service(Trigger, tare_service, self._on_tare)

        self.get_logger().info(
            f"Isaac FT sensor: {raw_topic} -> {wrench_topic} (frame '{self._frame_id}'), "
            f"tare on {tare_service}"
        )

    def _on_raw(self, msg):
        self._last = [
            msg.force.x, msg.force.y, msg.force.z,
            msg.torque.x, msg.torque.y, msg.torque.z,
        ]
        self._seen = True

        out = WrenchStamped()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = self._frame_id
        out.wrench.force.x = self._last[0] - self._offset[0]
        out.wrench.force.y = self._last[1] - self._offset[1]
        out.wrench.force.z = self._last[2] - self._offset[2]
        out.wrench.torque.x = self._last[3] - self._offset[3]
        out.wrench.torque.y = self._last[4] - self._offset[4]
        out.wrench.torque.z = self._last[5] - self._offset[5]
        self._publisher.publish(out)

    def _on_tare(self, request, response):
        if not self._seen:
            response.success = False
            response.message = 'no wrench received from Isaac yet'
            self.get_logger().warn('tare requested before any reading arrived')
            return response
        self._offset = list(self._last)
        response.success = True
        response.message = 'tared'
        self.get_logger().info(f'tared at {self._offset}')
        return response


def main(args=None):
    rclpy.init(args=args)
    node = IsaacFTSensor()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        # Stopping the launch: rclpy's signal handler shuts the context down, and spin() reports that
        # as ExternalShutdownException, or as KeyboardInterrupt when a SIGINT reaches Python first.
        pass
    except RuntimeError:
        # rclpy's RCLError, from a wait set or publish on a context that was shut down between two
        # cycles. Only that case is a normal exit; with the context alive it is a real error.
        if rclpy.ok():
            raise
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
