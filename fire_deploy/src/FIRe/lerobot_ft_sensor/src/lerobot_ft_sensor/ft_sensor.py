from __future__ import annotations

import threading
from typing import Dict

import numpy as np
from geometry_msgs.msg import WrenchStamped
from rclpy.node import Node

from lerobot_ft_sensor.configuration_ft_sensor import FTSensorConfig


class FTSensor:
    """Thin cache for force/torque data published on a ROS WrenchStamped topic."""

    def __init__(self, node: Node, config: FTSensorConfig) -> None:
        self.node = node
        self.config = config

        self._data: Dict[str, np.ndarray | float] = self._empty_sample()
        self._data_lock = threading.Lock()

        self._subscription = None
        self._is_connected = False
        self.is_initialized = False

    def connect(self) -> None:
        if self._is_connected:
            return

        self._subscription = self.node.create_subscription(
            WrenchStamped,
            self.config.wrench_topic,
            self._wrench_callback,
            self.config.queue_size,
        )
        self._is_connected = True
        self.node.get_logger().info(
            f"FT sensor subscribed to {self.config.wrench_topic}"
        )

    def disconnect(self) -> None:
        if not self._is_connected:
            return

        if self._subscription is not None:
            self.node.destroy_subscription(self._subscription)
            self._subscription = None
        self._is_connected = False

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def raw_data(self) -> Dict[str, np.ndarray | float]:
        return self.data

    @property
    def raw_force(self) -> np.ndarray:
        return self.force

    @property
    def raw_torque(self) -> np.ndarray:
        return self.torque

    @property
    def data(self) -> Dict[str, np.ndarray | float]:
        with self._data_lock:
            return self._copy_sample(self._data)

    @property
    def force(self) -> np.ndarray:
        with self._data_lock:
            return np.asarray(self._data["force"], dtype=np.float32).copy()

    @property
    def torque(self) -> np.ndarray:
        with self._data_lock:
            return np.asarray(self._data["torque"], dtype=np.float32).copy()

    @property
    def timestamp(self) -> float:
        with self._data_lock:
            return float(self._data["timestamp"])

    @staticmethod
    def _empty_sample() -> Dict[str, np.ndarray | float]:
        return {
            "force": np.zeros(3, dtype=np.float32),
            "torque": np.zeros(3, dtype=np.float32),
            "timestamp": 0.0,
        }

    @staticmethod
    def _copy_sample(sample: Dict[str, np.ndarray | float]) -> Dict[str, np.ndarray | float]:
        return {
            "force": np.asarray(sample["force"], dtype=np.float32).copy(),
            "torque": np.asarray(sample["torque"], dtype=np.float32).copy(),
            "timestamp": float(sample["timestamp"]),
        }

    @staticmethod
    def _sample_from_msg(msg: WrenchStamped) -> Dict[str, np.ndarray | float]:
        stamp = msg.header.stamp
        return {
            "force": np.array(
                [msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z],
                dtype=np.float32,
            ),
            "torque": np.array(
                [msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z],
                dtype=np.float32,
            ),
            "timestamp": float(stamp.sec) + float(stamp.nanosec) * 1e-9,
        }

    def _wrench_callback(self, msg: WrenchStamped) -> None:
        with self._data_lock:
            self._data = self._sample_from_msg(msg)
            self.is_initialized = True
