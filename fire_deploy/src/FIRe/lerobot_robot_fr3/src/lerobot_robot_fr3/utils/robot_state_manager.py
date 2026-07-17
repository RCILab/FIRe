import threading

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, Twist, PoseStamped, TwistStamped
from typing import Dict
import numpy as np
from rclpy.callback_groups import ReentrantCallbackGroup


class RobotStateManager:
    def __init__(self, node: Node) -> None:
        self.node = node

        self._joint_states = JointState()
        self._ee_pose = Pose()
        self._ee_vel = Twist()
        self._state_lock = threading.Lock()

        self.cb_group = ReentrantCallbackGroup()

        self._joint_states_sub = self.node.create_subscription(
            JointState, "/joint_states", self._update_joint_states, 10,
            callback_group=self.cb_group
        )
        self._ee_pose_sub = self.node.create_subscription(
            PoseStamped, "/ee_state/pose", self._update_ee_pose, 10,
            callback_group=self.cb_group
        )
        self._ee_twist_sub = self.node.create_subscription(
            TwistStamped, "/ee_state/twist", self._update_ee_vel, 10,
            callback_group=self.cb_group
        )

    @property
    def joint_states(self) -> Dict[str, np.ndarray]:
        # Snapshot the message reference so all three arrays come from the same update.
        with self._state_lock:
            snapshot = self._joint_states
        joint_positions = np.array(snapshot.position, dtype=np.float32)
        joint_velocities = np.array(snapshot.velocity, dtype=np.float32)
        joint_efforts = np.array(snapshot.effort, dtype=np.float32)
        return {
            "position": joint_positions[0:7],
            "velocity": joint_velocities[0:7],
            "effort":   joint_efforts[0:7]
        }
    
    @property
    def gripper_joint_states(self) -> Dict[str, np.ndarray]:
        # Snapshot the message reference so all three arrays come from the same update.
        with self._state_lock:
            snapshot = self._joint_states
        joint_positions = np.array(snapshot.position, dtype=np.float32)
        joint_velocities = np.array(snapshot.velocity, dtype=np.float32)
        joint_efforts = np.array(snapshot.effort, dtype=np.float32)
        return {
            "position": np.array([joint_positions[7], joint_positions[7]], dtype=np.float32),
            "velocity": np.array([joint_velocities[7], joint_velocities[7]], dtype=np.float32),
            "effort":   np.array([joint_efforts[7], joint_efforts[7]], dtype=np.float32),
        }

    @property
    def ee_pose(self) -> np.ndarray:
        """return np.array([x, y, z, qw, qx, qy, qz])"""
        # Snapshot so position and orientation come from the same message.
        with self._state_lock:
            snapshot = self._ee_pose
        pos = snapshot.position
        quat = snapshot.orientation
        return np.array([
            pos.x, pos.y, pos.z,
            quat.w, quat.x, quat.y, quat.z
        ], dtype=np.float32)

    @property
    def ee_vel(self) -> np.ndarray:
        """return np.array([lx, ly, lz, ax, ay, az])"""
        # Snapshot so linear and angular velocities come from the same message.
        with self._state_lock:
            snapshot = self._ee_vel
        lv = snapshot.linear
        av = snapshot.angular
        return np.array([
            lv.x, lv.y, lv.z,
            av.x, av.y, av.z
        ], dtype=np.float32)

    @property
    def ee_pos(self) -> np.ndarray:
        """return np.array([x, y, z])"""
        return self.ee_pose[0:3]

    @property
    def ee_quat(self) -> np.ndarray:
        """return np.array([qw, qx, qy, qz])"""
        return self.ee_pose[3:7]

    @property
    def ee_linvel(self) -> np.ndarray:
        """return np.array([lx, ly, lz])"""
        return self.ee_vel[0:3]

    @property
    def ee_angvel(self) -> np.ndarray:
        """return np.array([ax, ay, az])"""
        return self.ee_vel[3:6]
    
    @property
    def gripper_qpos(self) -> np.ndarray:
        return self.gripper_joint_states["position"]
    
    @property
    def gripper_qvel(self) -> np.ndarray:
        return self.gripper_joint_states["velocity"]

    def _update_joint_states(self, msg: JointState):
        with self._state_lock:
            self._joint_states = msg

    def _update_ee_pose(self, msg: PoseStamped):
        with self._state_lock:
            self._ee_pose = msg.pose

    def _update_ee_vel(self, msg: TwistStamped):
        with self._state_lock:
            self._ee_vel = msg.twist