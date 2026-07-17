import threading
import time
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Tuple
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.signals import SignalHandlerOptions
from std_msgs.msg import Header, Float64MultiArray
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from std_srvs.srv import Trigger
from rclpy.parameter import Parameter

try:
    from cho_interfaces.action import VisionLanguageAction
    from cho_interfaces.msg import ActionChunk
except ImportError:
    raise ImportError(
        "cho_interfaces package is not found. Please make sure it is built and sourced properly. "
        "Please source your ROS 2 workspace."
    )

from lerobot.robots import Robot
from lerobot.processor import RobotAction, RobotObservation
from .config_fr3 import FR3RobotConfig
from lerobot_ft_sensor import FTSensor
from .utils import RobotStateManager, CameraSensorManager
from .utils.rotation_utils import wxyz2xyzw
from .tasks import Task
from . import create_task


@dataclass
class TeleopAction:
    action: RobotAction
    action_space: str
    is_relative: bool


class FR3Robot(Robot):
    config_class = FR3RobotConfig
    name = "fr3_robot"

    def __init__(
        self, config: FR3RobotConfig, task_name: str,
    ):
        super().__init__(config)
        self.config: FR3RobotConfig = config
        self.task: Task = create_task(task_name)

        self.node = None
        self.executor_thread = None
        self._executor = None
        self._is_connected = False
        
        # Action Client-related variables
        self._action_client = None
        self._goal_handle = None
        self._goal_accepted = False
        self._success_signaled = False
        
        # update robot state & can get robot state from this
        self.robot_state_manager = None
        self.camera_sensor_manager = None
        self.ft_sensor = None

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            return

        if not rclpy.ok():
            # Don't let rclpy install its own SIGINT handler: it would shut the
            # context down the instant Ctrl+C is pressed, before disconnect() can
            # send the success/cancel signals — which then crashed (invalid
            # context) and left subprocesses (e.g. the Inverse3 bridge) wedged.
            # With NO, Ctrl+C raises a normal KeyboardInterrupt we handle in the
            # control loop, and the context stays valid for a clean disconnect.
            rclpy.init(signal_handler_options=SignalHandlerOptions.NO)

        self.node = rclpy.create_node(
            'lerobot_fr3_vla_client',
            parameter_overrides=[
                Parameter(
                    'use_sim_time',
                    Parameter.Type.BOOL,
                    self.config.use_sim_time,
                )
            ],
        )
        self.node.get_logger().info(
            f"Clock mode: {'sim_time' if self.config.use_sim_time else 'real_time'}"
        )

        self.action_cb_group = ReentrantCallbackGroup()

        # create robot state & sensor managers
        self.robot_state_manager = RobotStateManager(self.node)
        if self.config.use_cameras:
            self.camera_sensor_manager = CameraSensorManager(
                node=self.node, config=self.config.cameras, fps=30.0
            )
        if self.config.use_ft_sensor:
            self.ft_sensor = FTSensor(
                node=self.node, config=self.config.ft_sensor
            )

        # allocate managers to task
        self.task.allocate_managers(
            robot=self.robot_state_manager,
            ft_sensor=self.ft_sensor,
            camera_sensor=self.camera_sensor_manager,
        )

        self.pub_action_chunk = self.node.create_publisher(
            ActionChunk, self.config.action_topic, 10
        )

        # 2. Create the Action Client
        self._action_client = ActionClient(
            self.node, VisionLanguageAction, self.config.vla_action_server,
            callback_group=self.action_cb_group
        )

        # 3. Start the background ROS 2 thread (spin is required for Action communication)
        self.executor_thread = threading.Thread(target=self._spin_ros, daemon=True)
        self.executor_thread.start()

        # 4. Wait for the Action Server connection
        print(f"[{self.name}] Waiting for VLA Action Server: {self.config.vla_action_server}...")
        if not self._action_client.wait_for_server(timeout_sec=5.0):
            raise ConnectionError("VLA Action Server is not available.")

        # 5. Send the Goal and wait for acceptance
        self._send_vla_goal()

        print(f"[{self.name}] Waiting for first EE pose observation...")
        timeout = 5.0
        start_time = time.time()
        while time.time() - start_time < timeout:
            # The initial Pose object is filled with zeros, so a non-zero value means the TF has been updated
            if np.any(self.robot_state_manager.ee_pose != 0.0):
                break
            time.sleep(0.1)
        else:
            print("Warning: Did not receive EE pose (TF transform) within timeout.")

        # connect to sensors
        if self.camera_sensor_manager is not None:
            self.camera_sensor_manager.connect()
        if self.ft_sensor is not None:
            self.ft_sensor.connect()

        self.task.reset()

        self._is_connected = True
        print(f"[{self.name}] VLA Client Connected and Ready!")

    def send_new_vla_goal(self) -> None:
        """Send a fresh VLA action goal for a new episode on an existing connection.

        VLAActionServer::compute() (cho_controller_franka) marks the goal
        succeeded and stops reacting to ActionChunk messages once
        send_success_signal() fires — see `handle_success_trigger()`/`compute()`
        in vla_action_server.cpp. A new episode therefore needs a brand new
        accepted goal before any further action has effect; this is a no-op
        during the very first connect() (handled by _send_vla_goal there).
        """
        self._goal_handle = None
        self._goal_accepted = False
        self._send_vla_goal()

    def _send_vla_goal(self):
        """Request control authority (Goal) from the Action Server."""
        goal_msg = VisionLanguageAction.Goal()
        goal_msg.model_name = self.config.model_name
        goal_msg.inference_frequency = self.config.inference_frequency

        self._send_goal_future = self._action_client.send_goal_async(goal_msg)
        self._send_goal_future.add_done_callback(self._goal_response_callback)

        timeout = 5.0
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self._goal_accepted:
                return
            time.sleep(0.1)
        
        raise ConnectionError("Goal was not accepted by the VLA Action Server.")

    def _goal_response_callback(self, future):
        """Handle the server's response to the Goal request."""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.node.get_logger().error('VLA Goal rejected by server!')
            self._goal_accepted = False
            return

        self.node.get_logger().info('VLA Goal accepted! Robot control started.')
        self._goal_handle = goal_handle
        self._goal_accepted = True

    def _spin_ros(self):
        self._executor = MultiThreadedExecutor(num_threads=4)
        self._executor.add_node(self.node)
        try:
            self._executor.spin()
        except Exception:
            # executor.shutdown()/rclpy.shutdown() during disconnect raises here
            # (e.g. ExternalShutdownException); a clean exit, nothing to report.
            pass
        finally:
            try:
                self._executor.shutdown()
            except Exception:
                pass

    def get_observation(self) -> RobotObservation:
        if not self.is_connected:
            raise ConnectionError(f"{self.name} is not connected.")
        
        obs_dict = self.task.get_observation()

        return obs_dict
    
    def get_vla_observation(self) -> RobotObservation:
        if not self.is_connected:
            raise ConnectionError(f"{self.name} is not connected.")
        
        obs_dict = self.task.get_vla_observation()

        return obs_dict

    def send_action(self, action: RobotAction) -> RobotAction:
        if not self.is_connected:
            raise ConnectionError(f"{self.name} is not connected.")

        if not self._goal_accepted:
            self.node.get_logger().warn("Cannot send action: Goal not accepted yet.")
            return action
        
        arm_action = self.task.get_arm_action(action)
        gripper_action = self.task.get_gripper_action(action)

        processed_arm_action, processed_gripper_action = self.task.process_action(arm_action, gripper_action)

        self.apply_action(
            processed_arm_action,
            processed_gripper_action,
            action_space=self.task.control_action_space,
            arm_action_dim=self.task.control_arm_action_dim,
        )

        processed_action = {
            "processed_arm_action": processed_arm_action,
            "processed_gripper_action": processed_gripper_action,
        }

        return processed_action

    def send_teleop_action(self, teleop_action: TeleopAction) -> RobotAction:
        if not self.is_connected:
            raise ConnectionError(f"{self.name} is not connected.")

        if not self._goal_accepted:
            self.node.get_logger().warn("Cannot send action: Goal not accepted yet.")
            return teleop_action.action

        arm_action = self.task.get_arm_action(teleop_action.action)
        gripper_action = self.task.get_gripper_action(teleop_action.action)

        if teleop_action.action_space == "task_space" and teleop_action.is_relative:
            processed_arm_action, processed_gripper_action = self.task.process_action(
                arm_action, gripper_action
            )
            action_space = self.config.action_space
            is_relative = self.config.is_relative
        else:
            processed_arm_action = arm_action
            processed_gripper_action = gripper_action
            action_space = teleop_action.action_space
            is_relative = teleop_action.is_relative

        if np.allclose(processed_arm_action[:3], 0.0, atol=1e-6):
            self.node.get_logger().warn(
                "send_teleop_action: target_pos is near-zero. "
                "Replacing with current EE pose to prevent unsafe motion."
            )
            processed_arm_action = np.concatenate([
                self.robot_state_manager.ee_pos,
                self.robot_state_manager.ee_quat,
            ]).astype(np.float32)

        self.apply_action(
            processed_arm_action,
            processed_gripper_action,
            action_space=action_space,
            is_relative=is_relative,
        )

        return {
            "processed_arm_action": processed_arm_action,
            "processed_gripper_action": processed_gripper_action,
        }
    
    def send_processed_action(self, action:RobotAction) -> RobotAction:
        warnings.warn(
            "send_processed_action() is deprecated. Use send_teleop_action() "
            "with TeleopAction(action_space='task_space', is_relative=False).",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.send_teleop_action(
            TeleopAction(
                action=action,
                action_space="task_space",
                is_relative=False,
            )
        )
        


    def apply_action(
        self,
        processed_arm_action: np.ndarray,
        processed_gripper_action: np.ndarray = None,
        *,
        action_space: str | None = None,
        is_relative: bool | None = None,
        arm_action_dim: int | None = None,
    ) -> None:
        if not self.task.controls_gripper:
            processed_gripper_action = np.array([], dtype=np.float32)
            
        msg = self.wrap_action_to_msg(
            processed_arm_action,
            processed_gripper_action,
            action_space=action_space,
            is_relative=is_relative,
            arm_action_dim=arm_action_dim,
        )
        # send action chunk message to VLA Action Server
        self.pub_action_chunk.publish(msg)

    def reset_success_signal(self) -> None:
        """Re-arm send_success_signal() for the next episode.

        Needed when a single connection records multiple episodes back to
        back (continuous teleop sessions) — without this, only the first
        episode's success signal would ever be sent.
        """
        self._success_signaled = False

    def send_success_signal(self) -> None:
        """Tell the VLA Action Server that episode playback has finished.

        Calls the ``/vla/trigger_success`` service. Idempotent per connection:
        only the first call actually signals, so invoking it right after the
        control loop (before the success prompt) and again during ``disconnect``
        is safe.
        """
        if self._success_signaled:
            return
        if not rclpy.ok() or self.node is None:
            print(f"[{self.name}] ROS context already down; skipping success signaling.")
            return
        try:
            print(f"[{self.name}] Sending Task Success signal to VLA Action Server...")
            trigger_client = self.node.create_client(Trigger, '/vla/trigger_success')
            if trigger_client.wait_for_service(timeout_sec=2.0):
                future = trigger_client.call_async(Trigger.Request())
                t0 = time.time()
                while not future.done() and time.time() - t0 < 1.0:
                    time.sleep(0.1)
                if future.done():
                    try:
                        response = future.result()
                        print(f"[{self.name}] Trigger Response: {response.success} - {response.message}")
                    except Exception as e:
                        print(f"[{self.name}] Failed to get response from Trigger service: {e}")
            else:
                print(f"[{self.name}] Warning: /vla/trigger_success service is not available.")
            self._success_signaled = True
        except Exception as e:
            print(f"[{self.name}] ROS success signaling skipped: {e}")

    def disconnect(self) -> None:
        if not self._is_connected:
            return
        # Mark disconnected up-front so any re-entry (double finally, __del__,
        # interpreter shutdown) is a no-op and we never run this twice.
        self._is_connected = False

        # disconnect sensors (best-effort)
        for name, obj in (("camera", self.camera_sensor_manager),
                          ("ft_sensor", self.ft_sensor)):
            if obj is not None:
                try:
                    obj.disconnect()
                except Exception as e:
                    print(f"[{self.name}] {name} disconnect error: {e}")

        # ROS signaling only while the context is valid. Guarded so a shut-down
        # context can never crash us (which previously caused a segfault).
        if rclpy.ok():
            # Success signal is idempotent: if record.py already sent it right
            # after playback (before the success prompt), this is a no-op.
            self.send_success_signal()
            try:
                # Control-safe shutdown: if there is an active Goal, send a Cancel request
                if self._goal_handle is not None and self._goal_accepted:
                    print(f"[{self.name}] Canceling VLA Action goal for safe stop...")
                    cancel_future = self._goal_handle.cancel_goal_async()
                    t0 = time.time()
                    while not cancel_future.done() and time.time() - t0 < 1.0:
                        time.sleep(0.1)
            except Exception as e:
                print(f"[{self.name}] ROS shutdown signaling skipped: {e}")
        else:
            print(f"[{self.name}] ROS context already down; skipping success/cancel signaling.")

        # Stop the executor first so spin() returns and the thread exits before
        # we destroy the node — destroying a node still being spun crashes the
        # rclpy C++ layer (segfault) during teardown.
        if self._executor is not None:
            try:
                self._executor.shutdown()
            except Exception as e:
                print(f"[{self.name}] executor shutdown error: {e}")
        if self.executor_thread is not None:
            self.executor_thread.join(timeout=2.0)

        # Tear down node / context (each guarded independently).
        try:
            if self.node is not None:
                self.node.destroy_node()
        except Exception as e:
            print(f"[{self.name}] node destroy error: {e}")
        if rclpy.ok():
            try:
                rclpy.shutdown()
            except Exception as e:
                print(f"[{self.name}] rclpy shutdown error: {e}")

        if self.executor_thread is not None:
            self.executor_thread.join(timeout=1.0)

        print(f"[{self.name}] Disconnected.")

    @property
    def observation_features(self) -> dict[str, tuple]:
        return self.task.observation_features

    @property
    def action_features(self) -> dict[str, tuple]:
        return self.task.action_features
    
    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def wrap_action_to_msg(
        self,
        arm_action: np.ndarray,
        gripper_action: np.ndarray = None,
        *,
        action_space: str | None = None,
        is_relative: bool | None = None,
        arm_action_dim: int | None = None,
    ) -> ActionChunk:

        msg_action_space = action_space if action_space is not None else self.config.action_space
        msg_is_relative = is_relative if is_relative is not None else self.config.is_relative
        msg_arm_action_dim = arm_action_dim if arm_action_dim is not None else self.config.arm_action_dim

        if msg_action_space in {"task", "task_space"} and self.config.rotation_type == "quaternion":
            # Copy first: the caller (e.g. record.py teleop) reuses this same
            # array to log the action, and the controller expects xyzw on the
            # wire while the recorded action / state stay wxyz. Mutating in place
            # would silently store xyzw and break the recorded orientation.
            arm_action = arm_action.copy()
            arm_action[3:7] = wxyz2xyzw(arm_action[3:7])
        gripper_action = gripper_action if gripper_action is not None else np.array([])

        msg = ActionChunk()
        msg.header = Header(stamp=self.node.get_clock().now().to_msg(), frame_id="base_link")
        msg.action_space = msg_action_space
        msg.relative = msg_is_relative
        msg.rotation_type = self.config.rotation_type
        msg.chunk_size = len(arm_action) // msg_arm_action_dim

        msg.arm_actions = arm_action.tolist()
        msg.gripper_actions = gripper_action.tolist()

        return msg
