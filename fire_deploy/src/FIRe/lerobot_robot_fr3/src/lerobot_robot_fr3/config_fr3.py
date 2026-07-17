from dataclasses import dataclass, field
from lerobot.cameras import CameraConfig
from lerobot.robots import RobotConfig
from lerobot.cameras.zmq import ZMQCamera, ZMQCameraConfig
from typing import Dict
from lerobot_ft_sensor.configuration_ft_sensor import FTSensorConfig
import os


@dataclass
class FR3RobotConfig(RobotConfig):
    use_sim_time: bool = False
    
    ros_domain_id: int = 0
    
    # --- ROS 2 topics and action server names ---
    observation_topic: str = "/vla/observation/ee_pose"
    action_topic: str = "/vla/action/ee_pose"
    vla_action_server: str = "/controller_action_server/vla_controller"
    
    # --- VLA Action Goal settings ---
    model_name: str = "lerobot_vla_policy"
    inference_frequency: float = 15.0
    
    # --- VLA Chunk settings ---
    action_space: str = "task"    # Task space control
    is_relative: bool = False
    rotation_type: str = "axis_angle"  # "axis_angle", "euler", "quaternion", "rotation6d"
    arm_action_dim: int = 6

    # turn on / off sensors
    use_cameras: bool = False
    use_ft_sensor: bool = False
    
    ft_sensor: FTSensorConfig = FTSensorConfig()
    cameras: Dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "wrist": ZMQCameraConfig(
                server_address="127.0.0.1", port=5555, camera_name="wrist", timeout_ms=5000,
                width=256, height=256, fps=30
            ),
            "left":  ZMQCameraConfig(
                server_address="127.0.0.1", port=5555, camera_name="left", timeout_ms=5000,
                width=256, height=256, fps=30
            ),
            "right": ZMQCameraConfig(
                server_address="127.0.0.1", port=5555, camera_name="right", timeout_ms=5000,
                width=256, height=256, fps=30
            )
        }
    )