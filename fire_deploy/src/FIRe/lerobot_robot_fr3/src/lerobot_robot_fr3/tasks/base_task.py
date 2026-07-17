from typing import Any, Dict, Optional, Tuple
import numpy as np
from abc import ABC, abstractmethod
from lerobot_ft_sensor import FTSensor
from lerobot_robot_fr3.utils import RobotStateManager, CameraSensorManager
import os, sys


class Task(ABC):
    def __init__(self, name: str) -> None:
        self.name = name
        self._robot: RobotStateManager = None
        self._camera_sensor: CameraSensorManager = None
        self._ft_sensor: FTSensor = None

    @property
    def model_cfg_path(self) -> str:
        module = sys.modules[self.__class__.__module__]
        if hasattr(module, "__file__") and module.__file__ is not None:
            class_dir = os.path.dirname(os.path.abspath(module.__file__))
        else:
            class_dir = os.path.dirname(os.path.abspath(__file__))
            
        return os.path.join(class_dir, "agents", f"{self.name}.yaml")
    
    @property
    def controls_gripper(self) -> bool:
        return True

    @property
    def robot(self) -> RobotStateManager:
        return self._robot
    
    @property
    def camera_sensor(self) -> CameraSensorManager:
        return self._camera_sensor
    
    @property
    def ft_sensor(self) -> FTSensor:
        return self._ft_sensor

    def allocate_managers(
        self, 
        robot: RobotStateManager,
        camera_sensor: CameraSensorManager = None,
        ft_sensor: FTSensor = None,
    ) -> None:
        self._robot = robot
        self._camera_sensor = camera_sensor
        self._ft_sensor = ft_sensor
    
    @abstractmethod
    def create_config(self) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def create_buffer(self) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_observation(self) -> Dict[str, np.ndarray]:
        raise NotImplementedError
    
    @abstractmethod
    def get_vla_observation(self) -> Dict[str, np.ndarray]:
        raise NotImplementedError
    
    @abstractmethod
    def get_arm_action(self, action: Dict[str, np.ndarray]) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def get_gripper_action(self, action: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
        raise NotImplementedError
    
    @abstractmethod
    def get_log(self) -> Dict[str, np.ndarray]:
        raise NotImplementedError

    def get_reward(self) -> float:
        return 0.0

    def get_done(self) -> bool:
        return False

    def get_truncated(self) -> bool:
        return False

    def get_info(self) -> Dict[str, Any]:
        return {}

    # ── control metadata ──────────────────────────────────────────────────────
    # Defaults preserve the existing task-space behavior (Factory/Forge). Tasks that
    # drive the robot in a different space (e.g. joint position) override these so
    # FR3Robot publishes the ActionChunk with the right action_space / arm_action_dim.
    @property
    def control_action_space(self) -> str:
        return "task"

    @property
    def control_arm_action_dim(self) -> int:
        return 6

    @property
    def vla_action_spec(self) -> Dict[str, Any]:
        """GR00T ``action`` layout used when recording a VLA dataset.

        Default: 6-dim relative EE delta (position delta + axis-angle rotation
        delta) + gripper — the RL/relative task-space action used by Factory
        and Forge. Tasks whose recorded action is an absolute EE pose (e.g.
        Inverse3 teleop for pick_place, which bypasses process_action() and
        sends absolute task-space poses) override this so the stored action and
        the modality.json GR00T trains on match what was actually sent to the
        robot.

        Keys:
          - ``arm_dim``: number of arm action components stored (before gripper)
          - ``names``: LeRobot parquet column names for the ``action`` vector
          - ``info_names``: human-readable per-component names for info.json
          - ``modality``: GR00T modality.json ``action`` sub-dict
        """
        return {
            "arm_dim": 6,
            "names": [f"arm_action_{i}" for i in range(6)] + ["gripper_action"],
            "info_names": ["dx", "dy", "dz", "drx", "dry", "drz", "gripper_close"],
            "modality": {
                "eef_position_delta": {"start": 0, "end": 3},
                "eef_rotation_delta": {"start": 3, "end": 6, "rotation_type": "axis_angle"},
                "gripper_close": {"start": 6, "end": 7},
            },
        }


    @abstractmethod
    def process_action(
        self,
        arm_action: np.ndarray,
        gripper_action: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        raise NotImplementedError
    
    @property
    @abstractmethod
    def observation_features(self) -> Dict[str, Tuple[int, ...]]:
        raise NotImplementedError
    
    @property
    @abstractmethod
    def action_features(self) -> Dict[str, Tuple[int, ...]]:
        raise NotImplementedError
    
    @property
    @abstractmethod
    def log_features(self) -> Dict[str, Tuple[int, ...]]:
        raise NotImplementedError
