from __future__ import annotations

from abc import ABC, abstractmethod
import datetime
import shutil
from pathlib import Path
from typing import Any
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from .utils import (
    flat_feature_names, squeeze_image,
)


TASK_REGISTRY = {
    "forge-peg_insert": {
        "task_index": 0,
        "task": "Insert peg into the socket",
    },
    "peg_insert": {
        "task_index": 0,
        "task": "Insert peg into the socket",
    },
}

VALID_TASK = {"task_index": 1, "task": "valid"}


def _camera_shapes(robot) -> dict[str, tuple[int, int, int]]:
    if getattr(robot, "camera_sensor_manager", None) is not None:
        return dict(robot.camera_sensor_manager.shapes)
    return {
        name: (cfg.height, cfg.width, 3)
        for name, cfg in getattr(robot.config, "cameras", {}).items()
    }


class BaseRecorder(ABC):
    """Base recorder class responsible for common LeRobot dataset recording."""

    def __init__(
        self,
        robot,
        task: str,
        fps: int,
        repo_id: str | None = None,
        root: str | Path | None = None,
        task_text: str | None = None,
        resume: bool = False,
        use_videos: bool = True,
        image_writer_processes: int = 0,
        image_writer_threads_per_camera: int = 4,
        vcodec: str = "h264",
        streaming_encoding: bool = False,
        encoder_threads: int | None = None,
        batch_encoding_size: int = 1,
    ) -> None:
        self._robot = robot
        self._task_key = task
        self._task_info = self._resolve_task_info(task, task_text)
        self._task = self._task_info["task"]
        self._fps = fps
        self._frames = 0
        self._closed = False
        self._resume = resume
        self._use_videos = use_videos

        self._image_writer_processes = image_writer_processes
        self._image_writer_threads_per_camera = image_writer_threads_per_camera
        self._vcodec = vcodec
        self._streaming_encoding = streaming_encoding
        self._encoder_threads = encoder_threads
        self._batch_encoding_size = batch_encoding_size
        
        # Store configuration variables (so child classes can access them)
        self.camera_shapes = _camera_shapes(robot)
        self._image_dtype = "video" if use_videos else "image"
        
        self._obs_features = robot.task.observation_features
        self._action_features = robot.task.action_features
        self._log_features = robot.task.log_features
        self._rl_state_names = flat_feature_names("", self._obs_features)
        self._log_state_names = flat_feature_names("log_", self._log_features)
        
        # Common VLA state names (override in subclasses if they differ per model)
        self._vla_state_names = [
            "eef_position_0", "eef_position_1", "eef_position_2",
            "eef_quaternion_0", "eef_quaternion_1", "eef_quaternion_2", "eef_quaternion_3",
            "gripper_qpos_0", "gripper_qpos_1",
        ]

        self._dataset_root = self._resolve_root(root, repo_id, resume)
        # If there is no repo_id, use the folder name as repo_id (to satisfy LeRobotDataset's required argument)
        self._repo_id = repo_id or self._dataset_root.name
        self._recording_root = self._resolve_recording_root(self._dataset_root, resume)

        self._vla_dataset = self._create_dataset(
            repo_id=self._repo_id,
            root=self._recording_root,
            features=self._build_vla_dataset_features(),
        )

    # ── [Common helper methods: _resolve_root, _create_dataset, etc. same as before] ──
    @property
    def output_root(self) -> Path:
        return self._dataset_root

    def _resolve_root(self, root: str | Path | None, repo_id: str | None = None, resume: bool = False) -> Path:
        if root is not None:
            path = Path(root)
        elif repo_id:
            safe_repo_name = repo_id.replace("/", "_")
            path = Path("outputs/datasets") / safe_repo_name
        else:
            # If neither is provided, use a default date-based folder
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            path = Path("outputs/datasets") / f"dataset_{timestamp}"

        if resume:
            return path

        if path.exists():
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            candidate = path.with_name(f"{path.name}_{timestamp}")
            suffix = 1
            while candidate.exists():
                candidate = path.with_name(f"{path.name}_{timestamp}_{suffix}")
                suffix += 1
            print(f"[INFO] Dataset path '{path}' already exists, using new root: {candidate}")
            return candidate

        return path

    def _resolve_recording_root(self, dataset_root: Path, resume: bool) -> Path:
        return dataset_root
    
    def _resolve_task_info(self, task_key: str, task_text: str | None) -> dict[str, Any]:
        info = dict(TASK_REGISTRY.get(task_key, {"task_index": 0, "task": task_key}))
        if task_text is not None:
            info["task"] = task_text
        return info

    def _create_dataset(
        self,
        repo_id: str,
        root: str | Path | None,
        features: dict[str, dict],
    ) -> LeRobotDataset:
        return LeRobotDataset.create(
            repo_id=repo_id,
            fps=self._fps,
            root=root,
            robot_type=self._robot.name,
            features=features,
            use_videos=self._use_videos,
            image_writer_processes=self._image_writer_processes,
            image_writer_threads=self._image_writer_threads_per_camera * len(self.camera_shapes),
            vcodec=self._vcodec,
            streaming_encoding=self._streaming_encoding,
            encoder_threads=self._encoder_threads,
            batch_encoding_size=self._batch_encoding_size,
        )

    def _image_features(self, use_view_suffix: bool) -> dict[str, dict]:
        features: dict[str, dict] = {}
        for name, shape in self.camera_shapes.items():
            feature_name = f"{name}_view" if use_view_suffix else name
            features[f"observation.images.{feature_name}"] = {
                "dtype": self._image_dtype,
                "shape": shape,
                "names": ["height", "width", "channels"],
            }
        return features
    
    @abstractmethod
    def _build_vla_dataset_features(self) -> dict[str, dict]:
        raise NotImplementedError

    def _vla_state(self, vla_obs: dict[str, Any]) -> np.ndarray:
        parts = [
            np.asarray(vla_obs["state.eef_position"], dtype=np.float32).reshape(-1)[:3],
            np.asarray(vla_obs["state.eef_quaternion"], dtype=np.float32).reshape(-1)[:4],
            np.asarray(vla_obs["state.gripper_qpos"], dtype=np.float32).reshape(-1)[:2],
        ]
        return np.concatenate(parts).astype(np.float32, copy=False)
    
    @abstractmethod
    def _format_action(self, arm_action: np.ndarray, gripper_action: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def record(self, arm_action: np.ndarray, gripper_action: np.ndarray) -> None:
        vla_obs = self._robot.get_vla_observation()

        vla_image_frame = {
            f"observation.images.{name}_view": squeeze_image(
                vla_obs[f"video.{name}_view"],
                shape,
            )
            for name, shape in self.camera_shapes.items()
        }
        vla_frame: dict[str, Any] = {
            "observation.state": self._vla_state(vla_obs),
            "action": self._format_action(arm_action, gripper_action),
            "task": self._task,
        }
        vla_frame.update(vla_image_frame)

        self._vla_dataset.add_frame(vla_frame)
        self._frames += 1

    def _ask_episode_success(self) -> bool:
        while True:
            value = input("[INPUT] Episode success? 0 = fail, 1 = success: ").strip()
            if value in {"0", "1"}:
                return value == "1"
            print("[WARN] Please enter 0 or 1.")
    
    def _remove_images_dir(self, root: str | Path) -> None:
        images_dir = Path(root) / "images"
        if images_dir.exists():
            shutil.rmtree(images_dir)

    @abstractmethod
    def _post_save_processing(self, success: bool) -> None:
        raise NotImplementedError

    def save(self) -> None:
        if self._closed:
            return
        self._closed = True

        if self._frames == 0:
            print("[INFO] Recorder buffer empty - nothing to save.")
            self._vla_dataset.finalize()
            return

        success = self._ask_episode_success()
        self._vla_dataset.save_episode()
        self._vla_dataset.finalize()
        
        # Run model-specific post-processing logic
        self._post_save_processing(success)
        
        print(f"[INFO] Saved {self._frames} VLA frames to dataset: {self._vla_dataset.root}")
