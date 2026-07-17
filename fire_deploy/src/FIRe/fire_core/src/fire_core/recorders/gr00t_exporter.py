from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lerobot.datasets.video_utils import encode_video_frames

from .base_recorder import VALID_TASK


class GR00TExporter:
    def __init__(
        self,
        *,
        task_info: dict[str, Any],
        fps: int,
        camera_shapes: dict[str, tuple[int, int, int]],
        action_spec: dict[str, Any] | None = None,
    ) -> None:
        self._task_info = task_info
        self._fps = fps
        self._camera_shapes = camera_shapes
        # Task-provided action layout (names/info_names/modality). None only on
        # the video-only encode_deferred_videos() path, which never rewrites
        # metadata, so the delta fallbacks below are cosmetic there.
        self._action_spec = action_spec or {}

    def convert_episode(self, root: Path, success: bool, *, keep_images: bool = False) -> None:
        meta_dir = root / "meta"
        data_dir = root / "data" / "chunk-000"
        src_data = data_dir / "file-000.parquet"
        dst_data = data_dir / "episode_000000.parquet"

        df = pd.read_parquet(src_data)
        total_frames = len(df)
        gr00t_df = self._build_episode_dataframe(df, success, episode_index=0, start_index=0)
        gr00t_df.to_parquet(dst_data, index=False)
        if src_data != dst_data and src_data.exists():
            src_data.unlink()

        self._move_lerobot_videos_to_gr00t_layout(root, episode_index=0)
        self._remove_lerobot_metadata(meta_dir)
        self.rewrite_metadata(root)
        if not keep_images:
            self._remove_images_dir(root)

    def append_staging_episode(self, *, staging_root: Path, target_root: Path) -> int:
        next_episode_index = self._next_episode_index(target_root)
        existing_frames = self._total_frames(target_root)

        src_data = staging_root / "data" / "chunk-000" / "episode_000000.parquet"
        if not src_data.exists():
            raise FileNotFoundError(f"Missing staging parquet: {src_data}")

        target_data_dir = target_root / "data" / "chunk-000"
        target_data_dir.mkdir(parents=True, exist_ok=True)
        dst_data = target_data_dir / f"episode_{next_episode_index:06d}.parquet"

        df = pd.read_parquet(src_data)
        total_new_frames = len(df)
        df["episode_index"] = np.full(total_new_frames, next_episode_index, dtype=np.int64)
        df["index"] = np.arange(existing_frames, existing_frames + total_new_frames, dtype=np.int64)
        df.to_parquet(dst_data, index=False)

        for view_name in self._camera_view_names():
            src_video = (
                staging_root
                / "videos"
                / "chunk-000"
                / f"observation.images.{view_name}"
                / "episode_000000.mp4"
            )
            dst_dir = target_root / "videos" / "chunk-000" / f"observation.images.{view_name}"
            dst_dir.mkdir(parents=True, exist_ok=True)
            if src_video.exists():
                shutil.move(str(src_video), str(dst_dir / f"episode_{next_episode_index:06d}.mp4"))

            src_images = (
                staging_root
                / "images"
                / f"observation.images.{view_name}"
                / "episode-000000"
            )
            dst_images = (
                target_root
                / "images"
                / f"observation.images.{view_name}"
                / f"episode-{next_episode_index:06d}"
            )
            if src_images.exists():
                if dst_images.exists():
                    shutil.rmtree(dst_images)
                dst_images.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src_images), str(dst_images))

        self.rewrite_metadata(target_root)
        shutil.rmtree(staging_root)
        return next_episode_index

    @classmethod
    def encode_deferred_videos(
        cls,
        *,
        root: Path,
        last_episode: int | None,
        vcodec: str = "h264",
        encoder_threads: int | None = None,
        remove_images: bool = True,
    ) -> None:
        if last_episode is None:
            last_episode = cls._latest_episode_index(root)
            if last_episode < 0:
                print(f"[INFO] No episodes found to encode in dataset: {root}")
                return
        if last_episode < 0:
            raise ValueError("--last_episode must be >= 0")

        info_path = root / "meta" / "info.json"
        if not info_path.exists():
            raise FileNotFoundError(f"Missing GR00T metadata: {info_path}")

        with info_path.open("r") as f:
            info = json.load(f)

        fps = int(round(float(info["fps"])))
        video_keys = [
            key
            for key, feature in info.get("features", {}).items()
            if key.startswith("observation.images.") and feature.get("dtype") == "video"
        ]
        if not video_keys:
            raise ValueError(f"No video features found in metadata: {info_path}")

        for episode_index in range(last_episode + 1):
            for video_key in video_keys:
                image_dir = root / "images" / video_key / f"episode-{episode_index:06d}"
                video_path = (
                    root
                    / "videos"
                    / "chunk-000"
                    / video_key
                    / f"episode_{episode_index:06d}.mp4"
                )
                if video_path.exists():
                    print(f"[INFO] Video already exists, skipping: {video_path}")
                    continue
                if not image_dir.exists():
                    raise FileNotFoundError(f"Missing deferred image directory: {image_dir}")

                print(f"[INFO] Encoding {video_key} episode {episode_index:06d} -> {video_path}")
                try:
                    encode_video_frames(
                        image_dir,
                        video_path,
                        fps=fps,
                        vcodec=vcodec,
                        pix_fmt="yuv420p",
                        overwrite=True,
                        encoder_threads=encoder_threads,
                    )
                except Exception:
                    if video_path.exists():
                        video_path.unlink()
                    raise
                if remove_images:
                    shutil.rmtree(image_dir)

        images_root = root / "images"
        if remove_images and images_root.exists() and not any(images_root.rglob("frame-*.png")):
            shutil.rmtree(images_root)

    @classmethod
    def _latest_episode_index(cls, root: Path) -> int:
        indices: list[int] = []
        for path in sorted((root / "data" / "chunk-000").glob("episode_*.parquet")):
            try:
                indices.append(int(path.stem.removeprefix("episode_")))
            except ValueError:
                continue
        return max(indices, default=-1)

    def rewrite_metadata(self, root: Path) -> None:
        meta_dir = root / "meta"
        meta_dir.mkdir(parents=True, exist_ok=True)

        episode_files = self._episode_files(root)
        dataframes = [pd.read_parquet(path) for path in episode_files]
        total_frames = sum(len(df) for df in dataframes)

        with (meta_dir / "tasks.jsonl").open("w") as f:
            f.write(json.dumps(self._task_info) + "\n")
            f.write(json.dumps(VALID_TASK) + "\n")

        with (meta_dir / "episodes.jsonl").open("w") as f:
            for path, df in zip(episode_files, dataframes):
                episode_index = int(path.stem.removeprefix("episode_"))
                f.write(
                    json.dumps(
                        {
                            "episode_index": episode_index,
                            "tasks": [self._task_info["task"], VALID_TASK["task"]],
                            "length": len(df),
                        }
                    )
                    + "\n"
                )

        with (meta_dir / "modality.json").open("w") as f:
            json.dump(self.modality(), f, indent=4)
        with (meta_dir / "info.json").open("w") as f:
            json.dump(self.info(total_frames, total_episodes=len(episode_files)), f, indent=4)

        if dataframes:
            self.write_stats(pd.concat(dataframes, ignore_index=True), meta_dir)

    def modality(self) -> dict[str, Any]:
        return {
            "state": {
                "eef_position": {"start": 0, "end": 3},
                "eef_quaternion": {"start": 3, "end": 7, "rotation_type": "quaternion"},
                "gripper_qpos": {"start": 7, "end": 9},
            },
            "action": self._action_spec.get("modality") or {
                "eef_position_delta": {"start": 0, "end": 3},
                "eef_rotation_delta": {"start": 3, "end": 6, "rotation_type": "axis_angle"},
                "gripper_close": {"start": 6, "end": 7},
            },
            "video": {
                view_name: {"original_key": f"observation.images.{view_name}"}
                for view_name in self._camera_view_names()
            },
            "annotation": {
                "human.action.task_description": {},
                "human.validity": {},
            },
        }

    def info(self, total_frames: int, total_episodes: int = 1) -> dict[str, Any]:
        action_info_names = self._action_spec.get("info_names") or [
            "dx", "dy", "dz", "drx", "dry", "drz", "gripper_close"
        ]
        video_info = {
            "video.fps": float(self._fps),
            "video.codec": "h264",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "has_audio": False,
        }
        image_features = {
            f"observation.images.{view_name}": {
                "dtype": "video",
                "shape": list(self._camera_shapes[camera_name]),
                "names": ["height", "width", "channel"],
                "video_info": video_info,
            }
            for camera_name, view_name in zip(self._camera_shapes, self._camera_view_names())
        }
        features = {
            **image_features,
            "observation.state": {
                "dtype": "float64",
                "shape": [9],
                "names": [
                    "x",
                    "y",
                    "z",
                    "qw",
                    "qx",
                    "qy",
                    "qz",
                    "gripper_qpos1",
                    "gripper_qpos2",
                ],
            },
            "action": {
                "dtype": "float64",
                "shape": [len(action_info_names)],
                "names": action_info_names,
            },
            "timestamp": {"dtype": "float64", "shape": [1]},
            "annotation.human.action.task_description": {"dtype": "int64", "shape": [1]},
            "task_index": {"dtype": "int64", "shape": [1]},
            "annotation.human.validity": {"dtype": "int64", "shape": [1]},
            "episode_index": {"dtype": "int64", "shape": [1]},
            "index": {"dtype": "int64", "shape": [1]},
            "next.reward": {"dtype": "float64", "shape": [1]},
            "next.done": {"dtype": "bool", "shape": [1]},
        }
        return {
            "codebase_version": "v2.0",
            "robot_type": "franka_emika_panda",
            "total_episodes": total_episodes,
            "total_frames": total_frames,
            "total_tasks": 2,
            "total_videos": total_episodes * len(self._camera_shapes),
            "total_chunks": 1,
            "chunks_size": 64,
            "fps": float(self._fps),
            "splits": {"train": "0:1"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": features,
        }

    def write_stats(self, df: pd.DataFrame, meta_dir: Path) -> None:
        stats_keys = [
            "observation.state",
            "action",
            "timestamp",
            "annotation.human.action.task_description",
            "task_index",
            "episode_index",
            "index",
            "next.reward",
        ]
        stats = {key: self._stats_for_values(df[key]) for key in stats_keys if key in df}
        with (meta_dir / "stats.json").open("w") as f:
            json.dump(stats, f, indent=4)

    def _build_episode_dataframe(
        self,
        df: pd.DataFrame,
        success: bool,
        *,
        episode_index: int,
        start_index: int,
    ) -> pd.DataFrame:
        total_frames = len(df)
        reward = np.zeros(total_frames, dtype=np.float64)
        done = np.zeros(total_frames, dtype=bool)
        validity = np.zeros(total_frames, dtype=bool)
        if success and total_frames > 0:
            reward[-1] = 1.0
            done[-1] = True
            validity[-1] = True

        task_index = int(self._task_info["task_index"])
        return pd.DataFrame(
            {
                "observation.state": [
                    np.asarray(v, dtype=np.float64) for v in df["observation.state"].to_list()
                ],
                "action": [np.asarray(v, dtype=np.float64) for v in df["action"].to_list()],
                "timestamp": df["timestamp"].astype(np.float64).to_numpy(),
                "annotation.human.action.task_description": np.full(
                    total_frames, task_index, dtype=np.int64
                ),
                "task_index": np.full(total_frames, task_index, dtype=np.int64),
                "annotation.human.validity": validity,
                "episode_index": np.full(total_frames, episode_index, dtype=np.int64),
                "index": np.arange(start_index, start_index + total_frames, dtype=np.int64),
                "next.reward": reward,
                "next.done": done,
            }
        )

    def _move_lerobot_videos_to_gr00t_layout(self, root: Path, episode_index: int) -> None:
        video_root = root / "videos"
        for view_name in self._camera_view_names():
            src_video = video_root / f"observation.images.{view_name}" / "chunk-000" / "file-000.mp4"
            dst_dir = video_root / "chunk-000" / f"observation.images.{view_name}"
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst_video = dst_dir / f"episode_{episode_index:06d}.mp4"
            if src_video.exists():
                if dst_video.exists():
                    dst_video.unlink()
                shutil.move(str(src_video), str(dst_video))

        for stale_dir in video_root.glob("observation.images.*"):
            if stale_dir.is_dir():
                shutil.rmtree(stale_dir)

    def _remove_lerobot_metadata(self, meta_dir: Path) -> None:
        tasks_parquet = meta_dir / "tasks.parquet"
        if tasks_parquet.exists():
            tasks_parquet.unlink()
        episodes_dir = meta_dir / "episodes"
        if episodes_dir.exists():
            shutil.rmtree(episodes_dir)

    def _remove_images_dir(self, root: Path) -> None:
        images_dir = root / "images"
        if images_dir.exists():
            shutil.rmtree(images_dir)

    def _episode_files(self, root: Path) -> list[Path]:
        return sorted((root / "data" / "chunk-000").glob("episode_*.parquet"))

    def _next_episode_index(self, root: Path) -> int:
        indices: list[int] = []
        for path in self._episode_files(root):
            try:
                indices.append(int(path.stem.removeprefix("episode_")))
            except ValueError:
                continue
        return max(indices, default=-1) + 1

    def _total_frames(self, root: Path) -> int:
        return sum(len(pd.read_parquet(path)) for path in self._episode_files(root))

    def _camera_view_names(self) -> list[str]:
        return [f"{name}_view" for name in self._camera_shapes]

    def _stats_for_values(self, values: pd.Series) -> dict[str, Any]:
        first = values.iloc[0]
        if isinstance(first, np.ndarray):
            arr = np.stack(values.to_numpy()).astype(np.float64)
        elif isinstance(first, (list, tuple)):
            arr = np.asarray(values.to_list(), dtype=np.float64)
        else:
            arr = values.to_numpy(dtype=np.float64)

        def clean(value: np.ndarray | np.number) -> Any:
            if np.ndim(value) == 0:
                return float(value)
            return np.asarray(value).tolist()

        return {
            "mean": clean(np.mean(arr, axis=0)),
            "std": clean(np.std(arr, axis=0)),
            "min": clean(np.min(arr, axis=0)),
            "max": clean(np.max(arr, axis=0)),
            "q01": clean(np.quantile(arr, 0.01, axis=0)),
            "q99": clean(np.quantile(arr, 0.99, axis=0)),
        }
