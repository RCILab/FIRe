from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


FeatureMap = dict[str, tuple[int, ...]]
gridspec: Any = None
plt: Any = None


DEFAULT_STATE_NAMES: list[str] = [
    "eef_position_x",
    "eef_position_y",
    "eef_position_z",
    "eef_quaternion_w",
    "eef_quaternion_x",
    "eef_quaternion_y",
    "eef_quaternion_z",
    "gripper_qpos_0",
    "gripper_qpos_1",
]

DEFAULT_ACTION_NAMES: list[str] = [
    "eef_position_delta_x",
    "eef_position_delta_y",
    "eef_position_delta_z",
    "eef_rotation_delta_x",
    "eef_rotation_delta_y",
    "eef_rotation_delta_z",
    "gripper_close",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot a LeRobot/GR00T dataset recorded by scripts/record.py.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--root",
        type=Path,
        help="LeRobot dataset root containing data/chunk-000/episode_*.parquet.",
    )
    input_group.add_argument(
        "--data",
        type=Path,
        help="Single episode parquet or CSV file.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="forge-peg_insert",
        help="Task name used to instantiate task features without connecting to the robot.",
    )
    parser.add_argument(
        "--episode",
        type=int,
        default=-1,
        help="Episode index to plot. Use -1 for latest episode, or -2 for all episodes.",
    )
    parser.add_argument(
        "--save_path",
        type=Path,
        default=None,
        help="Output PNG path. Omit for interactive plt.show().",
    )
    parser.add_argument(
        "--max_points",
        type=int,
        default=0,
        help="Optional downsample target per plotted series. 0 disables downsampling.",
    )
    return parser.parse_args()


def load_task_features(task_name: str) -> tuple[FeatureMap, FeatureMap, FeatureMap]:
    from lerobot_robot_fr3.config_fr3 import FR3RobotConfig
    from lerobot_robot_fr3.fr3 import FR3Robot

    config = FR3RobotConfig()
    robot = FR3Robot(config, task_name=task_name)
    task = robot.task
    return task.observation_features, task.log_features, task.action_features


def total_dim(features: FeatureMap) -> int:
    return sum(int(np.prod(shape)) for shape in features.values())


def vector_labels(key: str, dim: int, names: list[str] | None = None) -> list[str]:
    if names is not None and len(names) == dim:
        return names
    if dim == 3:
        suffixes = ["x", "y", "z"]
    elif dim == 4:
        suffixes = ["w", "x", "y", "z"]
    else:
        suffixes = [str(index) for index in range(dim)]
    return [f"{key}_{suffix}" for suffix in suffixes]


def read_metadata_names(root: Path, key: str) -> list[str] | None:
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        return None
    with info_path.open("r") as file:
        info = json.load(file)
    feature = info.get("features", {}).get(key)
    names = feature.get("names") if isinstance(feature, dict) else None
    if isinstance(names, list) and all(isinstance(name, str) for name in names):
        return names
    return None


def episode_files(root: Path) -> list[Path]:
    files = sorted((root / "data" / "chunk-000").glob("episode_*.parquet"))
    if not files:
        files = sorted((root / "data" / "chunk-000").glob("file-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet episodes found under {root / 'data' / 'chunk-000'}")
    return files


def select_episode_files(root: Path, episode: int) -> list[Path]:
    files = episode_files(root)
    if episode == -2:
        return files
    if episode == -1:
        return [files[-1]]

    target = root / "data" / "chunk-000" / f"episode_{episode:06d}.parquet"
    if target.exists():
        return [target]
    if episode < 0 or episode >= len(files):
        raise IndexError(f"Episode {episode} is out of range. Available files: {len(files)}")
    return [files[episode]]


def read_dataframe(path: Path) -> pd.DataFrame:
    if path.suffix == ".csv":
        return pd.read_csv(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported data file extension: {path}")


def load_dataset(root: Path, episode: int) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in select_episode_files(root, episode):
        df = read_dataframe(path)
        if "episode_index" not in df.columns:
            episode_index = int(path.stem.removeprefix("episode_").removeprefix("file-"))
            df["episode_index"] = episode_index
        frames.append(df)
        print(f"[load] {len(df)} frames <- {path}")
    return pd.concat(frames, ignore_index=True)


def as_vector(value: Any) -> np.ndarray:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            return np.fromstring(stripped.strip("[]").replace(",", " "), sep=" ", dtype=np.float64)
    return np.asarray(value, dtype=np.float64).reshape(-1)


def vector_matrix(df: pd.DataFrame, key: str, fallback_prefixes: list[str]) -> np.ndarray | None:
    if key in df.columns:
        values = [as_vector(value) for value in df[key].to_list()]
        if not values:
            return None
        return np.vstack(values)

    for prefix in fallback_prefixes:
        cols = indexed_columns(df, prefix)
        if cols:
            return df[cols].to_numpy(dtype=np.float64)
    return None


def indexed_columns(df: pd.DataFrame, prefix: str) -> list[str]:
    cols: list[str] = []
    index = 0
    while f"{prefix}_{index}" in df.columns:
        cols.append(f"{prefix}_{index}")
        index += 1
    return cols


def scalar_series(df: pd.DataFrame, key: str) -> np.ndarray | None:
    if key not in df.columns:
        return None
    return df[key].to_numpy(dtype=np.float64)


def downsample(values: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    if max_points <= 0 or len(values) <= max_points:
        return np.arange(len(values)), values
    indices = np.linspace(0, len(values) - 1, max_points).astype(np.int64)
    return indices, values[indices]


def draw_vector_group(
    fig: Any,
    spec: Any,
    title: str,
    values: np.ndarray,
    labels: list[str],
    max_points: int,
) -> None:
    dim = values.shape[1]
    inner = gridspec.GridSpecFromSubplotSpec(dim, 1, subplot_spec=spec, hspace=0.10)
    axes = [fig.add_subplot(inner[index, 0]) for index in range(dim)]
    for ax in axes[1:]:
        ax.sharex(axes[0])
    for ax in axes[:-1]:
        plt.setp(ax.get_xticklabels(), visible=False)

    axes[0].set_title(title, fontsize=9, pad=4, loc="left", color="dimgray", style="italic")
    for index, ax in enumerate(axes):
        x_values, y_values = downsample(values[:, index], max_points)
        ax.plot(x_values, y_values, lw=1.0)
        ax.set_ylabel(labels[index] if index < len(labels) else f"{title}_{index}", fontsize=7, labelpad=2)
        ax.yaxis.set_tick_params(labelsize=6)
        ax.xaxis.set_tick_params(labelsize=6)
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("frame", fontsize=7)


def draw_scalar_group(
    fig: Any,
    spec: Any,
    title: str,
    values: np.ndarray,
    max_points: int,
) -> None:
    ax = fig.add_subplot(spec)
    x_values, y_values = downsample(values, max_points)
    ax.plot(x_values, y_values, lw=1.0)
    ax.set_title(title, fontsize=9, pad=4, loc="left", color="dimgray", style="italic")
    ax.set_xlabel("frame", fontsize=7)
    ax.set_ylabel(title, fontsize=7, labelpad=2)
    ax.yaxis.set_tick_params(labelsize=6)
    ax.xaxis.set_tick_params(labelsize=6)
    ax.grid(True, alpha=0.25)


def print_summary(name: str, values: np.ndarray | None) -> None:
    if values is None:
        print(f"[summary] {name}: missing")
        return
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        print(f"[summary] {name}: shape={values.shape}, no finite values")
        return
    print(
        f"[summary] {name}: shape={values.shape} "
        f"mean={finite_values.mean():.6g} min={finite_values.min():.6g} max={finite_values.max():.6g}"
    )


def episode_boundaries(df: pd.DataFrame) -> list[int]:
    if "episode_index" not in df.columns or df["episode_index"].nunique() <= 1:
        return []
    changes = df["episode_index"].to_numpy()[1:] != df["episode_index"].to_numpy()[:-1]
    return [int(index + 1) for index in np.flatnonzero(changes)]


def add_episode_boundaries(fig: Any, boundaries: list[int]) -> None:
    if not boundaries:
        return
    for ax in fig.axes:
        for boundary in boundaries:
            ax.axvline(boundary, color="black", lw=0.6, alpha=0.25, linestyle=":")


def main() -> None:
    global gridspec, plt

    args = parse_args()

    try:
        import matplotlib.gridspec as matplotlib_gridspec
        import matplotlib.pyplot as matplotlib_pyplot
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required for plotting. Install it in the fire environment "
            "with `pip install matplotlib`."
        ) from exc

    gridspec = matplotlib_gridspec
    plt = matplotlib_pyplot

    try:
        obs_features, log_features, action_features = load_task_features(args.task)
    except ImportError as exc:
        print(f"[warn] task feature loading skipped: {exc}")
        print("[warn] Source the ROS workspace first if you need task feature dims.")
        obs_features, log_features, action_features = {}, {}, {}

    if args.root is not None:
        root = args.root
        df = load_dataset(args.root, args.episode)
    else:
        data_path = args.data
        root = data_path.parent.parent.parent
        df = read_dataframe(data_path)

    state_names = read_metadata_names(root, "observation.state") or DEFAULT_STATE_NAMES
    action_names = read_metadata_names(root, "action") or DEFAULT_ACTION_NAMES

    state = vector_matrix(df, "observation.state", ["observation.state", "state"])
    action = vector_matrix(df, "action", ["action", "policy_action"])
    reward = scalar_series(df, "next.reward")
    done = scalar_series(df, "next.done")

    print(f"[info] task={args.task}")
    print(
        f"[info] task feature dims: obs={total_dim(obs_features)} "
        f"log={total_dim(log_features)} action={total_dim(action_features)}"
    )
    print_summary("observation.state", state)
    print_summary("action", action)
    print_summary("next.reward", reward)
    print_summary("next.done", done)

    panels: list[tuple[str, np.ndarray, list[str]]] = []
    if state is not None:
        panels.append(("Observation State", state, state_names))
    if action is not None:
        panels.append(("Action", action, action_names))

    scalars: list[tuple[str, np.ndarray]] = []
    if reward is not None:
        scalars.append(("next.reward", reward))
    if done is not None:
        scalars.append(("next.done", done))

    if not panels and not scalars:
        raise ValueError("No plottable LeRobot columns found.")

    rows = max(
        [values.shape[1] for _, values, _ in panels] + [1 for _ in scalars],
        default=1,
    )
    cols = len(panels) + len(scalars)
    fig = plt.figure(figsize=(8 * cols, max(rows * 0.75, 8)))
    fig.suptitle(f"{args.task} LeRobot dataset ({len(df)} frames)", fontsize=13)
    grid = gridspec.GridSpec(
        rows,
        cols,
        figure=fig,
        hspace=0.08,
        wspace=0.35,
        left=0.06,
        right=0.98,
        top=0.94,
        bottom=0.06,
    )

    col = 0
    for title, values, labels in panels:
        draw_vector_group(fig, grid[: values.shape[1], col], title, values, labels, args.max_points)
        col += 1
    for title, values in scalars:
        draw_scalar_group(fig, grid[0, col], title, values, args.max_points)
        col += 1

    add_episode_boundaries(fig, episode_boundaries(df))

    if args.save_path is not None:
        os.makedirs(args.save_path.parent if args.save_path.parent != Path("") else Path("."), exist_ok=True)
        fig.savefig(args.save_path, dpi=150, bbox_inches="tight")
        print(f"[plot] saved -> {args.save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
