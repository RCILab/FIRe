from __future__ import annotations

import argparse
import os
import re
import sys
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

from lerobot_robot_fr3.config_fr3 import FR3RobotConfig
from lerobot_robot_fr3.fr3 import FR3Robot


# ─── task feature loading ─────────────────────────────────────────────────────

def load_task_features(task_name: str) -> Tuple[Dict, Dict]:
    """Instantiate the Factory task and return observation_features, log_features.
    Accessible via __init__ alone, without robot.connect()."""
    try:
        from lerobot_robot_fr3.tasks.factory.factory import Factory
    except ImportError as e:
        print(f"[ERROR] Cannot import Factory task: {e}")
        sys.exit(1)
    
    config = FR3RobotConfig()
    robot = FR3Robot(config, task_name=task_name)
    task = robot.task
    return task.observation_features, task.log_features


# ─── sim CSV column name normalization ────────────────────────────────────────────────────
# sim CSV: x/y/z, w/x/y/z naming → _0/_1/_2, _0/_1/_2/_3 index naming

_AXIS_TO_IDX = {"x": 0, "y": 1, "z": 2}
_QUAT_TO_IDX = {"w": 0, "x": 1, "y": 2, "z": 3}


def _normalize_sim_df(df: pd.DataFrame) -> pd.DataFrame:
    """Rename the sim CSV's x/y/z, w/x/y/z column suffixes to _0/_1/... indices."""
    rename_map: Dict[str, str] = {}
    for col in df.columns:
        m = re.match(r"^(.+_quat)_([wxyz])$", col)
        if m:
            rename_map[col] = f"{m.group(1)}_{_QUAT_TO_IDX[m.group(2)]}"
            continue
        m = re.match(r"^(.+)_([xyz])$", col)
        if m:
            rename_map[col] = f"{m.group(1)}_{_AXIS_TO_IDX[m.group(2)]}"
    return df.rename(columns=rename_map)


def _indexed_dim(df: pd.DataFrame, prefix: str) -> int:
    i = 0
    while f"{prefix}_{i}" in df.columns:
        i += 1
    return i


def _copy_indexed_alias(df: pd.DataFrame, dst_prefix: str, src_prefix: str) -> pd.DataFrame:
    if _indexed_dim(df, dst_prefix) > 0 or _indexed_dim(df, src_prefix) == 0:
        return df

    out = df.copy()
    for i in range(_indexed_dim(df, src_prefix)):
        out[f"{dst_prefix}_{i}"] = out[f"{src_prefix}_{i}"]
    return out


def _ensure_comparison_aliases(df: pd.DataFrame) -> pd.DataFrame:
    # New names are explicit; old CSVs used normalized_action/raw_action.
    df = _copy_indexed_alias(df, "policy_action", "normalized_action")
    df = _copy_indexed_alias(df, "ema_action", "raw_action")
    return df


# ─── feature → column name / y-axis label helpers ───────────────────────────────────────

def features_total_dim(features: Dict[str, Tuple[int, ...]]) -> int:
    return sum(int(np.prod(s)) for s in features.values())


def feature_flat_cols(key: str, shape: Tuple[int, ...]) -> List[str]:
    """List of flattened column names for a single feature key.  e.g. ee_pos → [ee_pos_0, ee_pos_1, ee_pos_2]"""
    return [f"{key}_{i}" for i in range(int(np.prod(shape)))]


def feature_ylabels(key: str, shape: Tuple[int, ...]) -> List[str]:
    """Y-axis labels with an x/y/z or w/x/y/z suffix depending on dim."""
    dim = int(np.prod(shape))
    if dim == 3:
        suffixes = ["x", "y", "z"]
    elif dim == 4:
        suffixes = ["w", "x", "y", "z"]
    else:
        suffixes = [str(i) for i in range(dim)]
    return [f"{key}_{s}" for s in suffixes]


def make_obs_groups(obs_features: Dict[str, Tuple[int, ...]]) -> List[dict]:
    """Build obs_N-based subgroup definitions from observation_features."""
    groups: List[dict] = []
    idx = 0
    for key, shape in obs_features.items():
        dim = int(np.prod(shape))
        start, end = idx, idx + dim - 1
        groups.append({
            "key":      key,
            "title":    f"{key}  (obs {start}–{end})",
            "obs_cols": [f"obs_{idx + i}" for i in range(dim)],
            "ylabels":  feature_ylabels(key, shape),
        })
        idx += dim
    return groups


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="sim2real comparison driven by task features."
    )
    p.add_argument("--task", type=str, required=True,
                   help="Task name, e.g. factory-peg_insert or forge-peg_insert")
    p.add_argument("--sim",  type=str, required=True,
                   help="Sim CSV path (forge_env.py logging)")
    p.add_argument("--real", type=str, required=True,
                   help="Real CSV path (play.py --save_path)")
    p.add_argument("--save_path", type=str, default=None,
                   help="Output PNG path. Omit for interactive plt.show().")
    p.add_argument("--prev_action_sim_shift", type=int, default=0,
                   help=(
                       "Shift sim prev_actions rows before comparison. "
                       "Use 1 to compare real[i].prev_actions with sim[i+1].prev_actions."
                   ))
    p.add_argument("--prev_action_dims", type=str, default=None,
                   help=(
                       "Comma-separated prev_actions dims for numeric error summary. "
                       "Default: all dims, except forge-style 7D uses 0,1,2,5,6."
                   ))
    p.add_argument("--policy_action_dims", type=str, default=None,
                   help="Comma-separated policy_action dims for numeric error summary. Default: all dims.")
    return p.parse_args()


# ─── data loading ─────────────────────────────────────────────────────────────

def load_sim(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "episode" in df.columns and "step" in df.columns:
        if df["episode"].nunique() == len(df):
            df = df.sort_values("episode").reset_index(drop=True)
            print(f"[sim]  episode col treated as step index → {len(df)} rows")
        else:
            first_ep = df["episode"].iloc[0]
            df = df[df["episode"] == first_ep].sort_values("step").reset_index(drop=True)
            print(f"[sim]  episode={first_ep} → {len(df)} steps")
    else:
        print(f"[sim]  no episode/step cols → {len(df)} rows")
    df = _normalize_sim_df(df)
    df = _ensure_comparison_aliases(df)
    print(f"[sim]  loaded {len(df)} steps ← {os.path.basename(path)}")
    return df


def load_real(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = _ensure_comparison_aliases(df)
    print(f"[real] {len(df)} steps ← {os.path.basename(path)}")
    return df


def safe_extract(df: pd.DataFrame, cols: List[str], label: str) -> np.ndarray:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"[{label}] missing columns: {missing}")
    return df[cols].to_numpy(dtype=np.float64)


def _parse_dim_list(dim_text: str | None, dim: int, default_dims: List[int] | None = None) -> List[int]:
    if dim_text is None:
        return default_dims if default_dims is not None else list(range(dim))

    dims = [int(x.strip()) for x in dim_text.split(",") if x.strip()]
    bad = [i for i in dims if i < 0 or i >= dim]
    if bad:
        raise ValueError(f"dimension list out of range for dim={dim}: {bad}")
    return dims


def _align_by_sim_shift(
    sim_data: np.ndarray,
    real_data: np.ndarray,
    sim_shift: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """sim_shift=1 means compare sim[i+1] against real[i]."""
    if sim_shift > 0:
        sim_data = sim_data[sim_shift:]
    elif sim_shift < 0:
        real_data = real_data[-sim_shift:]

    n_min = min(len(sim_data), len(real_data))
    return sim_data[:n_min], real_data[:n_min]


def print_indexed_error_summary(
    sim_df: pd.DataFrame,
    real_df: pd.DataFrame,
    prefix: str,
    label: str,
    dim_text: str | None = None,
) -> None:
    dim = min(_indexed_dim(sim_df, prefix), _indexed_dim(real_df, prefix))
    if dim == 0:
        print(f"[{label}] no common {prefix}_* columns")
        return

    cols = [f"{prefix}_{i}" for i in range(dim)]
    sim_data = safe_extract(sim_df, cols, f"sim/{label}")
    real_data = safe_extract(real_df, cols, f"real/{label}")
    n_min = min(len(sim_data), len(real_data))
    sim_data = sim_data[:n_min]
    real_data = real_data[:n_min]

    dims = _parse_dim_list(dim_text, dim)
    diff = real_data[:, dims] - sim_data[:, dims]
    abs_diff = np.abs(diff)

    print(f"[{label}] compare real[i] vs sim[i] rows={len(diff)} dims={dims}")
    print(f"[{label}] abs diff: mean={abs_diff.mean():.6g}  max={abs_diff.max():.6g}")
    for col, dim_idx in enumerate(dims):
        print(
            f"  dim {dim_idx}: "
            f"mean={abs_diff[:, col].mean():.6g}  "
            f"max={abs_diff[:, col].max():.6g}"
        )


def print_obs_error_summary(
    obs_features: Dict[str, Tuple[int, ...]],
    sim_df: pd.DataFrame,
    real_df: pd.DataFrame,
) -> None:
    dim = features_total_dim(obs_features)
    cols = [f"obs_{i}" for i in range(dim)]
    sim_data = safe_extract(sim_df, cols, "sim/obs")
    real_data = safe_extract(real_df, cols, "real/obs")
    n_min = min(len(sim_data), len(real_data))
    diff = real_data[:n_min] - sim_data[:n_min]
    abs_diff = np.abs(diff)
    print(f"[obs] compare real[i] vs sim[i] rows={n_min} dims=all")
    print(f"[obs] abs diff: mean={abs_diff.mean():.6g}  max={abs_diff.max():.6g}")


def print_prev_action_error_summary(
    obs_features: Dict[str, Tuple[int, ...]],
    sim_df: pd.DataFrame,
    real_df: pd.DataFrame,
    sim_shift: int,
    dim_text: str | None,
) -> None:
    prev_group = next(
        (grp for grp in make_obs_groups(obs_features) if grp["key"] == "prev_actions"),
        None,
    )
    if prev_group is None:
        print("[prev_actions] task has no prev_actions observation")
        return

    sim_data = safe_extract(sim_df, prev_group["obs_cols"], "sim/prev_actions")
    real_data = safe_extract(real_df, prev_group["obs_cols"], "real/prev_actions")
    sim_data, real_data = _align_by_sim_shift(sim_data, real_data, sim_shift)

    dim = sim_data.shape[1]
    default_dims = [i for i in [0, 1, 2, 5, 6] if i < dim] if dim >= 7 else list(range(dim))
    dims = _parse_dim_list(dim_text, dim, default_dims)
    diff = real_data[:, dims] - sim_data[:, dims]
    abs_diff = np.abs(diff)

    shift_expr = f"i+{sim_shift}" if sim_shift >= 0 else f"i{sim_shift}"
    print(
        f"[prev_actions] compare real[i] vs sim[{shift_expr}] "
        f"rows={len(diff)} dims={dims}"
    )
    print(
        f"[prev_actions] abs diff: "
        f"mean={abs_diff.mean():.6g}  max={abs_diff.max():.6g}"
    )
    for col, dim_idx in enumerate(dims):
        print(
            f"  dim {dim_idx}: "
            f"mean={abs_diff[:, col].mean():.6g}  "
            f"max={abs_diff[:, col].max():.6g}"
        )


# ─── drawing primitives ───────────────────────────────────────────────────────

def _plot_pair(ax, sim_vals, real_vals, ylabel, show_legend=False):
    """Overlay one sim/real channel on a single ax."""
    n_min = min(len(sim_vals), len(real_vals))
    sim_x  = np.arange(len(sim_vals))
    real_x = np.arange(len(real_vals))

    ax.plot(sim_x[:n_min],  sim_vals[:n_min],  color="C0", lw=1.0,
            label="sim" if show_legend else None)
    if len(sim_vals) > n_min:
        ax.plot(sim_x[n_min:], sim_vals[n_min:], color="C0", lw=1.0,
                linestyle=":", alpha=0.5)

    ax.plot(real_x[:n_min], real_vals[:n_min], color="C1", lw=1.0,
            label="real" if show_legend else None)
    if len(real_vals) > n_min:
        ax.plot(real_x[n_min:], real_vals[n_min:], color="C1", lw=1.0,
                linestyle=":", alpha=0.5)

    ax.set_ylabel(ylabel, fontsize=7, labelpad=2)
    ax.yaxis.set_tick_params(labelsize=6)
    ax.xaxis.set_tick_params(labelsize=6)
    ax.grid(True, alpha=0.25)
    if show_legend:
        ax.legend(loc="upper right", fontsize=7)


def _draw_feature_panel(
    fig,
    gs_cell,
    groups: List[dict],        # [{title, cols, ylabels}, ...]
    sim_df: pd.DataFrame,
    real_df: pd.DataFrame,
    n_total_rows: int,
    group_sim_shifts: Dict[str, int] | None = None,
):
    """Common panel renderer: takes group definitions and draws each subgroup."""
    inner_gs = gridspec.GridSpecFromSubplotSpec(
        n_total_rows, 1,
        subplot_spec=gs_cell,
        hspace=0.10,
    )

    row_cursor  = 0
    first_group = True
    group_sim_shifts = group_sim_shifts or {}

    for grp in groups:
        cols    = grp["cols"]
        labels  = grp["ylabels"]
        dim     = len(cols)
        key     = grp.get("key", "")

        sim_data  = safe_extract(sim_df,  cols, f"sim/{grp['title']}")
        real_data = safe_extract(real_df, cols, f"real/{grp['title']}")
        sim_shift = group_sim_shifts.get(key, 0)
        if sim_shift:
            sim_data, real_data = _align_by_sim_shift(sim_data, real_data, sim_shift)

        axes = [fig.add_subplot(inner_gs[row_cursor + i, 0]) for i in range(dim)]
        for ax in axes[1:]:
            ax.sharex(axes[0])
        for ax in axes[:-1]:
            plt.setp(ax.get_xticklabels(), visible=False)

        title = grp["title"]
        if sim_shift:
            title = f"{title}  (real[i] vs sim[i+{sim_shift}])"
        axes[0].set_title(title, fontsize=8, pad=3,
                          loc="left", color="dimgray", style="italic")

        for i, (ax, ylabel) in enumerate(zip(axes, labels)):
            _plot_pair(
                ax,
                sim_data[:, i], real_data[:, i],
                ylabel=ylabel,
                show_legend=(first_group and i == 0),
            )
        if first_group:
            first_group = False

        axes[-1].set_xlabel("step", fontsize=7)
        row_cursor += dim


# ─── panel builders ───────────────────────────────────────────────────────────

def draw_log_panel(fig, gs_cell,
                   log_features: Dict[str, Tuple[int, ...]],
                   sim_df: pd.DataFrame, real_df: pd.DataFrame):
    """Left panel: draws each log_features key as a subgroup."""
    groups = [
        {
            "key":     key,
            "title":   key,
            "cols":    feature_flat_cols(key, shape),
            "ylabels": feature_ylabels(key, shape),
        }
        for key, shape in log_features.items()
    ]
    _draw_feature_panel(fig, gs_cell, groups, sim_df, real_df,
                        n_total_rows=features_total_dim(log_features))


def draw_obs_panel(fig, gs_cell,
                   obs_features: Dict[str, Tuple[int, ...]],
                   sim_df: pd.DataFrame, real_df: pd.DataFrame,
                   prev_action_sim_shift: int = 0):
    """Right panel: draws observation_features subgroups based on obs_N columns."""
    groups = [
        {
            "key":     grp["key"],
            "title":   grp["title"],
            "cols":    grp["obs_cols"],
            "ylabels": grp["ylabels"],
        }
        for grp in make_obs_groups(obs_features)
    ]
    group_sim_shifts = {}
    if prev_action_sim_shift:
        group_sim_shifts["prev_actions"] = prev_action_sim_shift
    _draw_feature_panel(fig, gs_cell, groups, sim_df, real_df,
                        n_total_rows=features_total_dim(obs_features),
                        group_sim_shifts=group_sim_shifts)


def draw_indexed_panel(fig, gs_cell,
                       prefix: str, title: str,
                       sim_df: pd.DataFrame, real_df: pd.DataFrame):
    dim = min(_indexed_dim(sim_df, prefix), _indexed_dim(real_df, prefix))
    if dim == 0:
        return
    groups = [{
        "key":     prefix,
        "title":   title,
        "cols":    [f"{prefix}_{i}" for i in range(dim)],
        "ylabels": [f"{prefix}_{i}" for i in range(dim)],
    }]
    _draw_feature_panel(fig, gs_cell, groups, sim_df, real_df, n_total_rows=dim)


# ─── entry point ──────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    obs_features, log_features = load_task_features(args.task)

    df_sim  = load_sim(args.sim)
    df_real = load_real(args.real)

    n_log_rows = features_total_dim(log_features)
    n_obs_rows = features_total_dim(obs_features)
    n_action_rows = min(_indexed_dim(df_sim, "policy_action"), _indexed_dim(df_real, "policy_action"))
    n_rows = max(n_log_rows, n_obs_rows, n_action_rows, 1)

    print(
        f"[info] task={args.task}  "
        f"log_rows={n_log_rows}  obs_rows={n_obs_rows}  policy_action_rows={n_action_rows}"
    )
    print_obs_error_summary(obs_features, df_sim, df_real)
    print_indexed_error_summary(
        df_sim,
        df_real,
        "policy_action",
        "policy_action",
        args.policy_action_dims,
    )
    print_prev_action_error_summary(
        obs_features,
        df_sim,
        df_real,
        args.prev_action_sim_shift,
        args.prev_action_dims,
    )

    # ── figure ────────────────────────────────────────────────────────────
    fig_h = max(n_rows * 0.85, 12)
    fig = plt.figure(figsize=(24, fig_h))
    fig.suptitle(args.task, fontsize=13)

    gs = gridspec.GridSpec(
        n_rows, 3,
        figure=fig,
        hspace=0.08, wspace=0.35,
        left=0.05, right=0.98,
        top=0.95,   bottom=0.04,
    )

    fig.text(0.18, 0.972, "Current EE Pose / Target EE Pose",
             ha="center", fontsize=12, fontweight="bold")
    fig.text(0.51, 0.972, "Observations",
             ha="center", fontsize=12, fontweight="bold")
    fig.text(0.84, 0.972, "Policy Actions",
             ha="center", fontsize=12, fontweight="bold")

    draw_log_panel(fig, gs[:n_log_rows, 0], log_features, df_sim, df_real)
    draw_obs_panel(
        fig,
        gs[:n_obs_rows, 1],
        obs_features,
        df_sim,
        df_real,
        prev_action_sim_shift=args.prev_action_sim_shift,
    )
    if n_action_rows:
        draw_indexed_panel(fig, gs[:n_action_rows, 2], "policy_action", "policy_action", df_sim, df_real)

    print(f"[plot] sim={len(df_sim)} steps  real={len(df_real)} steps")

    if args.save_path:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_path)) or ".", exist_ok=True)
        fig.savefig(args.save_path, dpi=150, bbox_inches="tight")
        print(f"[plot] saved → {args.save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
