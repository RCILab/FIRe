# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate RL-Games checkpoints: episode-end success rate and contact-force statistics.

Every checkpoint is evaluated with the same protocol (deterministic actions, the same
number of episodes, episode-end success), so the best, final, and last-k checkpoints of
FIRe and the baselines can be compared on one metric.

Example:
    python scripts/reinforcement_learning/rl_games/eval.py --task FireLab-VLA-Gr00t-Forge-PegInsert-Direct-v1 \
        --checkpoint logs/rl_games/Forge/gr00t/peg_insert/seed1/nn --num_envs 64 --num_rounds 4 \
        --out eval/fire_peg_seed1.json --headless --enable_cameras
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate checkpoints of an RL agent from RL-Games.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rl_games_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument(
    "--checkpoint",
    type=str,
    nargs="+",
    required=True,
    help="Checkpoint files and/or directories (every *.pth inside a directory is evaluated).",
)
parser.add_argument("--num_rounds", type=int, default=4, help="Episodes per environment for each checkpoint.")
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment.")
parser.add_argument("--force_threshold", type=float, default=10.0, help="Fixed threshold for force statistics [N].")
parser.add_argument("--out", type=str, required=True, help="Output JSON path (a per-episode CSV is written next to it).")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""


import glob
import json
import math
import os
import re

import gymnasium as gym
import numpy as np
import torch

from rl_games.common import env_configurations, vecenv
from rl_games.common.player import BasePlayer
from rl_games.torch_runner import Runner

from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import fire_lab.tasks  # noqa: F401

FORCE_EMA = 0.25  # same smoothing as the real-robot force signal


def expand_checkpoints(paths):
    files = []
    for p in paths:
        if os.path.isdir(p):
            files += sorted(glob.glob(os.path.join(p, "**", "*.pth"), recursive=True))
        else:
            files.append(retrieve_file_path(p))
    return files


def checkpoint_epoch(path):
    m = re.search(r"_ep_(\d+)_", os.path.basename(path))
    return int(m.group(1)) if m else None


def episode_force_metrics(force_norm, f_ref, dt, threshold):
    """force_norm: (T,) raw |F| per policy step for one episode."""
    smooth = np.zeros_like(force_norm)
    acc = 0.0
    for i, f in enumerate(force_norm):
        acc = FORCE_EMA * f + (1.0 - FORCE_EMA) * acc
        smooth[i] = acc
    out = {
        "peak_raw": float(force_norm.max()),
        "peak": float(smooth.max()),
        "t_above_thr": float((smooth > threshold).sum() * dt),
        "impulse_above_thr": float(np.clip(smooth - threshold, 0.0, None).sum() * dt),
    }
    if f_ref is not None:
        out["f_ref"] = float(f_ref)
        out["t_above_fref"] = float((smooth > f_ref).sum() * dt)
    return out


def evaluate(env, agent, checkpoint, num_rounds, threshold):
    unwrapped = env.unwrapped
    dt = unwrapped.step_dt
    has_force = hasattr(unwrapped, "force_sensor_body_idx")

    agent.restore(checkpoint)
    agent.reset()
    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    _ = agent.get_batch_size(obs, 1)
    if agent.is_rnn:
        agent.init_rnn()

    episodes = []
    forces = [[] for _ in range(unwrapped.num_envs)]
    rounds = 0
    while rounds < num_rounds and simulation_app.is_running():
        with torch.inference_mode():
            f_ref = unwrapped.contact_penalty_thresholds.clone() if has_force else None
            obs = agent.obs_to_torch(obs)
            actions = agent.get_action(obs, is_deterministic=True)
            obs, _, dones, _ = env.step(actions)
            done_ids = dones.nonzero(as_tuple=False).squeeze(-1).tolist()

            if has_force and len(done_ids) == 0:
                f = unwrapped._robot.root_physx_view.get_link_incoming_joint_force()[
                    :, unwrapped.force_sensor_body_idx, 0:3
                ]
                f = torch.norm(f, dim=-1).cpu().numpy()
                for i in range(unwrapped.num_envs):
                    forces[i].append(f[i])

            if len(done_ids) > 0:
                successes = unwrapped.last_episode_successes.cpu().numpy()
                for i in done_ids:
                    ep = {"round": rounds, "env": i, "success": int(successes[i])}
                    if has_force and len(forces[i]) > 0:
                        ep.update(episode_force_metrics(np.array(forces[i]), float(f_ref[i]), dt, threshold))
                    episodes.append(ep)
                    forces[i] = []
                if agent.is_rnn and agent.states is not None:
                    for s in agent.states:
                        s[:, dones, :] = 0.0
                rounds += 1
    return episodes


def summarize(episodes):
    summary = {"n_episodes": len(episodes), "success_rate": float(np.mean([e["success"] for e in episodes]))}
    for key in ["peak", "t_above_thr", "impulse_above_thr", "t_above_fref"]:
        vals = [e[key] for e in episodes if key in e]
        if vals:
            summary[f"{key}_median"] = float(np.median(vals))
            summary[f"{key}_mean"] = float(np.mean(vals))
    return summary


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Evaluate RL-Games checkpoints."""
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.device is not None:
        agent_cfg["params"]["config"]["device"] = args_cli.device
        agent_cfg["params"]["config"]["device_name"] = args_cli.device
    agent_cfg["params"]["seed"] = args_cli.seed
    env_cfg.seed = args_cli.seed

    checkpoints = expand_checkpoints(args_cli.checkpoint)
    if not checkpoints:
        raise ValueError(f"No checkpoints found in {args_cli.checkpoint}")

    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    obs_groups = agent_cfg["params"]["env"].get("obs_groups")
    concate_obs_groups = agent_cfg["params"]["env"].get("concate_obs_groups", True)

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions, obs_groups, concate_obs_groups)
    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = checkpoints[0]
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    runner = Runner()
    runner.load(agent_cfg)
    agent: BasePlayer = runner.create_player()

    results, rows = [], []
    for ckpt in checkpoints:
        print(f"[INFO] Evaluating {ckpt}")
        episodes = evaluate(env, agent, ckpt, args_cli.num_rounds, args_cli.force_threshold)
        summary = {"checkpoint": ckpt, "epoch": checkpoint_epoch(ckpt), **summarize(episodes)}
        print(f"[INFO] {json.dumps(summary)}")
        results.append(summary)
        rows += [{"checkpoint": ckpt, "epoch": summary["epoch"], **e} for e in episodes]

    os.makedirs(os.path.dirname(os.path.abspath(args_cli.out)), exist_ok=True)
    with open(args_cli.out, "w") as f:
        json.dump({"task": args_cli.task, "seed": args_cli.seed, "num_rounds": args_cli.num_rounds, "results": results}, f, indent=2)
    if rows:
        import csv

        keys = sorted({k for r in rows for k in r})
        with open(os.path.splitext(args_cli.out)[0] + "_episodes.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
    print(f"[INFO] Wrote {args_cli.out}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
