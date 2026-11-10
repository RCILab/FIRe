# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from RL-Games.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rl_games_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_last_checkpoint",
    action="store_true",
    help="When no checkpoint provided, use the last saved model. Otherwise use the best saved model.",
)
parser.add_argument(
    "--save_actions",
    action="store_true",
    default=False,
    help="Save per-step VLA / residual / final / applied actions to a CSV file (VLA residual envs only).",
)
parser.add_argument("--max_steps", type=int, default=None, help="Stop playing after this many env steps.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import csv
import datetime
import gymnasium as gym
import math
import os
import random
import torch

from rl_games.common import env_configurations, vecenv
from rl_games.common.player import BasePlayer
from rl_games.torch_runner import Runner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

import isaaclab_tasks
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import firelab.tasks


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):

    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.device is not None:
        agent_cfg["params"]["config"]["device"] = args_cli.device
        agent_cfg["params"]["config"]["device_name"] = args_cli.device

    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    agent_cfg["params"]["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["params"]["seed"]
    env_cfg.seed = agent_cfg["params"]["seed"]

    log_root_path = os.path.join("logs", "rl_games", agent_cfg["params"]["config"]["name"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.checkpoint is None:
        run_dir = agent_cfg["params"]["config"].get("full_experiment_name", ".*")
        if args_cli.use_last_checkpoint:
            checkpoint_file = ".*"
        else:
            checkpoint_file = f"{agent_cfg['params']['config']['name']}.pth"
        resume_path = get_checkpoint_path(log_root_path, run_dir, checkpoint_file, other_dirs=["nn"])
    else:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    log_dir = os.path.dirname(os.path.dirname(resume_path))

    env_cfg.log_dir = log_dir

    act_csv_file = None
    act_csv_writer = None
    act_csv_path = None
    if args_cli.save_actions:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        act_csv_path = os.path.join(log_dir, f"actions_{timestamp}.csv")
        os.makedirs(log_dir, exist_ok=True)
        act_csv_file = open(act_csv_path, "w", newline="")
        print(f"[INFO] Saving actions to: {act_csv_path}")

    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    obs_groups = agent_cfg["params"]["env"].get("obs_groups")
    concate_obs_groups = agent_cfg["params"]["env"].get("concate_obs_groups", True)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_root_path, log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions, obs_groups, concate_obs_groups)

    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = resume_path
    print(f"[INFO]: Loading model checkpoint from: {agent_cfg['params']['load_path']}")

    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    runner = Runner()
    runner.load(agent_cfg)
    agent: BasePlayer = runner.create_player()
    agent.restore(resume_path)
    agent.reset()

    if args_cli.save_actions:
        if not hasattr(env.unwrapped, "record_actions"):
            raise ValueError("--save_actions requires a VLA residual env (e.g. DirectRLVLAEnv).")
        env.unwrapped.record_actions = True

    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    timestep = 0
    step = 0
    _ = agent.get_batch_size(obs, 1)
    if agent.is_rnn:
        agent.init_rnn()
    while simulation_app.is_running():
        step += 1
        with torch.inference_mode():
            obs = agent.obs_to_torch(obs)
            actions = agent.get_action(obs, is_deterministic=agent.is_deterministic)
            obs, _, dones, _ = env.step(actions)

            if args_cli.save_actions:
                rec = env.unwrapped.recorded_actions
                keys = [k for k in rec if k != "ep_step"]
                if act_csv_writer is None:
                    act_csv_writer = csv.writer(act_csv_file)
                    header = ["step", "env_id", "ep_step"]
                    for key in keys:
                        header += [key] if rec[key].dim() == 1 else [f"{key}_{i}" for i in range(rec[key].shape[-1])]
                    act_csv_writer.writerow(header)
                cols = [rec["ep_step"].unsqueeze(-1).float()]
                cols += [(rec[k].unsqueeze(-1) if rec[k].dim() == 1 else rec[k]).float() for k in keys]
                rows = torch.cat(cols, dim=-1).cpu().tolist()
                for env_id, row in enumerate(rows):
                    row[0] = int(row[0])
                    act_csv_writer.writerow([step, env_id] + row)

            if len(dones) > 0:
                if agent.is_rnn and agent.states is not None:
                    for s in agent.states:
                        s[:, dones, :] = 0.0
        if args_cli.video:
            timestep += 1
            if timestep == args_cli.video_length:
                break
        if args_cli.max_steps is not None and step >= args_cli.max_steps:
            break

    if act_csv_file is not None:
        act_csv_file.close()
        print(f"[INFO] Actions saved to: {act_csv_path}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
