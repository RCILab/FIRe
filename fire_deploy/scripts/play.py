"""play.py

Franka FR3 VLA inference runner.

Step ordering — mirrors Isaac Lab DirectRLEnv
---------------------------------------------
Isaac Lab:
  reset()  →  obs_0  (zero action initialized)
  loop:
    action_t = policy(obs_t)     # deterministic policy output
    pre_physics_step(action_t)   # buffer action
    apply_action()               # send buffered action → robot/sim
    sim.step()                   # physics
    obs_t+1 = get_observations() # collect next obs

This runner:
  connect()  →  obs_0  (zero action initialized)
  loop:
    action_t = infer(obs_t)      # deterministic policy output
    send_action(action_t)        # apply buffered action first  ← pre_physics + apply_action
    obs_t+1 = get_observation()  # collect obs after physics    ← get_observations
    action_t+1 = infer(obs_t+1)  # compute next action (async)  ← policy

Architecture
------------
Main Process (A)  ←→  Inference Process (B)
  - ROS2 / robot I/O         - rl-games agent (GPU)
  - control loop @ N Hz      - spin-wait on obs_flag
  - CSV logging              - writes action to shared mem

Communication: shared torch tensors + RawValue spin-flags
  obs_flag  : 0=idle  1=obs_ready
  action_flag: 0=idle  1=action_ready  2=warmup_done
"""

from __future__ import annotations

import argparse
import ctypes
from multiprocessing import RawValue
from typing import List, Optional

import cv2
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp

from fire_core.checkpoints import resolve_checkpoint_path
from fire_core.utils import total_dim, flat_keys
from fire_core.inference import start_inference_process
from fire_core.strategies import (
    LiveInferenceStrategy,
    LiveInferenceWithVLAStrategy,
    VLAOnlyStrategy,
    ReplayRawStrategy,
    ReplayPoseStrategy,
)
from fire_core.logger import StepLogger
from fire_core.loop import run_control_loop


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--task",           default="peg_insert")
    checkpoint_group = p.add_mutually_exclusive_group()
    checkpoint_group.add_argument("--checkpoint", help="Local checkpoint path.")
    checkpoint_group.add_argument(
        "--hf_checkpoint",
        help="Hugging Face checkpoint path, e.g. user/repo or user/repo/path/to/model.pth.",
    )
    p.add_argument("--device",         default="cuda:0")
    p.add_argument("--control_hz",     type=float, default=15.0)
    p.add_argument("--episode_length", type=int, default=88, help="Max steps per episode (for logging only)")
    p.add_argument("--replay",         default=None, metavar="CSV")
    p.add_argument("--save_path",      default=None)
    p.add_argument("--use_cameras",    action="store_true")
    p.add_argument("--use_ft_sensor",  action="store_true")
    p.add_argument("--use_sim_time",   action="store_true")
    p.add_argument("--vla", type=str, default=None, choices=["gr00t", "pi05", "openvla"])
    p.add_argument(
        "--vla_chunk_size", type=int, default=None,
        help="Action chunk size. Defaults per backend: gr00t=16, pi05=8, openvla=8.",
    )
    p.add_argument("--host", type=str, default=None, choices=["localhost", "163.180.160.225", "163.180.164.55"])
    p.add_argument("--port", type=int, default=None)

    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--raw",  action="store_true", help="Replay: CSV obs → inference")
    mode.add_argument("--pose", action="store_true", help="Replay: CSV target pose → direct send")

    args = p.parse_args()
    if (args.raw or args.pose) and not args.replay:
        p.error("--raw / --pose requires --replay")
    if args.replay and not (args.raw or args.pose):
        p.error("--replay requires --raw or --pose")

    no_ckpt = args.checkpoint is None and args.hf_checkpoint is None
    if no_ckpt:
        if args.raw:
            p.error("--raw replay requires --checkpoint or --hf_checkpoint.")
        if not args.replay and args.vla is None:
            p.error(
                "Provide --checkpoint/--hf_checkpoint for an RL (or RL+VLA) run, "
                "or --vla alone for a VLA-only run with no RL policy."
            )
    return args


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    from lerobot_robot_fr3.config_fr3 import FR3RobotConfig
    from lerobot_robot_fr3.fr3 import FR3Robot
    from fire_core.vla_clients import (
        AsyncGr00tInferenceClient, AsyncPi05InferenceClient,
        AsyncOpenVLAInferenceClient,
    )

    args = parse_args()
    checkpoint_path = None
    if args.checkpoint or args.hf_checkpoint:
        checkpoint_path = resolve_checkpoint_path(
            checkpoint=args.checkpoint,
            hf_checkpoint=args.hf_checkpoint,
        )
    needs_inference = checkpoint_path is not None and not (args.replay and args.pose)

    # ── Robot object (access task features before connect) ───────────────────────────
    config = FR3RobotConfig(
        use_sim_time=args.use_sim_time,
        is_relative=False,
        rotation_type="quaternion",
        use_cameras=args.use_cameras,
        use_ft_sensor=args.use_ft_sensor,
    )
    robot = FR3Robot(config, task_name=args.task)

    obs_features    = robot.task.observation_features
    log_features    = robot.task.log_features
    action_features = robot.task.action_features
    obs_dim    = total_dim(obs_features)
    action_dim = total_dim(action_features)
    print(f"[INFO] obs_dim={obs_dim}  action_dim={action_dim}")

    # ── Shared memory ─────────────────────────────────────────────────────────
    obs_shm    = torch.zeros(1, obs_dim).share_memory_()
    action_shm = torch.zeros(1, action_dim).share_memory_()
    obs_flag    = RawValue(ctypes.c_int32, 0)
    action_flag = RawValue(ctypes.c_int32, 0)

    # ── VLA client ────────────────────────────────────────────────────────────
    VLA_DEFAULT_CHUNK_SIZE = {"gr00t": 16, "pi05": 8, "openvla": 8}
    if args.vla is not None and args.vla_chunk_size is None:
        args.vla_chunk_size = VLA_DEFAULT_CHUNK_SIZE[args.vla]

    if args.vla is not None:
        assert args.host is not None and args.port is not None
        if args.vla == "gr00t":
            vla_policy = AsyncGr00tInferenceClient(host=args.host, port=args.port)
        elif args.vla == "pi05":
            host = "127.0.0.1" if args.host == "localhost" else args.host
            vla_policy = AsyncPi05InferenceClient(host=host, port=args.port)
        elif args.vla == "openvla":
            host = "127.0.0.1" if args.host == "localhost" else args.host
            vla_policy = AsyncOpenVLAInferenceClient(
                host=host, port=args.port, task_name=args.task,
            )
    else:
        vla_policy = None

    # ── Inference process ─────────────────────────────────────────────────────
    infer_handle = None
    if needs_inference:
        infer_handle = start_inference_process(
            obs_shm, action_shm, obs_flag, action_flag,
            checkpoint=checkpoint_path,
            cfg_path=robot.task.model_cfg_path,
            obs_dim=obs_dim,
            action_dim=action_dim,
            device=args.device,
        )
        infer_handle.wait_ready(action_flag)

    # ── Robot connect ─────────────────────────────────────────────────────────
    print("[INFO] Connecting to robot ...")
    try:
        robot.connect()
    except Exception as e:
        print(f"[ERROR] {e}")
        if infer_handle:
            infer_handle.shutdown()
        return

    if not robot.is_connected:
        print("[ERROR] Failed to connect.")
        if infer_handle:
            infer_handle.shutdown()
        return

    # ── CSV load & column validation ──────────────────────────────────────────────────
    replay_data: Optional[pd.DataFrame] = None
    pose_cols: List[str] = []

    if args.replay:
        replay_data = pd.read_csv(args.replay)
        print(f"[INFO] Loaded {len(replay_data)} steps from {args.replay}")

        if args.raw:
            required = [f"obs_{i}" for i in range(obs_dim)]
        else:
            pose_cols = (
                flat_keys({"target_pos":  log_features["target_pos"]})
                + flat_keys({"target_quat": log_features["target_quat"]})
            )
            required = pose_cols

        missing = [c for c in required if c not in replay_data.columns]
        if missing:
            print(f"[ERROR] CSV missing columns: {missing}")
            if infer_handle:
                infer_handle.shutdown()
            robot.disconnect()
            return

    # ── Strategy selection ─────────────────────────────────────────────────────────
    obs_buf = obs_shm.numpy()[0]

    if args.replay and args.raw:
        strategy = ReplayRawStrategy(
            replay_data, obs_buf, obs_flag, action_flag,
            action_shm, obs_dim, action_dim,
            robot=robot, obs_features=obs_features,
        )
    elif args.replay and args.pose:
        strategy = ReplayPoseStrategy(replay_data, pose_cols, action_dim)
    elif vla_policy is not None and checkpoint_path is not None:
        strategy = LiveInferenceWithVLAStrategy(
            robot, obs_buf, obs_features,
            obs_flag, action_flag, action_shm, action_dim,
            vla_policy=vla_policy,
            vla_chunk_size=args.vla_chunk_size,
        )
    elif vla_policy is not None:
        # No RL checkpoint: drive the robot purely off the VLA server's chunk.
        strategy = VLAOnlyStrategy(
            robot, action_features,
            vla_policy=vla_policy,
            vla_chunk_size=args.vla_chunk_size,
        )
    else:
        strategy = LiveInferenceStrategy(
            robot, obs_buf, obs_features,
            obs_flag, action_flag, action_shm, action_dim,
        )

    # ── Logger ────────────────────────────────────────────────────────────────
    logger = StepLogger(robot, obs_features, log_features) if args.save_path else None

    # ── Control loop ──────────────────────────────────────────────────────────
    try:
        run_control_loop(robot, strategy, control_hz=args.control_hz, max_steps=args.episode_length, logger=logger)
    finally:
        if logger:
            logger.save(args.save_path)
        if infer_handle:
            infer_handle.shutdown()
        cv2.destroyAllWindows()
        robot.disconnect()
        print("[INFO] Robot disconnected.")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
