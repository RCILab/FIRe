from __future__ import annotations

import argparse
import ctypes
import sys
import time
from multiprocessing import RawValue
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.multiprocessing as mp
from scipy.spatial.transform import Rotation

from fire_core.checkpoints import resolve_checkpoint_path
from fire_core.utils import total_dim
from fire_core.inference import start_inference_process
from fire_core.strategies import LiveInferenceStrategy
from fire_core.logger import StepLogger
from fire_core.loop import run_control_loop


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _wxyz_to_rotation(q: np.ndarray) -> Rotation:
    """[qw, qx, qy, qz] → scipy Rotation (internal XYZW)."""
    return Rotation.from_quat([q[1], q[2], q[3], q[0]])


def _rotation_to_wxyz(r: Rotation) -> np.ndarray:
    x, y, z, w = r.as_quat()
    return np.array([w, x, y, z], dtype=np.float32)


_AXIS_CHOICES: tuple[str, ...] = ("+x", "-x", "+y", "-y", "+z", "-z")


def _normalize_axis_args(argv: list[str]) -> list[str]:
    normalized: list[str] = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg in {"--position_axes", "--rotation_axes"} and i + 3 < len(argv):
            axes = argv[i + 1:i + 4]
            if all(axis in _AXIS_CHOICES for axis in axes):
                normalized.append(f"{arg}={','.join(axes)}")
                i += 4
                continue
        normalized.append(arg)
        i += 1
    return normalized


def _parse_axes(value: str) -> tuple[str, str, str]:
    axes = tuple(part.strip() for part in value.replace(" ", ",").split(",") if part.strip())
    if len(axes) != 3 or any(axis not in _AXIS_CHOICES for axis in axes):
        raise argparse.ArgumentTypeError(
            "expected 3 signed axes from +x, -x, +y, -y, +z, -z"
        )
    axis_names = {axis[1] for axis in axes}
    if axis_names != {"x", "y", "z"}:
        raise argparse.ArgumentTypeError("axes must use x, y, z exactly once")
    return axes  # type: ignore[return-value]


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--task",         default="forge-peg_insert")
    p.add_argument("--control_hz",   type=float, default=15.0)
    p.add_argument("--use_cameras",  action="store_true")
    p.add_argument("--use_ft_sensor", action="store_true")
    p.add_argument("--use_sim_time", action="store_true")
    p.add_argument("--episode_length", type=int, default=88)
    p.add_argument("--lerobot_repo_id", default=None)
    p.add_argument("--lerobot_root",    default=None)
    p.add_argument("--lerobot_task",    default=None)
    p.add_argument("--obs_save_path",   default=None, help="Directory to save per-step observation CSV (like play.py --save_path).")
    p.add_argument("--resume", action="store_true")
    p.add_argument(
        "--last_episode",
        nargs="?", const=-1, type=int, default=None,
        help=(
            "After recording, encode deferred GR00T videos up to this episode "
            "index. Omit value to encode through the latest episode."
        ),
    )
    p.add_argument("--encoder_threads", type=int, default=None)

    # ── Mode ──────────────────────────────────────────────────────────────────
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--teleop", choices=["inverse3"], default=None,
        help="Use a teleop device for demonstration recording.",
    )

    # ── Model mode ────────────────────────────────────────────────────────────
    ckpt = p.add_mutually_exclusive_group()
    ckpt.add_argument("--checkpoint")
    ckpt.add_argument("--hf_checkpoint")
    p.add_argument("--device", default="cuda:0")

    # ── Inverse3 teleop options ───────────────────────────────────────────────
    inv3 = p.add_argument_group("Inverse3 Teleop")
    inv3.add_argument("--inv3_port",       default="/dev/inverse3_left")
    inv3.add_argument("--versegrip_port",  default="/dev/versegrip_left")
    inv3.add_argument("--position_scale",  type=float, default=3.0)
    inv3.add_argument(
        "--position_axes",
        type=_parse_axes,
        default=("-x", "-y", "+z"),
        metavar="ROBOT_X,ROBOT_Y,ROBOT_Z",
        help=(
            "Signed Inverse3 axes used for robot XYZ. Example: "
            "--position_axes +y -x +z maps robot x=+device y, robot y=-device x."
        ),
    )
    inv3.add_argument(
        "--rotation_axes",
        type=_parse_axes,
        default=("-y", "+x", "+z"),
        metavar="ROBOT_X,ROBOT_Y,ROBOT_Z",
        help=(
            "Right-handed signed VerseGrip frame mapping for robot XYZ. "
            "Must have determinant +1."
        ),
    )
    inv3.add_argument(
        "--absolute_teleop",
        action="store_true",
        default=True,
        help="Keep the initial device/robot home fixed instead of re-anchoring on enable.",
    )
    inv3.add_argument(
        "--require_calibration",
        action="store_true",
        default=True,
        help="Do not send motion until calibration_button has been pressed.",
    )
    inv3.add_argument("--enable_button",   type=int, default=-1)
    inv3.add_argument("--calibration_button", type=int, default=2,
                      help="VerseGrip button bit to re-home rotation (and position)")
    inv3.add_argument("--grasp_button", type=int, default=0)
    inv3.add_argument("--end_episode_button", type=int, default=1)
    inv3.add_argument("--gripper_open_value", type=float, default=1.0)
    inv3.add_argument("--gripper_close_value", type=float, default=-1.0)
    inv3.add_argument(
        "--haptic_feedback",
        action="store_true",
        help="Allow send_feedback() to apply external force to the Inverse3.",
    )

    args = p.parse_args(_normalize_axis_args(sys.argv[1:]))

    # ── Validation ────────────────────────────────────────────────────────────
    if args.last_episode is not None:
        if args.lerobot_root is None:
            p.error("--last_episode requires --lerobot_root")
        if args.last_episode < -1:
            p.error("--last_episode must be >= 0")

    if args.teleop is None:
        # Model mode: checkpoint required
        no_ckpt = args.checkpoint is None and args.hf_checkpoint is None
        if no_ckpt:
            if args.last_episode is not None:
                return args
            p.error("Model mode requires --checkpoint or --hf_checkpoint")
    else:
        # Teleop mode: checkpoint not needed
        if args.checkpoint is not None or args.hf_checkpoint is not None:
            print("[WARN] --checkpoint/--hf_checkpoint ignored in teleop mode.")

    return args


# ──────────────────────────────────────────────────────────────────────────────
# Shared: recorder factory
# ──────────────────────────────────────────────────────────────────────────────

def _build_recorder(
    args: argparse.Namespace,
    robot: Any,
    *,
    root: str | None = None,
    resume: bool | None = None,
) -> Any | None:
    root = args.lerobot_root if root is None else root
    if root is None:
        return None
    resume = args.resume if resume is None else resume
    from fire_core.recorders import GR00TRecorder
    return GR00TRecorder(
        robot=robot,
        repo_id=args.lerobot_repo_id,
        root=root,
        task=args.task,
        task_text=args.lerobot_task,
        fps=int(round(args.control_hz)),
        resume=resume,
        defer_video_encoding=True,
    )


def _validate_inverse3_ports(args: argparse.Namespace) -> None:
    missing_ports = [
        port for port in (args.inv3_port, args.versegrip_port)
        if not Path(port).exists()
    ]
    if not missing_ports:
        return

    raise SystemExit(
        "[ERROR] Inverse3 device port not found: "
        + ", ".join(missing_ports)
        + "\nCheck USB connection/permissions and udev symlinks, or pass "
        "--inv3_port/--versegrip_port with the current /dev/ttyACM* paths.\n"
        "Useful checks:\n"
        "  ls -l /dev/inverse3_left /dev/versegrip_left\n"
        "  ls -l /dev/ttyACM* /dev/ttyUSB*"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Encode helper
# ──────────────────────────────────────────────────────────────────────────────

def encode_deferred_videos(
    args: argparse.Namespace, root: Path | None = None
) -> None:
    from fire_core.recorders.gr00t_exporter import GR00TExporter
    last_episode = None if args.last_episode == -1 else args.last_episode
    GR00TExporter.encode_deferred_videos(
        root=root or Path(args.lerobot_root),
        last_episode=last_episode,
        vcodec="h264",
        encoder_threads=args.encoder_threads,
        remove_images=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Teleop control loop
# ──────────────────────────────────────────────────────────────────────────────

def run_teleop_loop(
    robot: Any,
    teleop: Any,
    recorder: Any | None,
    *,
    control_hz: float,
    max_steps: int,
    reanchor_on_enable: bool,
) -> None:
    from lerobot_robot_fr3.fr3 import TeleopAction

    dt = 1.0 / control_hz
    prev_enabled = False
    robot_home_pos = robot.robot_state_manager.ee_pos.copy()
    robot_home_quat = robot.robot_state_manager.ee_quat.copy()

    print(f"\n[INFO] Teleop loop @ {control_hz} Hz")
    print("[INFO] Press VerseGrip button 1 or Ctrl+C to stop and save.")

    step = 0
    try:
        while max_steps <= 0 or step < max_steps:
            t0 = time.time()

            action = teleop.get_action()
            enabled = bool(action["inv3.enabled"].item())
            calibrated = bool(action.get("inv3.calibrated", np.array([False])).item())
            end_episode = bool(action.get("inv3.end_episode", np.array([False])).item())
            buttons = int(action["inv3.buttons"][0])
            if end_episode:
                print("\n[INFO] End episode button pressed.")
                break

            if calibrated:
                robot_home_pos = robot.robot_state_manager.ee_pos.copy()
                robot_home_quat = robot.robot_state_manager.ee_quat.copy()
                print(f"\n[INFO] Teleop calibrated — home: {robot_home_pos}")

            # Rising edge: re-anchor robot home so robot stays put on engage.
            # Absolute teleop keeps the initial home fixed for the whole episode.
            if not calibrated and reanchor_on_enable and enabled and not prev_enabled:
                robot_home_pos = robot.robot_state_manager.ee_pos.copy()
                robot_home_quat = robot.robot_state_manager.ee_quat.copy()
                print(f"\n[INFO] Teleop engaged — home: {robot_home_pos}")

            prev_enabled = enabled

            # Live status so it's obvious whether the buttons are registering.
            dp = action["inv3.pos"]
            gripper_action = np.asarray(action["inv3.gripper"], dtype=np.float32).reshape(1)
            print(
                f"\r[teleop] enabled={int(enabled)} btn={buttons:08b} "
                f"dpos=({dp[0]:+.3f},{dp[1]:+.3f},{dp[2]:+.3f}) "
                f"grip={gripper_action[0]:+.1f} step={step}   ",
                end="", flush=True,
            )

            if not enabled:
                sleep_t = dt - (time.time() - t0)
                if sleep_t > 0:
                    time.sleep(sleep_t)
                continue

            # Compute absolute EEF target
            target_pos = robot_home_pos + action["inv3.pos"]
            delta_rot = _wxyz_to_rotation(action["inv3.rot"])
            home_rot = _wxyz_to_rotation(robot_home_quat)
            target_rot = delta_rot * home_rot
            target_quat = _rotation_to_wxyz(target_rot)
            arm_action = np.concatenate(
                [target_pos, target_quat]
            ).astype(np.float32)

            robot.send_teleop_action(TeleopAction(
                action={
                    "arm_actions": arm_action,
                    "gripper_actions": gripper_action,
                },
                action_space="task_space",
                is_relative=False,
            ))

            if recorder is not None:
                recorder.record(
                    arm_action=arm_action,
                    gripper_action=gripper_action,
                )

            step += 1
            sleep_t = dt - (time.time() - t0)
            if sleep_t > 0:
                time.sleep(sleep_t)

    except KeyboardInterrupt:
        print("\n[INFO] Teleop stopped by user.")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # ── Encode-only ───────────────────────────────────────────────────────────
    encode_only = args.checkpoint is None and args.hf_checkpoint is None
    if args.last_episode is not None and encode_only and args.teleop is None:
        encode_deferred_videos(args)
        return

    if args.teleop == "inverse3":
        _validate_inverse3_ports(args)

    from lerobot_robot_fr3.config_fr3 import FR3RobotConfig
    from lerobot_robot_fr3.fr3 import FR3Robot

    config = FR3RobotConfig(
        use_sim_time=args.use_sim_time,
        is_relative=False,
        rotation_type="quaternion",
        use_cameras=args.use_cameras,
        use_ft_sensor=args.use_ft_sensor,
    )
    robot = FR3Robot(config, task_name=args.task)

    # ── Teleop mode ───────────────────────────────────────────────────────────
    if args.teleop == "inverse3":
        from lerobot_teleoperator_inverse3 import (
            Inverse3Teleop, Inverse3TeleopConfig,
        )

        teleop_cfg = Inverse3TeleopConfig(
            inverse3_port=args.inv3_port,
            versegrip_port=args.versegrip_port,
            position_scale=args.position_scale,
            position_axes=tuple(args.position_axes),
            rotation_axes=tuple(args.rotation_axes),
            reanchor_on_enable=not args.absolute_teleop,
            require_calibration=args.require_calibration,
            enable_button=args.enable_button,
            calibration_button=args.calibration_button,
            grasp_button=args.grasp_button,
            end_episode_button=args.end_episode_button,
            gripper_open_value=args.gripper_open_value,
            gripper_close_value=args.gripper_close_value,
            haptic_feedback_enabled=args.haptic_feedback,
        )
        teleop = Inverse3Teleop(teleop_cfg)

        if args.lerobot_root is not None and not args.use_cameras:
            print(
                "[WARN] Recording GR00T dataset without --use_cameras. "
                "The dataset will contain zero-filled fallback images."
            )

        print("[INFO] Connecting to robot ...")
        robot.connect()

        print("[INFO] Connecting to Inverse3 ...")
        teleop.connect()

        # Continuous session: keep the robot/Inverse3 connection alive across
        # episodes so bringup (ROS controller, task_manager, Inverse3 bridge)
        # only has to happen once. Each loop iteration records one episode
        # into the same dataset root (first episode honors --resume; every
        # later episode in the session always appends).
        session_root: Path | None = None
        encode_root: Path | None = None
        try:
            while True:
                resume = args.resume if session_root is None else True
                root = str(session_root) if session_root is not None else args.lerobot_root
                recorder = _build_recorder(args, robot, root=root, resume=resume)

                try:
                    run_teleop_loop(
                        robot, teleop, recorder,
                        control_hz=args.control_hz,
                        max_steps=args.episode_length,
                        reanchor_on_enable=not args.absolute_teleop,
                    )
                finally:
                    # Episode is done — tell the VLA before we ask the operator
                    # whether it was a success (mirrors the model-mode ordering).
                    robot.send_success_signal()
                    if recorder:
                        recorder.save()
                        if session_root is None:
                            session_root = recorder.output_root
                        if args.last_episode is not None:
                            encode_root = recorder.output_root

                again = input("[INFO] Record another episode? (y/n): ").strip().lower()
                if again != "y":
                    break

                # Allow the next episode to signal success again
                # (send_success_signal() is otherwise idempotent per connection).
                robot.reset_success_signal()

                # The VLA action goal ends (succeeded) the instant the previous
                # send_success_signal() lands — the controller then ignores all
                # ActionChunk messages until a new goal is accepted. Re-arm it
                # now so the robot actually moves once teleop resumes. Requires
                # task_manager to have re-activated the VLA controller by now.
                print("[INFO] Requesting a new VLA goal for the next episode ...")
                while True:
                    try:
                        robot.send_new_vla_goal()
                        break
                    except ConnectionError as e:
                        input(
                            f"[WARN] {e} Make sure task_manager has re-launched and "
                            "activated the VLA controller, then press Enter to retry."
                        )

                # Debounce: don't let a still-held end_episode button instantly
                # end the next episode too.
                while bool(teleop.get_action()["inv3.end_episode"].item()):
                    time.sleep(0.05)

                print("[INFO] Reposition and press the calibration button to start the next episode.")
                teleop.calibrate()
        except KeyboardInterrupt:
            print("\n[INFO] Session stopped by user.")
        finally:
            teleop.disconnect()
            robot.disconnect()
            print("[INFO] Disconnected.")

        if encode_root is not None:
            encode_deferred_videos(args, root=encode_root)
        return

    # ── Model mode ────────────────────────────────────────────────────────────
    checkpoint_path = resolve_checkpoint_path(
        checkpoint=args.checkpoint,
        hf_checkpoint=args.hf_checkpoint,
    )

    obs_features    = robot.task.observation_features
    log_features    = robot.task.log_features
    action_features = robot.task.action_features
    obs_dim    = total_dim(obs_features)
    action_dim = total_dim(action_features)
    print(f"[INFO] obs_dim={obs_dim}  action_dim={action_dim}")

    if args.lerobot_root is not None and not args.use_cameras:
        print(
            "[WARN] Recording GR00T dataset without --use_cameras. "
            "The dataset will contain zero-filled fallback images."
        )

    obs_shm    = torch.zeros(1, obs_dim).share_memory_()
    action_shm = torch.zeros(1, action_dim).share_memory_()
    obs_flag    = RawValue(ctypes.c_int32, 0)
    action_flag = RawValue(ctypes.c_int32, 0)

    recorder = _build_recorder(args, robot)

    infer_handle = start_inference_process(
        obs_shm, action_shm, obs_flag, action_flag,
        checkpoint=checkpoint_path,
        cfg_path=robot.task.model_cfg_path,
        obs_dim=obs_dim,
        action_dim=action_dim,
        device=args.device,
    )
    infer_handle.wait_ready(action_flag)

    print("[INFO] Connecting to robot ...")
    try:
        robot.connect()
    except Exception as e:
        print(f"[ERROR] {e}")
        infer_handle.shutdown()
        return

    if not robot.is_connected:
        print("[ERROR] Failed to connect.")
        infer_handle.shutdown()
        return

    obs_buf = obs_shm.numpy()[0]
    strategy = LiveInferenceStrategy(
        robot, obs_buf, obs_features,
        obs_flag, action_flag, action_shm, action_dim,
    )

    # ── Logger ────────────────────────────────────────────────────────────────
    logger = StepLogger(robot, obs_features, log_features) if args.obs_save_path else None

    encode_root = None
    try:
        run_control_loop(
            robot, strategy,
            control_hz=args.control_hz,
            max_steps=args.episode_length,
            logger=logger,
            recorder=recorder,
        )
    finally:
        # Playback of episode_length steps is done — tell the VLA it has
        # finished before we ask the operator whether it was a success.
        robot.send_success_signal()
        try:
            if logger:
                logger.save(args.obs_save_path)
            if recorder:
                recorder.save()
                if args.last_episode is not None:
                    encode_root = recorder.output_root
        finally:
            infer_handle.shutdown()
            cv2.destroyAllWindows()
            robot.disconnect()
            print("[INFO] Robot disconnected.")

    if encode_root is not None:
        encode_deferred_videos(args, root=encode_root)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
