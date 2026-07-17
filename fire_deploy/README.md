# FIRe — Real-Robot Deployment (`fire_deploy`)

This is the **real-robot deployment side** of FIRe. It runs trained models on a physical Franka
FR3: it loads an RL residual policy (trained separately in [`fire_lab/`](../fire_lab)) and adds its
fine, force-aware corrections on top of a VLA model's action chunk, driving the arm through a ROS 2
controller. Interchangeable VLA backends (GR00T, pi05, OpenVLA) are supported.

> For the overall FIRe method, results, paper, and how the simulation and deployment sides fit
> together, see the **[top-level project README](../README.md)**.

**Scope of this component** (execution & data collection only — no training or simulator):

- **Run trained models** — `play.py` (VLA + RL residual, or either alone).
- **Collect demonstrations** — `record.py` into GR00T LeRobot datasets, including Inverse3 haptic
  teleoperation.
- **Sensing stack** — wrist force/torque sensor and RealSense cameras.

Training the RL residual policy and fine-tuning the VLA model happen in
[`fire_lab/`](../fire_lab), not here.

# Installation
- create conda env
```bash
conda create -n fire python=3.10
conda activate fire
git clone --recurse-submodules https://github.com/RCILab/FIRe.git
cd FIRe/fire_deploy
pip install -r requirements.txt
```

> **Submodule:** the Inverse3 teleoperator lives in a git submodule
> (`src/FIRe/lerobot_teleoperators/lerobot_teleoperator_inverse3`) that `requirements.txt`
> installs editable, so clone with `--recurse-submodules` — or, in an existing clone, run
> `git submodule update --init --recursive` **before** `pip install -r requirements.txt`.
> Its C++ bridge additionally needs the proprietary Haply SDK built separately; see the
> submodule's [`SETUP.md`](src/FIRe/lerobot_teleoperators/lerobot_teleoperator_inverse3/SETUP.md).

The vision (camera) server now runs in the same `fire` env — `requirements.txt` installs
`src/FIRe/vision_server` (with `pyrealsense2`), so no separate env is needed.

- install ros2 based controller: (https://github.com/chohh7391/cho_robot_project)

# How to run

- bringup vla controller
```bash
source ~/ros2_ws/install/setup.bash
ros2 launch cho_franka_bringup bringup_gazebo_robot.launch.py vla:=true control_mode:=torque
```

- vision server
```bash
conda activate fire
python scripts/run_vision_server.py
```

- run model
```bash
python scripts/play.py --task <TASK_NAME> --checkpoint <CHECKPOINT_PATH>
```

Add `--use_cameras` and/or `--use_ft_sensor` if the task/model uses camera or force-torque
inputs; when omitted, zeros are fed in their place.

You can also load a checkpoint from Hugging Face with `--hf_checkpoint <user/repo/path/to/model.pth>`
in place of a local `--checkpoint <CHECKPOINT_PATH>` (works for both `play.py` and `record.py`).

## Arguments

### `play.py`

| Argument | Description |
|---|---|
| `--task` | Task name — `forge-peg_insert`, `forge-gear_mesh`, `forge-nut_thread`, `pick_place`. |
| `--checkpoint` / `--hf_checkpoint` | RL checkpoint: local path, or Hugging Face `user/repo/path/to/model.pth` (mutually exclusive). |
| `--vla` | VLA backend: `gr00t` / `pi05` / `openvla`. Omit to run the RL policy only. |
| `--vla_chunk_size` | Action chunk size (default per backend: gr00t 16, pi05/openvla 8). |
| `--host` / `--port` | VLA server address (required with `--vla`). |
| `--use_cameras` | Feed camera views (required for a VLA server); zeros otherwise. |
| `--use_ft_sensor` | Feed force/torque readings; zeros otherwise. |
| `--control_hz` | Control-loop rate (default 15). |
| `--episode_length` | Max steps per episode (logging only; default 88). |
| `--device` | Torch device (default `cuda:0`). |
| `--use_sim_time` | Use ROS simulated time. |
| `--save_path` | Dump a per-step observation CSV. |
| `--replay <CSV>` + `--raw`/`--pose` | Replay a recorded CSV (obs → inference, or target pose → direct send). |

### `record.py`

Model mode (`--checkpoint`/`--hf_checkpoint`) drives the robot with an RL policy; teleop mode
(`--teleop inverse3`) records human demonstrations. A dataset is written **only when
`--lerobot_root` is set** (always GR00T format).

| Argument | Description |
|---|---|
| `--task` | Task name (same options as `play.py` above). |
| `--checkpoint` / `--hf_checkpoint` | Model-mode RL checkpoint (local or Hugging Face). |
| `--teleop inverse3` | Teleop mode — record demonstrations via the Inverse3 device. |
| `--lerobot_root` | Dataset save root. Recording happens only when this is set. |
| `--lerobot_task` | Natural-language task description stored in the dataset. |
| `--lerobot_repo_id` | Optional Hugging Face dataset repo id. |
| `--use_cameras` | Record camera views (recommended; zeros otherwise). |
| `--use_ft_sensor` | Record force/torque readings; zeros otherwise. |
| `--resume` | Append to an existing dataset root. |
| `--last_episode [N]` | After recording, encode deferred videos through episode `N` (or the latest). |
| `--obs_save_path` | Also dump a per-step observation CSV. |
| `--control_hz` / `--episode_length` / `--device` / `--use_sim_time` | As in `play.py`. |
| Inverse3 teleop flags (`--inv3_port`, `--versegrip_port`, `--position_scale`, button bits, …) | See [docs/inverse3_teleoperator.md](docs/inverse3_teleoperator.md). |

## VLA backends (`--vla`)

`play.py` can drive the
RL action path with different VLA servers via `--vla`. Supported backends:

| `--vla`   | server                                   | protocol        | chunk size |
|-----------|------------------------------------------|-----------------|------------|
| `gr00t`   | Isaac-GR00T `inference_service.py`       | ZMQ             | 16         |
| `pi05`    | openpi `serve_policy.py`                 | WebSocket       | 8          |
| `openvla` | openvla-oft `vla-scripts/deploy_batch.py`| HTTP            | 8          |

The chunk size is auto-selected per backend (override with `--vla_chunk_size`). FIRe speaks a
single gr00t-canonical observation/action format internally; each client
(`src/FIRe/fire_core/src/fire_core/vla_clients/`) translates to/from its server's format, so the
backends are interchangeable.

```bash
# pi05 (openpi WebSocket server)
python scripts/play.py --task forge-peg_insert --checkpoint <CHECKPOINT_PATH> \
  --vla pi05 --host <HOST> --port <PORT>

# openvla (openvla-oft HTTP server)
python scripts/play.py --task forge-peg_insert --checkpoint <CHECKPOINT_PATH> \
  --vla openvla --host <HOST> --port <PORT>
```

## Play a VLA model only (no RL checkpoint)

Drive the robot purely from the VLA server's action chunk — no RL policy. Pass `--vla` **without**
`--checkpoint`/`--hf_checkpoint` (e.g. the teleop-collected `pick_place` task). The task's action
convention decides how the chunk is applied:

- Absolute EE-pose tasks (e.g. `pick_place`): the server returns `eef_position` + `eef_quaternion`,
  sent to the robot as an absolute task-space pose (no `process_action` delta scaling).
- Relative-delta tasks: the server returns `eef_position_delta` + `eef_rotation_delta`, turned into
  an absolute pose by the task's `process_action`.

Prerequisites (each in its own sourced terminal): robot/controller bringup, vision server, task
manager for the task, and the GR00T inference server (`--host`/`--port` below point to it).

```bash
conda activate fire
source ~/ros2_ws/install/setup.bash
cd /home/home/FIRe

python scripts/play.py \
--task pick_place \
--vla gr00t \
--host <HOST> --port <PORT> \
--use_cameras \
--episode_length 384
```

`--use_cameras` is required (the VLA server needs the camera views). Swap `--vla`/`--host`/`--port`
for `pi05` or `openvla` as in the table above.

- record data
```bash
python scripts/record.py --task <TASK_NAME> --checkpoint <CHECKPOINT_PATH> --lerobot_root <PATH_TO_SAVE> --lerobot_task "<TASK_DESCRIPTION>"
```

> **Note:** `record.py` records **GR00T-format** LeRobot datasets only. To use the
> collected data with **pi05** or **openvla**, first record the GR00T dataset and then convert it to
> the target format separately — there is no built-in pi05/openvla recorder.

- record GR00T data and encode all deferred videos after the episode
```bash
python scripts/record.py --task <TASK_NAME> --checkpoint <CHECKPOINT_PATH> --lerobot_root <PATH_TO_SAVE> --lerobot_task "<TASK_DESCRIPTION>" --last_episode
```

Use `--resume` to append the next episode to the same dataset root. When `--last_episode` is set, `record.py` first records one episode without video encoding, then encodes every missing video in the dataset root after the robot disconnects. Existing `.mp4` files are skipped.

Add `--obs_save_path <DIR>` to also dump per-step observation info to a CSV (same format as `play.py --save_path`), independent of the LeRobot dataset:
```bash
python scripts/record.py --task <TASK_NAME> --checkpoint <CHECKPOINT_PATH> --lerobot_root <PATH_TO_SAVE> --lerobot_task "<TASK_DESCRIPTION>" --obs_save_path <OBS_CSV_DIR>
```

- Bota force/torque sensor setup

See the standalone module guide at [src/FIRe/lerobot_ft_sensor/README.md](src/FIRe/lerobot_ft_sensor/README.md).

Frequently-used concrete commands are collected in [docs/local_commands.md](docs/local_commands.md).

# Inverse3 Haptic Teleop

Collect demonstrations directly with the Haption Inverse3 + VerseGrip.
For initial setup (build / udev serial port configuration / troubleshooting), see the submodule's [SETUP.md](src/FIRe/lerobot_teleoperators/lerobot_teleoperator_inverse3/SETUP.md).
For the operation manual (running / buttons / defaults), see [docs/inverse3_teleoperator.md](docs/inverse3_teleoperator.md).

## Install

```bash
# Build the C++ bridge server (one-time)
cd src/FIRe/lerobot_teleoperators/lerobot_teleoperator_inverse3/\
src/lerobot_teleoperator_inverse3/inverse3_bridge
make
cd /home/home/FIRe

# Install the Python package
conda activate fire
pip install -e src/FIRe/lerobot_teleoperators/lerobot_teleoperator_inverse3
```

## Record demonstrations via teleop

```bash
conda activate fire
source ~/ros2_ws/install/setup.bash

python scripts/record.py \
--teleop inverse3 \
--task forge-peg_insert \
--lerobot_root /home/home/datasets/FIRe/teleop/peg_insert \
--lerobot_task "Insert peg into the socket" \
--use_cameras \
--inv3_port /dev/inverse3_left \
--versegrip_port /dev/versegrip_left \
--position_scale 3.0 \
--episode_length 200
```

Add `--resume` to append another episode to the same dataset root.

**How it works:**
- The robot follows only while the VerseGrip's enable button (bit-0 by default) is **held down**.
- **The instant the button is first pressed**, the Inverse3's current position is automatically matched to the robot's current position (no jump).
- Releasing the button pauses motion; pressing it again restarts from the current position.
- Stop recording with `Ctrl+C` → enter whether the episode succeeded, then save.
