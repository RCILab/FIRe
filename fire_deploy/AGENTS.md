# FIRe Project Agent Instructions

## 🎯 Project Purpose (WHY)
FIRe is a framework designed to perform force-sensitive robotic tasks (e.g., peg insert, gear mesh, nut thread) by combining actions from a VLA (Vision-Language-Action) model with residual actions from an RL (Reinforcement Learning) model.
- **Scope:** This project *executes* trained residual RL models (training happens in a separate project).
- **Data Pipeline:** Includes functionality to record demonstrations and play them back. Raw recording uses the `lerobot` writer, while GR00T datasets are exported to the GR00T-compatible LeRobot v2-style layout.
- **Future-proofing (VLA):** Currently implementing `gr00t` as the VLA model, but the architecture must remain agnostic to support future models like `pi05` or `openvla`. 

## 🗺️ Codebase Map (WHAT)
- `src/FIRe/fire_core/`: The core logic that bridges trained model execution with robot control.
- `src/FIRe/lerobot_robot_fr3/`: Handles robot control and task environments, primarily sending commands to a separate robot control project via ROS.
- `src/FIRe/lerobot_ft_sensor/`: Interfaces with the Force/Torque (F/T) sensor and fetches data, implemented in the `lerobot` format.
- *Note:* `vision_server` is currently low-priority/can be ignored.

## 🛠️ Development Workflow (HOW)
- **Package Management:** Use standard `pip` within a Conda environment. env name is "fire"
- **Testing:** Use `pytest` as the modern standard for all unit and integration testing. Run tests via `pytest test/` (or the specific test file).
- **Linting/Formatting:** Currently, no strict linters (like ruff or black) are enforced. Match the surrounding code style and do not run automated formatters unless explicitly requested.

## ▶️ Robot Execution Runbook
Run robot commands from a terminal with the `fire` Conda environment activated and the ROS workspace sourced:

```bash
conda activate fire
source ~/ros2_ws/install/setup.bash
```

Before running `scripts/play.py`, start the Gazebo/controller bringup first:

```bash
conda activate fire
source ~/ros2_ws/install/setup.bash
ros2 launch cho_franka_bringup bringup_gazebo_robot.launch.py vla:=true control_mode:=torque
```

Wait about 5 seconds for the controller/action server to come up. Then, in another sourced terminal, run:

```bash
conda activate fire
source ~/ros2_ws/install/setup.bash
python scripts/play.py --task forge-peg_insert --checkpoint /home/home/FIRe/checkpoints/Forge/VLA_RL-BL-forge-peg_insert/nn/Forge.pth
```

For GR00T dataset recording, use the same ROS sourcing in the recording terminal:

```bash
conda activate fire
source ~/ros2_ws/install/setup.bash
python scripts/record.py --task forge-peg_insert --checkpoint /home/home/FIRe/checkpoints/Forge/VLA_RL-BL-forge-peg_insert/nn/Forge.pth --vla gr00t --lerobot_repo_id chohh7391/gr00t_peg_insert --lerobot_root /home/home/datasets --lerobot_task "peg insert"
```

Use `--hf_checkpoint user/repo/path/to/model.pth` instead of `--checkpoint /local/path/model.pth` when loading a checkpoint from Hugging Face.

To append another episode to the same local GR00T dataset, add `--resume`.

## 🚨 Core Rules & Conventions
1. **Strict Type Hinting:** Python type hints (`typing` module) are **MANDATORY** for all function signatures, class attributes, and complex variables. Do not write Python code without type hints.
2. **VLA Abstraction:** When building VLA interfaces, use generic names (e.g., `VLAModel`, `BaseVLA`) instead of specific names like `Gr00tModel`. Do not hardcode "gr00t" in general naming conventions to ensure easy swapping of VLA backends.
3. **LeRobot Compatibility:** Any data recording or sensor logic must strictly conform to the `lerobot` format conventions.
4. **ROS Integration:** Remember that `lerobot_robot_fr3` acts as a ROS client sending commands to an external system, not directly controlling the hardware loop internally.
