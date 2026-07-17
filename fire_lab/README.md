# FIRe LAB

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.1.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/index.html)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/22.04/)

## Overview

This is the **simulation and training side** of FIRe, built on Isaac Lab. It uses a frozen VLA model
as the base policy and trains a force-aware **residual RL policy** on top to improve success on
contact-rich assembly. It also generates demonstrations for VLA fine-tuning and integrates the VLA
backends (GR00T, pi05, OpenVLA) as inference servers. The trained checkpoints are run on the real
robot from [`fire_deploy/`](../fire_deploy).

> For the overall FIRe method, results, paper, and how the simulation and deployment sides fit
> together, see the **[top-level project README](../README.md)**.

## Installation

```bash
# 1) Conda env
conda create -n fire_lab python=3.11
conda activate fire_lab

# 2) Clone the monorepo (fire_lab lives inside it)
cd $HOME
git clone --recurse-submodules https://github.com/chohh7391/FIRe.git
cd FIRe/fire_lab

# 3) Isaac Sim SDK (v5.1.0)
pip install --upgrade pip
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com

# 4) Pytorch (CUDA 12.8)
pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128

# 5) Isaac Lab (v2.3.0) — vendored as the _isaaclab submodule
#    Populated by the --recurse-submodules clone above; if _isaaclab is empty,
#    run `git submodule update --init --recursive` from the repo root first.
sudo apt install -y cmake build-essential
./_isaaclab/isaaclab.sh --install

# 6) Smoke test (headless sim)
python _isaaclab/scripts/tutorials/00_sim/create_empty.py --headless

# 7) Dev install
python -m pip install -e source/fire_lab

# 8) Extra dependencies
python -m pip install zmq scikit-learn pyarrow fastparquet av json_numpy
```

- Verify the extension is installed by listing the registered tasks
  (you should see the `FireLab-*` tasks):

    ```bash
    # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not in a venv/conda env
    python scripts/reinforcement_learning/list_envs.py
    ```

    If task names change, update the `"FireLab-"` search pattern in
    `scripts/reinforcement_learning/list_envs.py` so they are listed.


# VLA-RL

## 1. Save Demo

Roll out a trained base line policy to save GR00T demos. Requires a trained base line
checkpoint ([pretrained here](https://drive.google.com/drive/folders/1FtDIjFQs3Yy5Gnrr_lzFxlC8oiTt4yFa?usp=sharing)).
Set the output `dataset_path` in the matching `{}_gr00t_env_cfg.py`, then run:

```bash
python scripts/reinforcement_learning/rl_games/play.py \
    --task=<DEMO_SAVE_TASK_ID> \
    --headless --enable_cameras \
    --checkpoint=<BASE_LINE_CHECKPOINT_PATH>
```

Available Demo-Save task ids:

```text
FireLab-VLA-Gr00t-Factory-PegInsert-Demo-Save-Direct-v0
FireLab-VLA-Gr00t-Factory-GearMesh-Demo-Save-Direct-v0
FireLab-VLA-Gr00t-Factory-NutThread-Demo-Save-Direct-v0
FireLab-VLA-Gr00t-Forge-PegInsert-Demo-Save-Direct-v0
FireLab-VLA-Gr00t-Forge-GearMesh-Demo-Save-Direct-v0
FireLab-VLA-Gr00t-Forge-NutThread-Demo-Save-Direct-v0
```

Add `--huggingface` to push the dataset to the Hugging Face Hub.


## 2. Train RL with a VLA Server

Start the VLA inference server first, then run RL training with the matching task id.
Use `env.vla_host` and `env.vla_port` to point the Isaac Lab client to the server.

### GR00T

Smoke test:

```bash
python scripts/reinforcement_learning/rl_games/train.py \
    --task=FireLab-VLA-Gr00t-Factory-PegInsert-Direct-v1 \
    --headless \
    --enable_cameras \
    --num_envs=2 \
    --max_iterations=1 \
    env.vla_host=<GR00T_SERVER_IP> \
    env.vla_port=<GR00T_SERVER_PORT>
```

Training:

```bash
python scripts/reinforcement_learning/rl_games/train.py \
    --task=FireLab-VLA-Gr00t-Factory-PegInsert-Direct-v1 \
    --headless \
    --enable_cameras \
    env.vla_host=<GR00T_SERVER_IP> \
    env.vla_port=<GR00T_SERVER_PORT>
```

Available GR00T task ids:

```text
FireLab-VLA-Gr00t-Factory-PegInsert-Direct-v1
FireLab-VLA-Gr00t-Factory-GearMesh-Direct-v1
FireLab-VLA-Gr00t-Factory-NutThread-Direct-v1
FireLab-VLA-Gr00t-Forge-PegInsert-Direct-v1
FireLab-VLA-Gr00t-Forge-GearMesh-Direct-v1
FireLab-VLA-Gr00t-Forge-NutThread-Direct-v1
```

### Pi05

Smoke test:

```bash
python scripts/reinforcement_learning/rl_games/train.py \
    --task=FireLab-VLA-Pi05-Factory-PegInsert-Direct-v1 \
    --headless \
    --enable_cameras \
    --num_envs=2 \
    --max_iterations=1 \
    env.vla_host=<PI05_SERVER_IP> \
    env.vla_port=<PI05_SERVER_PORT>
```

Training:

```bash
python scripts/reinforcement_learning/rl_games/train.py \
    --task=FireLab-VLA-Pi05-Factory-PegInsert-Direct-v1 \
    --headless \
    --enable_cameras \
    env.vla_host=<PI05_SERVER_IP> \
    env.vla_port=<PI05_SERVER_PORT>
```

Available Pi05 task ids:

```text
FireLab-VLA-Pi05-Factory-PegInsert-Direct-v1
FireLab-VLA-Pi05-Factory-GearMesh-Direct-v1
FireLab-VLA-Pi05-Factory-NutThread-Direct-v1
FireLab-VLA-Pi05-Forge-PegInsert-Direct-v1
FireLab-VLA-Pi05-Forge-GearMesh-Direct-v1
FireLab-VLA-Pi05-Forge-NutThread-Direct-v1
```

### OpenVLA

Start the OpenVLA-OFT server with `openvla-oft/vla-scripts/deploy_batch.py`.
The client uses `POST /act_batch` and expects the server to return `{"actions": ...}`.

Smoke test:

```bash
python scripts/reinforcement_learning/rl_games/train.py \
    --task=FireLab-VLA-OpenVLA-Factory-PegInsert-Direct-v1 \
    --headless \
    --enable_cameras \
    --num_envs=2 \
    --max_iterations=1 \
    env.vla_host=<OPENVLA_SERVER_IP> \
    env.vla_port=<OPENVLA_SERVER_PORT>
```

Training:

```bash
python scripts/reinforcement_learning/rl_games/train.py \
    --task=FireLab-VLA-OpenVLA-Factory-PegInsert-Direct-v1 \
    --headless \
    --enable_cameras \
    env.vla_host=<OPENVLA_SERVER_IP> \
    env.vla_port=<OPENVLA_SERVER_PORT>
```

Available OpenVLA task ids:

```text
FireLab-VLA-OpenVLA-Factory-PegInsert-Direct-v1
FireLab-VLA-OpenVLA-Factory-GearMesh-Direct-v1
FireLab-VLA-OpenVLA-Factory-NutThread-Direct-v1
```

Notes:

- Isaac Sim startup can take a few minutes, especially with `--enable_cameras`.
- `env.vla_host` and `env.vla_port` override the defaults in the environment config.
- OpenVLA factory tasks default to 32 environments to reduce server-side GPU memory usage.
- If the server is reachable, the log should print `Initialize gr00t Client Node`, `Initialize pi05 Client Node`, or `Initialize openvla Client Node`.


## 3. Train Gr00t Model using Demo

- Install Isaac-GR00T

```bash
cd ~/

git clone https://github.com/NVIDIA/Isaac-GR00T.git
```

- install dependencies

```bash
cd Isaac-GR00T
conda create -n gr00t python=3.10
conda activate gr00t
pip install --upgrade setuptools
pip install -e .[base]
pip install --no-build-isolation flash-attn==2.7.1.post4 
```

- Replace the files at `Isaac-GR00T/gr00t/data/embodiment_tags.py` and `Isaac-GR00T/gr00t/experiment/data_config.py` with the files from the gr00t folder within the repository.
  
- change meta folder in dataset like `~/fire_lab/gr00t/meta`

- take care three things
  - 1st: In `episodes.jsonl`, edit "tasks" except "valid". And edit "length" to your total_frames
  - 2nd: In `info.json`, edit "total_episodes" and "total_frames"
  - 3rd: Change stats.json file using `fire_lab/gr00t/utils/get_stats.py`
  - 4th: In `tasks.jsonl`, edit "task" at "task_index": 0


- Train gr00t model

```bash
python scripts/gr00t_finetune.py \
    --dataset-path <DATASET_ROOT>/<TASK_NAME>/ \
    --num-gpus 1 \
    --batch-size 32 \
    --output-dir <OUTPUT_DIR>/<TASK_NAME> \
    --max-steps 20000 \
    --embodiment-tag franka \
    --data-config franka_triple_cam \
    --video-backend torchvision_av \
    --push_to_hub \
    --hub_model_id <HF_USERNAME>/<TASK_NAME>
```

- Eval gr00t model
```bash
python scripts/eval_policy.py \
    --plot \
    --embodiment_tag franka \
    --model_path  <HF_USERNAME>/<TASK_NAME> \
    --data_config franka_triple_cam \
    --embodiment_tag franka \
    --dataset_path <DATASET_ROOT>/<TASK_NAME>/ \
    --video_backend decord \
    --modality_keys eef_position_delta eef_rotation_delta gripper_close \
    --save_plot_path <PLOT_DIR>/<TASK_NAME>/eef_pose.png
```


## 4. Train VLA-RL Policy

- run gr00t server

```bash
cd ~/Isaac-GR00T

python scripts/inference_service.py --server --model_path <GR00T_MODEL_PATH> --embodiment-tag franka --data-config franka_triple_cam --denoising-steps 4 --port {PORT_ID}
```

- change port id to {PORT_ID} in source/fire_lab/fire_lab/envs/direct_rl_gr00t_env.py

- Train vla-rl policy
```bash
python scripts/reinforcement_learning/rl_games/train.py --task={TASK_NAME} --headless --enable_cameras --wandb-entity={YOUR_ENTITY} --wandb-project-name={WANDB_PROJECT_NAME} --wandb-name={RUN_NAME} --huggingface --repo_id={REPOSITORY_ID} --track
```

- If your task is 'Automate', run this terminal input
```bash
python source/isaaclab_tasks/isaaclab_tasks/direct/automate/run_w_id.py --assembly_id=ASSEMBLY_ID \
--train --headless
```

(e.g)
```bash
python scripts/reinforcement_learning/rl_games/train.py --task=FireLab-VLA-Gr00t-Forge-PegInsert-Direct-v1 --headless --enable_cameras --wandb-entity=chohh7391-kyung-hee-university --wandb-project-name=VLA_RL-VLA_RL-gr00t --wandb-name=forge-peg_insert --huggingface --repo_id=bhe1004/VLA_RL-VLA_RL-gr00t-forge-peg_insert
```
