# Local Commands

Frequently-used concrete commands for this FIRe setup. Pick the task-specific values from the
table below; the examples use peg insert.

## Tasks

| Task | `--task` (play / record) | task manager `task:=` | `--lerobot_task` |
|---|---|---|---|
| Peg insert | `forge-peg_insert` | `peg_insert` | `Insert peg into the socket` |
| Gear mesh  | `forge-gear_mesh`  | `gear_mesh`  | `Insert and mesh gear into the base with other gears` |
| Nut thread | `forge-nut_thread` | `nut_thread` | `Thread the nut onto the first 2 threads of the bolt` |
| Pick place | `pick_place`       | `pick_place` | `Put the green cube on the brown book into the sky-blue box` |

## Bringup robot
```bash
source ~/ros2_ws/install/setup.bash
ros2 launch cho_franka_bringup bringup_gazebo_robot.launch.py control_mode:=torque vla:=true
```

## Run vision server
```bash
conda activate fire
python scripts/run_vision_server.py
```

## Run task manager
```bash
source ~/ros2_ws/install/setup.bash
ros2 launch cho_task_manager run_task_manager.launch.py task:=peg_insert robot_type:=franka
```

Set `task:=` to one of `peg_insert`, `gear_mesh`, `nut_thread`, `pick_place` (see the table).

## Play
```bash
python scripts/play.py \
--task forge-peg_insert \
--checkpoint <CHECKPOINT>
```

## Record
```bash
python scripts/record.py \
--task forge-peg_insert \
--checkpoint <CHECKPOINT> \
--lerobot_root /home/home/datasets/FIRe/gr00t/peg_insert \
--lerobot_task "Insert peg into the socket" \
--last_episode
```

Swap `--task` / `--lerobot_task` per the table above. Add `--resume` to append another episode to
the same dataset root.

## Revision experiment logging

One `play.py` run = one trial. With `--save_path <DIR>` every run writes
`<DIR>/collected_data_<YYYYmmdd_HHMMSS>.csv` and appends one row to `<DIR>/episodes.csv`
(`csv_file, label, note, task, vla, checkpoint, episode_length, control_hz, start_time, end_time`).
`--ask_success` prompts for the success label (0/1) and an optional note right after the episode
(also after a Ctrl+C stop); without it the row is still written with an empty `label`.
`--force_threshold <N>` fixes the forge force threshold instead of sampling U[5, 10] N per episode.

```bash
python scripts/play.py \
--task forge-peg_insert \
--checkpoint <CHECKPOINT> \
--vla gr00t --host <HOST> --port <PORT> \
--use_cameras --use_ft_sensor \
--save_path <DIR> \
--ask_success \
--force_threshold 7.5
```

Columns added to the per-step CSV (all existing columns are unchanged; new ones are appended):

| Column | Meaning |
|---|---|
| `step` | Control-step index (replay `--raw` keeps the source CSV's `step`). |
| `t_wall` | `time.time()` at the start of the control step (in RL runs the row's obs/FT were sampled at the end of the previous step). |
| `ft_force_raw_0..2` / `ft_torque_raw_0..2` | Unsmoothed wrench, axes rotated into the robot base frame (forge only; zeros without `--use_ft_sensor`). Torque is about the sensor origin. |
| `ft_torque_0..2` | EMA-smoothed torque, same smoothing as `ft_force_0..2` (forge only). |
| `force_threshold` | The episode's force threshold [N] (forge only). |
| `residual_action_0..6` | RL policy output before the VLA chunk is added (RL and RL+VLA runs). |
| `vla_action_*` / `vla_chunk_idx` | VLA chunk row used at this step and its row index (RL+VLA: `0..6`, zero-padded; VLA-only: the raw chunk row). |

In RL+VLA runs `policy_action = residual_action + vla_action`. Columns that do not apply to a mode
are absent.

The CSV holds the wrench only at 15 Hz. Record the full-rate stream (raw, sensor frame) alongside
each session; timestamps share the system clock with `t_wall` (unless `--use_sim_time`):

```bash
source ~/ros2_ws/install/setup.bash
ros2 bag record -o <DIR>/wrench_$(date +%Y%m%d_%H%M%S) /bota_ft_sensor/wrench
```

The topic is `FTSensorConfig.wrench_topic` in
`src/FIRe/lerobot_ft_sensor/src/lerobot_ft_sensor/configuration_ft_sensor.py`.

Note: `episodes.csv` lives next to the trial CSVs, so loaders that glob `*.csv` in that directory
should glob `collected_data_*.csv` instead.
