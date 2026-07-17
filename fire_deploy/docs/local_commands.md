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
