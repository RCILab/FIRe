# fire_deploy/ros2_ws — FIRe's ROS 2 side, on cho_robot_project

The robot side of FIRe's deployment, as an overlay workspace on the core,
[cho_robot_project](https://github.com/chohh7391/cho_robot_project), at the tag in `fire.repos`: the FR3 with the Bota
SensONE force-torque sensor between its flange and its hand, the bringups that mount it, and the FORGE task trees.
The core's own robot has no sensor; everything about it is here. How a project builds on the core: the core's
`docs/usage_guidelines.md` and `templates/overlay_project/`.

## Packages

| Package | Holds |
| --- | --- |
| `fire_description` | the sensor as the core Franka xacro's `ee_connection()`, the robot with it as one URDF, its MuJoCo model and Isaac USD ([README](src/fire_description/README.md)) |
| `fire_bringup` | `bringup_{real,gz,mujoco,isaac}_robot.launch.py` over the core's, `bota_ft_sensor.launch.py` (the driver), `config/payload_ft_sensor.yaml`, the Isaac FT node ([README](src/fire_bringup/README.md)) |
| `fire_tasks` | `fire_peg_insert`, `fire_gear_mesh`, `fire_nut_thread` (the FORGE trees, registered with `cho_task_manager`) and the sensor tare they start with |
| `extern/bota_driver_ros2`, `extern/bota_driver_ros2_example` | Bota's driver and its example configuration, as submodules (`git submodule update --init --recursive`) |

## Build

```bash
# The core, at the tag in fire.repos (once, and again when the pin moves); build it as its docs/installation.md says
mkdir -p ~/core_ws/src && cd ~/core_ws
vcs import --recursive src < ~/FIRe/fire_deploy/ros2_ws/fire.repos

# This workspace, on top of it
source ~/core_ws/install/setup.bash
cd ~/FIRe/fire_deploy/ros2_ws
PYTHONNOUSERSITE=1 MAKEFLAGS="-j2 -l2" colcon build --parallel-workers 2 --symlink-install --base-paths src extern \
  --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
PYTHONNOUSERSITE=1 colcon test --base-paths src && colcon test-result --verbose
```

A core workspace already built serves as well, checked out at the pin. Every terminal sources this workspace's
`install/setup.bash`, which brings the core's. `PYTHONNOUSERSITE=1` is needed when `~/.local` holds numpy 2.x: it
breaks the pinocchio import. The Isaac USD is built once after the first build (fire_description `usd/README.md`).

## Run

```bash
# Isaac Sim, the FORGE tasks' simulator: the robot with the sensor, the emulated wrench and tare
ros2 launch fire_bringup bringup_isaac_robot.launch.py control_mode:=torque action_chunk:=true
ros2 launch cho_task_manager run_task_manager.launch.py robot_type:=franka task:=fire_gear_mesh \
  control_mode:=torque use_sim_time:=true

# The real FR3: the bringup and the sensor's driver
ros2 launch fire_bringup bringup_real_robot.launch.py control_mode:=torque action_chunk:=true
ros2 launch fire_bringup bota_ft_sensor.launch.py
```

`fire_gear_mesh` and `fire_nut_thread` tare the sensor first (`/bota_ft_sensor/tare`); `fire_peg_insert` does not.
On the real robot, Desk's end-effector load has to match `config/payload_ft_sensor.yaml` (0.97 kg).
