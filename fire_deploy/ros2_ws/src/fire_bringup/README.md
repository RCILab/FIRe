# fire_bringup

The core's Franka bringups (`cho_bringup_franka`) with the Bota SensONE mounted between the flange and the hand
(`fire_description`), the sensor's driver, and its emulation in Isaac Sim.

```bash
ros2 launch fire_bringup bringup_real_robot.launch.py   control_mode:=torque controller_name:=joint_space_impedance_controller
ros2 launch fire_bringup bota_ft_sensor.launch.py       # the real sensor's driver, beside the real bringup
ros2 launch fire_bringup bringup_isaac_robot.launch.py  control_mode:=torque action_chunk:=true
ros2 launch fire_bringup bringup_mujoco_robot.launch.py control_mode:=torque
ros2 launch fire_bringup bringup_gz_robot.launch.py     control_mode:=torque
```

Each `bringup_<backend>_robot` includes the core bringup of that name and passes every other argument through as
given (`--show-args` on the core's file lists them). What it sets:

| | real, Gazebo | MuJoCo | Isaac |
| --- | --- | --- | --- |
| the robot | `ee_connection_xacro:=` fire_description `xacro/bota_ft_sensor.xacro`, `special_connection:=bota_ft_sensor_wrench` | `xacro_file:=` fire_description's one-file URDF (its MuJoCo model with the sensor) | `xacro_file:=` the same URDF, `robot_usd:=` its USD |
| the payload | `payload_file:=config/payload_ft_sensor.yaml`, the hand and the sensor, 0.97 kg | the same | the same |
| the wrench | `bota_ft_sensor.launch.py`, the driver (real only) | none | `ft_link:=bota_ft_sensor_wrench` and `scripts/isaac_ft_sensor.py` |

On the real robot, Desk's end-effector load has to match the payload.

## The sensor's driver

`bota_ft_sensor.launch.py` starts only `bota_driver_node` (`bota_driver`, `../../extern/bota_driver_ros2`) with
`bota_driver_example`'s configuration (`../../extern/bota_driver_ros2_example`): the description already carries the
sensor's links, so a second `robot_state_publisher` and RViz would conflict with the bringup. For Bota's own
standalone views use their launches, e.g. `ros2 launch bota_driver_example plotjuggler.launch.py`.

| argument | default | |
| --- | --- | --- |
| `bota_ft_sensor_link_name` | `bota_ft_sensor` | sensor link prefix; also the driver node name and the topic prefix below |
| `config_file` | `bota_driver_example`'s `bota_config/bota_binary.json` | Bota driver JSON configuration (their serial example) |
| `output_rate` | `500.0` | driver output rate [Hz] |

With the default name it serves `/bota_ft_sensor/wrench` (`geometry_msgs/WrenchStamped`) and `/bota_ft_sensor/tare`
(`std_srvs/Trigger`, zeroes the sensor), and also `<name>/imu` and `<name>/bota_frame`. fire_tasks' tare calls the
service; the core's `SafetyMonitorBehavior` takes the wrench with `wrench_topic='/bota_ft_sensor/wrench'`. In Isaac,
`isaac_ft_sensor.py` serves the same two names from the emulated reading, so a task does not change between the robot
and simulation.

## Frames

`bota_ft_sensor_wrench` is `fr3_link8` shifted 0.035 m along z with no rotation, and `fr3_hand` hangs off it at
Rz(−π/4) (`fr3_hand_joint`). The transform from the sensor's frame to the TCP frame is `R_tcp_ft = Rx(π) × Rz(−π/4)`.
A real wrench is rotated into the world frame with `q_ft_to_world = quat_mul(ee_quat, q_ft_to_tcp)`, where
`ee_quat` is the pose on `/ee_state/pose` (the `fr3_hand_tcp` frame).
