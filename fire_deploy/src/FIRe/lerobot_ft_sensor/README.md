# LeRobot FT Sensor

`lerobot_ft_sensor` is a standalone LeRobot-compatible ROS 2 subscriber for force/torque data.

This package no longer connects to the Bota sensor directly. A separate process should publish
`geometry_msgs/msg/WrenchStamped` messages, and `FTSensor` caches the latest sample.

## Install

From this module directory:

```bash
cd path/to/lerobot_ft_sensor
pip install -e .
```

ROS 2 Python packages such as `rclpy` are expected to come from the system ROS installation,
not from `pip`.

Before running ROS scripts:

```bash
source ~/ros2_ws/install/setup.bash
```

## Configuration

Default topics:

```text
/bota_ft_sensor/wrench
```

The topic uses `geometry_msgs/msg/WrenchStamped`.

If your wrench topic has a different name, set:

```python
FTSensorConfig(wrench_topic="/your/wrench_topic")
```

Useful fields:

```python
FTSensorConfig(
    wrench_topic="/bota_ft_sensor/wrench",
    queue_size=1,
)
```

## ROS 2 Subscriber Test

Start the external publisher first, then run:

```bash
source ~/ros2_ws/install/setup.bash
python test/test_lerobot_ft_sensor.py
```

The script prints the latest force and torque exposed by `FTSensor`.
