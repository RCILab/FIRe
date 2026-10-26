# fire_description

The Franka FR3 with the Bota SensONE force-torque sensor between its flange (`fr3_link8`) and its hand, built on the
core's `cho_description_franka`:

| Path | What it is | Used by |
| --- | --- | --- |
| `xacro/bota_ft_sensor.xacro` | the sensor as the core Franka xacro's `ee_connection()` macro; the hand then hangs off `bota_ft_sensor_wrench`, 35 mm above the flange | the real and Gazebo bringups (`ee_connection_xacro:=`, `special_connection:=`) |
| `urdf/fr3_with_ft_sensor/fr3_franka_hand.urdf` | the same robot in one file, with its MuJoCo and Isaac `ros2_control` blocks | the MuJoCo (`xacro_file:=`) and Isaac bringups |
| `xml/fr3_with_ft_sensor/` | its MuJoCo model; the meshes are installed from the core's `xml/fr3/assets` plus `meshes/ft_sensor/mounting.stl` | the URDF's MuJoCo block |
| `usd/` | its Isaac USD, generated ([usd/README.md](usd/README.md)) | the Isaac bringup (`robot_usd:=`) |
| `meshes/ft_sensor/` | the SensONE mounting mesh, unmodified, from Bota's `bota_driver_ros2` (`meshes/BFT_SENS_M8/mounting.STL`, v1.1.14); BSD-3-Clause with Attribution Requirement, [LICENSE](meshes/ft_sensor/LICENSE). Attribution: Bota Systems AG, https://www.botasys.com | the URDF, the xacro, the MuJoCo model |

The sensor's wrench frame, `bota_ft_sensor_wrench`, is where the driver reports the wrench. It shares the flange's
axes; the hand is turned -45 deg about z from it, as from `fr3_link8` without the sensor. `fr3_hand_tcp` is 0.138 m
along z from `fr3_link8` with the sensor (0.1034 m without). `test/test_ft_description.py` checks the xacro and the
one-file URDF carry the same sensor.
