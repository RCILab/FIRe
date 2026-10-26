# Isaac Sim USD of the FR3 with the FT sensor

Generated, not tracked (`.gitignore`): rebuild it from `urdf/fr3_with_ft_sensor/fr3_franka_hand.urdf`
whenever that changes, with the core's converter (its options and why: cho_description_franka
`usd/README.md`):

```bash
source ~/fire_ws/install/setup.bash    # this workspace, over the core's
DESC=$(ros2 pkg prefix fire_description)/share/fire_description
CORE=$(ros2 pkg prefix cho_description_franka)/share/cho_description_franka
ISAAC=$(ros2 pkg prefix cho_simulation_isaac)/share/cho_simulation_isaac

~/isaacsim/python.sh $ISAAC/isaac/convert_urdf_to_usd.py \
    --urdf         $DESC/urdf/fr3_with_ft_sensor/fr3_franka_hand.urdf \
    --xacro-arg    hardware:=isaac \
    --usd-path     <this directory, in the source tree>/fr3_with_ft_sensor \
    --require-link fr3_hand_tcp --require-link bota_ft_sensor_wrench \
    --ros-package  cho_description_franka:$CORE --ros-package fire_description:$DESC
```

`--require-link bota_ft_sensor_wrench` matters: Isaac's FT emulation (`ft_link:=bota_ft_sensor_wrench`) reads that
link's incoming joint force, and a fixed joint merged away takes it with it. The result,
`fr3_with_ft_sensor/fr3_franka_hand/fr3_franka_hand.usda`, is what `fire_bringup`'s Isaac launch passes as
`robot_usd`. Rebuild the package once after the first conversion so the install has it.
