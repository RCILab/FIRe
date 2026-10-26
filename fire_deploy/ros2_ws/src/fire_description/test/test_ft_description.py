# Copyright 2026 Hyunho Cho
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The sensor sits where the real one does, in the xacro the real bringup builds and in the one-file URDF.

Run against the install (colcon test sources it).
"""

import os
import re
import xml.etree.ElementTree as ET

from ament_index_python.packages import get_package_share_directory
import pytest
import xacro

SHARE = get_package_share_directory('fire_description')
CORE = get_package_share_directory('cho_description_franka')
SENSOR_JOINTS = ('robot_mounting_link_to_bota_ft_sensor', 'bota_ft_sensor_mounting_to_wrench',
                 'bota_ft_sensor_mounting_to_mech1', 'bota_ft_sensor_mounting_to_mech2',
                 'bota_ft_sensor_mounting_to_mech3', 'fr3_hand_joint')


def _joints(root):
    joints = {}
    for joint in root.iter('joint'):
        if joint.find('parent') is None:
            continue    # a <ros2_control> joint
        origin = joint.find('origin')
        joints[joint.get('name')] = (
            joint.find('parent').get('link'), joint.find('child').get('link'),
            tuple(float(v) for v in (origin.get('xyz') or '0 0 0').split()) if origin is not None else (0, 0, 0),
            tuple(float(v) for v in (origin.get('rpy') or '0 0 0').split()) if origin is not None else (0, 0, 0))
    return joints


@pytest.fixture(scope='module')
def mounted():
    """The core FR3 xacro with this package's ee_connection(), as the real and Gazebo bringups build it."""
    document = xacro.process_file(
        os.path.join(CORE, 'robots', 'fr3', 'fr3.urdf.xacro'),
        mappings={'hand': 'true', 'ee_id': 'franka_hand',
                  'ee_connection_xacro': os.path.join(SHARE, 'xacro', 'bota_ft_sensor.xacro'),
                  'special_connection': 'bota_ft_sensor_wrench'})
    return _joints(ET.fromstring(document.toxml()))


def test_the_hand_hangs_off_the_sensor_35_mm_above_the_flange(mounted):
    assert mounted['robot_mounting_link_to_bota_ft_sensor'][:2] == ('fr3_link8', 'bota_ft_sensor_mounting')
    assert mounted['bota_ft_sensor_mounting_to_wrench'][2] == pytest.approx((0.0, 0.0, 0.035))
    assert mounted['fr3_hand_joint'][:2] == ('bota_ft_sensor_wrench', 'fr3_hand')


def test_the_one_file_urdf_carries_the_same_sensor(mounted):
    with open(os.path.join(SHARE, 'urdf', 'fr3_with_ft_sensor', 'fr3_franka_hand.urdf')) as stream:
        flat = _joints(ET.fromstring(re.sub(r'<\?xml[^>]*\?>', '', stream.read())))
    for name in SENSOR_JOINTS:
        assert flat[name][:2] == mounted[name][:2], name
        assert flat[name][2] == pytest.approx(mounted[name][2]), name
        assert flat[name][3] == pytest.approx(mounted[name][3], abs=1e-4), name


def test_the_urdf_names_this_packages_mujoco_models_and_they_have_their_meshes():
    with open(os.path.join(SHARE, 'urdf', 'fr3_with_ft_sensor', 'fr3_franka_hand.urdf')) as stream:
        text = stream.read()
    models = re.findall(r'\$\(find fire_description\)/(xml/fr3_with_ft_sensor/scene_\w+\.xml)', text)
    assert sorted(set(models)) == ['xml/fr3_with_ft_sensor/scene_position.xml',
                                   'xml/fr3_with_ft_sensor/scene_torque.xml',
                                   'xml/fr3_with_ft_sensor/scene_velocity.xml']
    for model in models:
        assert os.path.isfile(os.path.join(SHARE, model)), model
    assets = os.path.join(SHARE, 'xml', 'fr3_with_ft_sensor', 'assets')
    with open(os.path.join(SHARE, 'xml', 'fr3_with_ft_sensor', 'fr3.xml')) as stream:
        meshes = re.findall(r'<mesh [^>]*file="([^"]+)"', stream.read())
    assert 'mounting.stl' in meshes
    missing = [mesh for mesh in meshes if not os.path.isfile(os.path.join(assets, mesh))]
    assert missing == []
