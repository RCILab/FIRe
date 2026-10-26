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

"""Each launch includes a core bringup with arguments it has, naming files that exist."""

import os

from ament_index_python.packages import get_package_share_directory
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import get_launch_description_from_python_launch_file
import pytest
import yaml

SHARE = get_package_share_directory('fire_bringup')
CORE = os.path.join(get_package_share_directory('cho_bringup_franka'), 'launch')
BACKENDS = ('real', 'gz', 'mujoco', 'isaac')
#: Arguments a value of which is a file this workspace installs.
FILE_ARGUMENTS = {'ee_connection_xacro', 'payload_file', 'xacro_file'}


def _text(value):
    """A launch argument as the include holds it: a string, or text substitutions."""
    return value if isinstance(value, str) else ''.join(part.text for part in value)


def _include(backend):
    description = get_launch_description_from_python_launch_file(
        os.path.join(SHARE, 'launch', f'bringup_{backend}_robot.launch.py'))
    includes = [entity for entity in description.entities if isinstance(entity, IncludeLaunchDescription)]
    assert len(includes) == 1, backend
    return includes[0]


@pytest.mark.parametrize('backend', BACKENDS)
def test_it_sets_only_arguments_the_core_bringup_declares(backend):
    include = _include(backend)
    core = get_launch_description_from_python_launch_file(
        os.path.join(CORE, f'bringup_{backend}_robot.launch.py'))
    declared = {entity.name for entity in core.entities if isinstance(entity, DeclareLaunchArgument)}
    given = {_text(name): _text(value) for name, value in include.launch_arguments}
    assert set(given) <= declared, set(given) - declared
    for name in FILE_ARGUMENTS & set(given):
        assert os.path.isfile(given[name]), (name, given[name])


def test_the_payload_is_the_hand_with_the_sensor():
    with open(os.path.join(SHARE, 'config', 'payload_ft_sensor.yaml')) as stream:
        params = yaml.safe_load(stream)['/**']['ros__parameters']
    assert params['mass'] == pytest.approx(0.97)
