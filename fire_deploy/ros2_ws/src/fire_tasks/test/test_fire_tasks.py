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

"""The FORGE tasks as cho_task_manager finds them, checked as it checks its own.

Run against the install (colcon test): each task is found through the entry
point setup.py declares, not imported by hand.
"""

from cho_task_manager.behaviors.wait import WaitBehavior
from cho_task_manager.tasks import available_tasks
from cho_task_manager.testing import goal_problems, registry_tree
import pytest

from fire_tasks.ft_sensor import tare_ft_children
from fire_tasks.tare_ft_sensor import TareFTSensorServiceBehavior

TASKS = ('fire_peg_insert', 'fire_gear_mesh', 'fire_nut_thread')


@pytest.mark.parametrize('task', TASKS)
def test_each_task_is_registered_for_franka_only(task):
    assert task in available_tasks('franka')
    for robot in ('fr5', 'ur5e', 'openarm'):
        assert task not in available_tasks(robot)


@pytest.mark.parametrize('task', TASKS)
def test_every_goal_names_the_registry_frame_and_joints(task):
    tree, config = registry_tree(task, 'franka')
    assert goal_problems(tree, config) == []


def test_the_tasks_that_read_force_tare_the_sensor_first():
    for task, tares in (('fire_peg_insert', False), ('fire_gear_mesh', True), ('fire_nut_thread', True)):
        tree, _config = registry_tree(task, 'franka')
        found = [node for node in tree.iterate() if isinstance(node, TareFTSensorServiceBehavior)]
        assert bool(found) == tares, task
        if tares:
            assert found[0].service_name == '/bota_ft_sensor/tare'


def test_the_settle_after_the_tare_is_a_node_clock_wait():
    tare, settle = tare_ft_children(settle_sec=3.0)
    assert isinstance(tare, TareFTSensorServiceBehavior)
    assert isinstance(settle, WaitBehavior)
    assert (settle.name, settle.duration_sec) == ('Wait_After_Tare', 3.0)
