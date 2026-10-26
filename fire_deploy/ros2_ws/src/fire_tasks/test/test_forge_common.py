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

"""
Unit tests for the shared forge task helpers (fire_tasks.forge.common).

Covers the randomization bug that motivated extracting this module: offsets and yaw noise
used to be computed once at import time (module-level `range = [...]; x_offset =
np.random.uniform(...)`), so rebuilding a task tree never re-randomized. These tests assert
the replacement helpers sample fresh values on every call.
"""
import numpy as np
import py_trees

from fire_tasks.forge.common import (
    build_forge_tree,
    quat_mul,
    random_xy_offset,
    random_yaw_orientation,
)
from fire_tasks.forge.peg_insert import create_franka_peg_insert_tree
from fire_tasks.forge.gear_mesh import create_franka_gear_mesh_tree
from fire_tasks.forge.nut_thread import create_franka_nut_thread_tree


def test_quat_mul_identity():
    identity = [0.0, 0.0, 0.0, 1.0]
    assert quat_mul(identity, identity) == identity


def test_quat_mul_by_identity_is_noop():
    q = [0.6099, 0.7927, 0.0, 0.0]
    identity = [0.0, 0.0, 0.0, 1.0]
    assert quat_mul(q, identity) == q


def test_random_yaw_orientation_with_zero_range_is_deterministic_noop():
    base = [1.0, 0.0, 0.0, 0.0]
    assert random_yaw_orientation(base, [0.0, 0.0]) == base


def test_random_xy_offset_resamples_on_every_call():
    """Guards against the import-time-only randomization bug this module replaced."""
    np.random.seed(42)
    first = random_xy_offset([-0.01, 0.01])
    second = random_xy_offset([-0.01, 0.01])
    assert first != second  # two draws from the same seeded stream must differ

    np.random.seed(42)
    replay = random_xy_offset([-0.01, 0.01])
    assert replay == first  # reseeding proves each call actually draws fresh, not cached


def _mission(tree):
    """The mission sequence under the standard guarded root.

    The root is OneShot -> Selector(mission, safe abort); the abort branch is
    what a mid-mission failure runs instead of the tree simply dying with a
    controller still driving the arm.
    """
    assert isinstance(tree, py_trees.decorators.OneShot)
    guard = tree.decorated
    assert isinstance(guard, py_trees.composites.Selector)
    assert [child.name for child in guard.children][1] == "Report_Mission_Failure"
    return guard.children[0]


def _build_test_forge_tree():
    return build_forge_tree(
        task_label="Test_Task",
        approach_position_fn=lambda x, y: [0.6 + x, y, 0.1],
        base_orientation=[1.0, 0.0, 0.0, 0.0],
        yaw_range=[0.0, 0.0],
        grasp_params=dict(width=0.01, speed=0.05, force=50.0, epsilon_inner=0.005, epsilon_outer=0.005),
    )


def test_build_forge_tree_does_not_wire_in_finish_seq():
    mission_sequence = _mission(_build_test_forge_tree())
    child_names = [child.name for child in mission_sequence.children]
    assert child_names == ["1_Initialize", "2_Approach_Fixed_Object", "3_Start_Chunk_Stream"]
    assert "4_Finish" not in child_names


def test_forge_task_trees_still_build():
    for factory in (
        create_franka_peg_insert_tree,
        create_franka_gear_mesh_tree,
        create_franka_nut_thread_tree,
    ):
        tree = factory()
        assert isinstance(tree, py_trees.decorators.OneShot)


def test_forge_trees_hold_the_arm_when_the_mission_fails():
    """A forge mission that fails part-way used to leave whichever controller
    it was driving active with nothing putting the arm anywhere safe.
    """
    guard = _build_test_forge_tree().decorated
    abort = guard.children[1].decorated.decorated

    assert abort.name == "Safe_Abort"
    switch = abort.children[0]
    # Forge tasks are torque-mode only, so the hold must be the torque one.
    assert switch.make_request().activate_controllers == [
        "joint_space_impedance_controller"]
    assert abort.children[1].require_active == ["joint_space_impedance_controller"]


def _init_child_names(tree):
    init_seq = _mission(tree).children[0]
    assert init_seq.name == "1_Initialize"
    return [child.name for child in init_seq.children]


def test_ft_tare_is_per_task():
    # peg_insert never consumes FT data -> no tare and no settle wait;
    # gear_mesh/nut_thread are contact-rich -> tare must stay (pre-refactor behavior).
    peg_names = _init_child_names(create_franka_peg_insert_tree())
    assert "Tare_FT_Sensor" not in peg_names
    assert "Wait_After_Tare" not in peg_names

    for factory in (create_franka_gear_mesh_tree, create_franka_nut_thread_tree):
        names = _init_child_names(factory())
        assert names[:2] == ["Tare_FT_Sensor", "Wait_After_Tare"]
