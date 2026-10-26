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

"""The FORGE assembly tasks: peg insertion, gear meshing, nut threading.

Registered with cho_task_manager by setup.py's entry points, as fire_peg_insert,
fire_gear_mesh and fire_nut_thread: each name below is a {robot_type: builder}
mapping. A directory without this file is not a package, and find_packages()
would skip it.
"""

from .gear_mesh import create_franka_gear_mesh_tree
from .nut_thread import create_franka_nut_thread_tree
from .peg_insert import create_franka_peg_insert_tree

__all__ = [
    'create_franka_gear_mesh_tree',
    'create_franka_nut_thread_tree',
    'create_franka_peg_insert_tree',
]

#: The registrations setup.py points at.
PEG_INSERT = {'franka': create_franka_peg_insert_tree}
GEAR_MESH = {'franka': create_franka_gear_mesh_tree}
NUT_THREAD = {'franka': create_franka_nut_thread_tree}
