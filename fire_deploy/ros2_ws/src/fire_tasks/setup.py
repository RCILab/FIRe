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

from setuptools import find_packages, setup

package_name = 'fire_tasks'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Hyunho Cho',
    maintainer_email='chohh7391@gmail.com',
    description="FIRe's task trees, registered with cho_task_manager",
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        # Each task: <task name> = <module>:<{robot_type: tree builder}>. Task
        # names are global across every installed package, hence the prefix.
        'cho_task_manager.tasks': [
            'fire_peg_insert = fire_tasks.forge:PEG_INSERT',
            'fire_gear_mesh = fire_tasks.forge:GEAR_MESH',
            'fire_nut_thread = fire_tasks.forge:NUT_THREAD',
        ],
    },
)
