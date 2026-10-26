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

cmake_minimum_required(VERSION 3.8)
project(fire_description)

find_package(ament_cmake REQUIRED)
find_package(cho_description_franka REQUIRED)

install(DIRECTORY urdf xacro meshes xml DESTINATION share/${PROJECT_NAME})
# The USD is generated (usd/README.md) and not tracked; whatever is there is installed.
install(DIRECTORY usd DESTINATION share/${PROJECT_NAME} PATTERN ".gitignore" EXCLUDE)

# The MuJoCo model's meshes are the core FR3's (cho_description_franka
# xml/fr3/assets) plus the sensor's mounting, which the MJCF names by file in
# one asset directory. They are installed from the core rather than copied into
# this repository.
set(_core_assets "${cho_description_franka_DIR}/../xml/fr3/assets")
if(NOT EXISTS "${_core_assets}")
  message(FATAL_ERROR "cho_description_franka has no xml/fr3/assets (${_core_assets})")
endif()
install(DIRECTORY "${_core_assets}/" DESTINATION share/${PROJECT_NAME}/xml/fr3_with_ft_sensor/assets)
install(FILES meshes/ft_sensor/mounting.stl DESTINATION share/${PROJECT_NAME}/xml/fr3_with_ft_sensor/assets)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_ft_description test/test_ft_description.py TIMEOUT 120)
endif()

ament_package()
