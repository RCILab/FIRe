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

"""ament_flake8 over the package sources, with the rules in setup.cfg."""
import os

from ament_flake8.main import main_with_errors
import pytest

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGETS = [os.path.join(PACKAGE_ROOT, name) for name in ('fire_tasks', 'setup.py', 'test')]


@pytest.mark.flake8
@pytest.mark.linter
def test_flake8():
    rc, errors = main_with_errors(argv=['--config', os.path.join(PACKAGE_ROOT, 'setup.cfg')] + TARGETS)
    assert rc == 0, 'Found %d code style errors / warnings:\n' % len(errors) + '\n'.join(errors)
