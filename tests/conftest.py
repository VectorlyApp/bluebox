"""
tests/conftest.py

Configuration for pytest.
"""

from pathlib import Path
from typing import Any

import pytest

from web_hacker.data_models.routine.routine import Routine
from web_hacker.data_models.routine.operation import RoutineOperationUnion


@pytest.fixture(scope="session")
def tests_root() -> Path:
    """
    Root directory for tests.
    Returns:
        Path to the tests directory.
    """
    return Path(__file__).parent.resolve()


@pytest.fixture(scope="session")
def data_dir(tests_root: Path) -> Path:
    """
    Directory containing test data files.
    Returns:
        Path to tests/data.
    """
    d = tests_root / "data"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture(scope="session")
def input_data_dir(data_dir: Path) -> Path:
    """
    Directory containing input test data files.
    Returns:
        Path to tests/data/input.
    """
    d = data_dir / "input"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture
def make_routine():
    """
    Factory fixture to create Routine with hardcoded defaults.
    
    Usage:
        routine = make_routine(operations=[...])
        routine = make_routine(operations=[...], parameters=[...], name="custom")
    """
    def factory(operations: list[RoutineOperationUnion], **kwargs: Any) -> Routine:
        defaults = {
            "id": "Routine_test-00000000-0000-0000-0000-000000000000",
            "created_at": 1609459200,
            "updated_at": 1609459200,
            "metadata": None,
            "name": "test_routine",
            "description": "Test routine",
            "created_by": "test_user",
            "project_id": "test_project",
        }
        return Routine(operations=operations, **{**defaults, **kwargs})
    
    return factory
