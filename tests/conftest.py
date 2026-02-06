"""
tests/conftest.py

Configuration for pytest.
"""

from pathlib import Path
from collections.abc import Callable
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bluebox.data_models.routine.routine import Routine
from bluebox.data_models.routine.operation import RoutineOperationUnion


@pytest.fixture(autouse=True)
def mock_openai_clients() -> dict[str, MagicMock]:
    """
    Mock OpenAI clients to avoid needing real API keys in tests.

    This fixture automatically patches the OpenAI and AsyncOpenAI classes
    wherever they're imported in the openai_client module, so tests can
    instantiate OpenAIClient without a valid API key.
    """
    with (
        patch("bluebox.llms.openai_client.OpenAI") as mock_openai,
        patch("bluebox.llms.openai_client.AsyncOpenAI") as mock_async_openai,
    ):
        mock_openai.return_value = MagicMock()
        mock_async_openai.return_value = MagicMock()
        yield {"openai": mock_openai, "async_openai": mock_async_openai}


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
def make_routine() -> Callable[..., Routine]:
    """
    Factory fixture to create Routine with hardcoded defaults.

    Automatically ensures the routine ends with a proper return operation
    and has at least 2 operations (required by validation).

    Usage:
        routine = make_routine(operations=[...])
        routine = make_routine(operations=[...], parameters=[...], name="custom")
    """
    from bluebox.data_models.routine.operation import (
        RoutineFetchOperation,
        RoutineReturnOperation,
        RoutineReturnHTMLOperation,
    )
    from bluebox.data_models.routine.endpoint import Endpoint, HTTPMethod

    def factory(operations: list[RoutineOperationUnion], **kwargs: Any) -> Routine:
        defaults = {
            "name": "test_routine",
            "description": "Test routine",
        }

        # Check if operations end with a return operation
        has_return = operations and isinstance(
            operations[-1], (RoutineReturnOperation, RoutineReturnHTMLOperation)
        )

        # If no return operation, add a fetch + return
        if not has_return:
            # Add a fetch operation that sets a session_storage_key
            fetch_op = RoutineFetchOperation(
                endpoint=Endpoint(
                    url="https://test.example.com/api",
                    method=HTTPMethod.GET,
                    headers={},
                    body={}
                ),
                session_storage_key="test_result"
            )
            return_op = RoutineReturnOperation(session_storage_key="test_result")
            operations = list(operations) + [fetch_op, return_op]
        else:
            # Ensure the return operation's session_storage_key is set by a prior fetch/js_evaluate
            last_op = operations[-1]
            return_key = last_op.session_storage_key if hasattr(last_op, 'session_storage_key') else None
            if return_key:
                # Check if any prior operation sets this key
                has_setter = any(
                    hasattr(op, 'session_storage_key') and op.session_storage_key == return_key
                    for op in operations[:-1]
                )
                if not has_setter and len(operations) >= 2:
                    # Add a fetch operation before the return that sets the key
                    fetch_op = RoutineFetchOperation(
                        endpoint=Endpoint(
                            url="https://test.example.com/api",
                            method=HTTPMethod.GET,
                            headers={},
                            body={}
                        ),
                        session_storage_key=return_key
                    )
                    operations = list(operations[:-1]) + [fetch_op, operations[-1]]

        return Routine(
            operations=operations,
            **{**defaults, **kwargs}
        )
    return factory


@pytest.fixture
def mock_event_callback() -> AsyncMock:
    """Async callback that records calls for CDP monitor testing."""
    return AsyncMock()


@pytest.fixture
def mock_cdp_session() -> AsyncMock:
    """Mock AsyncCDPSession for testing monitors."""
    session = AsyncMock()
    session.send = AsyncMock(return_value=1)
    session.send_and_wait = AsyncMock(return_value={"result": {}})
    session.enable_domain = AsyncMock()
    session.page_session_id = "mock-session-id"
    return session
