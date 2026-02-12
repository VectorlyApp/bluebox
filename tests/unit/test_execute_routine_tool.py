"""
Unit tests for execute_routine_tool.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from bluebox.llms.tools.execute_routine_tool import execute_routine


@pytest.fixture
def mock_executor():
    """Mock RoutineExecutor to avoid Chrome dependency."""
    with patch("bluebox.llms.tools.execute_routine_tool.RoutineExecutor") as mock_class:
        mock_instance = MagicMock()
        mock_instance.execute.return_value = {"status": "completed", "data": {}}
        mock_class.return_value = mock_instance
        yield mock_instance


def test_execute_routine_invalid_json():
    """Test that invalid JSON string returns an error."""
    result = execute_routine(
        routine="not valid json",
        parameters={"test": "value"},
    )
    assert result["success"] is False
    assert "Invalid routine JSON" in result["error"]


def test_execute_routine_invalid_routine_from_json():
    """Test that invalid routine structure from JSON returns an error."""
    result = execute_routine(
        routine='{"invalid": "routine"}',
        parameters={"test": "value"},
    )
    assert result["success"] is False
    assert "Failed to parse routine" in result["error"]


def test_execute_routine_invalid_routine_from_dict():
    """Test that invalid routine dict returns an error."""
    result = execute_routine(
        routine={"invalid": "routine"},
        parameters={"test": "value"},
    )
    assert result["success"] is False
    assert "Failed to parse routine" in result["error"]


def test_execute_routine_valid_json_string(mock_executor):
    """Test that a valid routine JSON string can be parsed and executed."""
    routine = {
        "name": "test_routine",
        "description": "A test routine",
        "parameters": [
            {
                "name": "test_param",
                "description": "A test parameter",
                "type": "string",
                "required": True,
            }
        ],
        "operations": [
            {
                "type": "fetch",
                "endpoint": {
                    "url": "https://example.com/{{test_param}}",
                    "method": "GET",
                    "headers": {},
                    "body": {}
                },
                "session_storage_key": "result"
            },
            {
                "type": "return",
                "session_storage_key": "result",
            }
        ],
    }

    result = execute_routine(
        routine=json.dumps(routine),
        parameters={"test_param": "value"},
    )

    assert result["success"] is True
    assert "result" in result
    mock_executor.execute.assert_called_once()


def test_execute_routine_valid_dict(mock_executor):
    """Test that a valid routine dict can be parsed and executed."""
    routine = {
        "name": "test_routine",
        "description": "A test routine",
        "parameters": [
            {
                "name": "test_param",
                "description": "A test parameter",
                "type": "string",
                "required": True,
            }
        ],
        "operations": [
            {
                "type": "fetch",
                "endpoint": {
                    "url": "https://example.com/{{test_param}}",
                    "method": "GET",
                    "headers": {},
                    "body": {}
                },
                "session_storage_key": "result"
            },
            {
                "type": "return",
                "session_storage_key": "result",
            }
        ],
    }

    result = execute_routine(
        routine=routine,
        parameters={"test_param": "value"},
    )

    assert result["success"] is True
    assert "result" in result
    mock_executor.execute.assert_called_once()
