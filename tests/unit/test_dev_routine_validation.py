"""
tests/unit/test_dev_routine_validation.py

Unit tests for DevRoutine.validate method in dev_routine.py.
"""

import pytest
from web_hacker.data_models.routine.dev_routine import (
    DevRoutine,
    DevEndpoint,
    DevNavigateOperation,
    DevSleepOperation,
    DevFetchOperation,
    DevReturnOperation,
)
from web_hacker.data_models.routine.endpoint import HTTPMethod, CREDENTIALS
from web_hacker.data_models.routine.parameter import Parameter


def _make_endpoint(url: str = "https://api.example.com", headers: str = "{}", body: str = "{}") -> DevEndpoint:
    """Helper to create an endpoint."""
    return DevEndpoint(
        url=url,
        method=HTTPMethod.GET,
        headers=headers,
        body=body,
        description="Test endpoint",
        credentials=CREDENTIALS.SAME_ORIGIN
    )


def _make_fetch_op(key: str, url: str = "https://api.example.com", headers: str = "{}", body: str = "{}") -> DevFetchOperation:
    """Helper to create a fetch operation."""
    return DevFetchOperation(
        endpoint=_make_endpoint(url=url, headers=headers, body=body),
        session_storage_key=key
    )


def _make_return_op(key: str) -> DevReturnOperation:
    """Helper to create a return operation."""
    return DevReturnOperation(session_storage_key=key)


def _make_navigate_op(url: str = "https://example.com") -> DevNavigateOperation:
    """Helper to create a navigate operation."""
    return DevNavigateOperation(url=url)


class TestDevRoutineValidation:
    """Tests for DevRoutine.validate method."""

    def test_valid_routine_simple(self):
        """A simple valid routine (Navigate -> Fetch -> Return) should pass."""
        routine = DevRoutine(
            name="test",
            description="test",
            parameters=[],
            operations=[
                _make_navigate_op(),
                _make_fetch_op(key="result"),
                _make_return_op(key="result")
            ]
        )
        valid, errors, exc = routine.validate()
        assert valid is True
        assert not errors
        assert exc is None

    def test_valid_routine_chained_fetches(self):
        """A routine where fetch 1 output is used in fetch 2 should pass."""
        routine = DevRoutine(
            name="test",
            description="test",
            parameters=[],
            operations=[
                _make_navigate_op(),
                # Fetch 1 produces 'token'
                _make_fetch_op(key="token"),
                # Fetch 2 uses 'token'
                _make_fetch_op(key="result", headers='{"Authorization": "{{sessionStorage:token}}"}'),
                _make_return_op(key="result")
            ]
        )
        valid, errors, exc = routine.validate()
        assert valid is True, f"Errors: {errors}"

    def test_invalid_structure_too_few_ops(self):
        """Routine with < 3 operations should fail."""
        routine = DevRoutine(
            name="test",
            description="test",
            parameters=[],
            operations=[
                _make_navigate_op(),
                _make_return_op(key="result")
            ]
        )
        valid, errors, exc = routine.validate()
        assert valid is False
        assert "Must have at least 3 operations (navigate, fetch, return)" in errors

    def test_invalid_structure_first_not_navigate(self):
        """Routine starting with something other than Navigate should fail."""
        routine = DevRoutine(
            name="test",
            description="test",
            parameters=[],
            operations=[
                _make_fetch_op(key="res"),
                _make_fetch_op(key="res"),
                _make_return_op(key="res")
            ]
        )
        valid, errors, exc = routine.validate()
        assert valid is False
        assert "First operation should be a navigate operation" in errors

    def test_invalid_structure_last_not_return(self):
        """Routine ending with something other than Return should fail."""
        routine = DevRoutine(
            name="test",
            description="test",
            parameters=[],
            operations=[
                _make_navigate_op(),
                _make_fetch_op(key="res"),
                _make_fetch_op(key="res") # Not return
            ]
        )
        valid, errors, exc = routine.validate()
        assert valid is False
        assert "Last operation should be a return operation" in errors

    def test_invalid_structure_second_last_not_fetch(self):
        """Routine where second-to-last op is not Fetch should fail."""
        routine = DevRoutine(
            name="test",
            description="test",
            parameters=[],
            operations=[
                _make_navigate_op(),
                _make_navigate_op(), # Not fetch
                _make_return_op(key="res")
            ]
        )
        valid, errors, exc = routine.validate()
        assert valid is False
        assert "Second to last operation should be a fetch operation" in errors

    def test_invalid_return_key_mismatch(self):
        """Return key must match the last fetch key."""
        routine = DevRoutine(
            name="test",
            description="test",
            parameters=[],
            operations=[
                _make_navigate_op(),
                _make_fetch_op(key="result_A"),
                _make_return_op(key="result_B")
            ]
        )
        valid, errors, exc = routine.validate()
        assert valid is False
        assert "Last fetch session_storage_key should match return session_storage_key" in errors

    def test_parameter_validation_valid(self):
        """Routine using all defined parameters should pass."""
        params = [Parameter(name="user_id", description="id")]
        routine = DevRoutine(
            name="test",
            description="test",
            parameters=params,
            operations=[
                _make_navigate_op(),
                _make_fetch_op(key="res", url="https://api.com/{{user_id}}"),
                _make_return_op(key="res")
            ]
        )
        valid, errors, exc = routine.validate()
        assert valid is True

    def test_parameter_validation_unused(self):
        """Routine with defined but unused parameter should fail."""
        params = [Parameter(name="unused_param", description="unused")]
        routine = DevRoutine(
            name="test",
            description="test",
            parameters=params,
            operations=[
                _make_navigate_op(),
                _make_fetch_op(key="res"),
                _make_return_op(key="res")
            ]
        )
        valid, errors, exc = routine.validate()
        assert valid is False
        assert any("Parameter 'unused_param' is not used" in e for e in errors)

    def test_parameter_validation_undefined_placeholder(self):
        """Routine using undefined placeholder should fail."""
        routine = DevRoutine(
            name="test",
            description="test",
            parameters=[],
            operations=[
                _make_navigate_op(),
                _make_fetch_op(key="res", url="https://api.com/{{undefined_param}}"),
                _make_return_op(key="res")
            ]
        )
        valid, errors, exc = routine.validate()
        assert valid is False
        assert any("Placeholder 'undefined_param' has invalid prefix" in e for e in errors)

    def test_session_storage_unused_key(self):
        """Fetch producing a key that is NEVER used should fail."""
        routine = DevRoutine(
            name="test",
            description="test",
            parameters=[],
            operations=[
                _make_navigate_op(),
                _make_fetch_op(key="unused_key"), # Produced but never used
                _make_fetch_op(key="result"),     # Last fetch (implicitly used by return)
                _make_return_op(key="result")
            ]
        )
        valid, errors, exc = routine.validate()
        assert valid is False
        assert any("Fetch session storage key 'unused_key' is not used" in e for e in errors)

    def test_session_storage_return_key_valid(self):
        """Fetch producing a key that is ONLY used by return should pass."""
        # This covers the bug we just fixed: list.remove(x) error when x was only in return op
        routine = DevRoutine(
            name="test",
            description="test",
            parameters=[],
            operations=[
                _make_navigate_op(),
                _make_fetch_op(key="final_result"),
                _make_return_op(key="final_result")
            ]
        )
        valid, errors, exc = routine.validate()
        assert valid is True

    def test_session_storage_chained_valid(self):
        """Fetch producing key used by next fetch should pass."""
        routine = DevRoutine(
            name="test",
            description="test",
            parameters=[],
            operations=[
                _make_navigate_op(),
                _make_fetch_op(key="step1"),
                _make_fetch_op(key="step2", headers='{"x": "{{sessionStorage:step1}}"}'),
                _make_fetch_op(key="final", headers='{"y": "{{sessionStorage:step2}}"}'),
                _make_return_op(key="final")
            ]
        )
        valid, errors, exc = routine.validate()
        assert valid is True

    def test_duplicate_parameter_usage_valid(self):
        """Using the same parameter multiple times should be valid."""
        # This tests that we don't crash on removing/checking used params
        params = [Parameter(name="uid", description="id")]
        routine = DevRoutine(
            name="test",
            description="test",
            parameters=params,
            operations=[
                _make_navigate_op(),
                _make_fetch_op(key="res", url="https://api.com/{{uid}}", headers='{"u": "{{uid}}"}'),
                _make_return_op(key="res")
            ]
        )
        valid, errors, exc = routine.validate()
        assert valid is True

    def test_mixed_usage_valid(self):
        """Mixed usage of params and session storage."""
        params = [Parameter(name="q", description="query")]
        routine = DevRoutine(
            name="test",
            description="test",
            parameters=params,
            operations=[
                _make_navigate_op(),
                _make_fetch_op(key="auth_data"),
                _make_fetch_op(key="results", 
                               url="https://api.com?q={{q}}", 
                               headers='{"Auth": "{{sessionStorage:auth_data}}"}'),
                _make_return_op(key="results")
            ]
        )
        valid, errors, exc = routine.validate()
        assert valid is True

