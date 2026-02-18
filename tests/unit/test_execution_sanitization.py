"""Tests for RoutineExecutionResult.to_agent_dict() sanitization."""

from bluebox.data_models.routine.execution import (
    OperationExecutionMetadata,
    RoutineExecutionResult,
)


class TestToAgentDict:
    """Tests for RoutineExecutionResult.to_agent_dict()."""

    def test_strips_internal_fields(self) -> None:
        """Sanitized dict should not contain internal fields."""
        result = RoutineExecutionResult(
            ok=True,
            data={"price": 42},
            placeholder_resolution={"token": "secret123", "session": "abc"},
            operations_metadata=[
                OperationExecutionMetadata(type="fetch", duration_seconds=1.2),
            ],
            warnings=["rate limited"],
        )
        sanitized = result.to_agent_dict()
        assert "placeholder_resolution" not in sanitized
        assert "operations_metadata" not in sanitized
        assert "warnings" not in sanitized
        assert "has_failed_placeholders" not in sanitized
        assert "failed_placeholder_count" not in sanitized
        assert "has_operation_errors" not in sanitized
        assert "operation_error_summary" not in sanitized

    def test_preserves_business_fields(self) -> None:
        """Sanitized dict should preserve all agent-visible fields."""
        result = RoutineExecutionResult(
            ok=True,
            data={"items": [1, 2, 3]},
            error=None,
            is_base64=False,
            content_type="application/json",
            filename="results.json",
        )
        sanitized = result.to_agent_dict()
        assert sanitized["ok"] is True
        assert sanitized["data"] == {"items": [1, 2, 3]}
        assert sanitized["error"] is None
        assert sanitized["is_base64"] is False
        assert sanitized["content_type"] == "application/json"
        assert sanitized["filename"] == "results.json"

    def test_only_expected_keys(self) -> None:
        """Sanitized dict should contain exactly the expected keys."""
        result = RoutineExecutionResult(
            ok=True,
            data="some data",
            placeholder_resolution={"token": None},
            operations_metadata=[
                OperationExecutionMetadata(type="fetch", duration_seconds=1.0, error="403"),
            ],
        )
        sanitized = result.to_agent_dict()
        assert set(sanitized.keys()) == {"ok", "data", "error", "is_base64", "content_type", "filename"}

    def test_failed_execution(self) -> None:
        """Should correctly sanitize a failed execution."""
        result = RoutineExecutionResult(
            ok=False,
            error="Connection refused",
            data=None,
        )
        sanitized = result.to_agent_dict()
        assert sanitized["ok"] is False
        assert sanitized["error"] == "Connection refused"
        assert sanitized["data"] is None

    def test_base64_binary_result(self) -> None:
        """Should preserve base64 and content_type for binary results."""
        result = RoutineExecutionResult(
            ok=True,
            data="SGVsbG8gV29ybGQ=",
            is_base64=True,
            content_type="application/pdf",
            filename="report.pdf",
        )
        sanitized = result.to_agent_dict()
        assert sanitized["is_base64"] is True
        assert sanitized["content_type"] == "application/pdf"
        assert sanitized["filename"] == "report.pdf"
