"""Tests for RoutineExecutionResult.to_agent_dict() sanitization."""

from bluebox.data_models.routine.execution import (
    OperationExecutionMetadata,
    RoutineExecutionResult,
)


class TestToAgentDict:
    """Tests for RoutineExecutionResult.to_agent_dict()."""

    def test_strips_internal_fields(self) -> None:
        """Sanitized dict should not contain placeholder_resolution or operations_metadata."""
        result = RoutineExecutionResult(
            ok=True,
            data={"price": 42},
            placeholder_resolution={"token": "secret123", "session": "abc"},
            operations_metadata=[
                OperationExecutionMetadata(type="fetch", duration_seconds=1.2),
            ],
        )
        sanitized = result.to_agent_dict()
        assert "placeholder_resolution" not in sanitized
        assert "operations_metadata" not in sanitized

    def test_preserves_business_fields(self) -> None:
        """Sanitized dict should preserve all agent-visible fields."""
        result = RoutineExecutionResult(
            ok=True,
            data={"items": [1, 2, 3]},
            error=None,
            warnings=["rate limited"],
            is_base64=False,
            content_type="application/json",
            filename="results.json",
        )
        sanitized = result.to_agent_dict()
        assert sanitized["ok"] is True
        assert sanitized["data"] == {"items": [1, 2, 3]}
        assert sanitized["error"] is None
        assert sanitized["warnings"] == ["rate limited"]
        assert sanitized["is_base64"] is False
        assert sanitized["content_type"] == "application/json"
        assert sanitized["filename"] == "results.json"

    def test_failed_placeholders_detected(self) -> None:
        """Should detect and count failed (None-valued) placeholders."""
        result = RoutineExecutionResult(
            placeholder_resolution={
                "token": "abc123",
                "session_id": None,
                "auth_header": None,
            },
        )
        sanitized = result.to_agent_dict()
        assert sanitized["has_failed_placeholders"] is True
        assert sanitized["failed_placeholder_count"] == 2

    def test_no_failed_placeholders(self) -> None:
        """Should report no failures when all placeholders resolved."""
        result = RoutineExecutionResult(
            placeholder_resolution={"token": "abc", "id": "123"},
        )
        sanitized = result.to_agent_dict()
        assert sanitized["has_failed_placeholders"] is False
        assert sanitized["failed_placeholder_count"] == 0

    def test_empty_placeholder_resolution(self) -> None:
        """Should handle empty placeholder_resolution."""
        result = RoutineExecutionResult()
        sanitized = result.to_agent_dict()
        assert sanitized["has_failed_placeholders"] is False
        assert sanitized["failed_placeholder_count"] == 0

    def test_operation_errors_detected(self) -> None:
        """Should detect and summarize operation errors."""
        result = RoutineExecutionResult(
            operations_metadata=[
                OperationExecutionMetadata(type="navigate", duration_seconds=0.5),
                OperationExecutionMetadata(type="fetch", duration_seconds=1.0, error="403 Forbidden"),
                OperationExecutionMetadata(type="js_evaluate", duration_seconds=0.1, error="timeout"),
            ],
        )
        sanitized = result.to_agent_dict()
        assert sanitized["has_operation_errors"] is True
        assert sanitized["operation_error_summary"] == [
            "fetch: 403 Forbidden",
            "js_evaluate: timeout",
        ]

    def test_no_operation_errors(self) -> None:
        """Should report no errors when all operations succeeded."""
        result = RoutineExecutionResult(
            operations_metadata=[
                OperationExecutionMetadata(type="navigate", duration_seconds=0.5),
                OperationExecutionMetadata(type="fetch", duration_seconds=1.0),
            ],
        )
        sanitized = result.to_agent_dict()
        assert sanitized["has_operation_errors"] is False
        assert sanitized["operation_error_summary"] == []

    def test_empty_operations_metadata(self) -> None:
        """Should handle empty operations_metadata."""
        result = RoutineExecutionResult()
        sanitized = result.to_agent_dict()
        assert sanitized["has_operation_errors"] is False
        assert sanitized["operation_error_summary"] == []

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
