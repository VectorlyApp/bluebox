"""
tests/unit/data_models/routine/test_routine.py

Tests for the Routine model: fields, serialization, and public methods.

Covers:
- Routine construction and field defaults
- model_schema_markdown() static method
- compute_base_urls_from_operations()
- get_structure_warnings()

Note: Parameter/placeholder validation rules are tested in test_routine_validation.py.
"""

import json

import pytest

from bluebox.data_models.routine.endpoint import Endpoint, HTTPMethod
from bluebox.data_models.routine.operation import (
    RoutineClickOperation,
    RoutineDownloadOperation,
    RoutineFetchOperation,
    RoutineJsEvaluateOperation,
    RoutineNavigateOperation,
    RoutineReturnHTMLOperation,
    RoutineReturnOperation,
    RoutineSleepOperation,
)
from bluebox.data_models.routine.parameter import Parameter, ParameterType
from bluebox.data_models.routine.routine import Routine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_routine(**overrides) -> Routine:
    """Build a minimal valid Routine (navigate → fetch → return)."""
    defaults = dict(
        name="test",
        description="test routine",
        operations=[
            RoutineNavigateOperation(url="https://example.com"),
            RoutineFetchOperation(
                endpoint=Endpoint(
                    url="https://api.example.com/data",
                    method=HTTPMethod.GET,
                    headers={},
                    body={},
                ),
                session_storage_key="result",
            ),
            RoutineReturnOperation(session_storage_key="result"),
        ],
    )
    defaults.update(overrides)
    return Routine(**defaults)


# ---------------------------------------------------------------------------
# Construction and field defaults
# ---------------------------------------------------------------------------

class TestRoutineConstruction:
    """Test Routine model construction and default values."""

    def test_minimal_routine_creates_successfully(self) -> None:
        routine = _minimal_routine()
        assert routine.name == "test"
        assert routine.description == "test routine"
        assert len(routine.operations) == 3

    def test_parameters_defaults_empty(self) -> None:
        routine = _minimal_routine()
        assert routine.parameters == []

    def test_parameters_can_be_provided(self) -> None:
        routine = _minimal_routine(
            operations=[
                RoutineNavigateOperation(url="https://example.com/\"{{q}}\""),
                RoutineFetchOperation(
                    endpoint=Endpoint(
                        url="https://api.example.com/data",
                        method=HTTPMethod.GET, headers={}, body={},
                    ),
                    session_storage_key="result",
                ),
                RoutineReturnOperation(session_storage_key="result"),
            ],
            parameters=[
                Parameter(name="q", type=ParameterType.STRING, description="query"),
            ],
        )
        assert len(routine.parameters) == 1
        assert routine.parameters[0].name == "q"


# ---------------------------------------------------------------------------
# Serialization round-trip
# ---------------------------------------------------------------------------

class TestRoutineSerialization:
    """Test JSON serialization/deserialization."""

    def test_model_dump_json_round_trip(self) -> None:
        routine = _minimal_routine()
        json_str = routine.model_dump_json()
        parsed = json.loads(json_str)
        restored = Routine.model_validate(parsed)
        assert restored.name == routine.name
        assert len(restored.operations) == len(routine.operations)

    def test_model_dump_contains_expected_keys(self) -> None:
        routine = _minimal_routine()
        dumped = routine.model_dump()
        assert set(dumped.keys()) >= {"name", "description", "operations", "parameters"}


# ---------------------------------------------------------------------------
# model_schema_markdown()
# ---------------------------------------------------------------------------

class TestModelSchemaMarkdown:
    """Tests for Routine.model_schema_markdown() static method."""

    @pytest.fixture()
    def schema(self) -> str:
        return Routine.model_schema_markdown()

    def test_returns_string(self, schema: str) -> None:
        assert isinstance(schema, str)

    def test_has_header(self, schema: str) -> None:
        assert schema.startswith("## Routine Schema Reference")

    def test_has_routine_section(self, schema: str) -> None:
        assert "### Routine (top level)" in schema

    def test_has_parameter_section(self, schema: str) -> None:
        assert "### Parameter" in schema

    def test_has_endpoint_section(self, schema: str) -> None:
        assert "### Endpoint (used by fetch and download)" in schema

    def test_has_all_operation_types(self, schema: str) -> None:
        expected_ops = [
            "navigate", "sleep", "fetch", "return", "click",
            "input_text", "press", "get_cookies", "wait_for_url",
            "scroll", "return_html", "download", "js_evaluate",
        ]
        for op in expected_ops:
            assert f"### Operation: {op}" in schema, f"Missing operation: {op}"

    def test_operation_count(self, schema: str) -> None:
        """Should have exactly 13 operation sections."""
        count = schema.count("### Operation:")
        assert count == 13

    def test_routine_required_fields(self, schema: str) -> None:
        assert "- name: str (required)" in schema
        assert "- description: str (required)" in schema

    def test_navigate_fields(self, schema: str) -> None:
        assert "- url: str (required)" in schema
        assert "- sleep_after_navigation_seconds: float = 3.0" in schema

    def test_fetch_has_endpoint(self, schema: str) -> None:
        assert "- endpoint: Endpoint (required)" in schema
        assert "- session_storage_key: str? = null" in schema

    def test_endpoint_has_method_enum(self, schema: str) -> None:
        assert '"GET"' in schema
        assert '"POST"' in schema

    def test_endpoint_has_credentials(self, schema: str) -> None:
        assert '"same-origin" | "include" | "omit"' in schema

    def test_click_fields(self, schema: str) -> None:
        assert "- selector: str (required)" in schema
        assert '- button: "left" | "right" | "middle" = "left"' in schema
        assert "- ensure_visible: bool = true" in schema

    def test_js_evaluate_fields(self, schema: str) -> None:
        assert "- js: str (required)" in schema
        assert "- timeout_seconds: float = 5.0" in schema

    def test_no_type_discriminator_in_operations(self, schema: str) -> None:
        """The 'type' discriminator field should be skipped from operation sections."""
        in_operation = False
        for line in schema.splitlines():
            if line.startswith("### Operation:"):
                in_operation = True
            elif line.startswith("### "):
                in_operation = False
            if in_operation and line.startswith("- type:"):
                pytest.fail(f"'type' discriminator should be skipped: {line}")

    def test_parameter_observed_value_skipped(self, schema: str) -> None:
        """Internal field 'observed_value' should not appear."""
        assert "observed_value" not in schema

    def test_idempotent(self, schema: str) -> None:
        """Calling twice should return identical output."""
        assert Routine.model_schema_markdown() == schema

    def test_compact_size(self, schema: str) -> None:
        """Schema reference should be much smaller than model_json_schema()."""
        json_schema = json.dumps(Routine.model_json_schema(), indent=2)
        assert len(schema) < len(json_schema) / 3  # at least 3x smaller


# ---------------------------------------------------------------------------
# compute_base_urls_from_operations()
# ---------------------------------------------------------------------------

class TestComputeBaseUrls:
    """Tests for Routine.compute_base_urls_from_operations()."""

    def test_single_navigate(self) -> None:
        routine = _minimal_routine()
        result = routine.compute_base_urls_from_operations()
        assert result == "example.com"

    def test_navigate_and_fetch_same_domain(self) -> None:
        routine = _minimal_routine()
        # default ops have example.com navigate + api.example.com fetch
        result = routine.compute_base_urls_from_operations()
        assert result == "example.com"

    def test_multiple_domains_sorted(self) -> None:
        routine = _minimal_routine(
            operations=[
                RoutineNavigateOperation(url="https://www.zebra.com"),
                RoutineFetchOperation(
                    endpoint=Endpoint(
                        url="https://api.alpha.com/data",
                        method=HTTPMethod.GET, headers={}, body={},
                    ),
                    session_storage_key="result",
                ),
                RoutineReturnOperation(session_storage_key="result"),
            ],
        )
        result = routine.compute_base_urls_from_operations()
        assert result == "alpha.com,zebra.com"

    def test_download_url_included(self) -> None:
        routine = _minimal_routine(
            operations=[
                RoutineNavigateOperation(url="https://www.example.com"),
                RoutineDownloadOperation(
                    endpoint=Endpoint(
                        url="https://cdn.downloads.net/file.pdf",
                        method=HTTPMethod.GET, headers={}, body={},
                    ),
                    filename="file.pdf",
                ),
                RoutineFetchOperation(
                    endpoint=Endpoint(
                        url="https://api.example.com/data",
                        method=HTTPMethod.GET, headers={}, body={},
                    ),
                    session_storage_key="result",
                ),
                RoutineReturnOperation(session_storage_key="result"),
            ],
        )
        result = routine.compute_base_urls_from_operations()
        assert "downloads.net" in result
        assert "example.com" in result

    def test_no_url_operations_returns_none(self) -> None:
        routine = Routine(
            name="test",
            description="test",
            operations=[
                RoutineJsEvaluateOperation(
                    js='(function() { return "ok"; })()',
                    session_storage_key="result",
                ),
                RoutineReturnOperation(session_storage_key="result"),
            ],
        )
        assert routine.compute_base_urls_from_operations() is None

    def test_deduplicates_same_domain(self) -> None:
        routine = _minimal_routine(
            operations=[
                RoutineNavigateOperation(url="https://www.example.com"),
                RoutineFetchOperation(
                    endpoint=Endpoint(
                        url="https://api.example.com/one",
                        method=HTTPMethod.GET, headers={}, body={},
                    ),
                    session_storage_key="r1",
                ),
                RoutineFetchOperation(
                    endpoint=Endpoint(
                        url="https://api.example.com/two",
                        method=HTTPMethod.GET, headers={}, body={},
                    ),
                    session_storage_key="result",
                ),
                RoutineReturnOperation(session_storage_key="result"),
            ],
        )
        assert routine.compute_base_urls_from_operations() == "example.com"


# ---------------------------------------------------------------------------
# get_structure_warnings()
# ---------------------------------------------------------------------------

class TestGetStructureWarnings:
    """Tests for Routine.get_structure_warnings()."""

    def test_clean_routine_no_warnings(self) -> None:
        """Standard navigate → fetch → return should produce no warnings."""
        routine = _minimal_routine()
        assert routine.get_structure_warnings() == []

    def test_warns_first_op_not_navigate(self) -> None:
        routine = Routine(
            name="test",
            description="test",
            operations=[
                RoutineSleepOperation(timeout_seconds=1.0),
                RoutineFetchOperation(
                    endpoint=Endpoint(
                        url="https://api.example.com/data",
                        method=HTTPMethod.GET, headers={}, body={},
                    ),
                    session_storage_key="result",
                ),
                RoutineReturnOperation(session_storage_key="result"),
            ],
        )
        warnings = routine.get_structure_warnings()
        assert any("First operation" in w for w in warnings)

    def test_warns_second_to_last_not_fetch(self) -> None:
        routine = _minimal_routine(
            operations=[
                RoutineNavigateOperation(url="https://example.com"),
                RoutineFetchOperation(
                    endpoint=Endpoint(
                        url="https://api.example.com/data",
                        method=HTTPMethod.GET, headers={}, body={},
                    ),
                    session_storage_key="result",
                ),
                RoutineSleepOperation(timeout_seconds=1.0),
                RoutineReturnOperation(session_storage_key="result"),
            ],
        )
        warnings = routine.get_structure_warnings()
        assert any("Second-to-last" in w for w in warnings)

    def test_warns_no_navigate(self) -> None:
        routine = Routine(
            name="test",
            description="test",
            operations=[
                RoutineFetchOperation(
                    endpoint=Endpoint(
                        url="https://api.example.com/data",
                        method=HTTPMethod.GET, headers={}, body={},
                    ),
                    session_storage_key="result",
                ),
                RoutineReturnOperation(session_storage_key="result"),
            ],
        )
        warnings = routine.get_structure_warnings()
        assert any("no 'navigate'" in w for w in warnings)

    def test_warns_fetch_missing_session_storage_key(self) -> None:
        routine = _minimal_routine(
            operations=[
                RoutineNavigateOperation(url="https://example.com"),
                RoutineFetchOperation(
                    endpoint=Endpoint(
                        url="https://api.example.com/first",
                        method=HTTPMethod.GET, headers={}, body={},
                    ),
                    # no session_storage_key
                ),
                RoutineFetchOperation(
                    endpoint=Endpoint(
                        url="https://api.example.com/second",
                        method=HTTPMethod.GET, headers={}, body={},
                    ),
                    session_storage_key="result",
                ),
                RoutineReturnOperation(session_storage_key="result"),
            ],
        )
        warnings = routine.get_structure_warnings()
        assert any("no session_storage_key" in w for w in warnings)

    def test_warns_duplicate_session_storage_key(self) -> None:
        routine = _minimal_routine(
            operations=[
                RoutineNavigateOperation(url="https://example.com"),
                RoutineFetchOperation(
                    endpoint=Endpoint(
                        url="https://api.example.com/one",
                        method=HTTPMethod.GET, headers={}, body={},
                    ),
                    session_storage_key="result",
                ),
                RoutineFetchOperation(
                    endpoint=Endpoint(
                        url="https://api.example.com/two",
                        method=HTTPMethod.GET, headers={}, body={},
                    ),
                    session_storage_key="result",
                ),
                RoutineReturnOperation(session_storage_key="result"),
            ],
        )
        warnings = routine.get_structure_warnings()
        assert any("overwrite" in w for w in warnings)

    def test_warns_long_sleep(self) -> None:
        routine = _minimal_routine(
            operations=[
                RoutineNavigateOperation(url="https://example.com"),
                RoutineSleepOperation(timeout_seconds=60.0),
                RoutineFetchOperation(
                    endpoint=Endpoint(
                        url="https://api.example.com/data",
                        method=HTTPMethod.GET, headers={}, body={},
                    ),
                    session_storage_key="result",
                ),
                RoutineReturnOperation(session_storage_key="result"),
            ],
        )
        warnings = routine.get_structure_warnings()
        assert any("60.0s" in w for w in warnings)

    def test_warns_post_with_empty_body(self) -> None:
        routine = _minimal_routine(
            operations=[
                RoutineNavigateOperation(url="https://example.com"),
                RoutineFetchOperation(
                    endpoint=Endpoint(
                        url="https://api.example.com/data",
                        method=HTTPMethod.POST,
                        headers={},
                        body={},  # empty body for POST
                    ),
                    session_storage_key="result",
                ),
                RoutineReturnOperation(session_storage_key="result"),
            ],
        )
        warnings = routine.get_structure_warnings()
        assert any("empty body" in w for w in warnings)

    def test_no_warn_post_with_body(self) -> None:
        routine = _minimal_routine(
            operations=[
                RoutineNavigateOperation(url="https://example.com"),
                RoutineFetchOperation(
                    endpoint=Endpoint(
                        url="https://api.example.com/data",
                        method=HTTPMethod.POST,
                        headers={},
                        body={"key": "value"},
                    ),
                    session_storage_key="result",
                ),
                RoutineReturnOperation(session_storage_key="result"),
            ],
        )
        warnings = routine.get_structure_warnings()
        assert not any("empty body" in w for w in warnings)

    def test_no_warn_get_with_empty_body(self) -> None:
        """GET requests with empty body should NOT trigger the warning."""
        routine = _minimal_routine()
        warnings = routine.get_structure_warnings()
        assert not any("empty body" in w for w in warnings)

    def test_js_evaluate_as_second_to_last_no_warning(self) -> None:
        """js_evaluate before return should not trigger second-to-last warning."""
        routine = _minimal_routine(
            operations=[
                RoutineNavigateOperation(url="https://example.com"),
                RoutineJsEvaluateOperation(
                    js='(function() { return document.title; })()',
                    session_storage_key="result",
                ),
                RoutineReturnOperation(session_storage_key="result"),
            ],
        )
        warnings = routine.get_structure_warnings()
        assert not any("Second-to-last" in w for w in warnings)
