"""
tests/unit/agents/specialists/test_abstract_specialist.py

Comprehensive unit tests for AbstractSpecialist base class and @agent_tool decorator.
"""

from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from bluebox.agents.abstract_agent import agent_tool, _ToolMeta
from bluebox.agents.specialists.abstract_specialist import (
    AbstractSpecialist,
    AutonomousConfig,
    RunMode,
)


# =============================================================================
# Test fixtures and helper classes
# =============================================================================


class DummyResult(BaseModel):
    """Dummy result model for testing autonomous mode."""
    value: str


class ConcreteSpecialist(AbstractSpecialist):
    """
    Concrete implementation of AbstractSpecialist for testing.

    Has a mix of tools with different availability conditions and signatures.
    """

    def __init__(self, **kwargs: Any) -> None:
        self._autonomous_result: DummyResult | None = None
        super().__init__(**kwargs)

    def _get_system_prompt(self) -> str:
        return "Test system prompt"

    def _get_autonomous_system_prompt(self) -> str:
        return "Test autonomous prompt"

    def _get_autonomous_initial_message(self, task: str) -> str:
        return f"Task: {task}"

    def _check_autonomous_completion(self, tool_name: str) -> bool:
        return tool_name == "finalize" and self._autonomous_result is not None

    def _get_autonomous_result(self) -> BaseModel | None:
        return self._autonomous_result

    def _reset_autonomous_state(self) -> None:
        self._autonomous_result = None

    # --- Tools with various signatures and availability ---

    @agent_tool
    def _always_available(self, message: str) -> dict[str, Any]:
        """
        A tool that is always available.

        Args:
            message: The message to echo.
        """
        return {"echoed": message}

    @agent_tool
    def _no_params(self) -> dict[str, Any]:
        """A tool with no parameters."""
        return {"status": "ok"}

    @agent_tool
    def _with_optional_params(
        self,
        required_arg: str,
        optional_arg: int = 10,
        nullable_arg: str | None = None,
    ) -> dict[str, Any]:
        """
        A tool with required, optional, and nullable parameters.

        Args:
            required_arg: This one is required.
            optional_arg: This has a default value.
            nullable_arg: This can be None.
        """
        return {
            "required": required_arg,
            "optional": optional_arg,
            "nullable": nullable_arg,
        }

    @agent_tool
    def _with_list_param(self, items: list[str], count: int) -> dict[str, Any]:
        """
        A tool that accepts a list parameter.

        Args:
            items: List of string items.
            count: Number of items to process.
        """
        return {"items": items[:count]}

    @agent_tool(availability=False)
    def _never_available(self) -> dict[str, Any]:
        """A tool that is never available."""
        return {"should": "never see this"}

    @agent_tool(availability=lambda self: self.can_finalize)
    def _finalize_gated(self) -> dict[str, Any]:
        """A tool gated by can_finalize property."""
        return {"finalized": True}

    @agent_tool(
        description="Explicit description overrides docstring.",
        parameters={
            "type": "object",
            "properties": {
                "custom_field": {"type": "string", "description": "A custom field."}
            },
            "required": ["custom_field"],
        },
    )
    def _with_explicit_schema(self, custom_field: str) -> dict[str, Any]:
        """This docstring description is ignored because explicit description is provided."""
        return {"custom": custom_field}

    @agent_tool(availability=lambda self: self.can_finalize)
    def _finalize(self, value: str) -> dict[str, Any]:
        """
        Finalize the autonomous run.

        Args:
            value: The final value.
        """
        self._autonomous_result = DummyResult(value=value)
        return {"status": "finalized", "value": value}


@pytest.fixture
def mock_emit() -> MagicMock:
    """Mock emit_message_callable."""
    return MagicMock()


@pytest.fixture
def specialist(mock_emit: MagicMock) -> ConcreteSpecialist:
    """Create a ConcreteSpecialist instance for testing."""
    return ConcreteSpecialist(emit_message_callable=mock_emit)


@pytest.fixture
def autonomous_specialist(mock_emit: MagicMock) -> ConcreteSpecialist:
    """Create a ConcreteSpecialist in autonomous mode past min_iterations."""
    spec = ConcreteSpecialist(
        emit_message_callable=mock_emit,
        run_mode=RunMode.AUTONOMOUS,
    )
    spec._autonomous_iteration = 5  # past default min_iterations of 3
    return spec


# =============================================================================
# Tests for @agent_tool decorator
# =============================================================================


class TestAgentToolDecorator:
    """Tests for the @agent_tool decorator."""

    def test_extracts_tool_name_from_method_name(self, specialist: ConcreteSpecialist) -> None:
        """Tool name is derived by stripping leading underscores from method name."""
        tools = {meta.name: meta for meta, _ in specialist._collect_tools()}
        assert "always_available" in tools
        assert "no_params" in tools
        assert "with_optional_params" in tools

    def test_extracts_description_from_docstring(self, specialist: ConcreteSpecialist) -> None:
        """Description is extracted from docstring when not explicitly provided."""
        tools = {meta.name: meta for meta, _ in specialist._collect_tools()}

        # Single sentence docstring
        assert tools["no_params"].description == "A tool with no parameters."

        # Multi-paragraph docstring (before Args)
        assert "always available" in tools["always_available"].description

    def test_explicit_description_overrides_docstring(self, specialist: ConcreteSpecialist) -> None:
        """Explicit description parameter takes precedence over docstring."""
        tools = {meta.name: meta for meta, _ in specialist._collect_tools()}
        assert tools["with_explicit_schema"].description == "Explicit description overrides docstring."

    def test_auto_generates_parameters_schema(self, specialist: ConcreteSpecialist) -> None:
        """Parameters schema is auto-generated from type hints."""
        tools = {meta.name: meta for meta, _ in specialist._collect_tools()}
        schema = tools["always_available"].parameters

        assert schema["type"] == "object"
        assert "message" in schema["properties"]
        assert schema["properties"]["message"]["type"] == "string"
        assert schema["required"] == ["message"]

    def test_auto_generates_schema_with_optional_params(self, specialist: ConcreteSpecialist) -> None:
        """Optional parameters are not in required list."""
        tools = {meta.name: meta for meta, _ in specialist._collect_tools()}
        schema = tools["with_optional_params"].parameters

        assert schema["required"] == ["required_arg"]
        assert "optional_arg" in schema["properties"]
        assert "nullable_arg" in schema["properties"]
        assert "optional_arg" not in schema["required"]
        assert "nullable_arg" not in schema["required"]

    def test_auto_generates_schema_for_list_types(self, specialist: ConcreteSpecialist) -> None:
        """List types are correctly represented in schema."""
        tools = {meta.name: meta for meta, _ in specialist._collect_tools()}
        schema = tools["with_list_param"].parameters

        items_schema = schema["properties"]["items"]
        assert items_schema["type"] == "array"
        assert items_schema["items"]["type"] == "string"

    def test_explicit_schema_overrides_auto_generation(self, specialist: ConcreteSpecialist) -> None:
        """Explicit parameters schema takes precedence over auto-generation."""
        tools = {meta.name: meta for meta, _ in specialist._collect_tools()}
        schema = tools["with_explicit_schema"].parameters

        assert "custom_field" in schema["properties"]
        assert schema["properties"]["custom_field"]["description"] == "A custom field."

    def test_no_params_generates_empty_schema(self, specialist: ConcreteSpecialist) -> None:
        """Tool with no parameters has empty properties and required list."""
        tools = {meta.name: meta for meta, _ in specialist._collect_tools()}
        schema = tools["no_params"].parameters

        assert schema["type"] == "object"
        assert schema["properties"] == {}
        assert schema["required"] == []

    def test_availability_defaults_to_true(self, specialist: ConcreteSpecialist) -> None:
        """Tools without explicit availability are always available."""
        tools = {meta.name: meta for meta, _ in specialist._collect_tools()}
        assert tools["always_available"].availability is True

    def test_availability_false_is_captured(self, specialist: ConcreteSpecialist) -> None:
        """Tools with availability=False have that captured in metadata."""
        tools = {meta.name: meta for meta, _ in specialist._collect_tools()}
        assert tools["never_available"].availability is False

    def test_availability_callable_is_captured(self, specialist: ConcreteSpecialist) -> None:
        """Tools with callable availability have the callable captured."""
        tools = {meta.name: meta for meta, _ in specialist._collect_tools()}
        assert callable(tools["finalize_gated"].availability)

    def test_raises_error_without_description_or_docstring(self) -> None:
        """Decorator raises ValueError if no description and no docstring."""
        with pytest.raises(ValueError, match="no description and no docstring"):
            @agent_tool
            def _no_docs(self) -> dict[str, Any]:
                pass  # no docstring!


class TestToolMetaDataclass:
    """Tests for the _ToolMeta dataclass."""

    def test_tool_meta_is_frozen(self) -> None:
        """_ToolMeta instances are immutable."""
        meta = _ToolMeta(
            name="test",
            description="Test tool",
            parameters={"type": "object", "properties": {}, "required": []},
            availability=True,
        )
        with pytest.raises(AttributeError):
            meta.name = "changed"  # type: ignore[misc]

    def test_tool_meta_stores_all_fields(self) -> None:
        """_ToolMeta correctly stores all provided fields."""
        params = {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]}
        avail_fn = lambda self: True

        meta = _ToolMeta(
            name="my_tool",
            description="My description",
            parameters=params,
            availability=avail_fn,
        )

        assert meta.name == "my_tool"
        assert meta.description == "My description"
        assert meta.parameters == params
        assert meta.availability is avail_fn


# =============================================================================
# Tests for _collect_tools
# =============================================================================


class TestCollectTools:
    """Tests for the _collect_tools class method."""

    def test_finds_all_decorated_methods(self, specialist: ConcreteSpecialist) -> None:
        """_collect_tools finds all methods decorated with @agent_tool."""
        tools = specialist._collect_tools()
        tool_names = {meta.name for meta, _ in tools}

        expected = {
            "always_available",
            "no_params",
            "with_optional_params",
            "with_list_param",
            "never_available",
            "finalize_gated",
            "with_explicit_schema",
            "finalize",
            # Generic finalize tools from AbstractSpecialist (both with and without output schema)
            "finalize_with_output",
            "finalize_with_failure",
            "finalize_result",
            "finalize_failure",
            # Documentation tools from AbstractAgent
            "search_docs",
            "get_doc_file",
            "search_docs_by_terms",
            "search_docs_by_regex",
        }
        assert tool_names == expected

    def test_returns_tuple_of_tuples(self, specialist: ConcreteSpecialist) -> None:
        """_collect_tools returns immutable tuple (for caching)."""
        tools = specialist._collect_tools()
        assert isinstance(tools, tuple)
        assert all(isinstance(item, tuple) for item in tools)

    def test_result_is_cached(self) -> None:
        """_collect_tools result is cached per subclass."""
        # Clear cache first
        ConcreteSpecialist._collect_tools.cache_clear()

        # First call
        tools1 = ConcreteSpecialist._collect_tools()
        # Second call should return cached result
        tools2 = ConcreteSpecialist._collect_tools()

        assert tools1 is tools2  # same object (cached)

    def test_different_subclasses_have_separate_caches(self) -> None:
        """Different subclasses have independent tool caches."""

        class AnotherSpecialist(ConcreteSpecialist):
            @agent_tool
            def _extra_tool(self) -> dict[str, Any]:
                """An extra tool only in this subclass."""
                return {}

        concrete_tools = ConcreteSpecialist._collect_tools()
        another_tools = AnotherSpecialist._collect_tools()

        concrete_names = {meta.name for meta, _ in concrete_tools}
        another_names = {meta.name for meta, _ in another_tools}

        assert "extra_tool" not in concrete_names
        assert "extra_tool" in another_names


# =============================================================================
# Tests for _sync_tools
# =============================================================================


class TestSyncTools:
    """Tests for the _sync_tools method."""

    def test_registers_available_tools_only(self, specialist: ConcreteSpecialist) -> None:
        """Only tools with availability=True are registered."""
        specialist._sync_tools()

        # Check registered tool names
        registered = specialist._registered_tool_names

        assert "always_available" in registered
        assert "no_params" in registered
        assert "never_available" not in registered  # availability=False

    def test_callable_availability_evaluated(self, specialist: ConcreteSpecialist) -> None:
        """Callable availability is evaluated during sync."""
        # In conversational mode, can_finalize is False
        specialist._sync_tools()
        assert "finalize_gated" not in specialist._registered_tool_names

        # Switch to autonomous mode and advance past min_iterations
        specialist.run_mode = RunMode.AUTONOMOUS
        specialist._autonomous_iteration = 5
        specialist._sync_tools()
        assert "finalize_gated" in specialist._registered_tool_names

    def test_sync_clears_previous_tools(self, specialist: ConcreteSpecialist) -> None:
        """_sync_tools clears previously registered tools before re-registering."""
        specialist._sync_tools()
        first_count = len(specialist._registered_tool_names)

        # Sync again - should have same tools
        specialist._sync_tools()
        second_count = len(specialist._registered_tool_names)

        assert first_count == second_count

    def test_availability_cycle_available_unavailable_available(
        self, specialist: ConcreteSpecialist
    ) -> None:
        """Tool availability tracks state changes: available → unavailable → available."""
        # Phase 1: Make finalize_gated available (autonomous mode, past min_iterations)
        specialist.run_mode = RunMode.AUTONOMOUS
        specialist._autonomous_iteration = 5
        specialist._sync_tools()

        assert "finalize_gated" in specialist._registered_tool_names
        result = specialist._execute_tool("finalize_gated", {})
        assert result == {"finalized": True}

        # Phase 2: Make unavailable (back to conversational mode)
        specialist.run_mode = RunMode.CONVERSATIONAL
        specialist._sync_tools()

        assert "finalize_gated" not in specialist._registered_tool_names
        result = specialist._execute_tool("finalize_gated", {})
        assert "error" in result
        assert "not currently available" in result["error"]

        # Phase 3: Make available again (back to autonomous, past min_iterations)
        specialist.run_mode = RunMode.AUTONOMOUS
        specialist._autonomous_iteration = 10
        specialist._sync_tools()

        assert "finalize_gated" in specialist._registered_tool_names
        result = specialist._execute_tool("finalize_gated", {})
        assert result == {"finalized": True}

    def test_availability_changes_mid_iteration(self, specialist: ConcreteSpecialist) -> None:
        """Tool becomes available mid-session as iteration count increases."""
        specialist.run_mode = RunMode.AUTONOMOUS
        specialist._autonomous_config = AutonomousConfig(min_iterations=3, max_iterations=10)

        # Iteration 1: not available
        specialist._autonomous_iteration = 1
        specialist._sync_tools()
        assert "finalize_gated" not in specialist._registered_tool_names

        # Iteration 2: still not available
        specialist._autonomous_iteration = 2
        specialist._sync_tools()
        assert "finalize_gated" not in specialist._registered_tool_names

        # Iteration 3: now available (min_iterations reached)
        specialist._autonomous_iteration = 3
        specialist._sync_tools()
        assert "finalize_gated" in specialist._registered_tool_names

        # Iteration 4+: still available
        specialist._autonomous_iteration = 4
        specialist._sync_tools()
        assert "finalize_gated" in specialist._registered_tool_names


# =============================================================================
# Tests for _execute_tool
# =============================================================================


class TestExecuteTool:
    """Tests for the _execute_tool method."""

    # --- Success cases ---

    def test_executes_tool_with_required_param(self, specialist: ConcreteSpecialist) -> None:
        """Tool executes successfully with required parameters."""
        result = specialist._execute_tool("always_available", {"message": "hello"})
        assert result == {"echoed": "hello"}

    def test_executes_tool_with_no_params(self, specialist: ConcreteSpecialist) -> None:
        """Tool with no parameters executes successfully."""
        result = specialist._execute_tool("no_params", {})
        assert result == {"status": "ok"}

    def test_executes_tool_with_optional_params_omitted(self, specialist: ConcreteSpecialist) -> None:
        """Optional parameters use defaults when omitted."""
        result = specialist._execute_tool("with_optional_params", {"required_arg": "test"})
        assert result["required"] == "test"
        assert result["optional"] == 10  # default value
        assert result["nullable"] is None  # default value

    def test_executes_tool_with_optional_params_provided(self, specialist: ConcreteSpecialist) -> None:
        """Optional parameters can be overridden."""
        result = specialist._execute_tool(
            "with_optional_params",
            {"required_arg": "test", "optional_arg": 42, "nullable_arg": "provided"},
        )
        assert result["required"] == "test"
        assert result["optional"] == 42
        assert result["nullable"] == "provided"

    def test_executes_tool_with_list_param(self, specialist: ConcreteSpecialist) -> None:
        """Tool with list parameter executes correctly."""
        result = specialist._execute_tool(
            "with_list_param",
            {"items": ["a", "b", "c", "d"], "count": 2},
        )
        assert result == {"items": ["a", "b"]}

    # --- Error cases: unknown tool ---

    def test_unknown_tool_returns_error(self, specialist: ConcreteSpecialist) -> None:
        """Unknown tool name returns error dict."""
        result = specialist._execute_tool("nonexistent_tool", {})
        assert "error" in result
        assert "Unknown tool" in result["error"]
        assert "nonexistent_tool" in result["error"]

    # --- Error cases: availability ---

    def test_unavailable_tool_returns_error(self, specialist: ConcreteSpecialist) -> None:
        """Tool with availability=False returns error when called."""
        result = specialist._execute_tool("never_available", {})
        assert "error" in result
        assert "not currently available" in result["error"]

    def test_conditionally_unavailable_tool_returns_error(self, specialist: ConcreteSpecialist) -> None:
        """Tool with callable availability returns error when condition is False."""
        # In conversational mode, can_finalize is False
        result = specialist._execute_tool("finalize_gated", {})
        assert "error" in result
        assert "not currently available" in result["error"]

    def test_conditionally_available_tool_succeeds_when_available(
        self, autonomous_specialist: ConcreteSpecialist
    ) -> None:
        """Tool with callable availability succeeds when condition is True."""
        # autonomous_specialist has can_finalize=True
        result = autonomous_specialist._execute_tool("finalize_gated", {})
        assert result == {"finalized": True}

    # --- Error cases: missing required parameters ---

    def test_missing_required_param_returns_error(self, specialist: ConcreteSpecialist) -> None:
        """Missing required parameter returns descriptive error."""
        result = specialist._execute_tool("always_available", {})
        assert "error" in result
        assert "Missing required parameter(s)" in result["error"]
        assert "message" in result["error"]

    def test_missing_multiple_required_params_lists_all(self, specialist: ConcreteSpecialist) -> None:
        """All missing required parameters are listed in error."""
        result = specialist._execute_tool("with_list_param", {})
        assert "error" in result
        assert "items" in result["error"]
        assert "count" in result["error"]

    def test_none_value_for_required_param_returns_error(self, specialist: ConcreteSpecialist) -> None:
        """Explicitly passing None for required param is treated as missing."""
        result = specialist._execute_tool("always_available", {"message": None})
        assert "error" in result
        assert "Missing required parameter(s)" in result["error"]

    # --- Error cases: type validation ---

    def test_wrong_type_string_instead_of_int_returns_error(self, specialist: ConcreteSpecialist) -> None:
        """Passing string where int expected returns type error."""
        result = specialist._execute_tool(
            "with_optional_params",
            {"required_arg": "test", "optional_arg": "not_an_int"},
        )
        assert "error" in result
        assert "Invalid argument type" in result["error"]
        assert "optional_arg" in result["error"]

    def test_wrong_type_int_instead_of_string_returns_error(self, specialist: ConcreteSpecialist) -> None:
        """Passing int where string expected returns type error."""
        result = specialist._execute_tool("always_available", {"message": 12345})
        assert "error" in result
        assert "Invalid argument type" in result["error"]
        assert "message" in result["error"]

    def test_wrong_type_string_instead_of_list_returns_error(self, specialist: ConcreteSpecialist) -> None:
        """Passing string where list expected returns type error."""
        result = specialist._execute_tool(
            "with_list_param",
            {"items": "not_a_list", "count": 5},
        )
        assert "error" in result
        assert "Invalid argument type" in result["error"]

    def test_wrong_list_element_type_returns_error(self, specialist: ConcreteSpecialist) -> None:
        """List with wrong element types returns error."""
        result = specialist._execute_tool(
            "with_list_param",
            {"items": [1, 2, 3], "count": 2},  # ints instead of strings
        )
        assert "error" in result
        assert "Invalid argument type" in result["error"]

    def test_none_for_optional_param_passes_none(self, specialist: ConcreteSpecialist) -> None:
        """Explicit None for optional param passes None (not the default)."""
        # When you explicitly pass None, it's passed to the function.
        # The default (10) is only used when the argument is omitted entirely.
        # Type validation skips None values, so this passes through.
        result = specialist._execute_tool(
            "with_optional_params",
            {"required_arg": "test", "optional_arg": None},
        )
        assert result["optional"] is None  # explicit None passed through

    def test_correct_types_pass_validation(self, specialist: ConcreteSpecialist) -> None:
        """Correct types pass validation and tool executes."""
        result = specialist._execute_tool(
            "with_list_param",
            {"items": ["a", "b"], "count": 1},
        )
        assert "error" not in result
        assert result == {"items": ["a"]}


# =============================================================================
# Tests for can_finalize property and autonomous mode
# =============================================================================


class TestCanFinalizeProperty:
    """Tests for the can_finalize property."""

    def test_false_in_conversational_mode(self, specialist: ConcreteSpecialist) -> None:
        """can_finalize is False in conversational mode regardless of iteration."""
        specialist.run_mode = RunMode.CONVERSATIONAL
        specialist._autonomous_iteration = 100  # even with high iteration
        assert specialist.can_finalize is False

    def test_false_before_min_iterations(self, mock_emit: MagicMock) -> None:
        """can_finalize is False in autonomous mode before min_iterations."""
        specialist = ConcreteSpecialist(
            emit_message_callable=mock_emit,
            run_mode=RunMode.AUTONOMOUS,
        )
        specialist._autonomous_config = AutonomousConfig(min_iterations=5, max_iterations=10)
        specialist._autonomous_iteration = 3

        assert specialist.can_finalize is False

    def test_true_at_min_iterations(self, mock_emit: MagicMock) -> None:
        """can_finalize is True at exactly min_iterations."""
        specialist = ConcreteSpecialist(
            emit_message_callable=mock_emit,
            run_mode=RunMode.AUTONOMOUS,
        )
        specialist._autonomous_config = AutonomousConfig(min_iterations=5, max_iterations=10)
        specialist._autonomous_iteration = 5

        assert specialist.can_finalize is True

    def test_true_after_min_iterations(self, autonomous_specialist: ConcreteSpecialist) -> None:
        """can_finalize is True after min_iterations."""
        assert autonomous_specialist.can_finalize is True


class TestAutonomousConfig:
    """Tests for AutonomousConfig."""

    def test_default_values(self) -> None:
        """AutonomousConfig has sensible defaults."""
        config = AutonomousConfig()
        assert config.min_iterations == 3
        assert config.max_iterations == 10

    def test_custom_values(self) -> None:
        """AutonomousConfig accepts custom values."""
        config = AutonomousConfig(min_iterations=5, max_iterations=20)
        assert config.min_iterations == 5
        assert config.max_iterations == 20


# =============================================================================
# Tests for RunMode enum
# =============================================================================


class TestRunMode:
    """Tests for RunMode enum."""

    def test_conversational_value(self) -> None:
        """RunMode.CONVERSATIONAL has correct string value."""
        assert RunMode.CONVERSATIONAL == "conversational"

    def test_autonomous_value(self) -> None:
        """RunMode.AUTONOMOUS has correct string value."""
        assert RunMode.AUTONOMOUS == "autonomous"


# =============================================================================
# Tests for specialist lifecycle
# =============================================================================


class TestSpecialistLifecycle:
    """Tests for specialist initialization and reset."""

    def test_initial_state(self, specialist: ConcreteSpecialist) -> None:
        """Specialist initializes with correct default state."""
        assert specialist.run_mode == RunMode.CONVERSATIONAL
        assert specialist._autonomous_iteration == 0
        assert specialist.can_finalize is False

    def test_reset_clears_state(self, autonomous_specialist: ConcreteSpecialist) -> None:
        """reset() clears all state back to initial."""
        # Modify some state
        autonomous_specialist._autonomous_result = DummyResult(value="test")

        autonomous_specialist.reset()

        assert autonomous_specialist.run_mode == RunMode.CONVERSATIONAL
        assert autonomous_specialist._autonomous_iteration == 0
        assert autonomous_specialist._autonomous_result is None
        assert autonomous_specialist.can_finalize is False

    def test_reset_syncs_tools_for_conversational(self, autonomous_specialist: ConcreteSpecialist) -> None:
        """reset() syncs tools for conversational mode (finalize tools not available)."""
        # Before reset, finalize tools are available
        assert autonomous_specialist.can_finalize is True
        autonomous_specialist._sync_tools()
        assert "finalize_gated" in autonomous_specialist._registered_tool_names

        # After reset, finalize tools are not available
        autonomous_specialist.reset()
        assert "finalize_gated" not in autonomous_specialist._registered_tool_names


# =============================================================================
# Tests for edge cases and error handling
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and unusual inputs."""

    def test_empty_arguments_dict(self, specialist: ConcreteSpecialist) -> None:
        """Empty arguments dict works for no-param tools."""
        result = specialist._execute_tool("no_params", {})
        assert result == {"status": "ok"}

    def test_extra_arguments_return_error(self, specialist: ConcreteSpecialist) -> None:
        """Extra arguments not in tool schema return graceful error."""
        result = specialist._execute_tool(
            "no_params",
            {"extra_arg": "ignored"},
        )
        assert "error" in result
        assert "Unknown parameter(s)" in result["error"]
        assert "extra_arg" in result["error"]

    def test_unicode_in_arguments(self, specialist: ConcreteSpecialist) -> None:
        """Unicode strings in arguments work correctly."""
        result = specialist._execute_tool("always_available", {"message": "héllo 世界 🎉"})
        assert result == {"echoed": "héllo 世界 🎉"}

    def test_empty_string_argument(self, specialist: ConcreteSpecialist) -> None:
        """Empty string is valid for string parameters."""
        result = specialist._execute_tool("always_available", {"message": ""})
        assert result == {"echoed": ""}

    def test_empty_list_argument(self, specialist: ConcreteSpecialist) -> None:
        """Empty list is valid for list parameters."""
        result = specialist._execute_tool("with_list_param", {"items": [], "count": 5})
        assert result == {"items": []}

    def test_large_list_argument(self, specialist: ConcreteSpecialist) -> None:
        """Large lists are handled correctly."""
        large_list = [f"item_{i}" for i in range(1000)]
        result = specialist._execute_tool("with_list_param", {"items": large_list, "count": 3})
        assert result == {"items": ["item_0", "item_1", "item_2"]}

    def test_zero_for_int_parameter(self, specialist: ConcreteSpecialist) -> None:
        """Zero is valid for integer parameters."""
        result = specialist._execute_tool(
            "with_optional_params",
            {"required_arg": "test", "optional_arg": 0},
        )
        assert result["optional"] == 0

    def test_negative_int_parameter(self, specialist: ConcreteSpecialist) -> None:
        """Negative integers are valid for int parameters."""
        result = specialist._execute_tool(
            "with_optional_params",
            {"required_arg": "test", "optional_arg": -5},
        )
        assert result["optional"] == -5
