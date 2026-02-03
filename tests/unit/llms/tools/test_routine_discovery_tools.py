"""
tests/unit/llms/tools/test_routine_discovery_tools.py

Unit tests for routine_discovery_tools.py tool definitions and helper functions.
"""

import pytest

from bluebox.llms.tools.routine_discovery_tools import (
    TOOL_DEFINITIONS,
    get_tool_by_name,
    get_all_tool_names,
)


class TestToolDefinitionsStructure:
    """Tests for TOOL_DEFINITIONS structure and validity."""

    def test_tool_definitions_is_list(self) -> None:
        """TOOL_DEFINITIONS should be a list."""
        assert isinstance(TOOL_DEFINITIONS, list)

    def test_tool_definitions_not_empty(self) -> None:
        """TOOL_DEFINITIONS should contain tools."""
        assert len(TOOL_DEFINITIONS) > 0

    def test_all_tools_have_required_keys(self) -> None:
        """Each tool should have name, description, and parameters."""
        required_keys = {"name", "description", "parameters"}
        for tool in TOOL_DEFINITIONS:
            assert required_keys.issubset(tool.keys()), f"Tool missing keys: {tool.get('name', 'unknown')}"

    def test_all_tool_names_are_strings(self) -> None:
        """All tool names should be non-empty strings."""
        for tool in TOOL_DEFINITIONS:
            assert isinstance(tool["name"], str)
            assert len(tool["name"]) > 0

    def test_all_tool_descriptions_are_strings(self) -> None:
        """All tool descriptions should be non-empty strings."""
        for tool in TOOL_DEFINITIONS:
            assert isinstance(tool["description"], str)
            assert len(tool["description"]) > 0

    def test_all_parameters_are_valid_json_schema(self) -> None:
        """All tool parameters should be valid JSON schema objects."""
        for tool in TOOL_DEFINITIONS:
            params = tool["parameters"]
            assert isinstance(params, dict)
            assert params.get("type") == "object"
            assert "properties" in params
            assert "required" in params
            assert isinstance(params["properties"], dict)
            assert isinstance(params["required"], list)

    def test_unique_tool_names(self) -> None:
        """All tool names should be unique."""
        names = [tool["name"] for tool in TOOL_DEFINITIONS]
        assert len(names) == len(set(names)), "Duplicate tool names found"


class TestExpectedTools:
    """Tests that verify expected tools exist with correct structure."""

    EXPECTED_TOOLS = [
        "list_transactions",
        "get_transaction",
        "scan_for_value",
        "add_transaction_to_queue",
        "get_queue_status",
        "mark_transaction_complete",
        "record_identified_transaction",
        "record_extracted_variables",
        "record_resolved_variable",
        "execute_routine",
        "construct_routine",
    ]

    def test_all_expected_tools_exist(self) -> None:
        """All expected tools should be defined."""
        tool_names = get_all_tool_names()
        for expected in self.EXPECTED_TOOLS:
            assert expected in tool_names, f"Missing expected tool: {expected}"

    def test_tool_count(self) -> None:
        """Should have exactly the expected number of tools."""
        assert len(TOOL_DEFINITIONS) == len(self.EXPECTED_TOOLS)


class TestListTransactionsTool:
    """Tests for list_transactions tool definition."""

    def test_exists(self) -> None:
        """Tool should exist."""
        tool = get_tool_by_name("list_transactions")
        assert tool is not None

    def test_no_required_params(self) -> None:
        """list_transactions should have no required parameters."""
        tool = get_tool_by_name("list_transactions")
        assert tool["parameters"]["required"] == []

    def test_no_properties(self) -> None:
        """list_transactions should have no parameters."""
        tool = get_tool_by_name("list_transactions")
        assert tool["parameters"]["properties"] == {}


class TestGetTransactionTool:
    """Tests for get_transaction tool definition."""

    def test_exists(self) -> None:
        """Tool should exist."""
        tool = get_tool_by_name("get_transaction")
        assert tool is not None

    def test_requires_transaction_id(self) -> None:
        """get_transaction should require transaction_id."""
        tool = get_tool_by_name("get_transaction")
        assert "transaction_id" in tool["parameters"]["required"]

    def test_transaction_id_is_string(self) -> None:
        """transaction_id should be a string."""
        tool = get_tool_by_name("get_transaction")
        assert tool["parameters"]["properties"]["transaction_id"]["type"] == "string"


class TestScanForValueTool:
    """Tests for scan_for_value tool definition."""

    def test_exists(self) -> None:
        """Tool should exist."""
        tool = get_tool_by_name("scan_for_value")
        assert tool is not None

    def test_requires_value(self) -> None:
        """scan_for_value should require value parameter."""
        tool = get_tool_by_name("scan_for_value")
        assert "value" in tool["parameters"]["required"]

    def test_before_transaction_id_optional(self) -> None:
        """before_transaction_id should be optional."""
        tool = get_tool_by_name("scan_for_value")
        assert "before_transaction_id" not in tool["parameters"]["required"]
        assert "before_transaction_id" in tool["parameters"]["properties"]


class TestAddTransactionToQueueTool:
    """Tests for add_transaction_to_queue tool definition."""

    def test_exists(self) -> None:
        """Tool should exist."""
        tool = get_tool_by_name("add_transaction_to_queue")
        assert tool is not None

    def test_requires_transaction_id_and_reason(self) -> None:
        """Should require both transaction_id and reason."""
        tool = get_tool_by_name("add_transaction_to_queue")
        assert "transaction_id" in tool["parameters"]["required"]
        assert "reason" in tool["parameters"]["required"]


class TestGetQueueStatusTool:
    """Tests for get_queue_status tool definition."""

    def test_exists(self) -> None:
        """Tool should exist."""
        tool = get_tool_by_name("get_queue_status")
        assert tool is not None

    def test_no_required_params(self) -> None:
        """get_queue_status should have no required parameters."""
        tool = get_tool_by_name("get_queue_status")
        assert tool["parameters"]["required"] == []


class TestMarkTransactionCompleteTool:
    """Tests for mark_transaction_complete tool definition."""

    def test_exists(self) -> None:
        """Tool should exist."""
        tool = get_tool_by_name("mark_transaction_complete")
        assert tool is not None

    def test_requires_transaction_id(self) -> None:
        """Should require transaction_id."""
        tool = get_tool_by_name("mark_transaction_complete")
        assert "transaction_id" in tool["parameters"]["required"]


class TestRecordIdentifiedTransactionTool:
    """Tests for record_identified_transaction tool definition."""

    def test_exists(self) -> None:
        """Tool should exist."""
        tool = get_tool_by_name("record_identified_transaction")
        assert tool is not None

    def test_required_params(self) -> None:
        """Should require transaction_id, description, url, method."""
        tool = get_tool_by_name("record_identified_transaction")
        required = tool["parameters"]["required"]
        assert "transaction_id" in required
        assert "description" in required
        assert "url" in required
        assert "method" in required

    def test_method_is_enum(self) -> None:
        """method should be an enum of HTTP methods."""
        tool = get_tool_by_name("record_identified_transaction")
        method_prop = tool["parameters"]["properties"]["method"]
        assert "enum" in method_prop
        assert "GET" in method_prop["enum"]
        assert "POST" in method_prop["enum"]


class TestRecordExtractedVariablesTool:
    """Tests for record_extracted_variables tool definition."""

    def test_exists(self) -> None:
        """Tool should exist."""
        tool = get_tool_by_name("record_extracted_variables")
        assert tool is not None

    def test_requires_transaction_id_and_variables(self) -> None:
        """Should require transaction_id and variables."""
        tool = get_tool_by_name("record_extracted_variables")
        required = tool["parameters"]["required"]
        assert "transaction_id" in required
        assert "variables" in required

    def test_variables_is_array(self) -> None:
        """variables should be an array."""
        tool = get_tool_by_name("record_extracted_variables")
        variables_prop = tool["parameters"]["properties"]["variables"]
        assert variables_prop["type"] == "array"

    def test_variable_type_is_enum(self) -> None:
        """Variable type should be an enum."""
        tool = get_tool_by_name("record_extracted_variables")
        items = tool["parameters"]["properties"]["variables"]["items"]
        type_prop = items["properties"]["type"]
        assert "enum" in type_prop
        assert "parameter" in type_prop["enum"]
        assert "dynamic_token" in type_prop["enum"]
        assert "static_value" in type_prop["enum"]


class TestRecordResolvedVariableTool:
    """Tests for record_resolved_variable tool definition."""

    def test_exists(self) -> None:
        """Tool should exist."""
        tool = get_tool_by_name("record_resolved_variable")
        assert tool is not None

    def test_required_params(self) -> None:
        """Should require variable_name, transaction_id, source_type."""
        tool = get_tool_by_name("record_resolved_variable")
        required = tool["parameters"]["required"]
        assert "variable_name" in required
        assert "transaction_id" in required
        assert "source_type" in required

    def test_source_type_is_enum(self) -> None:
        """source_type should be an enum."""
        tool = get_tool_by_name("record_resolved_variable")
        source_type = tool["parameters"]["properties"]["source_type"]
        assert "enum" in source_type
        assert "storage" in source_type["enum"]
        assert "window_property" in source_type["enum"]
        assert "transaction" in source_type["enum"]
        assert "hardcode" in source_type["enum"]

    def test_optional_source_objects(self) -> None:
        """storage_source, window_property_source, transaction_source should be optional."""
        tool = get_tool_by_name("record_resolved_variable")
        props = tool["parameters"]["properties"]
        required = tool["parameters"]["required"]

        assert "storage_source" in props
        assert "storage_source" not in required
        assert "window_property_source" in props
        assert "window_property_source" not in required
        assert "transaction_source" in props
        assert "transaction_source" not in required


class TestExecuteRoutineTool:
    """Tests for execute_routine tool definition."""

    def test_exists(self) -> None:
        """Tool should exist."""
        tool = get_tool_by_name("execute_routine")
        assert tool is not None

    def test_requires_parameters(self) -> None:
        """Should require parameters."""
        tool = get_tool_by_name("execute_routine")
        assert "parameters" in tool["parameters"]["required"]

    def test_parameters_is_object(self) -> None:
        """parameters should be an object with additionalProperties."""
        tool = get_tool_by_name("execute_routine")
        params_prop = tool["parameters"]["properties"]["parameters"]
        assert params_prop["type"] == "object"
        assert params_prop.get("additionalProperties") is True


class TestConstructRoutineTool:
    """Tests for construct_routine tool definition."""

    def test_exists(self) -> None:
        """Tool should exist."""
        tool = get_tool_by_name("construct_routine")
        assert tool is not None

    def test_required_params(self) -> None:
        """Should require name, description, parameters, operations."""
        tool = get_tool_by_name("construct_routine")
        required = tool["parameters"]["required"]
        assert "name" in required
        assert "description" in required
        assert "parameters" in required
        assert "operations" in required

    def test_parameters_is_array(self) -> None:
        """parameters should be an array of parameter objects."""
        tool = get_tool_by_name("construct_routine")
        params_prop = tool["parameters"]["properties"]["parameters"]
        assert params_prop["type"] == "array"

    def test_operations_is_array(self) -> None:
        """operations should be an array."""
        tool = get_tool_by_name("construct_routine")
        ops_prop = tool["parameters"]["properties"]["operations"]
        assert ops_prop["type"] == "array"


class TestGetToolByName:
    """Tests for get_tool_by_name helper function."""

    def test_returns_tool_when_found(self) -> None:
        """Should return tool dict when name exists."""
        tool = get_tool_by_name("list_transactions")
        assert tool is not None
        assert tool["name"] == "list_transactions"

    def test_returns_none_when_not_found(self) -> None:
        """Should return None when name doesn't exist."""
        tool = get_tool_by_name("nonexistent_tool")
        assert tool is None

    def test_returns_correct_tool(self) -> None:
        """Should return the correct tool for each name."""
        for expected_tool in TOOL_DEFINITIONS:
            found = get_tool_by_name(expected_tool["name"])
            assert found == expected_tool


class TestGetAllToolNames:
    """Tests for get_all_tool_names helper function."""

    def test_returns_list(self) -> None:
        """Should return a list."""
        names = get_all_tool_names()
        assert isinstance(names, list)

    def test_returns_all_names(self) -> None:
        """Should return all tool names."""
        names = get_all_tool_names()
        assert len(names) == len(TOOL_DEFINITIONS)

    def test_returns_strings(self) -> None:
        """All returned names should be strings."""
        names = get_all_tool_names()
        for name in names:
            assert isinstance(name, str)

    def test_matches_tool_definitions(self) -> None:
        """Returned names should match TOOL_DEFINITIONS names."""
        names = get_all_tool_names()
        expected = [tool["name"] for tool in TOOL_DEFINITIONS]
        assert names == expected
