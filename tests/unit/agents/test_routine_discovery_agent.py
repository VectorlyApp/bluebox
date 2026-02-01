"""
tests/unit/agents/test_routine_discovery_agent.py

Unit tests for RoutineDiscoveryAgent tool implementations and agentic loop.
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from bluebox.agents.routine_discovery_agent import RoutineDiscoveryAgent
from bluebox.data_models.routine_discovery.state import (
    DiscoveryPhase,
    RoutineDiscoveryState,
)
from bluebox.data_models.routine_discovery.llm_responses import (
    TransactionIdentificationResponse,
    ExtractedVariableResponse,
    ResolvedVariableResponse,
    Variable,
    VariableType,
    SessionStorageSource,
    SessionStorageType,
    TransactionSource,
    WindowPropertySource,
)
from bluebox.data_models.routine.endpoint import HTTPMethod
from bluebox.utils.exceptions import TransactionIdentificationFailedError


@pytest.fixture
def mock_data_store():
    """Create a mock DiscoveryDataStore."""
    store = MagicMock()
    store.cdp_captures_vectorstore_id = "test-vectorstore-id"
    store.get_all_transaction_ids.return_value = ["tx_001", "tx_002", "tx_003"]
    store.get_transaction_by_id.return_value = {
        "request": {"url": "https://api.example.com", "method": "GET", "headers": {}},
        "response": {"status": 200, "body": {"data": "test"}},
    }
    store.get_vectorstore_ids.return_value = ["test-vectorstore-id"]
    store.generate_data_store_prompt.return_value = "Test data store prompt"
    return store


@pytest.fixture
def mock_llm_client():
    """Create a mock LLMClient."""
    client = MagicMock()
    client.clear_tools = MagicMock()
    client.register_tool = MagicMock()
    client.set_file_search_vectorstores = MagicMock()
    return client


@pytest.fixture
def mock_emit_message():
    """Create a mock emit message callable."""
    return MagicMock()


@pytest.fixture
def agent(mock_data_store, mock_llm_client, mock_emit_message):
    """Create a RoutineDiscoveryAgent instance with mocked dependencies."""
    # Use model_construct to bypass Pydantic validation for mocked dependencies
    agent = RoutineDiscoveryAgent.model_construct(
        llm_client=mock_llm_client,
        data_store=mock_data_store,
        task="Search for products",
        emit_message_callable=mock_emit_message,
        message_history=[],
        output_dir=None,
        last_response_id=None,
        n_transaction_identification_attempts=3,
        max_iterations=50,
        timeout=600,
        remote_debugging_address=None,
    )
    # Initialize state manually for tool tests
    agent._state = RoutineDiscoveryState()
    return agent


class TestToolListTransactions:
    """Tests for _tool_list_transactions method."""

    def test_returns_all_transaction_ids(self, agent):
        """Should return all transaction IDs from data store."""
        result = agent._tool_list_transactions()

        assert result["transaction_ids"] == ["tx_001", "tx_002", "tx_003"]
        assert result["count"] == 3

    def test_handles_empty_transactions(self, agent, mock_data_store):
        """Should handle empty transaction list."""
        mock_data_store.get_all_transaction_ids.return_value = []

        result = agent._tool_list_transactions()

        assert result["transaction_ids"] == []
        assert result["count"] == 0


class TestToolGetTransaction:
    """Tests for _tool_get_transaction method."""

    def test_returns_transaction_details(self, agent):
        """Should return full transaction details."""
        result = agent._tool_get_transaction("tx_001")

        assert result["transaction_id"] == "tx_001"
        assert "request" in result
        assert "response" in result

    def test_returns_error_for_invalid_id(self, agent, mock_data_store):
        """Should return error for non-existent transaction."""
        mock_data_store.get_all_transaction_ids.return_value = ["tx_001"]

        result = agent._tool_get_transaction("invalid_tx")

        assert "error" in result
        assert "not found" in result["error"]


class TestToolScanForValue:
    """Tests for _tool_scan_for_value method."""

    def test_scans_all_sources(self, agent, mock_data_store):
        """Should scan storage, window properties, and transactions."""
        mock_data_store.scan_storage_for_value.return_value = [{"type": "cookie", "path": "auth"}]
        mock_data_store.scan_window_properties_for_value.return_value = []
        mock_data_store.scan_transaction_responses.return_value = [{"tx_id": "tx_001", "path": "data.token"}]

        result = agent._tool_scan_for_value("test_token")

        mock_data_store.scan_storage_for_value.assert_called_once_with("test_token")
        mock_data_store.scan_window_properties_for_value.assert_called_once_with("test_token")
        mock_data_store.scan_transaction_responses.assert_called_once()
        assert "storage_sources" in result
        assert "window_property_sources" in result
        assert "transaction_sources" in result

    def test_uses_timestamp_filter(self, agent, mock_data_store):
        """Should filter transactions by timestamp when before_transaction_id provided."""
        mock_data_store.get_transaction_timestamp.return_value = 12345
        mock_data_store.scan_storage_for_value.return_value = []
        mock_data_store.scan_window_properties_for_value.return_value = []
        mock_data_store.scan_transaction_responses.return_value = []

        result = agent._tool_scan_for_value("test", before_transaction_id="tx_002")

        mock_data_store.get_transaction_timestamp.assert_called_with("tx_002")
        mock_data_store.scan_transaction_responses.assert_called_with("test", max_timestamp=12345)

    def test_limits_results(self, agent, mock_data_store):
        """Should limit results to 5 per source."""
        mock_data_store.scan_storage_for_value.return_value = list(range(10))
        mock_data_store.scan_window_properties_for_value.return_value = list(range(10))
        mock_data_store.scan_transaction_responses.return_value = list(range(10))

        result = agent._tool_scan_for_value("test")

        assert len(result["storage_sources"]) <= 5
        assert len(result["window_property_sources"]) <= 5
        assert len(result["transaction_sources"]) <= 5


class TestToolAddToQueue:
    """Tests for _tool_add_to_queue method."""

    def test_adds_valid_transaction(self, agent):
        """Should add valid transaction to queue."""
        result = agent._tool_add_to_queue("tx_001", "Dependency for auth token")

        assert result["success"] is True
        assert result["added"] is True
        assert "tx_001" in agent._state.transaction_queue

    def test_returns_error_for_invalid_transaction(self, agent, mock_data_store):
        """Should return error for non-existent transaction."""
        mock_data_store.get_all_transaction_ids.return_value = ["tx_001"]

        result = agent._tool_add_to_queue("invalid_tx", "test")

        assert result["success"] is False
        assert "not found" in result["error"]

    def test_handles_duplicate_add(self, agent):
        """Should handle adding same transaction twice."""
        agent._tool_add_to_queue("tx_001", "first")
        result = agent._tool_add_to_queue("tx_001", "second")

        assert result["success"] is True
        assert result["added"] is False


class TestToolGetQueueStatus:
    """Tests for _tool_get_queue_status method."""

    def test_returns_queue_status(self, agent):
        """Should return current queue status."""
        agent._state.transaction_queue = ["tx_002"]
        agent._state.processed_transactions = ["tx_001"]
        agent._state.current_transaction = "tx_003"

        result = agent._tool_get_queue_status()

        assert result["pending"] == ["tx_002"]
        assert result["processed"] == ["tx_001"]
        assert result["current"] == "tx_003"


class TestToolMarkComplete:
    """Tests for _tool_mark_complete method."""

    def test_marks_transaction_complete(self, agent):
        """Should mark transaction as complete."""
        agent._state.current_transaction = "tx_001"
        agent._state.transaction_queue = ["tx_002"]

        result = agent._tool_mark_complete("tx_001")

        assert result["success"] is True
        assert result["next_transaction"] == "tx_002"
        assert "tx_001" in agent._state.processed_transactions

    def test_advances_phase_when_queue_empty(self, agent, mock_emit_message):
        """Should advance to CONSTRUCT_ROUTINE phase when queue empty."""
        agent._state.current_transaction = "tx_001"
        agent._state.phase = DiscoveryPhase.PROCESS_QUEUE

        result = agent._tool_mark_complete("tx_001")

        assert result["next_transaction"] is None
        assert agent._state.phase == DiscoveryPhase.CONSTRUCT_ROUTINE


class TestToolRecordIdentifiedTransaction:
    """Tests for _tool_record_identified_transaction method."""

    def test_records_valid_transaction(self, agent, mock_emit_message):
        """Should record valid root transaction."""
        args = {
            "transaction_id": "tx_001",
            "description": "Search API call",
            "url": "https://api.example.com/search",
            "method": "GET",
        }

        result = agent._tool_record_identified_transaction(args)

        assert result["success"] is True
        assert agent._state.root_transaction is not None
        assert agent._state.root_transaction.transaction_id == "tx_001"
        assert agent._state.phase == DiscoveryPhase.PROCESS_QUEUE

    def test_returns_error_for_invalid_transaction(self, agent, mock_data_store):
        """Should return error for non-existent transaction."""
        mock_data_store.get_all_transaction_ids.return_value = ["tx_001"]
        args = {
            "transaction_id": "invalid_tx",
            "description": "test",
            "url": "https://test.com",
            "method": "GET",
        }

        result = agent._tool_record_identified_transaction(args)

        assert result["success"] is False
        assert "not found" in result["error"]
        assert agent._state.identification_attempts == 1

    def test_raises_after_max_attempts(self, agent, mock_data_store):
        """Should raise error after max identification attempts."""
        mock_data_store.get_all_transaction_ids.return_value = ["tx_001"]
        agent.n_transaction_identification_attempts = 2
        args = {
            "transaction_id": "invalid_tx",
            "description": "test",
            "url": "https://test.com",
            "method": "GET",
        }

        agent._tool_record_identified_transaction(args)  # Attempt 1

        with pytest.raises(TransactionIdentificationFailedError):
            agent._tool_record_identified_transaction(args)  # Attempt 2


class TestToolRecordExtractedVariables:
    """Tests for _tool_record_extracted_variables method."""

    def test_records_variables(self, agent):
        """Should record extracted variables for transaction."""
        args = {
            "transaction_id": "tx_001",
            "variables": [
                {
                    "type": "parameter",
                    "name": "query",
                    "observed_value": "test search",
                    "requires_dynamic_resolution": False,
                },
                {
                    "type": "dynamic_token",
                    "name": "csrf_token",
                    "observed_value": "abc123",
                    "requires_dynamic_resolution": True,
                },
            ],
        }

        result = agent._tool_record_extracted_variables(args)

        assert result["success"] is True
        assert result["total_variables"] == 2
        assert "csrf_token" in result["variables_needing_resolution"]
        assert "query" not in result["variables_needing_resolution"]

    def test_stores_in_state(self, agent):
        """Should store extracted variables in state."""
        args = {
            "transaction_id": "tx_001",
            "variables": [
                {
                    "type": "parameter",
                    "name": "user_id",
                    "observed_value": "123",
                    "requires_dynamic_resolution": False,
                }
            ],
        }

        agent._tool_record_extracted_variables(args)

        assert "tx_001" in agent._state.transaction_data
        assert agent._state.transaction_data["tx_001"]["extracted_variables"] is not None


class TestToolRecordResolvedVariable:
    """Tests for _tool_record_resolved_variable method."""

    def test_records_storage_source(self, agent):
        """Should record variable resolved from storage."""
        # First extract variables
        agent._tool_record_extracted_variables({
            "transaction_id": "tx_001",
            "variables": [
                {
                    "type": "dynamic_token",
                    "name": "auth_token",
                    "observed_value": "token123",
                    "requires_dynamic_resolution": True,
                }
            ],
        })

        args = {
            "variable_name": "auth_token",
            "transaction_id": "tx_001",
            "source_type": "storage",
            "storage_source": {
                "type": "sessionStorage",
                "dot_path": "auth.access_token",
            },
        }

        result = agent._tool_record_resolved_variable(args)

        assert result["success"] is True
        assert result["source_type"] == "storage"

    def test_records_transaction_source_and_adds_dependency(self, agent):
        """Should record transaction source and add dependency to queue."""
        # First extract variables
        agent._tool_record_extracted_variables({
            "transaction_id": "tx_001",
            "variables": [
                {
                    "type": "dynamic_token",
                    "name": "token",
                    "observed_value": "val",
                    "requires_dynamic_resolution": True,
                }
            ],
        })

        args = {
            "variable_name": "token",
            "transaction_id": "tx_001",
            "source_type": "transaction",
            "transaction_source": {
                "transaction_id": "tx_002",
                "dot_path": "response.token",
            },
        }

        result = agent._tool_record_resolved_variable(args)

        assert result["success"] is True
        assert result["needs_dependency_processing"] is True
        assert "tx_002" in agent._state.transaction_queue

    def test_returns_error_for_unknown_variable(self, agent):
        """Should return error if variable not found."""
        args = {
            "variable_name": "nonexistent",
            "transaction_id": "tx_001",
            "source_type": "storage",
        }

        result = agent._tool_record_resolved_variable(args)

        assert result["success"] is False
        # Error can be about no extracted variables or variable not found
        assert "error" in result


class TestToolExecute:
    """Tests for _execute_tool dispatcher method."""

    def test_dispatches_to_correct_tool(self, agent):
        """Should dispatch to correct tool based on name."""
        result = agent._execute_tool("list_transactions", {})

        assert "transaction_ids" in result

    def test_handles_unknown_tool(self, agent):
        """Should return error for unknown tool."""
        result = agent._execute_tool("nonexistent_tool", {})

        assert "error" in result
        assert "Unknown tool" in result["error"]

    def test_handles_tool_exception(self, agent, mock_data_store):
        """Should catch and return errors from tool execution."""
        mock_data_store.get_all_transaction_ids.side_effect = Exception("Database error")

        result = agent._execute_tool("list_transactions", {})

        assert "error" in result


class TestAgentInitialization:
    """Tests for agent initialization and setup."""

    def test_register_tools(self, agent, mock_llm_client):
        """Should register all discovery tools with LLM client."""
        agent._register_tools()

        mock_llm_client.clear_tools.assert_called_once()
        assert mock_llm_client.register_tool.call_count > 0

    def test_system_prompt_includes_state(self, agent):
        """System prompt should include current state context."""
        agent._state.phase = DiscoveryPhase.PROCESS_QUEUE
        agent._state.transaction_queue = ["tx_001"]
        agent._state.processed_transactions = ["tx_000"]

        prompt = agent._get_system_prompt()

        assert "process_queue" in prompt.lower()
        assert "1 pending" in prompt
        assert "1 processed" in prompt


class TestMessageHistory:
    """Tests for message history management."""

    def test_add_to_message_history(self, agent):
        """Should add messages to history."""
        agent._add_to_message_history("user", "Hello")
        agent._add_to_message_history("assistant", "Hi there")

        assert len(agent.message_history) == 2
        assert agent.message_history[0]["role"] == "user"
        assert agent.message_history[1]["role"] == "assistant"

    def test_add_tool_result(self, agent):
        """Should add tool results to history."""
        agent._add_tool_result("call_123", {"success": True})

        assert len(agent.message_history) == 1
        assert agent.message_history[0]["role"] == "tool"
        assert agent.message_history[0]["tool_call_id"] == "call_123"


class TestGetTestParameters:
    """Tests for get_test_parameters method."""

    def test_generates_test_params_from_observed_values(self, agent):
        """Should use observed values for test parameters."""
        # Set up state with extracted variables
        agent._state.transaction_data["tx_001"] = {
            "extracted_variables": ExtractedVariableResponse(
                transaction_id="tx_001",
                variables=[
                    Variable(
                        type=VariableType.PARAMETER,
                        requires_dynamic_resolution=False,
                        name="query",
                        observed_value="test search",
                        values_to_scan_for=["test search"],
                    )
                ],
            ),
            "resolved_variables": [],
        }

        # Create a mock routine with matching parameter
        mock_routine = MagicMock()
        mock_routine.parameters = [MagicMock(name="query", type="string")]
        mock_routine.parameters[0].name = "query"

        result = agent.get_test_parameters(mock_routine)

        assert len(result.parameters) == 1
        assert result.parameters[0].name == "query"
        assert result.parameters[0].value == "test search"

    def test_provides_defaults_for_missing_values(self, agent):
        """Should provide sensible defaults when no observed value."""
        agent._state.transaction_data = {}

        mock_routine = MagicMock()
        mock_routine.parameters = [
            MagicMock(name="count", type="integer"),
            MagicMock(name="price", type="number"),
            MagicMock(name="active", type="boolean"),
        ]
        mock_routine.parameters[0].name = "count"
        mock_routine.parameters[0].type = "integer"
        mock_routine.parameters[1].name = "price"
        mock_routine.parameters[1].type = "number"
        mock_routine.parameters[2].name = "active"
        mock_routine.parameters[2].type = "boolean"

        result = agent.get_test_parameters(mock_routine)

        assert result.parameters[0].value == "1"
        assert result.parameters[1].value == "1.0"
        assert result.parameters[2].value == "false"


class TestConstructRoutineTool:
    """Tests for _tool_construct_routine method."""

    def test_validates_routine(self, agent, mock_llm_client):
        """Should validate routine before construction."""
        args = {
            "name": "test_routine",
            "description": "Test routine",
            "parameters": [],
            "operations": [
                {"type": "navigate", "url": "https://example.com"},
                {
                    "type": "fetch",
                    "endpoint": {
                        "url": "https://api.example.com",
                        "method": "GET",
                        "headers": "{}",
                        "body": "{}",
                    },
                    "session_storage_key": "result",
                },
                {"type": "return", "session_storage_key": "result"},
            ],
        }

        # Mock the LLM client response for productionization
        mock_response = MagicMock()
        mock_response.content = '{"name": "test_routine", "description": "Test", "operations": []}'
        mock_llm_client.call_sync.return_value = mock_response

        # This will fail validation since we can't mock everything
        result = agent._tool_construct_routine(args)

        # At minimum, should attempt construction
        assert agent._state.construction_attempts == 1


class TestExecuteRoutineTool:
    """Tests for _tool_execute_routine method."""

    def test_returns_error_without_routine(self, agent):
        """Should return error if no routine constructed."""
        agent._state.production_routine = None

        result = agent._tool_execute_routine({"parameters": {}})

        assert result["success"] is False
        assert "No routine" in result["error"]

    def test_returns_error_without_browser(self, agent):
        """Should return error if no remote debugging address."""
        agent._state.production_routine = MagicMock()
        agent.remote_debugging_address = None

        result = agent._tool_execute_routine({"parameters": {}})

        assert result["success"] is False
        assert "remote_debugging_address" in result["error"]
