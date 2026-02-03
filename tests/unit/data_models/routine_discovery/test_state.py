"""
tests/unit/data_models/routine_discovery/test_state.py

Unit tests for RoutineDiscoveryState class and DiscoveryPhase enum.
"""

import pytest
from unittest.mock import MagicMock

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
)
from bluebox.data_models.routine.endpoint import HTTPMethod


class TestDiscoveryPhase:
    """Tests for DiscoveryPhase enum."""

    def test_phase_values(self) -> None:
        """All phases should have expected string values."""
        assert DiscoveryPhase.IDENTIFY_TRANSACTION.value == "identify_transaction"
        assert DiscoveryPhase.PROCESS_QUEUE.value == "process_queue"
        assert DiscoveryPhase.CONSTRUCT_ROUTINE.value == "construct_routine"
        assert DiscoveryPhase.VALIDATE_ROUTINE.value == "validate_routine"
        assert DiscoveryPhase.COMPLETE.value == "complete"

    def test_phase_count(self) -> None:
        """Should have exactly 5 phases."""
        assert len(DiscoveryPhase) == 5


class TestRoutineDiscoveryStateInit:
    """Tests for RoutineDiscoveryState initialization."""

    def test_default_initialization(self) -> None:
        """State should initialize with sensible defaults."""
        state = RoutineDiscoveryState()

        assert state.root_transaction is None
        assert state.transaction_queue == []
        assert state.processed_transactions == []
        assert state.current_transaction is None
        assert state.transaction_data == {}
        assert state.all_resolved_variables == []
        assert state.dev_routine is None
        assert state.production_routine is None
        assert state.phase == DiscoveryPhase.IDENTIFY_TRANSACTION
        assert state.identification_attempts == 0
        assert state.construction_attempts == 0
        assert state.validation_attempts == 0


class TestAddToQueue:
    """Tests for add_to_queue method."""

    def test_add_new_transaction(self) -> None:
        """Adding a new transaction should succeed."""
        state = RoutineDiscoveryState()

        added, position = state.add_to_queue("tx_001")

        assert added is True
        assert position == 0
        assert "tx_001" in state.transaction_queue

    def test_add_multiple_transactions(self) -> None:
        """Adding multiple transactions should assign correct positions."""
        state = RoutineDiscoveryState()

        added1, pos1 = state.add_to_queue("tx_001")
        added2, pos2 = state.add_to_queue("tx_002")
        added3, pos3 = state.add_to_queue("tx_003")

        assert added1 is True and pos1 == 0
        assert added2 is True and pos2 == 1
        assert added3 is True and pos3 == 2
        assert state.transaction_queue == ["tx_001", "tx_002", "tx_003"]

    def test_add_duplicate_transaction(self) -> None:
        """Adding a transaction already in queue should return existing position."""
        state = RoutineDiscoveryState()
        state.add_to_queue("tx_001")
        state.add_to_queue("tx_002")

        added, position = state.add_to_queue("tx_001")

        assert added is False
        assert position == 0  # Already at position 0
        assert state.transaction_queue.count("tx_001") == 1

    def test_add_already_processed_transaction(self) -> None:
        """Adding a transaction that was already processed should fail."""
        state = RoutineDiscoveryState()
        state.processed_transactions.append("tx_001")

        added, position = state.add_to_queue("tx_001")

        assert added is False
        assert position == -1
        assert "tx_001" not in state.transaction_queue


class TestGetNextTransaction:
    """Tests for pop_next_transaction method."""

    def test_get_next_from_populated_queue(self) -> None:
        """Should return and set the next transaction as current."""
        state = RoutineDiscoveryState()
        state.transaction_queue = ["tx_001", "tx_002", "tx_003"]

        result = state.pop_next_transaction()

        assert result == "tx_001"
        assert state.current_transaction == "tx_001"
        assert state.transaction_queue == ["tx_002", "tx_003"]

    def test_get_next_from_empty_queue(self) -> None:
        """Should return None when queue is empty."""
        state = RoutineDiscoveryState()

        result = state.pop_next_transaction()

        assert result is None
        assert state.current_transaction is None

    def test_get_next_exhausts_queue(self) -> None:
        """Should handle getting all transactions from queue."""
        state = RoutineDiscoveryState()
        state.transaction_queue = ["tx_001", "tx_002"]

        first = state.pop_next_transaction()
        second = state.pop_next_transaction()
        third = state.pop_next_transaction()

        assert first == "tx_001"
        assert second == "tx_002"
        assert third is None
        assert state.transaction_queue == []


class TestMarkTransactionComplete:
    """Tests for mark_transaction_complete method."""

    def test_mark_complete_adds_to_processed(self) -> None:
        """Completed transaction should be added to processed list."""
        state = RoutineDiscoveryState()
        state.current_transaction = "tx_001"

        state.mark_transaction_complete("tx_001")

        assert "tx_001" in state.processed_transactions
        assert state.current_transaction is None

    def test_mark_complete_returns_next_transaction(self) -> None:
        """Should return the next transaction from queue."""
        state = RoutineDiscoveryState()
        state.current_transaction = "tx_001"
        state.transaction_queue = ["tx_002", "tx_003"]

        next_tx = state.mark_transaction_complete("tx_001")

        assert next_tx == "tx_002"
        assert state.current_transaction == "tx_002"
        assert state.processed_transactions == ["tx_001"]

    def test_mark_complete_with_empty_queue(self) -> None:
        """Should return None when queue is empty after completion."""
        state = RoutineDiscoveryState()
        state.current_transaction = "tx_001"

        next_tx = state.mark_transaction_complete("tx_001")

        assert next_tx is None
        assert state.current_transaction is None

    def test_mark_complete_idempotent(self) -> None:
        """Marking the same transaction complete twice should not duplicate."""
        state = RoutineDiscoveryState()
        state.current_transaction = "tx_001"

        state.mark_transaction_complete("tx_001")
        state.mark_transaction_complete("tx_001")

        assert state.processed_transactions.count("tx_001") == 1

    def test_mark_complete_different_transaction(self) -> None:
        """Marking a different transaction should still process it and call get_next."""
        state = RoutineDiscoveryState()
        state.current_transaction = "tx_001"
        state.transaction_queue = ["tx_003"]  # Add something to queue

        state.mark_transaction_complete("tx_002")

        assert "tx_002" in state.processed_transactions
        # Current transaction changes to next in queue (pop_next_transaction is always called)
        assert state.current_transaction == "tx_003"


class TestStoreTransactionData:
    """Tests for store_transaction_data method."""

    def test_store_request_data(self) -> None:
        """Should store request data for a transaction."""
        state = RoutineDiscoveryState()
        request = {"url": "https://api.example.com", "method": "GET"}

        state.store_transaction_data("tx_001", request=request)

        assert state.transaction_data["tx_001"]["request"] == request

    def test_store_extracted_variables(self) -> None:
        """Should store extracted variables for a transaction."""
        state = RoutineDiscoveryState()
        extracted = ExtractedVariableResponse(
            transaction_id="tx_001",
            variables=[
                Variable(
                    type=VariableType.PARAMETER,
                    requires_dynamic_resolution=False,
                    name="user_id",
                    observed_value="123",
                    values_to_scan_for=["123"],
                )
            ],
        )

        state.store_transaction_data("tx_001", extracted_variables=extracted)

        assert state.transaction_data["tx_001"]["extracted_variables"] == extracted

    def test_store_resolved_variable(self) -> None:
        """Should store resolved variable and add to global list."""
        state = RoutineDiscoveryState()
        variable = Variable(
            type=VariableType.DYNAMIC_TOKEN,
            requires_dynamic_resolution=True,
            name="csrf_token",
            observed_value="abc123",
            values_to_scan_for=["abc123"],
        )
        resolved = ResolvedVariableResponse(
            variable=variable,
            short_explanation="Resolved from storage",
        )

        state.store_transaction_data("tx_001", resolved_variable=resolved)

        assert resolved in state.transaction_data["tx_001"]["resolved_variables"]
        assert resolved in state.all_resolved_variables

    def test_store_multiple_resolved_variables(self) -> None:
        """Should accumulate multiple resolved variables."""
        state = RoutineDiscoveryState()
        var1 = Variable(
            type=VariableType.DYNAMIC_TOKEN,
            requires_dynamic_resolution=True,
            name="token1",
            observed_value="val1",
            values_to_scan_for=["val1"],
        )
        var2 = Variable(
            type=VariableType.DYNAMIC_TOKEN,
            requires_dynamic_resolution=True,
            name="token2",
            observed_value="val2",
            values_to_scan_for=["val2"],
        )
        resolved1 = ResolvedVariableResponse(variable=var1, short_explanation="test")
        resolved2 = ResolvedVariableResponse(variable=var2, short_explanation="test")

        state.store_transaction_data("tx_001", resolved_variable=resolved1)
        state.store_transaction_data("tx_001", resolved_variable=resolved2)

        assert len(state.transaction_data["tx_001"]["resolved_variables"]) == 2
        assert len(state.all_resolved_variables) == 2

    def test_store_data_creates_entry_if_missing(self) -> None:
        """Should initialize transaction data structure if not present."""
        state = RoutineDiscoveryState()

        state.store_transaction_data("tx_new", request={"test": "data"})

        assert "tx_new" in state.transaction_data
        assert state.transaction_data["tx_new"]["request"] == {"test": "data"}
        assert state.transaction_data["tx_new"]["extracted_variables"] is None
        assert state.transaction_data["tx_new"]["resolved_variables"] == []


class TestGetQueueStatus:
    """Tests for get_queue_status method."""

    def test_empty_state_status(self) -> None:
        """Should return correct status for empty state."""
        state = RoutineDiscoveryState()

        status = state.get_queue_status()

        assert status["pending"] == []
        assert status["processed"] == []
        assert status["current"] is None
        assert status["pending_count"] == 0
        assert status["processed_count"] == 0

    def test_populated_state_status(self) -> None:
        """Should return correct status for populated state."""
        state = RoutineDiscoveryState()
        state.transaction_queue = ["tx_002", "tx_003"]
        state.processed_transactions = ["tx_000"]
        state.current_transaction = "tx_001"

        status = state.get_queue_status()

        assert status["pending"] == ["tx_002", "tx_003"]
        assert status["processed"] == ["tx_000"]
        assert status["current"] == "tx_001"
        assert status["pending_count"] == 2
        assert status["processed_count"] == 1


class TestGetOrderedTransactions:
    """Tests for get_ordered_transactions method."""

    def test_reverse_order(self) -> None:
        """Should return transactions in reverse processing order."""
        state = RoutineDiscoveryState()
        state.processed_transactions = ["root", "dep1", "dep2"]
        state.transaction_data = {
            "root": {"request": {"url": "/root"}},
            "dep1": {"request": {"url": "/dep1"}},
            "dep2": {"request": {"url": "/dep2"}},
        }

        ordered = state.get_ordered_transactions()

        # Keys should be in reverse order (dependencies first)
        keys = list(ordered.keys())
        assert keys == ["dep2", "dep1", "root"]

    def test_handles_missing_transaction_data(self) -> None:
        """Should return empty dict for transactions without stored data."""
        state = RoutineDiscoveryState()
        state.processed_transactions = ["tx_001", "tx_002"]
        state.transaction_data = {"tx_001": {"request": {"url": "/test"}}}

        ordered = state.get_ordered_transactions()

        assert ordered["tx_001"] == {"request": {"url": "/test"}}
        assert ordered["tx_002"] == {}


class TestReset:
    """Tests for reset method."""

    def test_reset_clears_all_state(self) -> None:
        """Reset should return state to initial values."""
        state = RoutineDiscoveryState()
        # Populate state
        state.root_transaction = MagicMock()
        state.transaction_queue = ["tx_001", "tx_002"]
        state.processed_transactions = ["tx_000"]
        state.current_transaction = "tx_001"
        state.transaction_data = {"tx_000": {"data": "test"}}
        state.all_resolved_variables = [MagicMock()]
        state.dev_routine = MagicMock()
        state.production_routine = MagicMock()
        state.phase = DiscoveryPhase.COMPLETE
        state.identification_attempts = 3
        state.construction_attempts = 2
        state.validation_attempts = 1

        state.reset()

        assert state.root_transaction is None
        assert state.transaction_queue == []
        assert state.processed_transactions == []
        assert state.current_transaction is None
        assert state.transaction_data == {}
        assert state.all_resolved_variables == []
        assert state.dev_routine is None
        assert state.production_routine is None
        assert state.phase == DiscoveryPhase.IDENTIFY_TRANSACTION
        assert state.identification_attempts == 0
        assert state.construction_attempts == 0
        assert state.validation_attempts == 0


class TestBFSWorkflow:
    """Integration tests for typical BFS workflow patterns."""

    def test_typical_workflow(self) -> None:
        """Test a typical BFS workflow with root and dependencies."""
        state = RoutineDiscoveryState()

        # Phase 1: Identify root transaction
        state.add_to_queue("root_tx")
        state.pop_next_transaction()
        assert state.current_transaction == "root_tx"

        # Phase 2: Process root, discover dependency
        state.add_to_queue("dep_tx")
        state.mark_transaction_complete("root_tx")
        assert state.current_transaction == "dep_tx"
        assert "root_tx" in state.processed_transactions

        # Phase 3: Process dependency
        state.mark_transaction_complete("dep_tx")
        assert state.current_transaction is None
        assert state.transaction_queue == []
        assert state.processed_transactions == ["root_tx", "dep_tx"]

    def test_complex_dependency_chain(self) -> None:
        """Test processing a more complex dependency graph."""
        state = RoutineDiscoveryState()

        # Add root
        state.add_to_queue("root")
        state.pop_next_transaction()

        # Root depends on dep1 and dep2
        state.add_to_queue("dep1")
        state.add_to_queue("dep2")
        state.mark_transaction_complete("root")

        # dep1 depends on dep3
        assert state.current_transaction == "dep1"
        state.add_to_queue("dep3")
        state.mark_transaction_complete("dep1")

        # Process dep2
        assert state.current_transaction == "dep2"
        state.mark_transaction_complete("dep2")

        # Process dep3
        assert state.current_transaction == "dep3"
        state.mark_transaction_complete("dep3")

        # All done
        assert state.current_transaction is None
        assert len(state.processed_transactions) == 4
        assert state.processed_transactions == ["root", "dep1", "dep2", "dep3"]

        # Execution order should be reversed
        ordered = state.get_ordered_transactions()
        assert list(ordered.keys()) == ["dep3", "dep2", "dep1", "root"]
