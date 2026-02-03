"""
bluebox/data_models/routine_discovery/state.py

State management for the LLM-driven routine discovery agent.
"""

from enum import StrEnum

from pydantic import BaseModel, Field, ConfigDict

from bluebox.data_models.routine_discovery.llm_responses import (
    TransactionIdentificationResponse,
    ExtractedVariableResponse,
    ResolvedVariableResponse,
)
from bluebox.data_models.routine.routine import Routine
from bluebox.data_models.routine.dev_routine import DevRoutine


class DiscoveryPhase(StrEnum):
    """Current phase of the routine discovery process."""
    IDENTIFY_TRANSACTION = "identify_transaction"
    PROCESS_QUEUE = "process_queue"
    CONSTRUCT_ROUTINE = "construct_routine"
    VALIDATE_ROUTINE = "validate_routine"
    COMPLETE = "complete"


class RoutineDiscoveryState(BaseModel):
    """
    Manages state for the routine discovery agent across tool calls.

    Tracks the BFS transaction queue, extracted/resolved variables,
    and the final routine being constructed.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Root transaction identification
    root_transaction: TransactionIdentificationResponse | None = Field(
        default=None,
        description="The identified root transaction that matches the user's task"
    )

    # BFS transaction queue management
    transaction_queue: list[str] = Field(
        default_factory=list,
        description="Pending transactions to process"
    )
    processed_transactions: list[str] = Field(
        default_factory=list,
        description="Transactions that have been fully processed (in BFS order)"
    )
    current_transaction: str | None = Field(
        default=None,
        description="Transaction currently being processed"
    )

    # Per-transaction data storage
    # Structure: {tx_id: {"request": dict, "extracted_variables": ..., "resolved_variables": [...]}}
    transaction_data: dict[str, dict] = Field(
        default_factory=dict,
        description="Stored data for each processed transaction"
    )

    # All resolved variables (for routine construction)
    all_resolved_variables: list[ResolvedVariableResponse] = Field(
        default_factory=list,
        description="All resolved variables across all transactions"
    )

    # Final output
    dev_routine: DevRoutine | None = Field(
        default=None,
        description="Intermediate routine format"
    )
    production_routine: Routine | None = Field(
        default=None,
        description="Final production routine"
    )

    # Progress tracking
    phase: DiscoveryPhase = Field(
        default=DiscoveryPhase.IDENTIFY_TRANSACTION,
        description="Current phase of discovery"
    )

    # Retry counters
    identification_attempts: int = Field(default=0)
    construction_attempts: int = Field(default=0)
    validation_attempts: int = Field(default=0)

    def add_to_queue(self, transaction_id: str) -> tuple[bool, int]:
        """
        Add a transaction to the queue if not already processed.

        Returns:
            Tuple of (was_added, queue_position).
            If already processed or currently being processed, returns (False, -1).
        """
        if transaction_id == self.current_transaction:
            return False, -1
        if transaction_id in self.processed_transactions:
            return False, -1
        if transaction_id in self.transaction_queue:
            return False, self.transaction_queue.index(transaction_id)
        self.transaction_queue.append(transaction_id)
        return True, len(self.transaction_queue) - 1

    def pop_next_transaction(self) -> str | None:
        """
        Pop the next transaction from the queue and set it as current.

        Returns:
            The next transaction ID, or None if queue is empty.
        """
        if not self.transaction_queue:
            self.current_transaction = None
            return None
        self.current_transaction = self.transaction_queue.pop(0)
        return self.current_transaction

    def mark_transaction_complete(self, transaction_id: str) -> None:
        """
        Mark a transaction as complete.

        Adds the transaction to processed_transactions and clears current_transaction
        if it matches. Call pop_next_transaction() separately to get the next one.
        """
        if transaction_id not in self.processed_transactions:
            self.processed_transactions.append(transaction_id)
        if self.current_transaction == transaction_id:
            self.current_transaction = None

    def store_transaction_data(
        self,
        transaction_id: str,
        request: dict | None = None,
        extracted_variables: ExtractedVariableResponse | None = None,
        resolved_variable: ResolvedVariableResponse | None = None,
    ) -> None:
        """Store data for a transaction."""
        if transaction_id not in self.transaction_data:
            self.transaction_data[transaction_id] = {
                "request": None,
                "extracted_variables": None,
                "resolved_variables": [],
            }

        if request is not None:
            self.transaction_data[transaction_id]["request"] = request
        if extracted_variables is not None:
            self.transaction_data[transaction_id]["extracted_variables"] = extracted_variables
        if resolved_variable is not None:
            self.transaction_data[transaction_id]["resolved_variables"].append(resolved_variable)
            self.all_resolved_variables.append(resolved_variable)

    def get_queue_status(self) -> dict:
        """Get a summary of the queue status."""
        return {
            "pending": list(self.transaction_queue),
            "processed": list(self.processed_transactions),
            "current": self.current_transaction,
            "pending_count": len(self.transaction_queue),
            "processed_count": len(self.processed_transactions),
        }

    def get_ordered_transactions(self) -> dict[str, dict]:
        """
        Get transactions in execution order (dependencies first, root last).

        The BFS processes root -> dep1 -> dep2, but execution needs
        dep2 -> dep1 -> root. This reverses the order.
        """
        # Get transactions in the order they were processed
        ordered_ids = list(self.processed_transactions)
        # Reverse so dependencies come first
        ordered_ids.reverse()
        return {tx_id: self.transaction_data.get(tx_id, {}) for tx_id in ordered_ids}

    def reset(self) -> None:
        """Reset all state for a fresh discovery run."""
        self.root_transaction = None
        self.transaction_queue = []
        self.processed_transactions = []
        self.current_transaction = None
        self.transaction_data = {}
        self.all_resolved_variables = []
        self.dev_routine = None
        self.production_routine = None
        self.phase = DiscoveryPhase.IDENTIFY_TRANSACTION
        self.identification_attempts = 0
        self.construction_attempts = 0
        self.validation_attempts = 0
