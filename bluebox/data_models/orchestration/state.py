"""
bluebox/data_models/orchestration/state.py

State management for the SuperDiscoveryAgent orchestrator.

Tracks the discovery phase, transaction queues, tasks, subagents,
and the routine being constructed.
"""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, ConfigDict

from bluebox.data_models.orchestration.task import Task, SubAgent
from bluebox.data_models.routine.placeholder import ExtractedPlaceholder
from bluebox.data_models.routine.routine import Routine


class SuperDiscoveryPhase(StrEnum):
    """Current phase of the super discovery process."""
    PLANNING = "planning"             # Analyzing task, planning approach
    DISCOVERING = "discovering"       # Delegating discovery tasks to specialists
    CONSTRUCTING = "constructing"     # Building the routine from discoveries
    VALIDATING = "validating"         # Testing the constructed routine
    COMPLETE = "complete"             # Discovery finished successfully
    FAILED = "failed"                 # Discovery failed


class SuperDiscoveryState(BaseModel):
    """
    Manages state for the SuperDiscoveryAgent orchestrator.

    Tracks the discovery phase, transaction queues, delegated tasks,
    subagent instances, and the routine being constructed.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Current phase
    phase: SuperDiscoveryPhase = Field(default=SuperDiscoveryPhase.PLANNING)

    # Transaction tracking (BFS queue like RoutineDiscoveryState)
    root_transaction_id: str | None = Field(
        default=None,
        description="The root transaction that matches the user's task"
    )
    queued_transaction_ids: list[str] = Field(
        default_factory=list,
        description="Transactions pending processing"
    )
    processed_transaction_ids: list[str] = Field(
        default_factory=list,
        description="Transactions that have been fully processed"
    )

    # Placeholder tracking
    queued_placeholders: list[ExtractedPlaceholder] = Field(
        default_factory=list,
        description="Placeholders pending resolution"
    )
    processed_placeholders: list[ExtractedPlaceholder] = Field(
        default_factory=list,
        description="Placeholders that have been resolved"
    )

    # Task and subagent management
    tasks: dict[str, Task] = Field(
        default_factory=dict,
        description="All delegated tasks by ID"
    )
    subagents: dict[str, SubAgent] = Field(
        default_factory=dict,
        description="All subagent instances by ID"
    )

    # Routine construction
    current_routine: Routine | None = Field(
        default=None,
        description="The routine being constructed"
    )
    current_test_parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Test parameters for routine validation"
    )

    # Retry counters
    construction_attempts: int = Field(default=0)
    validation_attempts: int = Field(default=0)

    ## Task management methods

    def add_task(self, task: Task) -> Task:
        """Add a task to the state."""
        self.tasks[task.id] = task
        return task

    def get_pending_tasks(self) -> list[Task]:
        """Get all pending tasks."""
        return [t for t in self.tasks.values() if t.status == "pending"]

    def get_in_progress_tasks(self) -> list[Task]:
        """Get all in-progress tasks."""
        return [t for t in self.tasks.values() if t.status == "in_progress"]

    def get_paused_tasks(self) -> list[Task]:
        """Get all paused tasks (hit max_loops, can resume)."""
        return [t for t in self.tasks.values() if t.status == "paused"]

    def get_completed_tasks(self) -> list[Task]:
        """Get all completed tasks."""
        return [t for t in self.tasks.values() if t.status == "completed"]

    def get_failed_tasks(self) -> list[Task]:
        """Get all failed tasks."""
        return [t for t in self.tasks.values() if t.status == "failed"]

    ## Transaction queue methods (similar to RoutineDiscoveryState)

    def add_to_transaction_queue(self, transaction_id: str) -> tuple[bool, int]:
        """
        Add a transaction to the queue if not already processed.

        Returns:
            Tuple of (was_added, queue_position).
            If already processed, returns (False, -1).
        """
        if transaction_id in self.processed_transaction_ids:
            return False, -1
        if transaction_id in self.queued_transaction_ids:
            return False, self.queued_transaction_ids.index(transaction_id)
        self.queued_transaction_ids.append(transaction_id)
        return True, len(self.queued_transaction_ids) - 1

    def pop_next_transaction(self) -> str | None:
        """Pop the next transaction from the queue."""
        if not self.queued_transaction_ids:
            return None
        return self.queued_transaction_ids.pop(0)

    def mark_transaction_processed(self, transaction_id: str) -> None:
        """Mark a transaction as processed."""
        if transaction_id not in self.processed_transaction_ids:
            self.processed_transaction_ids.append(transaction_id)

    def get_queue_status(self) -> dict[str, Any]:
        """Get a summary of the queue and task status."""
        return {
            "phase": self.phase.value,
            "queued_transactions": len(self.queued_transaction_ids),
            "processed_transactions": len(self.processed_transaction_ids),
            "pending_tasks": len(self.get_pending_tasks()),
            "in_progress_tasks": len(self.get_in_progress_tasks()),
            "paused_tasks": len(self.get_paused_tasks()),
            "completed_tasks": len(self.get_completed_tasks()),
            "failed_tasks": len(self.get_failed_tasks()),
        }

    ## Reset

    def reset(self) -> None:
        """Reset all state for a fresh discovery run."""
        self.phase = SuperDiscoveryPhase.PLANNING
        self.root_transaction_id = None
        self.queued_transaction_ids = []
        self.processed_transaction_ids = []
        self.queued_placeholders = []
        self.processed_placeholders = []
        self.tasks = {}
        self.subagents = {}
        self.current_routine = None
        self.current_test_parameters = {}
        self.construction_attempts = 0
        self.validation_attempts = 0
