"""
bluebox/data_models/orchestration/state.py

State management for the RoutineDiscoveryAgentBeta orchestrator.

Tracks the discovery phase, delegated tasks, subagent instances,
and the routine being constructed.
"""

from typing import Any

from pydantic import BaseModel, Field, ConfigDict

from bluebox.data_models.orchestration.task import Task, SubAgent, TaskStatus


class AgentOrchestrationState(BaseModel):
    """
    Manages state for agent orchestration workflows.

    Tracks the orchestration phase, delegated tasks, subagent instances,
    and the routine being constructed. Unlike RoutineDiscoveryState,
    this orchestrator delegates transaction/placeholder analysis to
    specialist agents rather than doing it directly.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Task and subagent management
    tasks: dict[str, Task] = Field(
        default_factory=dict,
        description="All delegated tasks by ID"
    )
    subagents: dict[str, SubAgent] = Field(
        default_factory=dict,
        description="All subagent instances by ID"
    )

    ## Task management methods

    def add_task(self, task: Task) -> Task:
        """Add a task to the state."""
        self.tasks[task.id] = task
        return task

    def get_pending_tasks(self) -> list[Task]:
        """Get all pending tasks."""
        return [t for t in self.tasks.values() if t.status == TaskStatus.PENDING]

    def get_in_progress_tasks(self) -> list[Task]:
        """Get all in-progress tasks."""
        return [t for t in self.tasks.values() if t.status == TaskStatus.IN_PROGRESS]

    def get_paused_tasks(self) -> list[Task]:
        """Get all paused tasks (hit max_loops, can resume)."""
        return [t for t in self.tasks.values() if t.status == TaskStatus.PAUSED]

    def get_completed_tasks(self) -> list[Task]:
        """Get all completed tasks."""
        return [t for t in self.tasks.values() if t.status == TaskStatus.COMPLETED]

    def get_failed_tasks(self) -> list[Task]:
        """Get all failed tasks."""
        return [t for t in self.tasks.values() if t.status == TaskStatus.FAILED]

    def get_queue_status(self) -> dict[str, Any]:
        """Get a summary of task status for system prompt."""
        return {
            "pending_tasks": len(self.get_pending_tasks()),
            "in_progress_tasks": len(self.get_in_progress_tasks()),
            "paused_tasks": len(self.get_paused_tasks()),
            "completed_tasks": len(self.get_completed_tasks()),
            "failed_tasks": len(self.get_failed_tasks()),
        }

    ## Reset

    def reset(self) -> None:
        """Reset all state for a fresh orchestration run."""
        self.tasks = {}
        self.subagents = {}
