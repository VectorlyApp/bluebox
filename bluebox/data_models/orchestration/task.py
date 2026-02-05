"""
bluebox/data_models/orchestration/task.py

Task and SubAgent models for the SuperDiscoveryAgent orchestrator.

Tasks represent units of work delegated to specialist subagents.
SubAgents are persistent specialist instances that can handle multiple tasks.
"""

from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class TaskStatus(StrEnum):
    """Status of a delegated task."""
    PENDING = "pending"           # Waiting to be picked up
    IN_PROGRESS = "in_progress"   # Currently being processed by a subagent
    PAUSED = "paused"             # Hit max_loops, can resume later
    COMPLETED = "completed"       # Successfully completed
    FAILED = "failed"             # Failed with an error


class AgentType(StrEnum):
    """Types of specialist agents available for task delegation."""
    JS_SPECIALIST = "js_specialist"
    NETWORK_SPY = "network_spy"
    TRACE_HOUND = "trace_hound"
    INTERACTION_SPECIALIST = "interaction_specialist"
    DOCS_DIGGER = "docs_digger"


class Task(BaseModel):
    """
    A unit of work delegated to a specialist subagent.

    Tasks track the full lifecycle from creation through completion,
    including pause/resume support for long-running operations.
    """
    id: str = Field(default_factory=lambda: str(uuid4()))
    agent_type: AgentType = Field(description="Type of specialist to handle this task")
    agent_id: str | None = Field(
        default=None,
        description="ID of the subagent to use. None means create new instance."
    )
    prompt: str = Field(description="Task prompt/instructions for the specialist")
    status: TaskStatus = Field(default=TaskStatus.PENDING)
    result: Any = Field(default=None, description="Task result on completion")
    error: str | None = Field(default=None, description="Error message if failed")

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now)
    started_at: datetime | None = Field(default=None)
    completed_at: datetime | None = Field(default=None)

    # Iteration management for pause/resume
    max_loops: int = Field(
        default=5,
        description="Max LLM iterations before pausing/returning"
    )
    loops_used: int = Field(
        default=0,
        description="Total loops used across all runs of this task"
    )

    # Additional context for the specialist
    context: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context data for the specialist"
    )


class SubAgent(BaseModel):
    """
    A persistent specialist agent instance that can handle multiple tasks.

    SubAgents maintain state across tasks, allowing conversation history
    and learned context to persist.
    """
    id: str = Field(default_factory=lambda: str(uuid4()))
    type: AgentType = Field(description="The type of specialist this agent is")
    llm_model: str = Field(description="The LLM model used by this agent")
    task_ids: list[str] = Field(
        default_factory=list,
        description="IDs of tasks this agent has handled"
    )
    created_at: datetime = Field(default_factory=datetime.now)
