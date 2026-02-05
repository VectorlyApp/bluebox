"""
bluebox/data_models/orchestration/task.py

Task and SubAgent models for the SuperDiscoveryAgent orchestrator.

Tasks represent units of work delegated to specialist subagents.
SubAgents are persistent specialist instances that can handle multiple tasks.
"""

import secrets
import string
from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


# Short ID alphabet: alphanumeric, no ambiguous chars (0/O, 1/l/I)
_SHORT_ID_ALPHABET = string.ascii_lowercase + string.digits
_SHORT_ID_ALPHABET = _SHORT_ID_ALPHABET.replace("0", "").replace("o", "").replace("l", "").replace("1", "")


def generate_short_id(length: int = 6) -> str:
    """
    Generate a short, LLM-friendly ID.

    Uses lowercase alphanumeric chars (excluding ambiguous 0/o/l/1).
    6 chars = ~2 billion combinations, sufficient for task tracking.

    Args:
        length: Number of characters (default 6).

    Returns:
        Short alphanumeric ID like "k7x9m2".
    """
    return "".join(secrets.choice(_SHORT_ID_ALPHABET) for _ in range(length))


class TaskStatus(StrEnum):
    """Status of a delegated task."""
    PENDING = "pending"           # Waiting to be picked up
    IN_PROGRESS = "in_progress"   # Currently being processed by a subagent
    PAUSED = "paused"             # Hit max_loops, can resume later
    COMPLETED = "completed"       # Successfully completed
    FAILED = "failed"             # Failed with an error


class SpecialistAgentType(StrEnum):
    """Types of specialist agents available for task delegation."""
    JS_SPECIALIST = "js_specialist"
    NETWORK_SPECIALIST = "network_specialist"
    VALUE_TRACE_RESOLVER = "value_trace_resolver"
    INTERACTION_SPECIALIST = "interaction_specialist"


class Task(BaseModel):
    """
    A unit of work delegated to a specialist subagent.

    Tasks track the full lifecycle from creation through completion,
    including pause/resume support for long-running operations.

    Note: agent_type values must match AbstractSpecialist.AGENT_TYPE on specialist classes.
    Use AbstractSpecialist.get_all_agent_types() for runtime discovery of valid types.
    """
    id: str = Field(default_factory=generate_short_id)
    agent_type: SpecialistAgentType = Field(description="Type of specialist to handle this task")
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

    # Output schema (orchestrator-defined)
    output_schema: dict[str, Any] | None = Field(
        default=None,
        description="JSON Schema defining the expected output structure from the specialist"
    )
    output_description: str | None = Field(
        default=None,
        description="Human-readable description of what output the specialist should return"
    )


class SubAgent(BaseModel):
    """
    A persistent specialist agent instance that can handle multiple tasks.

    SubAgents maintain state across tasks, allowing conversation history
    and learned context to persist.
    """
    id: str = Field(default_factory=generate_short_id)
    type: SpecialistAgentType = Field(description="The type of specialist this agent is")
    llm_model: str = Field(description="The LLM model used by this agent")
    task_ids: list[str] = Field(
        default_factory=list,
        description="IDs of tasks this agent has handled"
    )
    created_at: datetime = Field(default_factory=datetime.now)
