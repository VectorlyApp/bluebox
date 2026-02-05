"""
bluebox/data_models/orchestration/result.py

Universal result wrapper for specialist agents.

SpecialistResultWrapper is a standardized container for all specialist outputs,
allowing the orchestrator (SuperDiscoveryAgent) to dynamically define what
structure a specialist should return.
"""

from typing import Any

from pydantic import BaseModel, Field


class SpecialistResultWrapper(BaseModel):
    """
    Universal wrapper for specialist results.

    The orchestrator defines the expected output schema at task creation time.
    The specialist returns data matching that schema, wrapped in this container
    which adds metadata like success status, notes, and failure reasons.
    """

    output: dict[str, Any] | None = Field(
        default=None,
        description="The actual result matching the orchestrator's expected schema",
    )
    success: bool = Field(
        default=True,
        description="Whether the specialist completed successfully",
    )
    notes: list[str] = Field(
        default_factory=list,
        description="Notes, complaints, warnings, or errors encountered during execution",
    )
    failure_reason: str | None = Field(
        default=None,
        description="If success=False, explains why the task failed",
    )
