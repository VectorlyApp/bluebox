"""
bluebox/data_models/browser_agent.py

Data models for browser agent SSE stream events.

Mirrors the event shapes produced by the server's buagent subapp
(servers/src/data_models/buagent.py + subapp enrichment).
"""

from pydantic import BaseModel, Field


class BrowserAgentStepEvent(BaseModel):
    """SSE 'step' event payload."""
    step_number: int = Field(description="Current step number (1-based)")
    next_goal: str | None = Field(default=None, description="Agent's stated goal for the next action")
    is_done: bool = Field(default=False, description="Whether the agent marked the task as done on this step")


class BrowserAgentDoneEvent(BaseModel):
    """SSE 'done' event payload (enriched by the subapp with execution metadata)."""
    is_done: bool = Field(description="Whether the agent marked the task as done")
    is_successful: bool | None = Field(default=None, description="Whether the agent self-reported success")
    final_result: str | None = Field(default=None, description="Agent's final extracted result text")
    errors: list[str] = Field(default_factory=list, description="Errors encountered during execution")
    n_steps: int = Field(default=0, description="Number of steps executed")
    execution_id: str | None = Field(default=None, description="Unique execution identifier")
    duration_seconds: float | None = Field(default=None, description="Total execution duration in seconds")


class BrowserAgentErrorEvent(BaseModel):
    """SSE 'error' event payload."""
    error: str = Field(description="Error message")
    execution_id: str | None = Field(default=None, description="Unique execution identifier")
