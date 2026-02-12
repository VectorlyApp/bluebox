"""
bluebox/data_models/browser_agent.py

Data models for browser agent SSE stream events.

Mirrors the SSE event models from the server 
using a discriminated union on the ``type`` field 
for robust parsing.
"""

from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field, TypeAdapter


class BrowserAgentSSEEvent(BaseModel):
    """Base class for all browser agent SSE events."""
    type: str


class BrowserAgentStepEvent(BrowserAgentSSEEvent):
    """SSE 'step' event — emitted after each agent step."""
    type: Literal["step"] = "step"
    step_number: int = Field(description="Current step number (1-based)")
    next_goal: str | None = Field(default=None, description="Agent's stated goal for the next action")
    is_done: bool = Field(default=False, description="Whether the agent marked the task as done on this step")


class BrowserAgentDoneEvent(BrowserAgentSSEEvent):
    """SSE 'done' event — emitted when the agent finishes."""
    type: Literal["done"] = "done"
    is_done: bool = Field(description="Whether the agent marked the task as done")
    is_successful: bool | None = Field(default=None, description="Whether the agent self-reported success")
    final_result: str | None = Field(default=None, description="Agent's final extracted result text")
    errors: list[str] = Field(default_factory=list, description="Errors encountered during execution")
    n_steps: int = Field(default=0, description="Number of steps executed")
    execution_id: str | None = Field(default=None, description="Unique execution identifier")
    duration_seconds: float | None = Field(default=None, description="Total execution duration in seconds")


class BrowserAgentErrorEvent(BrowserAgentSSEEvent):
    """SSE 'error' event — emitted on failure."""
    type: Literal["error"] = "error"
    error: str = Field(description="Error message")
    execution_id: str | None = Field(default=None, description="Unique execution identifier")


BrowserAgentSSEEventUnion = Annotated[
    Union[BrowserAgentStepEvent, BrowserAgentDoneEvent, BrowserAgentErrorEvent],
    Field(discriminator="type"),
]

sse_event_adapter: TypeAdapter[BrowserAgentSSEEventUnion] = TypeAdapter(BrowserAgentSSEEventUnion)
