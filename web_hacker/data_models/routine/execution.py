"""
web_hacker/data_models/routine/execution.py

Execution context and result models for routine runs.

Contains:
- OperationExecutionMetadata: Per-operation timing, details, errors
- RoutineExecutionResult: Final result with data, warnings, operation metadata
- RoutineExecutionContext: Mutable state passed to operations (CDP session, parameters)
- FetchExecutionResult: Response data from fetch operations
"""

import re
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field
from websocket import WebSocket

from web_hacker.data_models.routine.endpoint import MimeType


class OperationExecutionMetadata(BaseModel):
    """Metadata collected during a single operation's execution."""
    type: str = Field(description="The operation type (e.g., 'navigate', 'fetch', 'click')")
    duration_seconds: float = Field(description="How long the operation took to execute")
    details: dict = Field(default_factory=dict, description="Operation-specific data")
    error: str | None = Field(default=None, description="Error message from the operation execution.")


class RoutineExecutionResult(BaseModel):
    """
    Result of a routine execution.
    Args:
        ok (bool): Whether the routine execution was successful.
        data (dict | list | str | None): The result of the routine execution.
        placeholder_resolution (dict[str, str | None]): The placeholder resolution of the routine execution.
        warnings (list[str] | None): Warnings from the routine execution.
        error (str | None): Error message from the routine execution.
        is_base64 (bool): Whether the data is base64-encoded binary content (from RoutineDownloadOperation).
        content_type (str | None): MIME type of the data (e.g., 'application/pdf', 'image/png').
        filename (str | None): Suggested filename for the data.
    """
    ok: bool = Field(default=True, description="Whether the routine execution was successful.")
    error: str | None = Field(default=None, description="Error message from the routine execution.")
    warnings: list[str] = Field(default_factory=list, description="Warnings from the routine execution.")
    operations_metadata: list[OperationExecutionMetadata] = Field(default_factory=list, description="Metadata for each operation executed in order")
    placeholder_resolution: dict[str, str | None] = Field(default_factory=dict, description="The placeholder resolution of the routine execution.")
    is_base64: bool = Field(default=False, description="Whether the data is base64-encoded binary content.")
    content_type: MimeType | str | None = Field(default=None, description="MIME type of the data (e.g., 'application/pdf', 'image/png').")
    filename: str | None = Field(default=None, description="Suggested filename for the data.")
    data: dict | list | str | None = Field(default=None, description="The result of the routine execution.")


class RoutineExecutionContext(BaseModel):
    """
    Context passed to operation.execute() containing all necessary state and helpers.

    Operations modify result directly (e.g., result.data, result.placeholder_resolution).
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Required inputs
    session_id: str
    ws: WebSocket | None = None
    send_cmd: Callable
    recv_until: Callable

    # Optional inputs with defaults
    parameters_dict: dict = Field(default_factory=dict)
    timeout: float = 180.0

    # Current page URL (updated by navigate operations, used by fetch to detect blank page)
    current_url: str = Field(default="about:blank", description="Current page URL, updated by navigate operations")

    # Result (operations update this directly)
    result: RoutineExecutionResult = Field(default_factory=RoutineExecutionResult)

    # Current operation metadata (set by execute(), operations can add to details)
    current_operation_metadata: OperationExecutionMetadata | None = None

class FetchExecutionResult(BaseModel):
    """
    Result of a fetch execution.
    """
    ok: bool = Field(description="Whether the fetch execution was successful.")
    result: dict | str | None = Field(default=None, description="The result of the fetch execution.")
    error: str | None = Field(default=None, description="Error message from the fetch execution.")
    resolved_values: dict[str, str | None] = Field(default_factory=dict, description="The placeholder resolution of the fetch execution.")
