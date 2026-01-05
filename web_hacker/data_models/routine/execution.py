"""
web_hacker/data_models/routine/execution.py

Execution-related data models for routines.
"""

import re
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field
from websocket import WebSocket

from web_hacker.data_models.routine.endpoint import MimeType


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

class FetchExecutionResult(BaseModel):
    """
    Result of a fetch execution.
    """
    ok: bool = Field(description="Whether the fetch execution was successful.")
    result: dict | str | None = Field(default=None, description="The result of the fetch execution.")
    error: str | None = Field(default=None, description="Error message from the fetch execution.")
    resolved_values: dict[str, str | None] = Field(default_factory=dict, description="The placeholder resolution of the fetch execution.")

class RoutineExecutionState: 
    # NOTE: THIS IS A LEGACY CLASS USED BY BROWSER_HANDLER
    # NOTE: servers repo still uses BROWSER_HANDLER so we cannot remove it yet
    """
    Per-routine execution state for network interception tracking.
    Holds compiled regex patterns and captured request/response data keyed by requestId.
    """

    def __init__(self) -> None:
        self.active: bool = False
        self.patterns: list[tuple[re.Pattern[str], str]] = []  # (compiled_regex, session_storage_key)
        self.requests: dict[str, dict[str, Any]] = {}
        self.matches: dict[str, str] = {}
        self.responses: dict[str, dict[str, Any]] = {}

