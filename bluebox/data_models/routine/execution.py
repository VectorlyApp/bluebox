"""
bluebox/data_models/routine/execution.py

Execution context and result models for routine runs.

Contains:
- OperationExecutionMetadata: Per-operation timing, details, errors
- RoutineExecutionResult: API-facing result with data
- RoutineExecutionResultWithMetadata: Internal result with warnings, operation metadata
- RoutineExecutionContext: Mutable state passed to operations (CDP session, parameters)
- FetchExecutionResult: Response data from fetch operations
"""

import re
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field, model_validator
from websocket import WebSocket

from bluebox.data_models.routine.endpoint import MimeType
from bluebox.utils.data_utils import strip_proxy_credentials


class OperationExecutionMetadata(BaseModel):
    """Metadata collected during a single operation's execution."""
    type: str = Field(description="The operation type (e.g., 'navigate', 'fetch', 'click')")
    duration_seconds: float = Field(description="How long the operation took to execute")
    details: dict = Field(default_factory=dict, description="Operation-specific data")
    error: str | None = Field(default=None, description="Error message from the operation execution.")


class RoutineExecutionResult(BaseModel):
    """
    Result of a routine execution (API-facing).

    Contains only the fields needed by external consumers (server API, SDK clients).
    For internal backend metadata (warnings, operation timing, placeholder resolution),
    see RoutineExecutionResultWithMetadata.
    """
    ok: bool = Field(default=True, description="Whether the routine execution was successful.")
    error: str | None = Field(default=None, description="Error message from the routine execution.")
    is_base64: bool = Field(default=False, description="Whether the data is base64-encoded binary content.")
    content_type: MimeType | str | None = Field(default=None, description="MIME type of the data (e.g., 'application/pdf', 'image/png').")
    filename: str | None = Field(default=None, description="Suggested filename for the data.")
    data: dict | list | str | None = Field(default=None, description="The result of the routine execution.")


class RoutineExecutionResultWithMetadata(RoutineExecutionResult):
    """
    Result of a routine execution with internal metadata.

    Extends RoutineExecutionResult with fields needed by the backend:
    warnings, per-operation metadata, and placeholder resolution details.
    """
    warnings: list[str] = Field(default_factory=list, description="Warnings from the routine execution.")
    operations_metadata: list[OperationExecutionMetadata] = Field(default_factory=list, description="Metadata for each operation executed in order")
    placeholder_resolution: dict[str, str | None] = Field(default_factory=dict, description="The placeholder resolution of the routine execution.")
    proxy_address: str | None = Field(default=None, description="Proxy used for this execution (host:port, credentials and scheme stripped).")

    @model_validator(mode="after")
    def _strip_proxy_credentials(self) -> "RoutineExecutionResultWithMetadata":
        """Strip username/password from proxy_address, keep only scheme://host:port."""
        if self.proxy_address:
            self.proxy_address = strip_proxy_credentials(self.proxy_address)
        return self


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
    param_type_map: dict[str, str] = Field(default_factory=dict, description="Maps parameter names to their type strings for typed coercion")
    timeout: float = 180.0

    # Current page URL (updated by navigate operations, used by fetch to detect blank page)
    current_url: str = Field(default="about:blank", description="Current page URL, updated by navigate operations")

    # Result (operations update this directly)
    result: RoutineExecutionResultWithMetadata = Field(default_factory=RoutineExecutionResultWithMetadata)

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
