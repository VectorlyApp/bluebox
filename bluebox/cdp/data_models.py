"""
bluebox/cdp/data_models.py

Data models for async CDP events.

NOTE: Interaction events use UiInteractionEvent.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, ConfigDict


class BaseCDPEvent(BaseModel):
    """
    Base model for all CDP event details.
    """
    timestamp: int = Field(
        default_factory=lambda: int(datetime.now().timestamp()),
        description="Unix timestamp (seconds) when the event occurred"
    )


class NetworkTransactionEvent(BaseCDPEvent):
    """
    Model for network transaction monitoring events.
    Captures detailed HTTP request and response information.
    """
    model_config = ConfigDict(extra='allow')

    request_id: str = Field(
        ...,
        description="Unique identifier for the network request",
        examples=["interception-job-1.0", "15BF081D76D2923D4AA7E645C41FB876"]
    )
    url: str = Field(
        ...,
        description="The requested URL",
        examples=["https://ycombinator.com/", "https://www.ycombinator.com/"]
    )
    method: str = Field(
        ...,
        description="HTTP method used",
        examples=["GET", "POST", "PUT"]
    )
    type: str | None = Field(
        default=None,
        description="Resource type (Document, Script, Image, etc.)",
        examples=["Document", "Script", "XHR"]
    )
    status: int | None = Field(
        default=None,
        description="HTTP response status code",
        examples=[200, 301, 404, 500]
    )
    status_text: str | None = Field(
        default=None,
        description="HTTP response status text",
        examples=["OK", "Moved Permanently", "Not Found"]
    )
    request_headers: dict[str, str] | None = Field(
        default=None,
        description="HTTP request headers",
    )
    response_headers: dict[str, str] | None = Field(
        default=None,
        description="HTTP response headers",
    )
    post_data: str | None = Field(
        default=None,
        description="Request body data (for POST/PUT requests)",
    )
    response_body: str = Field(
        default="",
        description="Response body content",
    )
    response_body_base64: bool = Field(
        default=False,
        description="Whether response body is base64 encoded",
    )
    mime_type: str = Field(
        default="",
        description="MIME type of the response",
    )
    # Error fields for failed requests
    errorText: str | None = Field(
        default=None,
        description="Error text if the request failed",
    )
    failed: bool = Field(
        default=False,
        description="Whether the request failed",
    )


class StorageEvent(BaseCDPEvent):
    """
    Model for browser storage monitoring events.
    Handles cookies, localStorage, sessionStorage, and IndexedDB events.
    """
    model_config = ConfigDict(extra='allow')

    type: str = Field(
        ...,
        description="Type of storage event",
        examples=[
            "initialCookies", "cookieChange",
            "localStorageCleared", "localStorageItemAdded", "sessionStorageItemAdded",
            "indexedDBEvent",
        ]
    )
    source: str | None = Field(
        default=None,
        description="Source of the event (for cookie events)",
    )
    origin: str | None = Field(
        default=None,
        description="Origin/domain for storage events",
    )
    # Cookie-specific fields
    count: int | None = Field(
        default=None,
        description="Number of cookies (for initialCookies events)",
    )
    cookies: list[dict[str, Any]] | None = Field(
        default=None,
        description="List of cookies (for initialCookies events)",
    )
    triggered_by: str | None = Field(
        default=None,
        description="What triggered the cookie change",
    )
    added: list[dict[str, Any]] | None = Field(
        default=None,
        description="Added cookies/items (for change events)",
    )
    modified: list[dict[str, Any]] | None = Field(
        default=None,
        description="Modified cookies/items (for change events)",
    )
    removed: list[dict[str, Any]] | None = Field(
        default=None,
        description="Removed cookies/items (for change events)",
    )
    total_count: int | None = Field(
        default=None,
        description="Total count of items after change",
    )
    # Storage item fields
    key: str | None = Field(
        default=None,
        description="Storage key",
    )
    value: Any | None = Field(
        default=None,
        description="Storage value (for add/update events)",
    )
    old_value: Any | None = Field(
        default=None,
        description="Previous storage value (for update events)",
    )
    new_value: Any | None = Field(
        default=None,
        description="New storage value (for update events)",
    )
    # IndexedDB fields
    params: dict[str, Any] | None = Field(
        default=None,
        description="Raw IndexedDB event parameters",
    )


class WindowPropertyChange(BaseModel):
    """
    Model for a single window property change.
    """
    path: str = Field(
        ...,
        description="The property path that changed",
    )
    value: Any | None = Field(
        ...,
        description="The new value of the property (None for deletions)",
    )
    change_type: str = Field(
        ...,
        description="Type of change: 'added', 'changed', or 'deleted'",
    )


class WindowPropertyEvent(BaseCDPEvent):
    """
    Model for window property monitoring events.
    Emitted when window properties are collected and changes are detected.
    """
    url: str = Field(
        ...,
        description="The current page URL",
    )
    changes: list[WindowPropertyChange] = Field(
        ...,
        description="List of property changes detected",
    )
    total_keys: int = Field(
        ...,
        description="Total number of unique properties tracked in history"
    )
