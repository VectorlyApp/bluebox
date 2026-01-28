"""
bluebox/data_models/cdp.py

Data models for async CDP events.
"""

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, ConfigDict

from bluebox.data_models.ui_elements import UIElement


## Base event

class BaseCDPEvent(BaseModel):
    """
    Base model for all CDP event details.
    """
    timestamp: float = Field(
        default_factory=lambda: datetime.now(tz=timezone.utc).timestamp(),
        description="Unix timestamp (seconds) when the event occurred"
    )

## Network models

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
    post_data: str | dict | list | None = Field(
        default=None,
        description="Request body data (for POST/PUT requests). May be parsed JSON (dict/list) or raw string.",
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

## Storage models

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

## Window property models

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

## UI interaction models

class InteractionType(StrEnum):
    """Types of UI interactions that match real DOM event names."""

    # Mouse events
    CLICK = "click"
    MOUSEDOWN = "mousedown"
    MOUSEUP = "mouseup"
    DBLCLICK = "dblclick"
    CONTEXTMENU = "contextmenu"
    MOUSEOVER = "mouseover"

    # Keyboard events
    KEYDOWN = "keydown"
    KEYUP = "keyup"
    KEYPRESS = "keypress"  # deprecated but still emitted by browsers

    # Form events
    INPUT = "input"
    CHANGE = "change"

    # Focus events
    FOCUS = "focus"
    BLUR = "blur"


class Interaction(BaseModel):
    """
    Details about how an interaction occurred.
    
    Contains browser event properties like mouse coordinates, keyboard keys,
    and modifier keys. These details provide the "how" of an interaction,
    while InteractionType provides the "what".
    """
    # Mouse properties
    mouse_button: int | None = Field(
        default=None,
        description="Mouse button pressed (0=left, 1=middle, 2=right). None for non-mouse interactions."
    )
    mouse_x_viewport: int | None = Field(
        default=None,
        description="X coordinate relative to viewport. None for non-mouse interactions."
    )
    mouse_y_viewport: int | None = Field(
        default=None,
        description="Y coordinate relative to viewport. None for non-mouse interactions."
    )
    mouse_x_page: int | None = Field(
        default=None,
        description="X coordinate relative to page (includes scroll). None for non-mouse interactions."
    )
    mouse_y_page: int | None = Field(
        default=None,
        description="Y coordinate relative to page (includes scroll). None for non-mouse interactions."
    )

    # Keyboard properties
    key_value: str | None = Field(
        default=None,
        description="The key value pressed (e.g., 'a', 'Enter', 'Shift'). None for non-keyboard interactions."
    )
    key_code: str | None = Field(
        default=None,
        description="The physical key code (e.g., 'KeyA', 'Enter', 'ShiftLeft'). None for non-keyboard interactions."
    )
    key_code_deprecated: int | None = Field(
        default=None,
        description="Deprecated numeric key code. None for non-keyboard interactions."
    )
    key_which_deprecated: int | None = Field(
        default=None,
        description="Deprecated numeric key code. None for non-keyboard interactions."
    )

    # Modifier keys (apply to both mouse and keyboard interactions)
    ctrl_pressed: bool = Field(
        default=False,
        description="Whether the Ctrl key was pressed during the interaction."
    )
    shift_pressed: bool = Field(
        default=False,
        description="Whether the Shift key was pressed during the interaction."
    )
    alt_pressed: bool = Field(
        default=False,
        description="Whether the Alt key was pressed during the interaction."
    )
    meta_pressed: bool = Field(
        default=False,
        description="Whether the Meta/Cmd key was pressed during the interaction."
    )


class UIInteractionEvent(BaseCDPEvent):
    """
    Complete UI interaction event record.
    
    Represents a single user interaction with a web element, including:
    - What type of interaction occurred
    - When it occurred (timestamp)
    - What element was interacted with (UIElement)
    - How it occurred (Interaction) - mouse position, keys pressed, modifiers, etc.
    - Page context (URL)
    """
    # Interaction type
    type: InteractionType

    # How the interaction occurred (mouse coordinates, keyboard keys, modifiers, etc.)
    interaction: Interaction | None = Field(
        default=None,
        description="Details about how the interaction occurred (mouse position, keys pressed, modifiers, etc.)."
    )

    # Element that was interacted with
    element: UIElement

    # Page context
    url: str = Field(
        description="URL of the page where the interaction occurred."
    )
