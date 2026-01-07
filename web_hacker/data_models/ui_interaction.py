"""
web_hacker/data_models/monitor/ui_interactions.py

UI interaction data models for tracking user interactions with web elements.
"""

from enum import StrEnum
from typing import Optional
from pydantic import BaseModel, Field

from web_hacker.data_models.ui_elements import UiElement


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
    KEYPRESS = "keypress"  # Deprecated but still emitted by browsers

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
    mouse_button: Optional[int] = Field(
        default=None,
        description="Mouse button pressed (0=left, 1=middle, 2=right). None for non-mouse interactions."
    )
    mouse_x_viewport: Optional[int] = Field(
        default=None,
        description="X coordinate relative to viewport. None for non-mouse interactions."
    )
    mouse_y_viewport: Optional[int] = Field(
        default=None,
        description="Y coordinate relative to viewport. None for non-mouse interactions."
    )
    mouse_x_page: Optional[int] = Field(
        default=None,
        description="X coordinate relative to page (includes scroll). None for non-mouse interactions."
    )
    mouse_y_page: Optional[int] = Field(
        default=None,
        description="Y coordinate relative to page (includes scroll). None for non-mouse interactions."
    )
    
    # Keyboard properties
    key_value: Optional[str] = Field(
        default=None,
        description="The key value pressed (e.g., 'a', 'Enter', 'Shift'). None for non-keyboard interactions."
    )
    key_code: Optional[str] = Field(
        default=None,
        description="The physical key code (e.g., 'KeyA', 'Enter', 'ShiftLeft'). None for non-keyboard interactions."
    )
    key_code_deprecated: Optional[int] = Field(
        default=None,
        description="Deprecated numeric key code. None for non-keyboard interactions."
    )
    key_which_deprecated: Optional[int] = Field(
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


class UiInteractionEvent(BaseModel):
    """
    Complete UI interaction event record.
    
    Represents a single user interaction with a web element, including:
    - What type of interaction occurred
    - When it occurred (timestamp)
    - What element was interacted with (UiElement)
    - How it occurred (Interaction) - mouse position, keys pressed, modifiers, etc.
    - Page context (URL)
    """
    # Interaction type
    type: InteractionType
    
    # Timestamp
    timestamp: int = Field(
        description="Client-side timestamp (milliseconds since epoch) when the interaction occurred."
    )
    
    # How the interaction occurred (mouse coordinates, keyboard keys, modifiers, etc.)
    interaction: Interaction | None = Field(
        default=None,
        description="Details about how the interaction occurred (mouse position, keys pressed, modifiers, etc.)."
    )
    
    # Element that was interacted with
    element: UiElement
    
    # Page context
    url: str = Field(
        description="URL of the page where the interaction occurred."
    )

