
"""
web_hacker/data_models/ui_elements.py

UI element data models for robust element identification and replay.
"""

from enum import StrEnum
from typing import Dict, List
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)


class SelectorType(StrEnum):
    CSS = "css"
    XPATH = "xpath"
    TEXT = "text"          # e.g. "button with label X"
    ROLE = "role"          # e.g. role+name/aria-label
    NAME = "name"          # input[name="..."]
    ID = "id"              # #id


# Default priority mapping for selector types (lower = higher priority)
DEFAULT_SELECTOR_PRIORITIES: Dict[SelectorType, int] = {
    SelectorType.ID: 10,           # Highest priority - IDs are unique
    SelectorType.NAME: 20,         # Form controls by name are very stable
    SelectorType.CSS: 30,          # CSS selectors (with stable attributes)
    SelectorType.ROLE: 40,         # ARIA roles + labels
    SelectorType.TEXT: 50,         # Text-based matching
    SelectorType.XPATH: 80,        # XPath (often brittle, last resort)
}


class Selector(BaseModel):
    """
    A single way to locate an element.
    `value` is the raw string (CSS, XPath, etc.)
    `type` tells the executor how to interpret it.
    `priority` controls which selector to try first (lower = higher priority).
    If not specified, uses the default priority for the selector type.
    """
    type: SelectorType
    value: str
    priority: int | None = Field(
        default=None,
        description="Priority for this selector (lower = higher priority). If None, uses default for selector type.",
    )

    description: str | None = Field(
        default=None,
        description="Human readable note (e.g. 'primary stable selector').",
    )
    
    def get_priority(self) -> int:
        """Get the effective priority, using default if not set."""
        if self.priority is not None:
            return self.priority
        return DEFAULT_SELECTOR_PRIORITIES.get(self.type, 100)


class BoundingBox(BaseModel):
    x: float
    y: float
    width: float
    height: float


class UiElement(BaseModel):
    """
    Unified description of a UI element sufficient for robust replay.

    - Raw DOM data (tag, attributes, text)
    - Multiple selectors (CSS, XPath, text-based, etc.)
    - Context (URL, frame)
    """
    # Context
    url: str | None = Field(
        default=None,
        description="Page URL where this element was observed.",
    )

    # Core DOM identity
    tag_name: str
    id: str | None = None
    name: str | None = None
    class_names: List[str] | None = Field(default=None, description="List of CSS class names.")

    # Common attributes
    type_attr: str | None = Field(default=None, description="Input type, button type, etc.")
    role: str | None = None
    aria_label: str | None = None
    placeholder: str | None = None
    title: str | None = None
    href: str | None = None
    src: str | None = None
    value: str | None = None

    # Full attribute map for anything else (data-*, etc.)
    attributes: Dict[str, str] | None = Field(
        default=None,
        description="All raw attributes from the DOM element.",
    )

    # Content
    text: str | None = Field(
        default=None,
        description="Trimmed inner text (useful for text-based selectors).",
    )

    # Geometry
    bounding_box: BoundingBox | None = None

    # Locators (multiple ways to find it again)
    selectors: List[Selector] | None = Field(
        default=None,
        description="Ordered list of selectors to try when locating this element.",
    )

    # Convenience accessors for most common selectors
    css_path: str | None = None    # from getElementPath
    xpath: str | None = None       # full xpath

    def build_default_selectors(self) -> None:
        """
        Populate `selectors` from known fields if empty.
        Call this once after constructing from raw DOM.
        """
        if self.selectors is None:
            self.selectors = []
        elif self.selectors:
            return
        
        # Ensure attributes is a dict for easier access
        if self.attributes is None:
            self.attributes = {}
        
        # Ensure class_names is a list
        if self.class_names is None:
            self.class_names = []

        # Highest priority: ID (uses default priority from DEFAULT_SELECTOR_PRIORITIES)
        if self.id:
            self.selectors.append(
                Selector(
                    type=SelectorType.ID,
                    value=self.id,
                    priority=DEFAULT_SELECTOR_PRIORITIES[SelectorType.ID],
                    description="Locate by DOM id",
                )
            )

        # Name attribute - if it exists, use it
        if self.name:
            self.selectors.append(
                Selector(
                    type=SelectorType.NAME,
                    value=self.name,
                    priority=DEFAULT_SELECTOR_PRIORITIES[SelectorType.NAME],
                    description="Locate by name attribute",
                )
            )

        # Placeholder attribute - if it exists, use it
        if self.placeholder:
            self.selectors.append(
                Selector(
                    type=SelectorType.CSS,
                    value=f'{self.tag_name.lower()}[placeholder="{self.placeholder}"]',
                    priority=DEFAULT_SELECTOR_PRIORITIES[SelectorType.CSS],
                    description="Locate by placeholder",
                )
            )

        # Role - if it exists, use it
        if self.role:
            self.selectors.append(
                Selector(
                    type=SelectorType.ROLE,
                    value=self.role,
                    priority=DEFAULT_SELECTOR_PRIORITIES[SelectorType.ROLE],
                    description=f"Locate by role={self.role}",
                )
            )

        # Text - if it exists, use it
        if self.text:
            snippet = self.text.strip()
            if snippet:
                self.selectors.append(
                    Selector(
                        type=SelectorType.TEXT,
                        value=snippet,
                        priority=DEFAULT_SELECTOR_PRIORITIES[SelectorType.TEXT],
                        description="Locate by text content",
                    )
                )

        # Direct CSS and XPath if we have them
        if self.css_path:
            self.selectors.append(
                Selector(
                    type=SelectorType.CSS,
                    value=self.css_path,
                    priority=DEFAULT_SELECTOR_PRIORITIES[SelectorType.CSS],
                    description="Recorded CSS path",
                )
            )
        if self.xpath:
            self.selectors.append(
                Selector(
                    type=SelectorType.XPATH,
                    value=self.xpath,
                    priority=DEFAULT_SELECTOR_PRIORITIES[SelectorType.XPATH],
                    description="Full XPath (last resort)",
                )
            )

        # Fallback: first stable-looking class
        if not self.selectors and self.class_names:
            
            # Filter out classes that are likely to be unstable
            stable_classes = [
                c for c in self.class_names
                if not c.startswith("sc-")
                and not c.startswith("css-")
                and (not c.isalnum() or len(c) < 10)
            ]
            
            # If there are stable classes, use the first one
            if stable_classes:
                cls = stable_classes[0]
                self.selectors.append(
                    Selector(
                        type=SelectorType.CSS,
                        value=f".{cls}",
                        priority=DEFAULT_SELECTOR_PRIORITIES[SelectorType.CSS],
                        description="Fallback by single stable-looking class",
                    )
                )
                
        if not self.selectors: 
            logger.warning("No selectors found for element %s", self.model_dump_json())
                