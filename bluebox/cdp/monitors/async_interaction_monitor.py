"""
bluebox/cdp/monitors/async_interaction_monitor.py

Async interaction monitor for CDP.
Tracks mouse clicks, keyboard events, and element details via JavaScript injection.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from bluebox.cdp.monitors.abstract_async_monitor import AbstractAsyncMonitor
from bluebox.data_models.ui_elements import UIElement, BoundingBox
from bluebox.data_models.cdp import UIInteractionEvent, InteractionType, Interaction
from bluebox.utils.logger import get_logger

if TYPE_CHECKING:
    from bluebox.cdp.async_cdp_session import AsyncCDPSession
    from bluebox.data_models.cdp import BaseCDPEvent

logger = get_logger(name=__name__)


class AsyncInteractionMonitor(AbstractAsyncMonitor):
    """
    Async interaction monitor for CDP.
    Tracks mouse clicks, keyboard events, and element details via JavaScript injection.
    """

    # Class constant
    BINDING_NAME = "__webHackerInteractionLog"

    # Class methods ______________________________________________________________________________________________

    @classmethod
    def get_ws_event_summary(cls, detail: dict[str, Any]) -> dict[str, Any]:
        """Extract lightweight summary for WebSocket streaming."""
        return {
            "type": cls.get_monitor_category(),
            "interaction_type": detail.get("type"),
            "element_tag": detail.get("element", {}).get("tag_name") if detail.get("element") else None,
            "url": detail.get("url"),
        }

    # Static methods _____________________________________________________________________________________________

    @staticmethod
    def consolidate_interactions(
        interaction_events_path: str,
        output_path: str | None = None,
    ) -> dict[str, Any]:
        """
        Consolidate all interactions from JSONL file into a single JSON structure.

        Args:
            interaction_events_path: Path to interaction events JSONL file.
            output_path: Optional path to write consolidated JSON file.

        Returns:
            Dict with structure:
            {
                "interactions": [...],
                "summary": {
                    "total": 123,
                    "by_type": {...},
                    "by_url": {...}
                }
            }
        """
        if not os.path.exists(interaction_events_path):
            return {"interactions": [], "summary": {"total": 0, "by_type": {}, "by_url": {}}}

        interactions = []
        by_type: dict[str, int] = defaultdict(int)
        by_url: dict[str, int] = defaultdict(int)

        try:
            with open(interaction_events_path, mode="r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        interaction = json.loads(line)
                        interactions.append(interaction)

                        # Update statistics
                        interaction_type = interaction.get("type", "unknown")
                        by_type[interaction_type] += 1
                        url = interaction.get("url", "unknown")
                        by_url[url] += 1
                    except json.JSONDecodeError as e:
                        logger.warning("Failed to parse interaction line: %s", e)
                        continue
        except Exception as e:
            logger.error("Failed to read interaction log: %s", e)
            return {"interactions": [], "summary": {"total": 0, "by_type": {}, "by_url": {}}}

        consolidated = {
            "interactions": interactions,
            "summary": {
                "total": len(interactions),
                "by_type": dict(by_type),
                "by_url": dict(by_url),
            },
        }

        # Save to file if output path provided
        if output_path:
            try:
                with open(output_path, mode="w", encoding="utf-8") as f:
                    json.dump(consolidated, f, indent=2, ensure_ascii=False)
                logger.info("Consolidated interactions saved to: %s", output_path)
            except Exception as e:
                logger.error("Failed to save consolidated interactions: %s", e)

        return consolidated


    # Magic methods _____________________________________________________________________________________________

    def __init__(self, event_callback_fn: Callable[[str, BaseCDPEvent], Awaitable[None]]) -> None:
        """
        Initialize AsyncInteractionMonitor.
        Args:
            event_callback_fn: Async callback for emitting events.
        """
        self.event_callback_fn = event_callback_fn

        # Statistics tracking
        self.interaction_count: int = 0
        self.interaction_types: dict[str, int] = defaultdict(int)
        self.interactions_by_url: dict[str, int] = defaultdict(int)

        # Pending DOM commands (for element enrichment if needed)
        self.pending_dom_commands: dict[int, dict[str, Any]] = {}


    # Private methods ___________________________________________________________________________________________

    async def _inject_interaction_listeners(self, cdp_session: AsyncCDPSession) -> None:
        """Inject JavaScript listeners for mouse and keyboard events."""
        interaction_script = f"""
(function() {{
    'use strict';

    const bindingName = '{self.BINDING_NAME}';

    // Wait for binding to be available (with timeout)
    function waitForBinding(callback, maxWait = 1000) {{
        const startTime = Date.now();
        function check() {{
            if (typeof window[bindingName] === 'function') {{
                callback();
            }} else if (Date.now() - startTime < maxWait) {{
                setTimeout(check, 50);
            }} else {{
                console.warn('Web Hacker interaction binding not available after timeout');
            }}
        }}
        check();
    }}

    // Helper function to get element details (UIElement format)
    function getElementDetails(element) {{
        if (!element) return null;

        // Collect all attributes
        const attributes = {{}};
        if (element.attributes) {{
            for (let i = 0; i < element.attributes.length; i++) {{
                const attr = element.attributes[i];
                attributes[attr.name] = attr.value;
            }}
        }}

        // Parse class names into array
        const classNames = element.className && typeof element.className === 'string'
            ? element.className.split(/\\s+/).filter(c => c)
            : [];

        const details = {{
            tag_name: (element.tagName || '').toLowerCase(),
            id: element.id || null,
            name: element.name || null,
            class_names: classNames.length > 0 ? classNames : null,
            type_attr: element.type || null,
            role: element.getAttribute('role') || null,
            aria_label: element.getAttribute('aria-label') || null,
            placeholder: element.placeholder || null,
            title: element.title || null,
            href: element.href || null,
            src: element.src || null,
            value: element.value || null,
            text: element.textContent ? element.textContent.trim().substring(0, 200) : null,
            attributes: Object.keys(attributes).length > 0 ? attributes : null,
        }};

        // Improved selector generation
        function getElementPath(el) {{
            if (!el || el.nodeType !== 1) return '';
            const path = [];
            let current = el;

            while (current && current.nodeType === 1) {{
                let selector = current.tagName.toLowerCase();

                // 1. ID is gold standard
                if (current.id) {{
                    selector += '#' + current.id;
                    path.unshift(selector);
                    break; // ID is usually unique enough
                }}

                // 2. Stable attributes
                const stableAttrs = ['name', 'data-testid', 'data-test-id', 'data-cy', 'role', 'placeholder', 'aria-label', 'title'];
                let foundStable = false;
                for (const attr of stableAttrs) {{
                    const val = current.getAttribute(attr);
                    if (val) {{
                        selector += `[${{attr}}="${{val.replace(/"/g, '\\"')}}"]`;
                        foundStable = true;
                        break;
                    }}
                }}

                // 3. Classes (careful filtering)
                if (!foundStable && current.className && typeof current.className === 'string') {{
                    // Filter out likely generated classes
                    const classes = current.className.split(/\\s+/)
                        .filter(c => c)
                        .filter(c => !c.startsWith('sc-')) // Styled Components
                        .filter(c => !c.match(/^[a-zA-Z0-9]{{10,}}$/)) // Long random strings
                        .filter(c => !c.match(/css-/)); // Emotion/CSS-in-JS

                    if (classes.length > 0) {{
                        selector += '.' + classes.join('.');
                    }}
                }}

                // 4. Nth-child fallback if no unique traits
                if (!foundStable && !current.id) {{
                    let sibling = current;
                    let index = 1;
                    while (sibling = sibling.previousElementSibling) {{
                        if (sibling.tagName === current.tagName) index++;
                    }}
                    if (index > 1) selector += `:nth-of-type(${{index}})`;
                }}

                path.unshift(selector);
                current = current.parentElement;
                if (path.length > 5) break; // Limit depth
            }}
            return path.join(' > ');
        }}

        details.css_path = getElementPath(element);

        // Get XPath (Full structural path like /html/body/div[1]/input[1])
        function getXPath(el) {{
            if (!el || el.nodeType !== 1) return '';

            const parts = [];
            while (el && el.nodeType === 1) {{
                let part = el.tagName.toLowerCase();

                // Count all previous siblings with the same tag name (1-based indexing)
                let index = 1;
                let sibling = el.previousElementSibling;
                while (sibling) {{
                    if (sibling.nodeType === 1 && sibling.tagName === el.tagName) {{
                        index++;
                    }}
                    sibling = sibling.previousElementSibling;
                }}

                // Always include index (XPath is 1-based)
                part += `[${{index}}]`;
                parts.unshift(part);

                el = el.parentElement;
            }}
            return '/' + parts.join('/');
        }}

        details.xpath = getXPath(element);
        details.url = window.location.href;

        // Get bounding box
        try {{
            const rect = element.getBoundingClientRect();
            details.bounding_box = {{
                x: rect.x,
                y: rect.y,
                width: rect.width,
                height: rect.height
            }};
        }} catch (e) {{
            details.bounding_box = null;
        }}

        return details;
    }}

    // Helper function to log interaction
    function logInteraction(type, event, element) {{
        const details = getElementDetails(element);
        const data = {{
            type: type,
            timestamp: Date.now(),
            event: {{
                mouse_button: event.button !== undefined ? event.button : null,
                key_value: event.key || null,
                key_code: event.code || null,
                key_code_deprecated: event.keyCode || null,
                key_which_deprecated: event.which || null,
                ctrl_pressed: event.ctrlKey || false,
                shift_pressed: event.shiftKey || false,
                alt_pressed: event.altKey || false,
                meta_pressed: event.metaKey || false,
                mouse_x_viewport: event.clientX || null,
                mouse_y_viewport: event.clientY || null,
                mouse_x_page: event.pageX || null,
                mouse_y_page: event.pageY || null,
            }},
            element: details,
            url: window.location.href
        }};

        try {{
            // Call CDP binding - bindings are accessed as functions
            if (typeof window[bindingName] === 'function') {{
                window[bindingName](JSON.stringify(data));
            }}
        }} catch (e) {{
            console.error('Failed to log interaction:', e);
        }}
    }}

    // Setup listeners after binding is available
    waitForBinding(function() {{
        // Mouse event listeners
        document.addEventListener('click', function(event) {{
            logInteraction('click', event, event.target);
        }}, true);

        document.addEventListener('mousedown', function(event) {{
            logInteraction('mousedown', event, event.target);
        }}, true);

        document.addEventListener('mouseup', function(event) {{
            logInteraction('mouseup', event, event.target);
        }}, true);

        document.addEventListener('dblclick', function(event) {{
            logInteraction('dblclick', event, event.target);
        }}, true);

        document.addEventListener('contextmenu', function(event) {{
            logInteraction('contextmenu', event, event.target);
        }}, true);

        document.addEventListener('mouseover', function(event) {{
            logInteraction('mouseover', event, event.target);
        }}, true);

        // Keyboard event listeners
        document.addEventListener('keydown', function(event) {{
            logInteraction('keydown', event, event.target);
        }}, true);

        document.addEventListener('keyup', function(event) {{
            logInteraction('keyup', event, event.target);
        }}, true);

        document.addEventListener('keypress', function(event) {{
            logInteraction('keypress', event, event.target);
        }}, true);

        // Input events (for form fields)
        document.addEventListener('input', function(event) {{
            logInteraction('input', event, event.target);
        }}, true);

        document.addEventListener('change', function(event) {{
            logInteraction('change', event, event.target);
        }}, true);

        // Focus events
        document.addEventListener('focus', function(event) {{
            logInteraction('focus', event, event.target);
        }}, true);

        document.addEventListener('blur', function(event) {{
            logInteraction('blur', event, event.target);
        }}, true);

        console.log('Web Hacker interaction monitoring enabled');
    }});
}})();
"""

        try:
            # Inject for all future documents
            await cdp_session.send("Page.addScriptToEvaluateOnNewDocument", {
                "source": interaction_script
            })

            # Inject for current page
            await cdp_session.send("Runtime.evaluate", {
                "expression": interaction_script,
                "includeCommandLineAPI": False
            })

            logger.info("Interaction monitoring script injected")
        except Exception as e:
            logger.warning("Failed to inject interaction script: %s", e)

    def _parse_interaction_event(self, raw_data: dict) -> UIInteractionEvent | None:
        """Parse raw JS data into UIInteractionEvent model."""
        try:
            element_data = raw_data.get("element")
            if not element_data:
                logger.warning("Missing element data for interaction")
                return None

            # Build BoundingBox
            bounding_box = None
            if element_data.get("bounding_box"):
                bb = element_data["bounding_box"]
                bounding_box = BoundingBox(
                    x=bb.get("x", 0),
                    y=bb.get("y", 0),
                    width=bb.get("width", 0),
                    height=bb.get("height", 0)
                )

            # Build UIElement
            ui_element = UIElement(
                tag_name=element_data.get("tag_name", ""),
                id=element_data.get("id"),
                name=element_data.get("name"),
                class_names=element_data.get("class_names"),
                type_attr=element_data.get("type_attr"),
                role=element_data.get("role"),
                aria_label=element_data.get("aria_label"),
                placeholder=element_data.get("placeholder"),
                title=element_data.get("title"),
                href=element_data.get("href"),
                src=element_data.get("src"),
                value=element_data.get("value"),
                text=element_data.get("text"),
                attributes=element_data.get("attributes"),
                bounding_box=bounding_box,
                css_path=element_data.get("css_path"),
                xpath=element_data.get("xpath"),
                url=element_data.get("url") or raw_data.get("url"),
            )
            ui_element.build_default_Identifiers()

            # Build Interaction details
            event_raw = raw_data.get("event", {})
            interaction = Interaction(
                mouse_button=event_raw.get("mouse_button"),
                key_value=event_raw.get("key_value"),
                key_code=event_raw.get("key_code"),
                key_code_deprecated=event_raw.get("key_code_deprecated"),
                key_which_deprecated=event_raw.get("key_which_deprecated"),
                ctrl_pressed=event_raw.get("ctrl_pressed", False),
                shift_pressed=event_raw.get("shift_pressed", False),
                alt_pressed=event_raw.get("alt_pressed", False),
                meta_pressed=event_raw.get("meta_pressed", False),
                mouse_x_viewport=event_raw.get("mouse_x_viewport"),
                mouse_y_viewport=event_raw.get("mouse_y_viewport"),
                mouse_x_page=event_raw.get("mouse_x_page"),
                mouse_y_page=event_raw.get("mouse_y_page"),
            )

            # Get interaction type
            interaction_type_str = raw_data.get("type", "unknown")
            try:
                interaction_type = InteractionType(interaction_type_str)
            except ValueError:
                logger.warning("Unknown interaction type: %s", interaction_type_str)
                return None

            return UIInteractionEvent(
                type=interaction_type,
                timestamp=raw_data.get("timestamp", 0),
                interaction=interaction,
                element=ui_element,
                url=raw_data.get("url", ""),
            )

        except Exception as e:
            logger.warning("Failed to parse interaction event: %s", e)
            return None

    async def _on_binding_called(self, msg: dict) -> bool:
        """Handle Runtime.bindingCalled event from JavaScript."""
        try:
            params = msg.get("params", {})
            name = params.get("name")
            payload = params.get("payload", "")

            if name != self.BINDING_NAME:
                return False

            # Parse interaction data
            raw_data = json.loads(payload)

            # Try to convert to UIInteractionEvent
            ui_interaction_event = self._parse_interaction_event(raw_data)

            if ui_interaction_event is not None:
                # Successfully parsed - use structured data
                interaction_data = ui_interaction_event.model_dump()
                interaction_type_str = ui_interaction_event.type.value
                url = ui_interaction_event.url
            else:
                # Fallback to raw data if structured parsing fails
                logger.debug("Using raw interaction data (structured parsing failed)")
                interaction_data = raw_data
                interaction_type_str = raw_data.get("type", "unknown")
                url = raw_data.get("url", "unknown")

            # Update statistics
            self.interaction_count += 1
            self.interaction_types[interaction_type_str] += 1
            self.interactions_by_url[url] += 1

            # Emit event via callback
            try:
                await self.event_callback_fn(
                    self.get_monitor_category(),
                    interaction_data
                )
            except Exception as e:
                logger.error("Error in event callback: %s", e, exc_info=True)

            return True

        except Exception as e:
            logger.warning("Error handling binding call: %s", e)
            return False


    # Public methods __________________________________________________________________________________________

    async def setup_interaction_monitoring(self, cdp_session: AsyncCDPSession) -> None:
        """Setup interaction monitoring via CDP session."""
        logger.info("🔧 Setting up interaction monitoring...")

        # Enable required domains
        await cdp_session.enable_domain("Runtime")
        await cdp_session.enable_domain("DOM")
        await cdp_session.enable_domain("Page")

        # Create binding for JavaScript to call
        await cdp_session.send("Runtime.addBinding", {"name": self.BINDING_NAME})

        # Inject interaction listeners
        await self._inject_interaction_listeners(cdp_session)

        logger.info("✅ Interaction monitoring setup complete")

    async def handle_interaction_message(self, msg: dict, cdp_session: AsyncCDPSession) -> bool:
        """
        Handle interaction-related CDP messages.
        Returns True if handled, False otherwise.
        """
        method = msg.get("method")

        if method == "Runtime.bindingCalled":
            return await self._on_binding_called(msg)

        # Page navigation - script auto-injected via addScriptToEvaluateOnNewDocument
        if method == "Page.frameNavigated":
            return False  # Don't swallow

        if method == "DOM.documentUpdated":
            return False  # Don't swallow

        return False

    async def handle_interaction_command_reply(self, msg: dict) -> bool:
        """
        Handle CDP command replies for interaction monitoring.
        Returns True if handled, False otherwise.
        """
        cmd_id = msg.get("id")
        if cmd_id is None:
            return False

        if cmd_id in self.pending_dom_commands:
            self.pending_dom_commands.pop(cmd_id)
            return True

        return False

    def get_interaction_summary(self) -> dict[str, Any]:
        """Get summary of interaction monitoring."""
        return {
            "interactions_logged": self.interaction_count,
            "interactions_by_type": dict(self.interaction_types),
            "interactions_by_url": dict(self.interactions_by_url),
        }
