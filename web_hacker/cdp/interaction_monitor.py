"""
web_hacker/cdp/interaction_monitor.py

Interaction monitoring for CDP — tracks mouse and keyboard events with element details.
"""

import logging
import os
import time
import json
from collections import defaultdict

from web_hacker.config import Config
from web_hacker.utils.cdp_utils import write_jsonl, write_json_file

# Import UiElement models
from web_hacker.data_models.ui_elements import (
    UiElement, Selector, SelectorType, BoundingBox
)

logging.basicConfig(level=Config.LOG_LEVEL, format=Config.LOG_FORMAT, datefmt=Config.LOG_DATE_FORMAT)
logger = logging.getLogger(__name__)


class InteractionMonitor:
    """
    Interaction monitor for CDP — tracks mouse clicks, keyboard events, and element details.
    """

    def __init__(self, output_dir, paths):
        self.output_dir = output_dir
        self.paths = paths
        
        # Interaction directory and log path
        interaction_dir = self.paths.get('interaction_dir', os.path.join(output_dir, "interaction"))
        os.makedirs(interaction_dir, exist_ok=True)
        
        self.interaction_log_path = self.paths.get(
            'interaction_jsonl_path',
            os.path.join(interaction_dir, "events.jsonl")
        )
        
        # Track pending DOM commands for element details
        self.pending_dom_commands = {}
        
        # Track interaction counts and statistics
        self.interaction_count = 0
        self.interaction_types = defaultdict(int)
        self.interactions_by_url = defaultdict(int)
        
        # Binding name for JavaScript to call
        self.binding_name = "__webHackerInteractionLog"
    
    # ------------------------ Setup ------------------------
    def setup_interaction_monitoring(self, cdp_session):
        """Setup interaction monitoring via CDP session."""
        
        # Enable Runtime domain for binding and script injection
        cdp_session.send("Runtime.enable")
        
        # Enable DOM domain for element details
        cdp_session.send("DOM.enable")
        
        # Enable Page domain for navigation events
        cdp_session.send("Page.enable")
        
        # Create a binding that JavaScript can call
        cdp_session.send("Runtime.addBinding", {
            "name": self.binding_name
        })
        
        # Inject interaction listeners script
        self._inject_interaction_listeners(cdp_session)
    
    def _inject_interaction_listeners(self, cdp_session):
        """Inject JavaScript listeners for mouse and keyboard events."""
        
        # JavaScript code to inject
        interaction_script = f"""
(function() {{
    'use strict';
    
    const bindingName = '{self.binding_name}';
    
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
    
    // Helper function to get element details (UiElement format)
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
                type: event.type,
                button: event.button !== undefined ? event.button : null,
                key: event.key || null,
                code: event.code || null,
                keyCode: event.keyCode || null,
                which: event.which || null,
                ctrlKey: event.ctrlKey || false,
                shiftKey: event.shiftKey || false,
                altKey: event.altKey || false,
                metaKey: event.metaKey || false,
                clientX: event.clientX || null,
                clientY: event.clientY || null,
                pageX: event.pageX || null,
                pageY: event.pageY || null,
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
            logInteraction('mouse_click', event, event.target);
        }}, true);
        
        document.addEventListener('mousedown', function(event) {{
            logInteraction('mouse_down', event, event.target);
        }}, true);
        
        document.addEventListener('mouseup', function(event) {{
            logInteraction('mouse_up', event, event.target);
        }}, true);
        
        document.addEventListener('dblclick', function(event) {{
            logInteraction('mouse_double_click', event, event.target);
        }}, true);
        
        document.addEventListener('contextmenu', function(event) {{
            logInteraction('mouse_context_menu', event, event.target);
        }}, true);
        
        document.addEventListener('mouseover', function(event) {{
            logInteraction('mouse_over', event, event.target);
        }}, true);
        
        // Keyboard event listeners
        document.addEventListener('keydown', function(event) {{
            logInteraction('key_down', event, event.target);
        }}, true);
        
        document.addEventListener('keyup', function(event) {{
            logInteraction('key_up', event, event.target);
        }}, true);
        
        document.addEventListener('keypress', function(event) {{
            logInteraction('key_press', event, event.target);
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
        
        # Inject script to run on every new document
        try:
            cdp_session.send("Page.addScriptToEvaluateOnNewDocument", {
                "source": interaction_script
            })
            
            # Also inject immediately for current page
            cdp_session.send("Runtime.evaluate", {
                "expression": interaction_script,
                "includeCommandLineAPI": False
            })
            
            logger.info("Interaction monitoring script injected")
        except Exception as e:
            logger.info("Failed to inject interaction monitoring script: %s", e)
    
    # ------------------------ Dispatch ------------------------
    def handle_interaction_message(self, msg, cdp_session):
        """Handle interaction-related CDP messages."""
        method = msg.get("method")
        
        # Handle Runtime.bindingCalled - this is triggered when JavaScript calls our binding
        if method == "Runtime.bindingCalled":
            return self._on_binding_called(msg, cdp_session)
        
        # Handle page navigation - re-inject script if needed
        if method == "Page.frameNavigated":
            # Script will be auto-injected via Page.addScriptToEvaluateOnNewDocument
            return True
        
        # Handle DOM events (optional - for additional element details)
        if method == "DOM.documentUpdated":
            # Document updated, script will be re-injected
            return True
        
        return False
    
    def handle_interaction_command_reply(self, msg, cdp_session):
        """Handle CDP command replies for interaction monitoring."""
        cmd_id = msg.get("id")
        if cmd_id is None:
            return False
        
        # Handle DOM command replies if we're waiting for element details
        if cmd_id in self.pending_dom_commands:
            self._on_dom_command_reply(cmd_id, msg)
            return True
        
        return False
    
    # ------------------------ Event Handlers ------------------------
    def _on_binding_called(self, msg, cdp_session):
        """Handle Runtime.bindingCalled event from JavaScript."""
        try:
            params = msg.get("params", {})
            name = params.get("name")
            payload = params.get("payload", "")
            
            if name != self.binding_name:
                return False
            
            # Parse the interaction data from JavaScript
            interaction_data = json.loads(payload)
            
            # Convert element details to UiElement format if available
            element_data = interaction_data.get("element")
            if element_data:
                try:
                    # Convert bounding_box if present
                    bounding_box = None
                    if element_data.get("bounding_box"):
                        bb_data = element_data["bounding_box"]
                        bounding_box = BoundingBox(
                            x=bb_data.get("x", 0),
                            y=bb_data.get("y", 0),
                            width=bb_data.get("width", 0),
                            height=bb_data.get("height", 0)
                        )
                    
                    # Create UiElement
                    ui_element = UiElement(
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
                        url=element_data.get("url") or interaction_data.get("url"),
                    )
                    
                    # Build default selectors
                    ui_element.build_default_selectors()
                    
                    # Replace element data with UiElement dict representation
                    interaction_data["element"] = ui_element.model_dump()
                except Exception as e:
                    logger.info("Failed to convert element to UiElement format: %s", e)
                    # Keep original element data if conversion fails
            
            # Add server-side timestamp
            interaction_data["server_timestamp"] = time.time()
            
            # Update statistics
            self.interaction_count += 1
            interaction_type = interaction_data.get("type", "unknown")
            self.interaction_types[interaction_type] += 1
            url = interaction_data.get("url", "unknown")
            self.interactions_by_url[url] += 1
            
            # Log the interaction
            self._log_interaction(interaction_data)
            
            return True
            
        except Exception as e:
            logger.info("Error handling binding call: %s", e)
            return False
    
    def _on_dom_command_reply(self, cmd_id, msg):
        """Handle DOM command replies (for getting additional element details)."""
        command_info = self.pending_dom_commands.pop(cmd_id, None)
        if not command_info:
            return
        
        # Process DOM response if needed
        # This can be used to enrich element details if necessary
        pass
    
    # ------------------------ Helpers ------------------------
    def _log_interaction(self, interaction_data):
        """Log interaction event to JSONL file."""
        try:
            write_jsonl(self.interaction_log_path, interaction_data)
        except Exception as e:
            logger.info("Failed to log interaction: %s", e)
    
    def consolidate_interactions(self, output_file_path=None):
        """
        Consolidate all interactions from JSONL file into a single JSON file.
        
        Returns dict with structure:
        {
            "interactions": [...],
            "summary": {
                "total": 123,
                "by_type": {...},
                "by_url": {...}
            }
        }
        """
        if not os.path.exists(self.interaction_log_path):
            return {"interactions": [], "summary": {"total": 0, "by_type": {}, "by_url": {}}}
        
        interactions = []
        by_type = defaultdict(int)
        by_url = defaultdict(int)
        
        try:
            with open(self.interaction_log_path, 'r', encoding='utf-8') as f:
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
                        logger.info("Failed to parse interaction line: %s", e)
                        continue
        except Exception as e:
            logger.info("Failed to read interaction log: %s", e)
            return {"interactions": [], "summary": {"total": 0, "by_type": {}, "by_url": {}}}
        
        consolidated = {
            "interactions": interactions,
            "summary": {
                "total": len(interactions),
                "by_type": dict(by_type),
                "by_url": dict(by_url)
            }
        }
        
        # Save to file if output path provided
        if output_file_path:
            try:
                write_json_file(output_file_path, consolidated)
                logger.info("Consolidated interactions saved to: %s", output_file_path)
            except Exception as e:
                logger.info("Failed to save consolidated interactions: %s", e)
        
        return consolidated
    
    def get_interaction_summary(self):
        """Get summary of interaction monitoring."""
        return {
            "interactions_logged": self.interaction_count,
            "interactions_by_type": dict(self.interaction_types),
            "interactions_by_url": dict(self.interactions_by_url),
            "log_path": self.interaction_log_path
        }

