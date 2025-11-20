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
    
    // Helper function to get element details
    function getElementDetails(element) {{
        if (!element) return null;
        
        const details = {{
            tagName: element.tagName || '',
            id: element.id || '',
            className: element.className || '',
            name: element.name || '',
            type: element.type || '',
            value: element.value || '',
            text: element.textContent ? element.textContent.substring(0, 200) : '',
            href: element.href || '',
            src: element.src || '',
            role: element.getAttribute('role') || '',
            ariaLabel: element.getAttribute('aria-label') || '',
            title: element.title || '',
            placeholder: element.placeholder || '',
        }};
        
        // Get XPath-like path
        function getElementPath(el) {{
            if (!el || el.nodeType !== 1) return '';
            const path = [];
            while (el && el.nodeType === 1) {{
                let selector = el.tagName.toLowerCase();
                if (el.id) {{
                    selector += '#' + el.id;
                }} else if (el.className) {{
                    const classes = el.className.split(' ').filter(c => c).slice(0, 3).join('.');
                    if (classes) selector += '.' + classes;
                }}
                path.unshift(selector);
                el = el.parentElement;
                if (path.length > 5) break; // Limit depth
            }}
            return path.join(' > ');
        }}
        
        details.path = getElementPath(element);
        
        // Get bounding box
        try {{
            const rect = element.getBoundingClientRect();
            details.boundingBox = {{
                x: Math.round(rect.x),
                y: Math.round(rect.y),
                width: Math.round(rect.width),
                height: Math.round(rect.height)
            }};
        }} catch (e) {{
            details.boundingBox = null;
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

