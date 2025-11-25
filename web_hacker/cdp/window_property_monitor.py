#!/usr/bin/env python3
"""
Window property monitoring functionality for CDP web scraping.
Tracks window properties over time, collecting on a schedule and on navigation events.
"""

import json
import logging
import os
import time
import threading
from pathlib import Path

from web_hacker.data_models.window_property import WindowProperty, WindowPropertyValue

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Native browser API prefixes - used to identify native vs application objects
NATIVE_PREFIXES = [
    "HTML", "SVG", "MathML", "RTC", "IDB", "Media", "Audio", "Video",
    "WebGL", "Canvas", "Crypto", "File", "Blob", "Form", "Input",
    "Mutation", "Intersection", "Resize", "Performance", "Navigation",
    "Storage", "Location", "History", "Navigator", "Screen", "Window",
    "Document", "Element", "Node", "Event", "Promise", "Array",
    "String", "Number", "Boolean", "Date", "RegExp", "Error", "Function",
    "Map", "Set", "WeakMap", "WeakSet", "Proxy", "Reflect", "Symbol",
    "Intl", "JSON", "Math", "Console", "TextEncoder", "TextDecoder",
    "ReadableStream", "WritableStream", "TransformStream", "AbortController",
    "URL", "URLSearchParams", "Headers", "Request", "Response", "Fetch",
    "Worker", "SharedWorker", "ServiceWorker", "BroadcastChannel",
    "MessageChannel", "MessagePort", "ImageData", "ImageBitmap",
    "OffscreenCanvas", "Path2D", "CanvasGradient", "CanvasPattern",
    "Geolocation", "Notification", "PushManager", "Cache", "IndexedDB"
]


class WindowPropertyMonitor:
    """Monitors window properties using CDP, tracking changes over time."""
    
    def __init__(self, output_dir, paths):
        self.output_dir = output_dir
        self.paths = paths
        
        # Window properties history: dict[path, WindowProperty]
        self.history_db: dict[str, WindowProperty] = {}
        self.last_seen_keys = set()  # Track keys from previous collection to detect deletions
        
        # Collection state
        self.collection_interval = 10.0  # seconds
        self.last_collection_time = 0
        self.navigation_detected = False
        self.page_ready = False  # Track if page is ready for collection
        self.collection_thread = None
        self.collection_lock = threading.Lock()
        self.pending_navigation = False  # Track if navigation happened during collection
        self.abort_collection = False  # Flag to abort ongoing collection on navigation
        
        # Output path (handled like storage and other monitors)
        self.output_file = paths.get('window_properties_json_path', 
                                     os.path.join(output_dir, "window_properties", "window_properties.json"))
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
    
    def _save_history(self):
        """Save window properties history to file."""
        try:
            # Save as dict[path, WindowProperty.model_dump()]
            serializable_dict = {
                path: window_prop.model_dump(mode='json')
                for path, window_prop in self.history_db.items()
            }
            with open(self.output_file, "w", encoding="utf-8") as f:
                json.dump(serializable_dict, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving window properties: {e}")
    
    def setup_window_property_monitoring(self, cdp_session):
        """Setup window property monitoring via CDP session."""
        # Enable Page domain for navigation events
        cdp_session.send("Page.enable")
        
        # Enable Runtime domain for property access
        cdp_session.send("Runtime.enable")
        
        # Check if page is already loaded (non-blocking, fail-fast)
        try:
            result = cdp_session.send_and_wait("Runtime.evaluate", {
                "expression": "document.readyState",
                "returnByValue": True
            }, timeout=0.5)  # Very short timeout - fail fast
            if result and result.get("result", {}).get("value") == "complete":
                self.page_ready = True
        except Exception:
            # Too bad, so sad - page not ready yet, will check later
            pass
    
    def handle_window_property_message(self, msg, cdp_session):
        """Handle window property-related CDP messages."""
        method = msg.get("method")
        
        # Detect navigation events
        if method == "Runtime.executionContextsCleared":
            self.page_ready = False
            self.navigation_detected = True
            
            # If collection is running, signal it to abort (don't block the event loop!)
            if self.collection_thread and self.collection_thread.is_alive():
                self.abort_collection = True
                self.pending_navigation = True
            
            return True
        
        elif method == "Page.frameNavigated":
            self.navigation_detected = True
            self.page_ready = True
            
            # Only trigger if no collection is running
            if not (self.collection_thread and self.collection_thread.is_alive()):
                self._trigger_collection_thread(cdp_session)
            else:
                # Collection is running, mark navigation as pending
                self.pending_navigation = True
            
            return True
        
        elif method == "Page.domContentEventFired":
            self.page_ready = True
            self.navigation_detected = True
            
            # Only trigger if no collection is running
            if not (self.collection_thread and self.collection_thread.is_alive()):
                self._trigger_collection_thread(cdp_session)
            else:
                # Collection is running, mark navigation as pending
                self.pending_navigation = True
            
            return True
        
        elif method == "Page.loadEventFired":
            self.page_ready = True
            self.navigation_detected = True
            
            # Only trigger if no collection is running
            if not (self.collection_thread and self.collection_thread.is_alive()):
                self._trigger_collection_thread(cdp_session)
            else:
                # Collection is running, mark navigation as pending
                self.pending_navigation = True
            
            return True
        
        return False
    
    def _is_application_object(self, className, name):
        """Heuristically determine if an object is an application object."""
        if not name:
            return False
        
        # First, check if className matches native patterns
        if className:
            for prefix in NATIVE_PREFIXES:
                if className.startswith(prefix):
                    return False
        
        # If name looks like a native API, it's native
        if name.startswith(("HTML", "SVG", "RTC", "IDB", "WebGL", "Media", "Audio", "Video")):
            return False
        
        # Skip common native browser globals
        native_globals = [
            "window", "self", "top", "parent", "frames", "document", "navigator",
            "location", "history", "screen", "console", "localStorage", "sessionStorage",
            "indexedDB", "caches", "performance", "fetch", "XMLHttpRequest", "WebSocket",
            "Blob", "File", "FileReader", "FormData", "URL", "URLSearchParams",
            "Headers", "Request", "Response", "AbortController", "Event", "CustomEvent",
            "Promise", "Map", "Set", "WeakMap", "WeakSet", "Proxy", "Reflect",
            "Symbol", "Intl", "JSON", "Math", "Date", "RegExp", "Error", "Array",
            "String", "Number", "Boolean", "Object", "Function", "ArrayBuffer",
            "DataView", "Int8Array", "Uint8Array", "Int16Array", "Uint16Array",
            "Int32Array", "Uint32Array", "Float32Array", "Float64Array"
        ]
        if name in native_globals:
            return False
        
        # If className is "Object" or empty, and it passed the blacklist checks above, it is likely an application object
        if className == "Object" or not className:
            return True
        
        return True
    
    def _fully_resolve_object_flat(self, cdp_session, object_id, base_path, flat_dict, visited=None, depth=0, max_depth=10):
        """Recursively resolve an object and add all properties to a flat dictionary with dot paths. Non-blocking, fail-fast."""
        # Check abort flag at start
        if self.abort_collection:
            return
        
        if visited is None:
            visited = set()
        
        if depth > max_depth or object_id in visited:
            return
        
        visited.add(object_id)
        
        try:
            # Very short timeout - if page changed, object IDs are invalid, just skip
            props_result = cdp_session.send_and_wait("Runtime.getProperties", {
                "objectId": object_id,
                "ownProperties": True
            }, timeout=0.5)  # Fail fast - if page changed, too bad so sad
            
            # Check abort flag after CDP call
            if self.abort_collection:
                return
            
            props_list = props_result.get("result", [])
            
            for prop in props_list:
                # Check abort flag periodically during processing
                if self.abort_collection:
                    return
                name = prop["name"]
                value = prop.get("value", {})
                value_type = value.get("type", "unknown")
                className = value.get("className", "")
                
                # Skip native APIs at deeper levels
                is_app_obj = self._is_application_object(className, name)
                if depth > 0 and not is_app_obj:
                    continue
                
                prop_path = f"{base_path}.{name}" if base_path else name
                
                # Only store actual values, no metadata
                if value_type == "string":
                    flat_dict[prop_path] = value.get("value")
                elif value_type in ["number", "boolean"]:
                    flat_dict[prop_path] = value.get("value")
                elif value_type == "object":
                    if value.get("subtype") == "null":
                        flat_dict[prop_path] = None
                    elif value.get("objectId"):
                        nested_obj_id = value.get("objectId")
                        if is_app_obj:
                            self._fully_resolve_object_flat(cdp_session, nested_obj_id, prop_path, flat_dict, visited.copy(), depth + 1, max_depth)
                elif value_type == "function":
                    pass  # Skip functions
                else:
                    flat_dict[prop_path] = value.get("value")
        
        except Exception as e:
            # During navigation, object IDs become invalid - this is expected
            # Too bad, so sad - just skip it, don't wait, don't log
            error_str = str(e)
            if "-32000" in error_str or "Cannot find context" in error_str or "context" in error_str.lower() or "TimeoutError" in str(type(e).__name__):
                # Silently skip - object ID became invalid due to navigation or timeout
                return
            # Only log truly unexpected errors (not timeouts or navigation errors)
            if "TimeoutError" not in str(type(e).__name__):
                logger.debug(f"Error resolving object {base_path}: {e}")
    
    def _get_current_url(self, cdp_session):
        """Get current page URL using CDP. Non-blocking, fail-fast."""
        # Check abort flag first
        if self.abort_collection:
            return "unknown"
        
        try:
            # Try Page.getFrameTree first - this works even if JavaScript isn't ready
            # Very short timeout - fail fast if page changed
            frame_tree = cdp_session.send_and_wait("Page.getFrameTree", {}, timeout=0.5)
            if frame_tree and "frameTree" in frame_tree:
                current_url = frame_tree.get("frameTree", {}).get("frame", {}).get("url")
                if current_url:
                    return current_url
        except Exception:
            # Too bad, so sad - skip fallbacks if first attempt fails
            return "unknown"
        
        # Only try one fallback with very short timeout
        if self.abort_collection:
            return "unknown"
        
        try:
            result = cdp_session.send_and_wait("Runtime.evaluate", {
                "expression": "window.location.href",
                "returnByValue": True
            }, timeout=0.5)  # Very short timeout
            if result and "result" in result:
                current_url = result["result"].get("value")
                if current_url:
                    return current_url
        except Exception:
            # Too bad, so sad - can't get URL
            pass
        
        return "unknown"
    
    def _collect_window_properties(self, cdp_session):
        """Collect all window properties into a flat dictionary. Fully non-blocking, fail-fast."""
        # Reset abort flag at start of collection
        self.abort_collection = False
        
        try:
            # Check if Runtime context is ready (very short timeout - fail fast)
            if self.abort_collection:
                return
            
            try:
                test_result = cdp_session.send_and_wait("Runtime.evaluate", {
                    "expression": "1+1",
                    "returnByValue": True
                }, timeout=0.5)  # Very short timeout
                if not test_result:
                    return
                if isinstance(test_result, dict):
                    if "error" in test_result or "result" not in test_result:
                        return
            except (TimeoutError, Exception):
                # Too bad, so sad - Runtime not ready, skip collection
                return
            
            # Check abort flag before continuing
            if self.abort_collection:
                return
            
            current_url = self._get_current_url(cdp_session)
            
            # Check abort flag
            if self.abort_collection:
                return
            
            # Get window object (very short timeout)
            try:
                result = cdp_session.send_and_wait("Runtime.evaluate", {
                    "expression": "window",
                    "returnByValue": False
                }, timeout=0.5)  # Very short timeout - fail fast
            except (TimeoutError, Exception):
                # Too bad, so sad - can't get window object, skip
                return
            
            if not result or not result.get("result", {}).get("objectId"):
                return
            
            # Check abort flag
            if self.abort_collection:
                return
            
            window_obj = result["result"]["objectId"]
            
            # Get all properties of window (short timeout - this is the biggest operation)
            if self.abort_collection:
                return
            
            try:
                props_result = cdp_session.send_and_wait("Runtime.getProperties", {
                    "objectId": window_obj,
                    "ownProperties": True
                }, timeout=1.0)  # Short timeout - if page changed, too bad so sad
            except (TimeoutError, Exception) as e:
                # If navigation happens during collection, object IDs become invalid
                # Too bad, so sad - just abort collection silently
                error_str = str(e)
                if "-32000" in error_str or "Cannot find context" in error_str or "TimeoutError" in str(type(e).__name__):
                    return  # Silently abort collection
                # Only log truly unexpected errors
                logger.debug(f"Error getting window properties: {e}")
                return
            
            # Check abort flag after getting properties
            if self.abort_collection:
                return
            
            flat_dict = {}
            all_props = props_result.get("result", [])
            
            total_props = len(all_props)
            
            skipped_count = 0
            processed_count = 0
            
            for prop in all_props:
                # Check abort flag frequently during processing
                if self.abort_collection:
                    return
                name = prop["name"]
                value = prop.get("value", {})
                value_type = value.get("type", "unknown")
                className = value.get("className", "")
                
                is_app_object = self._is_application_object(className, name)
                
                if not is_app_object:
                    skipped_count += 1
                    continue
                
                # Only store actual values, no metadata
                if value_type == "string":
                    flat_dict[name] = value.get("value")
                elif value_type in ["number", "boolean"]:
                    flat_dict[name] = value.get("value")
                elif value_type == "object" and value.get("objectId"):
                    # Check abort before recursive call
                    if self.abort_collection:
                        return
                    obj_id = value.get("objectId")
                    # Recursive resolution with fail-fast timeout (handled inside)
                    self._fully_resolve_object_flat(cdp_session, obj_id, name, flat_dict, max_depth=10)
                    # Check abort after recursive call
                    if self.abort_collection:
                        return
                elif value_type == "function":
                    pass  # Skip functions
                else:
                    flat_dict[name] = value.get("value")
                
                processed_count += 1
            
            # Update history
            current_ts = time.time()
            changes_count = 0
            
            # Update history with new/changed values
            current_keys = set()
            for key, value in flat_dict.items():
                current_keys.add(key)
                if key not in self.history_db:
                    # New key - create WindowProperty with first value
                    window_prop_value = WindowPropertyValue(
                        timestamp=current_ts,
                        value=value,
                        url=current_url
                    )
                    self.history_db[key] = WindowProperty(
                        path=key,
                        values=[window_prop_value]
                    )
                    changes_count += 1
                else:
                    # Existing key, check if value changed
                    window_property = self.history_db[key]
                    last_entry = window_property.values[-1]
                    if last_entry.value != value:
                        # Value changed, add new entry
                        window_prop_value = WindowPropertyValue(
                            timestamp=current_ts,
                            value=value,
                            url=current_url
                        )
                        window_property.values.append(window_prop_value)
                        changes_count += 1
            
            # Check for deleted keys (only check keys from previous collection, not all history!)
            for key in self.last_seen_keys:
                if key not in current_keys:
                    # Key was deleted since last collection
                    if key in self.history_db:
                        window_property = self.history_db[key]
                        last_entry = window_property.values[-1]
                        if last_entry.value is not None:
                            # Add deletion marker (None value)
                            window_prop_value = WindowPropertyValue(
                                timestamp=current_ts,
                                value=None,
                                url=current_url
                            )
                            window_property.values.append(window_prop_value)
                            changes_count += 1
            
            # Update last_seen_keys for next collection
            self.last_seen_keys = current_keys
            
            if changes_count > 0 or not os.path.exists(self.output_file):
                self._save_history()
            
        except Exception as e:
            logger.error(f"Error collecting window properties: {e}")
        finally:
            # Clear abort flag and thread reference since collection is done
            was_aborted = self.abort_collection
            self.abort_collection = False
            
            with self.collection_lock:
                self.collection_thread = None
            
            # After collection finishes, check if navigation is pending
            # If so, trigger a new collection for the new page
            if self.pending_navigation:
                self.pending_navigation = False
                # Small delay to let new page settle
                time.sleep(0.5)
                # Reset navigation flag and trigger new collection
                self.navigation_detected = True
                self._trigger_collection_thread(cdp_session)
    
    def _trigger_collection_thread(self, cdp_session):
        """Trigger collection in a separate thread."""
        with self.collection_lock:
            if self.collection_thread and self.collection_thread.is_alive():
                return
            
            self.collection_thread = threading.Thread(
                target=self._collect_window_properties,
                args=(cdp_session,)
            )
            self.collection_thread.daemon = True
            self.collection_thread.start()

    def check_and_collect(self, cdp_session):
        """Check if it's time to collect and collect if needed (runs in background thread)."""
        # Don't collect until page is ready (after first navigation)
        if not self.page_ready:
            return
        
        current_time = time.time()
        
        # Check if a collection is already running
        if self.collection_thread and self.collection_thread.is_alive():
            return

        # Collect on navigation or if interval has passed
        should_collect = (
            self.navigation_detected or
            (current_time - self.last_collection_time) >= self.collection_interval
        )
        
        if should_collect:
            self.navigation_detected = False
            self.last_collection_time = current_time
            self._trigger_collection_thread(cdp_session)

    def force_collect(self, cdp_session):
        """Force immediate collection of window properties (non-blocking)."""
        # Just trigger the thread. If it's running, great. If not, start it.
        # We do NOT wait for it to complete.
        self._trigger_collection_thread(cdp_session)
    
    def get_window_property_summary(self):
        """Get summary of window property monitoring."""
        total_keys = len(self.history_db)
        total_entries = sum(len(window_prop.values) for window_prop in self.history_db.values())
        
        return {
            "total_keys": total_keys,
            "total_history_entries": total_entries,
            "output_file": self.output_file
        }

