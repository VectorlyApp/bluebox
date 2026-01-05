#!/usr/bin/env python3
"""
Window property monitoring functionality for CDP web scraping.
Tracks window properties over time, collecting on a schedule and on navigation events.
"""

import json
import os
import time
import threading
from pathlib import Path

from web_hacker.data_models.window_property import WindowProperty, WindowPropertyValue
from web_hacker.utils.logger import get_logger

logger = get_logger(__name__)

# Native browser API prefixes - used to identify native vs application objects (moved to JS)
# See _collect_window_properties for the JS implementation

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

    def _get_current_url(self, cdp_session):
        """Get current page URL safely."""
        try:
            result = cdp_session.send_and_wait("Runtime.evaluate", {
                "expression": "location.href",
                "returnByValue": True
            }, timeout=1.0)
            
            if result and "result" in result and "value" in result["result"]:
                return result["result"]["value"]
        except Exception:
            pass
        return "unknown"

    def _collect_window_properties(self, cdp_session):
        """Collect all window properties into a flat dictionary using a single JS evaluation.
        
        This runs entirely in the browser context to avoid thousands of CDP roundtrips
        that would otherwise freeze the page.
        """
        # Reset abort flag at start of collection
        self.abort_collection = False
        
        try:
            # Check if Runtime context is ready (very short timeout - fail fast)
            if self.abort_collection:
                return
            
            try:
                # Simple check if runtime is responsive
                cdp_session.send_and_wait("Runtime.evaluate", {
                    "expression": "1+1",
                    "returnByValue": True
                }, timeout=0.5)
            except (TimeoutError, Exception):
                return
            
            # Check abort flag before continuing
            if self.abort_collection:
                return
            
            current_url = self._get_current_url(cdp_session)
            
            if self.abort_collection:
                return
            
            # JavaScript script to traverse and serialize window properties efficiently
            # This replicates the logic of the previous Python implementation but runs in-browser
            js_script = r"""
            (function() {
                const MAX_DEPTH = 10;
                const START_TIME = Date.now();
                const TIME_LIMIT = 500; // 500ms hard limit to ensure we never freeze the page
                
                const NATIVE_PREFIXES = [
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
                ];
                
                const IGNORED_GLOBALS = new Set([
                    "window", "self", "top", "parent", "frames", "document", "navigator",
                    "location", "history", "screen", "console", "localStorage", "sessionStorage",
                    "indexedDB", "caches", "performance", "fetch", "XMLHttpRequest", "WebSocket",
                    "Blob", "File", "FileReader", "FormData", "URL", "URLSearchParams",
                    "Headers", "Request", "Response", "AbortController", "Event", "CustomEvent",
                    "Promise", "Map", "Set", "WeakMap", "WeakSet", "Proxy", "Reflect",
                    "Symbol", "Intl", "JSON", "Math", "Date", "RegExp", "Error", "Array",
                    "String", "Number", "Boolean", "Object", "Function", "ArrayBuffer",
                    "DataView", "Int8Array", "Uint8Array", "Int16Array", "Uint16Array",
                    "Int32Array", "Uint32Array", "Float32Array", "Float64Array",
                    "alert", "confirm", "prompt", "print", "postMessage", "close", "stop", "focus", "blur", "open"
                ]);

                const flatDict = {};
                const visited = new Set();

                function isNative(obj) {
                    if (obj === null || obj === undefined) return false;
                    const ctor = obj.constructor;
                    if (!ctor || !ctor.name) return false;
                    const name = ctor.name;
                    for (let i = 0; i < NATIVE_PREFIXES.length; i++) {
                        if (name.startsWith(NATIVE_PREFIXES[i])) return true;
                    }
                    return false;
                }

                function traverse(obj, path, depth) {
                    if (depth > MAX_DEPTH) return;
                    // Check time limit every few iterations to ensure non-blocking
                    if (depth === 0 && Date.now() - START_TIME > TIME_LIMIT) return;
                    
                    if (visited.has(obj)) return;
                    visited.add(obj);
                    
                    let keys = [];
                    try { 
                        keys = Object.getOwnPropertyNames(obj); 
                    } catch(e) { return; }
                    
                    for (const key of keys) {
                        // Skip internal properties
                        if (key.startsWith("__") || key === "constructor" || key === "prototype") continue;
                        
                        let val;
                        try { val = obj[key]; } catch(e) { continue; }
                        
                        const valType = typeof val;
                        const newPath = path ? path + "." + key : key;
                        
                        if (val === null) {
                            flatDict[newPath] = null;
                        } else if (valType === 'string' || valType === 'number' || valType === 'boolean') {
                            flatDict[newPath] = val;
                        } else if (valType === 'object') {
                            if (!isNative(val)) {
                                traverse(val, newPath, depth + 1);
                            }
                        }
                    }
                }

                // Top level window scan
                try {
                    const keys = Object.getOwnPropertyNames(window);
                    for (const key of keys) {
                        if (IGNORED_GLOBALS.has(key) || key.startsWith("on") || key.startsWith("webkit")) continue;
                        
                        let val;
                        try { val = window[key]; } catch(e) { continue; }
                        
                        if (val === null || val === undefined) continue;
                        
                        // Check if it's a native type instance
                        if (typeof val === 'object' && isNative(val)) continue;
                        
                        // If it passed checks, traverse or add
                        const valType = typeof val;
                        if (valType === 'object') {
                            traverse(val, key, 0);
                        } else if (valType !== 'function') {
                            flatDict[key] = val;
                        }
                    }
                } catch(e) {}
                
                return flatDict;
            })()
            """

            # Execute the script - single round trip!
            try:
                result = cdp_session.send_and_wait("Runtime.evaluate", {
                    "expression": js_script,
                    "returnByValue": True,
                    "timeout": 1000  # Allow script to run
                }, timeout=2.0) # Overall timeout
            except (TimeoutError, Exception):
                return

            if self.abort_collection:
                return
                
            if not result or "result" not in result or "value" not in result["result"]:
                return

            flat_dict = result["result"]["value"]
            
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
