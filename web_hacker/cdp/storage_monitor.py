#!/usr/bin/env python3
"""
Storage monitoring functionality for CDP web scraping.
Tracks cookies, localStorage, sessionStorage, and other storage mechanisms using NATIVE CDP events.
"""

import time
from web_hacker.utils.cdp_utils import write_jsonl


class StorageMonitor:
    """Monitors browser storage changes using native CDP events (no JavaScript injection)."""
    
    def __init__(self, output_dir, paths):
        self.output_dir = output_dir
        self.paths = paths
        
        # Storage state tracking
        self.cookies_state = {}
        self.local_storage_state = {}
        self.session_storage_state = {}
        self.indexed_db_state = {}
        self.cache_storage_state = {}
        
        # Storage command tracking
        self.pending_storage_commands = {}
        
        # Storage log paths
        self.storage_log_path = self.paths.get('storage_jsonl_path', 
                                             f"{output_dir}/storage_events.jsonl")
        
        # Debouncing for native cookie checks
        self._last_native_cookie_check = 0
    
    def setup_storage_monitoring(self, cdp_session):
        """Setup storage monitoring via CDP session using NATIVE events only."""
        
        # Enable Network domain for Set-Cookie header detection
        cdp_session.send("Network.enable", {
            "maxTotalBufferSize": 10000000,
            "maxResourceBufferSize": 5000000,
            "maxPostDataSize": 65536
        })
        
        # Enable Runtime domain for console events (optional)
        cdp_session.send("Runtime.enable")
        
        # Enable Page domain for navigation events
        cdp_session.send("Page.enable")
        
        # Enable Storage domain for native cookie operations
        try:
            cdp_session.send("Storage.getCookies")
        except:
            pass  # Fallback to Network.getAllCookies
        
        # Enable DOM storage tracking
        cdp_session.send("DOMStorage.enable")
        
        # Enable database tracking
        cdp_session.send("Database.enable")
        
        # Enable IndexedDB tracking
        cdp_session.send("IndexedDB.enable")
        
        # Enable cache storage tracking (if available)
        try:
            cdp_session.send("CacheStorage.enable")
        except:
            pass  # Not all browsers support this
        
        # Get initial cookie state using native CDP
        self._get_initial_cookies(cdp_session)
        
        # Get initial DOM storage state
        self._get_initial_dom_storage(cdp_session)
        
    
    def handle_storage_message(self, msg, cdp_session):
        """Handle storage-related CDP messages using NATIVE events only."""
        method = msg.get("method")
        
        # 🔥 NATIVE CDP COOKIE EVENTS (No polling, no injection!)
        
        # Network response with Set-Cookie headers (REAL-TIME!)
        if method == "Network.responseReceived":
            self._handle_network_response_for_cookies(msg, cdp_session)
            return True
        
        # Extra info often contains cookie headers (REAL-TIME!)
        elif method == "Network.responseReceivedExtraInfo":
            self._handle_network_response_extra_info_for_cookies(msg, cdp_session)
            return True
        
        # Page navigation events (cookies might change on navigation)
        elif method == "Page.frameNavigated":
            self._trigger_native_cookie_check(cdp_session)
            # Don't swallow this event, let other monitors see it
            return False
        
        elif method == "Page.loadEventFired":
            self._trigger_native_cookie_check(cdp_session)
            # Don't swallow this event, let other monitors see it
            return False
        
        # Optional: Runtime console events for document.cookie detection
        elif method == "Runtime.consoleAPICalled":
            self._handle_console_for_cookie_operations(msg, cdp_session)
            return True
        
        # Handle DOM storage events
        elif method == "DOMStorage.domStorageItemsCleared":
            self._handle_dom_storage_cleared(msg)
            return True
        elif method == "DOMStorage.domStorageItemRemoved":
            self._handle_dom_storage_removed(msg)
            return True
        elif method == "DOMStorage.domStorageItemAdded":
            self._handle_dom_storage_added(msg)
            return True
        elif method == "DOMStorage.domStorageItemUpdated":
            self._handle_dom_storage_updated(msg)
            return True
        
        # Handle database events
        elif method == "Database.addDatabase":
            self._handle_database_added(msg)
            return True
        
        # Handle IndexedDB events
        elif method == "IndexedDB.databaseCreated":
            self._handle_indexeddb_added(msg)
            return True
        elif method == "IndexedDB.databaseDeleted":
            self._handle_indexeddb_added(msg)  # Reuse same handler for now
            return True
        
        # Handle cache storage events
        elif method == "CacheStorage.cacheCreated":
            self._handle_cache_storage_added(msg)
            return True
        elif method == "CacheStorage.cacheDeleted":
            self._handle_cache_storage_added(msg)  # Reuse same handler for now
            return True
        
        return False  # Message not handled
    
    def _handle_network_response_for_cookies(self, msg, cdp_session):
        """Handle Network.responseReceived for Set-Cookie headers (NATIVE)."""
        params = msg.get("params", {})
        response = params.get("response", {})
        headers = response.get("headers", {})
        
        # Check for Set-Cookie headers (case-insensitive)
        for header_name, header_value in headers.items():
            if header_name.lower() == "set-cookie":
                self._trigger_native_cookie_check(cdp_session)
                break
    
    def _handle_network_response_extra_info_for_cookies(self, msg, cdp_session):
        """Handle Network.responseReceivedExtraInfo for cookie headers (NATIVE)."""
        params = msg.get("params", {})
        headers = params.get("headers", {})
        
        # Check for Set-Cookie headers in extra info
        for header_name, header_value in headers.items():
            if header_name.lower() == "set-cookie":
                self._trigger_native_cookie_check(cdp_session)
                break
    
    def _handle_console_for_cookie_operations(self, msg, cdp_session):
        """Optional: Handle Runtime console events for document.cookie operations (NATIVE)."""
        params = msg.get("params", {})
        args = params.get("args", [])
        
        # Look for cookie-related console messages
        for arg in args:
            if isinstance(arg, dict) and "value" in arg:
                value = str(arg["value"]).lower()
                if any(keyword in value for keyword in ["cookie", "document.cookie", "set-cookie"]):
                    self._trigger_native_cookie_check(cdp_session)
                    break
    
    def _trigger_native_cookie_check(self, cdp_session):
        """Trigger immediate cookie check using native CDP (with debouncing)."""
        current_time = time.time()
        
        # Debounce to prevent spam (max once per 500ms)
        if (current_time - self._last_native_cookie_check) > 0.5:
            self._last_native_cookie_check = current_time
            
            # Use native CDP to get current cookies
            try:
                cmd_id = cdp_session.send("Storage.getCookies")
                self.pending_storage_commands[cmd_id] = {
                    "type": "getCookies",
                    "timestamp": current_time,
                    "triggered_by": "native_event"
                }
            except:
                # Fallback to Network.getAllCookies
                cmd_id = cdp_session.send("Network.getAllCookies")
                self.pending_storage_commands[cmd_id] = {
                    "type": "getAllCookies",
                    "timestamp": current_time,
                    "triggered_by": "native_event_fallback"
                }
    
    def handle_storage_command_reply(self, msg, cdp_session):
        """Handle CDP command replies."""
        cmd_id = msg.get("id")
        if cmd_id in self.pending_storage_commands:
            command_info = self.pending_storage_commands.pop(cmd_id)
            command_type = command_info.get("type")
            
            if command_type in ["getAllCookies", "getCookies"]:
                self._handle_get_cookies_reply(msg, command_info)
                return True
            elif command_type == "getDOMStorageItems":
                self._handle_get_dom_storage_reply(msg, command_info)
                return True
        
        return False  # Command not handled
    
    def monitor_cookie_changes(self, cdp_session):
        """Trigger native cookie monitoring (used by external callers)."""
        self._trigger_native_cookie_check(cdp_session)
    
    def _get_initial_cookies(self, cdp_session):
        """Get initial cookie state using native CDP."""
        try:
            cmd_id = cdp_session.send("Storage.getCookies")
            self.pending_storage_commands[cmd_id] = {
                "type": "getCookies",
                "timestamp": time.time(),
                "initial": True
            }
        except:
            # Fallback to Network.getAllCookies
            cmd_id = cdp_session.send("Network.getAllCookies")
            self.pending_storage_commands[cmd_id] = {
                "type": "getAllCookies",
                "timestamp": time.time(),
                "initial": True
            }
    
    def _get_initial_dom_storage(self, cdp_session):
        """Get initial DOM storage state."""
        # We'll discover storage IDs through events or by scanning the page
        # For now, we'll set up listeners for storage events
        pass
    
    def _handle_get_cookies_reply(self, msg, command_info):
        """Handle Storage.getCookies or Network.getAllCookies reply (NATIVE)."""
        result = msg.get("result", {})
        cookies = result.get("cookies", [])
        timestamp = command_info.get("timestamp")
        is_initial = command_info.get("initial", False)
        triggered_by = command_info.get("triggered_by", "unknown")
        
        # Compare with previous state
        current_cookies = {
            f"{cookie.get('domain', '')}:{cookie.get('name', '')}": cookie
            for cookie in cookies
        }
        
        if is_initial:
            self.cookies_state = current_cookies
            self._log_storage_event({
                "type": "initialCookies",
                "count": len(cookies),
                "cookies": cookies,
                "timestamp": timestamp,
                "source": "native_cdp"
            })
        else:
            # Check for changes
            added_cookies = []
            modified_cookies = []
            removed_cookies = []
            
            # Check for added/modified cookies
            for key, cookie in current_cookies.items():
                if key not in self.cookies_state:
                    added_cookies.append(cookie)
                elif self.cookies_state[key] != cookie:
                    modified_cookies.append({
                        "old": self.cookies_state[key],
                        "new": cookie
                    })
            
            # Check for removed cookies
            for key, cookie in self.cookies_state.items():
                if key not in current_cookies:
                    removed_cookies.append(cookie)
            
            # Update state
            self.cookies_state = current_cookies
            
            # Log changes if any
            if added_cookies or modified_cookies or removed_cookies:
                change_data = {
                    "type": "cookieChange",
                    "timestamp": timestamp,
                    "source": "native_cdp",
                    "triggered_by": triggered_by,
                    "added": added_cookies,
                    "modified": modified_cookies,
                    "removed": removed_cookies,
                    "total_count": len(cookies)
                }
                self._log_storage_event(change_data)
                
    
    # ... [Include all other existing storage methods unchanged] ...
    
    def _handle_dom_storage_cleared(self, msg):
        """Handle DOMStorage.domStorageItemsCleared event."""
        params = msg.get("params", {})
        storage_id = params.get("storageId", {})
        origin = storage_id.get("securityOrigin", "")
        is_local = storage_id.get("isLocalStorage", True)
        
        storage_type = "localStorage" if is_local else "sessionStorage"
        
        if is_local:
            if origin in self.local_storage_state:
                del self.local_storage_state[origin]
        else:
            if origin in self.session_storage_state:
                del self.session_storage_state[origin]
        
        self._log_storage_event({
            "type": f"{storage_type}Cleared",
            "origin": origin,
            "timestamp": time.time()
        })
    
    def _handle_dom_storage_removed(self, msg):
        """Handle DOMStorage.domStorageItemRemoved event."""
        params = msg.get("params", {})
        storage_id = params.get("storageId", {})
        origin = storage_id.get("securityOrigin", "")
        is_local = storage_id.get("isLocalStorage", True)
        key = params.get("key", "")
        
        storage_type = "localStorage" if is_local else "sessionStorage"
        
        if is_local:
            if origin in self.local_storage_state and key in self.local_storage_state[origin]:
                del self.local_storage_state[origin][key]
        else:
            if origin in self.session_storage_state and key in self.session_storage_state[origin]:
                del self.session_storage_state[origin][key]
        
        self._log_storage_event({
            "type": f"{storage_type}ItemRemoved",
            "origin": origin,
            "key": key,
            "timestamp": time.time()
        })
    
    def _handle_dom_storage_added(self, msg):
        """Handle DOMStorage.domStorageItemAdded event."""
        params = msg.get("params", {})
        storage_id = params.get("storageId", {})
        origin = storage_id.get("securityOrigin", "")
        is_local = storage_id.get("isLocalStorage", True)
        key = params.get("key", "")
        new_value = params.get("newValue", "")
        
        storage_type = "localStorage" if is_local else "sessionStorage"
        
        if is_local:
            if origin not in self.local_storage_state:
                self.local_storage_state[origin] = {}
            self.local_storage_state[origin][key] = new_value
        else:
            if origin not in self.session_storage_state:
                self.session_storage_state[origin] = {}
            self.session_storage_state[origin][key] = new_value
        
        self._log_storage_event({
            "type": f"{storage_type}ItemAdded",
            "origin": origin,
            "key": key,
            "value": new_value,
            "timestamp": time.time()
        })
    
    def _handle_dom_storage_updated(self, msg):
        """Handle DOMStorage.domStorageItemUpdated event."""
        params = msg.get("params", {})
        storage_id = params.get("storageId", {})
        origin = storage_id.get("securityOrigin", "")
        is_local = storage_id.get("isLocalStorage", True)
        key = params.get("key", "")
        old_value = params.get("oldValue", "")
        new_value = params.get("newValue", "")
        
        storage_type = "localStorage" if is_local else "sessionStorage"
        
        if is_local:
            if origin not in self.local_storage_state:
                self.local_storage_state[origin] = {}
            self.local_storage_state[origin][key] = new_value
        else:
            if origin not in self.session_storage_state:
                self.session_storage_state[origin] = {}
            self.session_storage_state[origin][key] = new_value
        
        self._log_storage_event({
            "type": f"{storage_type}ItemUpdated",
            "origin": origin,
            "key": key,
            "oldValue": old_value,
            "newValue": new_value,
            "timestamp": time.time()
        })
    
    def _handle_get_dom_storage_reply(self, msg, command_info):
        """Handle DOMStorage.getDOMStorageItems reply."""
        result = msg.get("result", {})
        entries = result.get("entries", [])
        storage_id = command_info.get("storageId", {})
        origin = storage_id.get("securityOrigin", "")
        is_local = storage_id.get("isLocalStorage", True)
        
        # Convert entries to dictionary
        storage_data = {entry[0]: entry[1] for entry in entries}
        
        if is_local:
            self.local_storage_state[origin] = storage_data
        else:
            self.session_storage_state[origin] = storage_data
    
    def _handle_database_added(self, msg):
        """Handle Database.addDatabase event."""
        params = msg.get("params", {})
        database = params.get("database", {})
        
        self._log_storage_event({
            "type": "databaseAdded",
            "database": database,
            "timestamp": time.time()
        })
    
    def _handle_indexeddb_added(self, msg):
        """Handle IndexedDB events."""
        params = msg.get("params", {})
        
        self._log_storage_event({
            "type": "indexedDBEvent",
            "params": params,
            "timestamp": time.time()
        })
    
    def _handle_cache_storage_added(self, msg):
        """Handle CacheStorage events."""
        params = msg.get("params", {})
        
        self._log_storage_event({
            "type": "cacheStorageEvent",
            "params": params,
            "timestamp": time.time()
        })
    
    def _log_storage_event(self, event_data):
        """Log storage event to file."""
        write_jsonl(self.storage_log_path, event_data)
    
    def get_storage_summary(self):
        """Get summary of current storage state."""
        return {
            "cookies_count": len(self.cookies_state),
            "local_storage_origins": list(self.local_storage_state.keys()),
            "session_storage_origins": list(self.session_storage_state.keys()),
            "local_storage_items": sum(len(items) for items in self.local_storage_state.values()),
            "session_storage_items": sum(len(items) for items in self.session_storage_state.values()),
        }
    