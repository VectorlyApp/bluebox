"""
web_hacker/cdp/cdp_session.py

CDP Session management for web scraping with Chrome DevTools Protocol.
"""

import json
import logging
import os
import websocket
import threading
import time

from web_hacker.config import Config
from web_hacker.cdp.network_monitor import NetworkMonitor
from web_hacker.cdp.storage_monitor import StorageMonitor
from web_hacker.cdp.interaction_monitor import InteractionMonitor

logging.basicConfig(level=Config.LOG_LEVEL, format=Config.LOG_FORMAT, datefmt=Config.LOG_DATE_FORMAT)
logger = logging.getLogger(__name__)


class CDPSession:
    """
    "Manages CDP WebSocket connection and coordinates monitoring components.
    """
    
    def __init__(
        self, 
        ws_url, 
        output_dir, 
        paths, 
        capture_resources=None, 
        block_patterns=None, 
        clear_cookies=False, 
        clear_storage=False
    ) -> None:
        self.ws = websocket.create_connection(ws_url)
        self.seq = 0
        self.output_dir = output_dir
        self.paths = paths
        self.clear_cookies = clear_cookies
        self.clear_storage = clear_storage
        
        # Response tracking for synchronous commands
        self.pending_responses = {}
        self.response_lock = threading.Lock()
        
        # Initialize monitoring components
        self.network_monitor = NetworkMonitor(
            output_dir=output_dir,
            paths=paths,
            capture_resources=capture_resources or set(),
            block_patterns=block_patterns or []
        )
        
        self.storage_monitor = StorageMonitor(
            output_dir=output_dir,
            paths=paths
        )
        
        self.interaction_monitor = InteractionMonitor(
            output_dir=output_dir,
            paths=paths
        )
        
    
    def send(self, method, params=None):
        """Send CDP command and return sequence ID."""
        self.seq += 1
        self.ws.send(json.dumps({"id": self.seq, "method": method, "params": params or {}}))
        return self.seq
    
    def send_and_wait(self, method, params=None, timeout=10):
        """Send CDP command and wait for response."""
        cmd_id = self.send(method, params)
        
        # Create a condition variable for this specific command
        condition = threading.Condition()
        response_data = {"result": None, "error": None, "received": False}
        
        # Store the condition in pending_responses
        with self.response_lock:
            self.pending_responses[cmd_id] = (condition, response_data)
        
        # Wait for response
        with condition:
            if not condition.wait(timeout):
                # Timeout occurred
                with self.response_lock:
                    if cmd_id in self.pending_responses:
                        del self.pending_responses[cmd_id]
                raise TimeoutError(f"CDP command {method} timed out after {timeout} seconds")
        
        # Check for errors
        if response_data["error"]:
            raise Exception(f"CDP command {method} failed: {response_data['error']}")
        
        return response_data["result"]
    
    def setup_cdp(self, navigate_to=None):
        """Setup CDP domains and configuration."""
        # Enable basic domains
        self.send("Page.enable")
        self.send("Runtime.enable")
        time.sleep(0.1)  # Small delay to ensure Runtime is ready
        
        # Clear cookies if requested
        if self.clear_cookies:
            logger.info("Clearing all browser cookies...")
            self.send("Network.clearBrowserCookies")
            
            # Also clear cookie store
            try:
                self.send("Storage.clearCookies")
            except:
                pass  # Not all browsers support this
        
        # Clear storage if requested
        if self.clear_storage:
            logger.info("Clearing localStorage and sessionStorage...")
            try:
                # Clear all storage for all origins
                self.send("Storage.clearDataForOrigin", {
                    "origin": "*",
                    "storageTypes": "local_storage,session_storage,indexeddb,cache_storage"
                })
            except:
                # Fallback: try to clear storage via Runtime evaluation
                try:
                    self.send("Runtime.enable")
                    # Clear localStorage
                    self.send("Runtime.evaluate", {
                        "expression": "localStorage.clear(); sessionStorage.clear();",
                        "includeCommandLineAPI": True
                    })
                except:
                    logger.info("Warning: Could not clear storage automatically")
        
        # Setup monitoring components
        self.network_monitor.setup_network_monitoring(self)
        self.storage_monitor.setup_storage_monitoring(self)
        self.interaction_monitor.setup_interaction_monitoring(self)
        
        # Optional navigate
        if navigate_to:
            self.send("Page.navigate", {"url": navigate_to})
    
    def handle_message(self, msg):
        """Handle incoming CDP message by delegating to appropriate monitors."""
        # Try network monitor first
        if self.network_monitor.handle_network_message(msg, self):
            return
        
        # Try storage monitor
        if self.storage_monitor.handle_storage_message(msg, self):
            return
        
        # Try interaction monitor
        if self.interaction_monitor.handle_interaction_message(msg, self):
            return
        
        # Handle command replies
        if "id" in msg:
            self._handle_command_reply(msg)
    
    def _handle_command_reply(self, msg):
        """Handle CDP command replies by delegating to monitors."""
        cmd_id = msg.get("id")
        
        # Check if this is a pending response for send_and_wait
        if cmd_id is not None:
            with self.response_lock:
                if cmd_id in self.pending_responses:
                    condition, response_data = self.pending_responses[cmd_id]
                    
                    # Store the result or error
                    if "result" in msg:
                        response_data["result"] = msg["result"]
                    elif "error" in msg:
                        response_data["error"] = msg["error"]
                    
                    response_data["received"] = True
                    del self.pending_responses[cmd_id]
                    
                    # Notify waiting thread
                    with condition:
                        condition.notify()
                    return True
        
        # Try network monitor first
        if self.network_monitor.handle_network_command_reply(msg, self):
            return True
        
        # Try storage monitor
        if self.storage_monitor.handle_storage_command_reply(msg, self):
            return True
        
        # Try interaction monitor
        if self.interaction_monitor.handle_interaction_command_reply(msg, self):
            return True
        
        return False
    
    def run(self):
        """Main message processing loop."""
        logger.info("Blocking trackers & capturing network/storage… Press Ctrl+C to stop.")
        
        try:
            while True:
                msg = json.loads(self.ws.recv())
                self.handle_message(msg)
        except KeyboardInterrupt:
            logger.info("\nStopped. Saving assets...")
            # Final cookie sync using native CDP (no delay needed)
            self.storage_monitor.monitor_cookie_changes(self)
            
            # Consolidate all transactions into a single JSON file
            consolidated_path = f"{self.output_dir}/consolidated_transactions.json"
            self.network_monitor.consolidate_transactions(consolidated_path)
            
            # Generate HAR file from consolidated transactions
            har_path = f"{self.output_dir}/network.har"
            self.network_monitor.generate_har_from_transactions(har_path, "Web Hacker Session")
            
            # Consolidate all interactions into a single JSON file
            interaction_dir = self.paths.get('interaction_dir', f"{self.output_dir}/interaction")
            consolidated_interactions_path = os.path.join(interaction_dir, "consolidated_interactions.json")
            self.interaction_monitor.consolidate_interactions(consolidated_interactions_path)
        finally:
            try:
                self.ws.close()
            except:
                pass
    
    def get_monitoring_summary(self):
        """Get summary of all monitoring activities."""
        # Trigger final cookie check using native CDP (no delay needed)
        try:
            self.storage_monitor.monitor_cookie_changes(self)
        except:
            pass
            
        storage_summary = self.storage_monitor.get_storage_summary()
        network_summary = self.network_monitor.get_network_summary()
        interaction_summary = self.interaction_monitor.get_interaction_summary()
        
        return {
            "network": network_summary,
            "storage": storage_summary,
            "interaction": interaction_summary,
        }