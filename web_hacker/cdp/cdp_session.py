"""
web_hacker/cdp/cdp_session.py

CDP session orchestration for browser monitoring.

Contains:
- CDPSession: Manages WebSocket connection, monitors (network, storage, interaction, window)
- start(): Begin monitoring session
- stop(): End session and save captured data
- Coordinates: NetworkMonitor, StorageMonitor, InteractionMonitor, WindowPropertyMonitor
"""

import json
import os
import websocket
from websocket._exceptions import WebSocketConnectionClosedException
import threading
import time

from web_hacker.cdp.network_monitor import NetworkMonitor
from web_hacker.cdp.storage_monitor import StorageMonitor
from web_hacker.cdp.interaction_monitor import InteractionMonitor
from web_hacker.cdp.window_property_monitor import WindowPropertyMonitor
from web_hacker.utils.logger import get_logger

logger = get_logger(__name__)


class CDPSession:
    """
    Manages CDP WebSocket connection and coordinates monitoring components.
    """

    def __init__(
        self,
        output_dir: str,
        paths: dict,
        ws: websocket.WebSocket | None = None,
        ws_url: str | None = None,
        capture_resources: set | None = None,
        block_patterns: list | None = None,
        clear_cookies: bool = False,
        clear_storage: bool = False,
    ) -> None:
        # Accept existing WebSocket or create new one from URL
        if ws is not None:
            self.ws = ws
        elif ws_url is not None:
            self.ws = websocket.create_connection(ws_url)
        else:
            raise ValueError("Either ws or ws_url must be provided")
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
        
        self.window_property_monitor = WindowPropertyMonitor(
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
        self.window_property_monitor.setup_window_property_monitoring(self)
        
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
        
        # Try window property monitor
        if self.window_property_monitor.handle_window_property_message(msg, self):
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
    
    def _finalize_session(self):
        """Finalize session by consolidating transactions, generating HAR, etc."""
        logger.info("Finalizing session...")
        
        # Final cookie sync using native CDP (no delay needed)
        logger.info("Syncing cookies...")
        try:
            self.storage_monitor.monitor_cookie_changes(self)
            logger.info("✓ Cookies synced")
        except Exception as e:
            logger.error(f"Failed to sync cookies: {e}", exc_info=True)
        
        # Force final window property collection (non-blocking)
        logger.info("Triggering final window property collection...")
        try:
            self.window_property_monitor.force_collect(self)
        except Exception as e:
            logger.error(f"Could not trigger window property collection: {e}", exc_info=True)
        
        # Consolidate all transactions into a single JSON file
        logger.info("Starting transaction consolidation...")
        try:
            network_dir = self.paths.get('network_dir', os.path.join(self.output_dir, "network"))
            consolidated_path = self.paths.get('consolidated_transactions_json_path', 
                                               os.path.join(network_dir, "consolidated_transactions.json"))
            logger.info(f"Consolidating transactions to {consolidated_path}...")
            result = self.network_monitor.consolidate_transactions(consolidated_path)
            logger.info(f"Consolidate method returned, checking file...")
            if os.path.exists(consolidated_path):
                file_size = os.path.getsize(consolidated_path)
                logger.info(f"✓ Consolidated transactions saved to {consolidated_path} ({file_size} bytes)")
            else:
                logger.error(f"✗ Consolidated transactions file NOT created at {consolidated_path}")
        except Exception as e:
            logger.error(f"Failed to consolidate transactions: {e}", exc_info=True)
        
        # Generate HAR file from consolidated transactions
        logger.info("Starting HAR file generation...")
        try:
            network_dir = self.paths.get('network_dir', os.path.join(self.output_dir, "network"))
            har_path = self.paths.get('network_har_path', 
                                     os.path.join(network_dir, "network.har"))
            logger.info(f"Generating HAR file at {har_path}...")
            self.network_monitor.generate_har_from_transactions(har_path, "Web Hacker Session")
            logger.info(f"HAR method returned, checking file...")
            if os.path.exists(har_path):
                file_size = os.path.getsize(har_path)
                logger.info(f"✓ HAR file saved to {har_path} ({file_size} bytes)")
            else:
                logger.error(f"✗ HAR file NOT created at {har_path}")
        except Exception as e:
            logger.error(f"Failed to generate HAR file: {e}", exc_info=True)
        
        # Consolidate all interactions into a single JSON file
        logger.info("Consolidating interactions...")
        try:
            interaction_dir = self.paths.get('interaction_dir', os.path.join(self.output_dir, "interaction"))
            consolidated_interactions_path = self.paths.get('consolidated_interactions_json_path',
                                                           os.path.join(interaction_dir, "consolidated_interactions.json"))
            self.interaction_monitor.consolidate_interactions(consolidated_interactions_path)
            logger.info("✓ Interactions consolidated")
        except Exception as e:
            logger.error(f"Failed to consolidate interactions: {e}", exc_info=True)
        
        logger.info("Asset saving complete.")
    
    def run(self):
        """Main message processing loop."""
        logger.info("Blocking trackers & capturing network/storage… Press Ctrl+C to stop.")
        
        last_check_time = 0
        check_interval = 1.0  # Check every 1 second
        
        try:
            while True:
                try:
                    msg = json.loads(self.ws.recv())
                    self.handle_message(msg)
                    
                    # Check for periodic window property collection (throttled)
                    current_time = time.time()
                    if current_time - last_check_time >= check_interval:
                        try:
                            self.window_property_monitor.check_and_collect(self)
                        except Exception as e:
                            # Don't let window property collection errors crash the session
                            logger.debug(f"Window property collection error (non-fatal): {e}")
                        last_check_time = current_time
                except (WebSocketConnectionClosedException, ConnectionResetError, OSError) as e:
                    logger.info(f"WebSocket connection lost (tab may have been closed): {type(e).__name__}. Ending session.")
                    break
        except KeyboardInterrupt:
            logger.info("\nStopped. Saving assets...")
        finally:
            # Always finalize session (consolidate transactions, generate HAR, etc.)
            self._finalize_session()
            
            # Close WebSocket connection
            try:
                logger.info("Closing WebSocket connection...")
                self.ws.close()
                logger.info("WebSocket closed")
            except Exception as e:
                logger.warning(f"Error closing WebSocket: {e}")
    
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
        window_property_summary = self.window_property_monitor.get_window_property_summary()
        
        return {
            "network": network_summary,
            "storage": storage_summary,
            "interaction": interaction_summary,
            "window_properties": window_property_summary,
        }
