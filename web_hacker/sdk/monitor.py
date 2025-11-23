"""
Browser monitoring SDK wrapper.
"""

from typing import Optional, Set
from pathlib import Path
import logging
import sys
import time
import threading

from ..cdp.cdp_session import CDPSession
from ..cdp.tab_managements import cdp_new_tab, dispose_context
from ..data_models.network import ResourceType
from ..utils.exceptions import BrowserConnectionError

logger = logging.getLogger(__name__)


class BrowserMonitor:
    """
    High-level interface for monitoring browser activity.
    
    Example:
        >>> monitor = BrowserMonitor(output_dir="./captures")
        >>> with monitor:
        ...     # User performs actions in browser
        ...     pass
        >>> summary = monitor.get_summary()
    """
    
    def __init__(
        self,
        remote_debugging_address: str = "http://127.0.0.1:9222",
        output_dir: str = "./cdp_captures",
        url: str = "about:blank",
        incognito: bool = True,
        block_patterns: Optional[list[str]] = None,
        capture_resources: Optional[Set] = None,
        create_tab: bool = True,
        clear_cookies: bool = False,
        clear_storage: bool = False,
    ):
        self.remote_debugging_address = remote_debugging_address
        self.output_dir = output_dir
        self.url = url
        self.incognito = incognito
        self.block_patterns = block_patterns
        self.capture_resources = capture_resources or {
            ResourceType.XHR,
            ResourceType.FETCH,
            ResourceType.DOCUMENT,
            ResourceType.SCRIPT,
            ResourceType.IMAGE,
            ResourceType.MEDIA
        }
        self.create_tab = create_tab
        self.clear_cookies = clear_cookies
        self.clear_storage = clear_storage
        
        self.session: Optional[CDPSession] = None
        self.context_id: Optional[str] = None
        self.created_tab = False
        self.start_time: Optional[float] = None
        self._run_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
    
    def start(self) -> None:
        """Start monitoring session."""
        self.start_time = time.time()
        
        # Create output directory structure
        paths = {
            "output_dir": self.output_dir,
            "network_dir": str(Path(self.output_dir) / "network"),
            "transactions_dir": str(Path(self.output_dir) / "network" / "transactions"),
            "storage_dir": str(Path(self.output_dir) / "storage"),
            "interaction_dir": str(Path(self.output_dir) / "interaction"),
        }
        
        # Create directories
        for path in paths.values():
            Path(path).mkdir(parents=True, exist_ok=True)
        
        # Get or create browser tab
        if self.create_tab:
            try:
                target_id, browser_context_id, ws = cdp_new_tab(
                    remote_debugging_address=self.remote_debugging_address,
                    incognito=self.incognito,
                    url=self.url,
                )
                self.context_id = browser_context_id
                self.created_tab = True
                ws_url = ws
            except Exception as e:
                raise BrowserConnectionError(f"Failed to create browser tab: {e}")
        else:
            # Connect to existing browser
            try:
                import requests
                ver = requests.get(f"{self.remote_debugging_address}/json/version", timeout=5)
                ver.raise_for_status()
                data = ver.json()
                ws_url = data.get("webSocketDebuggerUrl")
                if not ws_url:
                    raise BrowserConnectionError("Could not get WebSocket URL from browser")
            except Exception as e:
                raise BrowserConnectionError(f"Failed to connect to browser: {e}")
        
        # Initialize CDP session
        self.session = CDPSession(
            ws_url=ws_url,
            output_dir=paths["network_dir"],  # Use network directory for response bodies
            paths=paths,
            capture_resources=self.capture_resources,
            block_patterns=self.block_patterns or [],
            clear_cookies=self.clear_cookies,
            clear_storage=self.clear_storage,
        )
        
        self.session.setup_cdp(self.url if self.create_tab else None)
        
        # Start the monitoring loop in a separate thread
        self._stop_event.clear()
        self._run_thread = threading.Thread(target=self._run_monitoring_loop, daemon=True)
        self._run_thread.start()
        
        logger.info(f"Browser monitoring started. Output directory: {self.output_dir}")
    
    def _run_monitoring_loop(self):
        """Run the monitoring loop in a separate thread."""
        if not self.session:
            return
        
        try:
            import json
            
            # Set a timeout on the websocket to allow checking stop event
            if hasattr(self.session.ws, 'settimeout'):
                self.session.ws.settimeout(1.0)
            
            while not self._stop_event.is_set():
                try:
                    msg = json.loads(self.session.ws.recv())
                    self.session.handle_message(msg)
                except Exception as e:
                    if self._stop_event.is_set():
                        break
                    # Check if it's a timeout (which is expected)
                    if "timed out" in str(e).lower() or "timeout" in str(e).lower():
                        continue
                    logger.warning(f"Error in monitoring loop: {e}")
                    break
        except KeyboardInterrupt:
            pass
        finally:
            # Final cookie sync
            try:
                if self.session:
                    self.session.storage_monitor.monitor_cookie_changes(self.session)
            except:
                pass
            
            # Consolidate transactions
            try:
                if self.session:
                    consolidated_path = f"{self.output_dir}/network/consolidated_transactions.json"
                    self.session.network_monitor.consolidate_transactions(consolidated_path)
            except:
                pass
            
            # Generate HAR file
            try:
                if self.session:
                    har_path = f"{self.output_dir}/network/network.har"
                    self.session.network_monitor.generate_har_from_transactions(har_path, "Web Hacker Session")
            except:
                pass
            
            # Consolidate interactions
            try:
                if self.session:
                    interaction_dir = self.session.paths.get('interaction_dir', f"{self.output_dir}/interaction")
                    consolidated_interactions_path = str(Path(interaction_dir) / "consolidated_interactions.json")
                    self.session.interaction_monitor.consolidate_interactions(consolidated_interactions_path)
            except:
                pass
    
    def stop(self) -> dict:
        """Stop monitoring and return summary."""
        if not self.session:
            return {}
        
        # Signal stop
        self._stop_event.set()
        
        # Wait for thread to finish (with timeout)
        if self._run_thread and self._run_thread.is_alive():
            self._run_thread.join(timeout=5.0)
        
        # Close WebSocket
        try:
            if self.session.ws:
                self.session.ws.close()
        except Exception as e:
            logger.warning(f"Error closing WebSocket: {e}")
        
        summary = self.get_summary()
        
        # Cleanup
        if self.created_tab and self.context_id:
            try:
                dispose_context(self.remote_debugging_address, self.context_id)
            except Exception as e:
                logger.warning(f"Could not dispose browser context: {e}")
        
        end_time = time.time()
        summary["duration"] = end_time - (self.start_time or end_time)
        
        logger.info("Browser monitoring stopped.")
        return summary
    
    def get_summary(self) -> dict:
        """Get current monitoring summary without stopping."""
        if not self.session:
            return {}
        return self.session.get_monitoring_summary() if self.session else {}
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()