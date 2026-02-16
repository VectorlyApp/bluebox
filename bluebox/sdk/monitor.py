"""
bluebox/sdk/monitor.py

Browser monitoring SDK wrapper.

Contains:
- BrowserMonitor: Wraps AsyncCDPSession for easy browser capture
- start(): Begin capturing network, storage, interactions
- stop(): End capture, save data to output directory
- Outputs: network/events.jsonl, storage/events.jsonl, etc.
"""

import asyncio
import time
from typing import Awaitable, Any, Callable
from datetime import datetime, timezone

import requests

from bluebox.cdp.async_cdp_session import AsyncCDPSession
from bluebox.cdp.file_event_writer import FileEventWriter
from bluebox.cdp.connection import cdp_new_tab, dispose_context
from bluebox.utils.exceptions import BrowserConnectionError
from bluebox.utils.logger import get_logger

logger = get_logger(__name__)


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
        create_tab: bool = True,
        event_callback_fn: Callable[[str, dict], Awaitable[None]] | None = None,
    ):
        self.remote_debugging_address = remote_debugging_address
        self.output_dir = output_dir
        self.url = url
        self.incognito = incognito
        self.create_tab = create_tab
        self.event_callback_fn = event_callback_fn

        self.session: AsyncCDPSession | None = None
        self.context_id: str | None = None
        self.created_tab = False
        self.start_time: float | None = None
        self._run_task: asyncio.Task | None = None
        self._finalized = False

    def start(self) -> None:
        """Start monitoring session. Must be called from within an async context."""
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.astart())

    async def astart(self) -> None:
        """Start monitoring session (async)."""
        self.start_time = time.time()
        self._finalized = False

        # Set up file writer (creates output directory structure)
        writer = FileEventWriter.create_from_output_dir(self.output_dir)

        # If caller provided a custom callback, wrap it to also write to files
        if self.event_callback_fn:
            original_callback = self.event_callback_fn

            async def combined_callback(category: str, event: Any) -> None:
                await writer.write_event(category, event)
                await original_callback(category, event)

            callback = combined_callback
        else:
            callback = writer.write_event

        # Get or create browser tab
        ws_url = await self._resolve_ws_url()

        # Initialize async CDP session
        self.session = AsyncCDPSession(
            ws_url=ws_url,
            session_start_dtm=datetime.now(timezone.utc).isoformat(),
            event_callback_fn=callback,
            paths=writer.paths,
        )

        # Start the monitoring loop as an async task
        self._run_task = asyncio.create_task(self.session.run())

        logger.info(f"Browser monitoring started. Output directory: {self.output_dir}")

    async def _resolve_ws_url(self) -> str:
        """Create or find a browser tab and return the page-level WebSocket URL."""
        if self.create_tab:
            try:
                target_id, browser_context_id, browser_ws = cdp_new_tab(
                    remote_debugging_address=self.remote_debugging_address,
                    incognito=self.incognito,
                    url=self.url,
                )
                try:
                    browser_ws.close()
                except Exception:
                    pass
                self.context_id = browser_context_id
                self.created_tab = True
            except Exception as e:
                raise BrowserConnectionError(f"Failed to create browser tab: {e}")
        else:
            # Attach to existing tab or create one if none exist
            try:
                resp = requests.get(f"{self.remote_debugging_address}/json/list", timeout=5)
                resp.raise_for_status()
                tabs = resp.json()
                page_tabs = [t for t in tabs if t.get("type") == "page"]

                if page_tabs:
                    target_id = page_tabs[0]["id"]
                else:
                    logger.info("No existing page tabs found, creating a new tab...")
                    target_id, browser_context_id, browser_ws = cdp_new_tab(
                        remote_debugging_address=self.remote_debugging_address,
                        incognito=self.incognito,
                        url=self.url,
                    )
                    try:
                        browser_ws.close()
                    except Exception:
                        pass
                    self.context_id = browser_context_id
                    self.created_tab = True
            except BrowserConnectionError:
                raise
            except Exception as e:
                raise BrowserConnectionError(f"Failed to connect to browser: {e}")

        host_port = self.remote_debugging_address.replace("http://", "").replace("https://", "")
        return f"ws://{host_port}/devtools/page/{target_id}"

    def _is_browser_connected(self) -> bool:
        """Check if browser is still connected and responsive."""
        try:
            response = requests.get(
                f"{self.remote_debugging_address}/json/version",
                timeout=1
            )
            return response.status_code == 200
        except Exception:
            return False

    async def _finalize_session(self) -> None:
        """Finalize session: consolidate data files."""
        if self._finalized:
            return
        self._finalized = True

        if not self.session:
            return

        try:
            await self.session.finalize()
        except Exception as e:
            logger.warning(f"Could not finalize session: {e}")

    def stop(self) -> dict:
        """Stop monitoring and return summary. Must be called from within an async context."""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.astop())

    async def astop(self) -> dict:
        """Stop monitoring and return summary (async)."""
        if not self.session:
            return {}

        # Cancel the run task
        if self._run_task and not self._run_task.done():
            self._run_task.cancel()
            try:
                await self._run_task
            except (asyncio.CancelledError, Exception):
                pass

        # Finalize (consolidate transactions, HAR, etc.)
        await self._finalize_session()

        summary = self.get_summary()

        # Cleanup browser context if we created one and browser is still up
        if self.created_tab and self.context_id and self._is_browser_connected():
            try:
                dispose_context(browser_context_id=self.context_id, remote_debugging_address=self.remote_debugging_address)
            except Exception as e:
                logger.debug(f"Could not dispose browser context: {e}")

        end_time = time.time()
        summary["duration"] = end_time - (self.start_time or end_time)

        logger.info("Browser monitoring stopped.")
        return summary

    @property
    def is_alive(self) -> bool:
        """Check if the monitoring task is still running."""
        return self._run_task is not None and not self._run_task.done()

    def get_summary(self) -> dict:
        """Get current monitoring summary without stopping."""
        if not self.session:
            return {}
        return self.session.get_monitoring_summary()

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()

    async def __aenter__(self):
        """Async context manager entry."""
        await self.astart()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.astop()
