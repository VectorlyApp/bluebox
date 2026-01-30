"""
bluebox/cdp/monitors/async_dom_monitor.py

Async DOM monitor for CDP.
Captures full DOM snapshots on page load using DOMSnapshot.captureSnapshot.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Awaitable, Callable, ClassVar

from bluebox.cdp.monitors.abstract_async_monitor import AbstractAsyncMonitor
from bluebox.data_models.dom import DOMSnapshotEvent
from bluebox.utils.logger import get_logger

if TYPE_CHECKING:  # avoid circular import
    from bluebox.cdp.async_cdp_session import AsyncCDPSession

logger = get_logger(name=__name__)


class AsyncDOMMonitor(AbstractAsyncMonitor):
    """
    Async DOM monitor for CDP.
    Captures full DOM tree snapshots on page load.
    Uses callback pattern to emit events (no file saving).
    """

    # Class attributes _____________________________________________________________________________________________________

    # CSS properties to capture in computed styles
    COMPUTED_STYLE_PROPERTIES: ClassVar[list[str]] = [
        "display",
        "visibility",
        "opacity",
    ]

    # for streaming/storage limits
    URL_MAX_CHARS: ClassVar[int] = 150


    # Abstract method implementations ______________________________________________________________________________________

    @classmethod
    def get_ws_event_summary(cls, detail: dict[str, Any]) -> dict[str, Any]:
        """
        Extract a lightweight summary of a DOM snapshot event for WebSocket streaming.
        Args:
            detail: The full event detail dict emitted by the monitor.
        Returns:
            A simplified dict with fields relevant for real-time DOM monitoring.
        """
        documents = detail.get("documents", [])
        node_count = sum(
            len(doc.get("nodes", {}).get("nodeName", []))
            for doc in documents
        )
        return {
            "type": cls.get_monitor_category(),
            "url": (detail.get("url") or "")[:cls.URL_MAX_CHARS],
            "title": detail.get("title"),
            "document_count": len(documents),
            "node_count": node_count,
        }


    # Magic methods ________________________________________________________________________________________________________

    def __init__(
        self,
        event_callback_fn: Callable[[str, dict], Awaitable[None]]
    ) -> None:
        """
        Initialize AsyncDOMMonitor.
        Args:
            event_callback_fn: Async callback function that takes (category: str, detail: BaseCDPEvent).
                Called when DOM snapshots are captured.
        """
        self.event_callback_fn = event_callback_fn

        # tracking
        self.snapshot_count: int = 0
        self.pending_snapshot_cmd: dict[int, str] = {}  # cmd_id -> url
        self.current_url: str | None = None  # cached URL from Page.frameNavigated


    # Private methods ______________________________________________________________________________________________________

    async def _capture_snapshot(self, cdp_session: AsyncCDPSession, url: str) -> None:
        """
        Capture a DOM snapshot and emit it via callback.
        Args:
            cdp_session: The CDP session to use.
            url: The URL of the page being captured.
        """
        try:
            logger.info("📸 Capturing DOM snapshot for: %s", url[:100])

            # Get page title
            title: str | None = None
            try:
                title_result = await cdp_session.send_and_wait(
                    method="Runtime.evaluate",
                    params={
                        "expression": "document.title",
                        "returnByValue": True,
                    },
                    timeout=3.0,
                )
                if isinstance(title_result, dict):
                    title = title_result.get("result", {}).get("value")
            except Exception as e:
                logger.debug("⚠️ Could not get page title: %s", e)

            # Capture DOM snapshot
            snapshot_result = await cdp_session.send_and_wait(
                method="DOMSnapshot.captureSnapshot",
                params={
                    "computedStyles": self.COMPUTED_STYLE_PROPERTIES,
                    "includeDOMRects": True,
                    "includePaintOrder": False,
                    "includeBlendedBackgroundColors": False,
                    "includeTextColorOpacities": False,
                },
                timeout=30.0,  # DOM snapshots can take time on large pages
            )

            if not snapshot_result:
                logger.warning("⚠️ Empty DOMSnapshot.captureSnapshot result")
                return

            # Create event
            event = DOMSnapshotEvent(
                url=url,
                title=title,
                documents=snapshot_result.get("documents", []),
                strings=snapshot_result.get("strings", []),
                computed_styles=self.COMPUTED_STYLE_PROPERTIES,
            )

            # Emit event
            self.snapshot_count += 1
            await self.event_callback_fn(
                self.get_monitor_category(),
                event.model_dump(),
            )
            logger.info(
                "✅ DOM snapshot captured: %d documents, %d strings",
                len(event.documents),
                len(event.strings),
            )

        except Exception as e:
            logger.error("❌ Failed to capture DOM snapshot: %s", e, exc_info=True)


    # Public methods _______________________________________________________________________________________________________

    async def setup_dom_monitoring(self, cdp_session: AsyncCDPSession) -> None:
        """
        Setup DOM snapshot monitoring.
        Enables DOMSnapshot domain.
        Args:
            cdp_session: The CDP session to use.
        """
        logger.info("🔧 Setting up DOM snapshot monitoring...")

        # Enable DOMSnapshot domain
        await cdp_session.enable_domain("DOMSnapshot")

        logger.info("✅ DOM snapshot monitoring setup complete")

    async def handle_dom_message(self, msg: dict, cdp_session: AsyncCDPSession) -> bool:
        """
        Handle CDP messages for DOM monitoring.
        Tracks URL from Page.frameNavigated and triggers snapshot capture on Page.loadEventFired.
        Args:
            msg: The CDP message dict.
            cdp_session: The CDP session for sending commands.
        Returns:
            True if the message was handled and should not be processed further.
        """
        method = msg.get("method")

        # Track URL from navigation events (for main frame only)
        if method == "Page.frameNavigated":
            params = msg.get("params", {})
            frame = params.get("frame", {})
            # Only track main frame navigations (no parentId)
            if not frame.get("parentId"):
                self.current_url = frame.get("url")
                logger.debug("📍 DOM monitor tracking URL: %s", self.current_url)
            return False

        # Capture snapshot when page finishes loading
        # Use asyncio.create_task to avoid blocking the message handler (deadlock)
        if method == "Page.loadEventFired":
            if self.current_url:
                # Schedule snapshot capture as background task to avoid deadlock
                asyncio.create_task(self._capture_snapshot(cdp_session, self.current_url))
            else:
                logger.warning("⚠️ Page.loadEventFired but no URL tracked from frameNavigated")
            return False  # allow other handlers to process this event too

        return False

    async def handle_dom_command_reply(self, msg: dict, cdp_session: AsyncCDPSession) -> bool:
        """
        Handle CDP command replies for DOM monitoring.
        Args:
            msg: The CDP message dict.
            cdp_session: The CDP session.
        Returns:
            True if the message was handled.
        """
        # Currently no pending commands to track
        # DOMSnapshot.captureSnapshot responses are handled via send_and_wait
        return False

    def get_dom_summary(self) -> dict[str, Any]:
        """
        Get summary of DOM monitoring activity.
        Returns:
            Dictionary with DOM monitoring statistics.
        """
        return {
            "snapshot_count": self.snapshot_count,
        }
