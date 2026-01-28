"""
bluebox/cdp/async_cdp_session.py

Single comprehensive class for asynchronous CDP session monitoring.
"""

import asyncio
import json
from typing import Any, Awaitable, Callable

from websockets.asyncio.client import connect, ClientConnection

from bluebox.cdp.monitors.async_interaction_monitor import AsyncInteractionMonitor
from bluebox.cdp.monitors.async_network_monitor import AsyncNetworkMonitor
from bluebox.cdp.monitors.async_storage_monitor import AsyncStorageMonitor
from bluebox.cdp.monitors.async_window_property_monitor import AsyncWindowPropertyMonitor
from bluebox.utils.logger import get_logger

logger = get_logger(name=__name__)


class AsyncCDPSession:
    """
    Single comprehensive class for testing async CDP session monitoring.
    Handles WebSocket, CDP commands, network events, and callbacks all in one place.
    """

    # Magic methods ________________________________________________________________________________________________________

    def __init__(
        self,
        ws_url: str,
        session_start_dtm: str,
        event_callback_fn: Callable[[str, dict], Awaitable[None]],
        paths: dict[str, str] | None = None,
    ) -> None:
        """
        Initialize AsyncCDPSession.
        Args:
            ws_url: WebSocket URL to connect to the browser session.
            session_start_dtm: Session start datetime in format YYYY-MM-DDTHH-MM-SSZ.
            event_callback_fn: Async callback function that takes (category: str, detail: BaseCDPEvent).
                Called when CDP events are captured. Caller can use this to store events, stream them, etc.
            paths: Optional dict of file paths for output. Used by finalize() for consolidation.
                Expected keys: 'network_events_path', 'consolidated_transactions_json_path',
                'network_har_path'. If not provided, finalize() will skip file operations.
        NOTE:
            The CDP sessionId will be obtained automatically in run() after connecting.
            CDP sessionIds are only valid for the specific WebSocket connection where Target.attachToTarget was called.
        """
        logger.info("🔧 Initializing AsyncCDPSession")

        self.ws_url = ws_url
        self.event_callback_fn = event_callback_fn
        self.session_start_dtm = session_start_dtm
        self.paths = paths or {}
        self.ws: ClientConnection | None = None
        self.seq = 0  # sequence ID for CDP commands

        # response tracking for CDP commands
        self.pending_responses: dict[int, asyncio.Future] = {}  # command ID -> future

        # track enabled CDP domains to avoid duplicate enables
        self._enabled_domains: set[str] = set()  # e.g., {"Page", "Runtime", "Network"}

        # page-level session ID (Chrome's CDP sessionId, _not_ OnKernel's session_id)
        # will be obtained in run() after connecting to the WebSocket
        # needed for Page, Runtime, Network domains when using Target.setAutoAttach with flatten:True
        self.page_session_id: str | None = None
        self._session_id_event = asyncio.Event()  # event to signal when sessionId is captured

        # initialize monitors
        self.network_monitor = AsyncNetworkMonitor(event_callback_fn=self.event_callback_fn)
        self.storage_monitor = AsyncStorageMonitor(event_callback_fn=self.event_callback_fn)
        self.window_property_monitor = AsyncWindowPropertyMonitor(event_callback_fn=self.event_callback_fn)
        self.interaction_monitor = AsyncInteractionMonitor(event_callback_fn=self.event_callback_fn)


    # Private methods ______________________________________________________________________________________________________

    async def _get_ws_cdp_session_id(self) -> None:
        """
        Get CDP sessionId from the current WebSocket connection.
        Must be called after connecting and starting the message receiver.
        """
        logger.info("🔍 Getting CDP session ID from current WebSocket connection...")
        try:
            # Step 1: Get targets to find the page targetId
            targets_result = await self.send_and_wait(method="Target.getTargets", timeout=5.0)
            cdp_target_id: str | None = None
            if targets_result and "targetInfos" in targets_result:
                for target_info in targets_result["targetInfos"]:
                    if target_info.get("type") == "page":
                        cdp_target_id = target_info.get("targetId")
                        logger.info("✅ Found page targetId: %s (url: %s)", cdp_target_id, target_info.get("url", "unknown"))
                        break

            if not cdp_target_id:
                logger.error("❌ No page target found in Target.getTargets result")
                raise RuntimeError("No page target found")

            # Step 2: Attach to the page target to get the CDP sessionId
            attach_result = await self.send_and_wait(
                method="Target.attachToTarget",
                params={"targetId": cdp_target_id, "flatten": True},
                timeout=5.0
            )
            if attach_result and "sessionId" in attach_result:
                self.page_session_id = attach_result["sessionId"]
                self._session_id_event.set()
                logger.debug("✅ Got CDP sessionId from current connection: %s", self.page_session_id)
                logger.debug("   This is Chrome's CDP sessionId (NOT OnKernel's session_id)")
            else:
                logger.error("❌ No sessionId in Target.attachToTarget response")
                raise RuntimeError("No sessionId in Target.attachToTarget response")
        except Exception as e:
            logger.error("❌ Failed to get CDP session ID: %s", e, exc_info=True)
            raise

    async def _handle_command_reply(self, msg: dict) -> None:
        """Handle CDP command replies."""
        cmd_id = msg.get("id")

        # try network monitor first
        handled = await self.network_monitor.handle_network_command_reply(msg, self)
        if handled:
            return

        # try storage monitor
        handled = await self.storage_monitor.handle_storage_command_reply(msg)
        if handled:
            return

        # try interaction monitor
        handled = await self.interaction_monitor.handle_interaction_command_reply(msg)
        if handled:
            return

        # handle general command replies
        if cmd_id is not None and cmd_id in self.pending_responses:
            future = self.pending_responses.pop(cmd_id)
            #logger.info("📥 Found pending response for id=%s", cmd_id)

            if "result" in msg:
                future.set_result(msg["result"])
            elif "error" in msg:
                logger.error("📥 Setting error: %s", json.dumps(msg["error"], indent=2))
                future.set_exception(Exception(f"CDP error: {msg['error']}"))
            else:
                future.set_result(None)
            return

        logger.info("📥 Command reply not handled: id=%s", cmd_id)


    # Public methods _______________________________________________________________________________________________________

    async def enable_domain(
        self, 
        domain: str, 
        params: dict | None = None, 
        timeout: float = 2.0,
        wait_for_response: bool = True,
    ) -> None:
        """
        Enable a CDP domain idempotently (skip if already enabled).
        Args:
            domain: The CDP domain name (e.g., "Page", "Network", "Runtime").
            params: Optional parameters for the enable command.
            timeout: Timeout in seconds (only used if wait_for_response=True).
            wait_for_response: If True, wait for response. If False, fire-and-forget.
        """
        # check if already enabled
        if domain in self._enabled_domains:
            logger.debug("⏭️ Domain %s already enabled, skipping", domain)
            return

        # enable the domain
        method = f"{domain}.enable"
        try:
            if wait_for_response:
                await self.send_and_wait(
                    method=method,
                    params=params,
                    timeout=timeout,
                )
            else:
                await self.send(
                    method=method,
                    params=params,
                )

            # add to enabled domains on success
            self._enabled_domains.add(domain)
            logger.debug("✅ Domain %s enabled", domain)
        except Exception as e:
            logger.warning("⚠️ Failed to enable domain %s: %s", domain, e)
            # don't add to enabled set on failure

    async def wait_for_page_session_id(self, timeout: float = 5.0) -> str | None:
        """
        Wait for page sessionId to be captured from Target.attachedToTarget.
        Args:
            timeout: Maximum time to wait in seconds.
        Returns:
            The page sessionId if captured, None if timeout.
        """
        if self.page_session_id:
            return self.page_session_id
        try:
            await asyncio.wait_for(
                fut=self._session_id_event.wait(),
                timeout=timeout,
            )
            return self.page_session_id
        except asyncio.TimeoutError:
            logger.warning("⏱️ Timeout waiting for page sessionId after %s seconds", timeout)
            return None

    async def send(self, method: str, params: dict | None = None) -> int:
        """
        Send CDP command and return sequence ID.
        Args:
            method (str): The CDP method to send. For example, "Page.setViewportSize".
            params (dict | None): The parameters to send with the command.
        Returns:
            int: The sequence ID of the command.
        """
        if not self.ws:
            raise RuntimeError("WebSocket not connected")

        self.seq += 1
        cmd_id = self.seq

        # warn if trying to use page-level domain without sessionId;        
        # page-level domains need sessionId; browser-level domains (Target, Fetch) do not
        page_level_domains = {"Page", "Runtime", "Network", "DOMStorage", "IndexedDB"}
        domain_name = method.split(".")[0] if "." in method else None
        is_page_level = domain_name in page_level_domains if domain_name else False
        if is_page_level and not self.page_session_id:
            logger.warning(
                "⚠️ Sending page-level command %s without sessionId (may fail). "
                "SessionId should be obtained via Target.attachToTarget first.",
                method
            )

        # always include sessionId if we have one (works for both page-level and browser-level domains)
        msg = {
            "id": cmd_id,
            "method": method,
            "params": params or {},
        }
        if self.page_session_id:
            msg["sessionId"] = self.page_session_id
        msg_json = json.dumps(msg)

        await self.ws.send(msg_json)
        return cmd_id
    
    async def send_and_wait(
        self,
        method: str,
        params: dict | None = None,
        timeout: float = 10.0,
    ) -> dict | None:
        """
        Send CDP command and wait for response asynchronously.
        Args:
            method: The CDP method to send.
            params: The parameters to send with the command.
            timeout: Timeout in seconds.
        Returns:
            The result from the CDP command.
        """
        cmd_id = await self.send(method, params)

        # create a future for this command
        future = asyncio.Future()
        self.pending_responses[cmd_id] = future
        try:
            # wait for response with timeout
            result = await asyncio.wait_for(fut=future, timeout=timeout)
            return result
        except asyncio.TimeoutError:
            # clean up on timeout
            self.pending_responses.pop(cmd_id, None)
            raise TimeoutError(f"CDP command {method} timed out after {timeout} seconds")
        except Exception as e:
            # clean up on error
            self.pending_responses.pop(cmd_id, None)
            raise e

    async def setup_cdp(self) -> None:
        """Setup CDP domains and configuration."""
        logger.info("🔧 Setting up CDP domains...")

        # get CDP sessionId from this WebSocket connection (must be from the same connection!)
        # sets self.page_session_id
        await self._get_ws_cdp_session_id()

        # enable basic domains (idempotent)
        await self.enable_domain("Page")
        await self.enable_domain("Runtime")

        # setup monitoring
        await self.network_monitor.setup_network_monitoring(self)
        await self.storage_monitor.setup_storage_monitoring(self)
        await self.window_property_monitor.setup_window_property_monitoring(self)
        await self.interaction_monitor.setup_interaction_monitoring(self)
        logger.info("✅ CDP domain setup complete")

    async def get_current_url(self, timeout: float = 3.0) -> str | None:
        """
        Return the current page URL using CDP. Uses navigation history first, then JS evaluation.
        Args:
            timeout: Timeout per CDP call.
        """
        try:
            # try navigation history first
            browser_history = await self.send_and_wait(
                method="Page.getNavigationHistory",
                timeout=timeout,
            )
            current_url: str | None = None
            if browser_history:
                current_index = browser_history.get("currentIndex", 0)
                entries = browser_history.get("entries", [])
                current_entry = (
                    entries[current_index]
                    if 0 <= current_index < len(entries)
                    else None
                )
                current_url = current_entry.get("url") if current_entry else None
            # fallback to JS evaluation if navigation history is not available
            if not current_url:
                eval_result = await self.send_and_wait(
                    method="Runtime.evaluate",
                    params={
                        "expression": "window.location.href",
                        "returnByValue": True,
                    },
                    timeout=timeout,
                )
                if isinstance(eval_result, dict):
                    current_url = eval_result.get("result", {}).get("value")

            return current_url
        except Exception as e:
            logger.debug("⚠️ Failed to get current URL: %s", e)
            return None

    async def handle_message(self, msg: dict) -> None:
        """Handle incoming CDP message."""
        method = msg.get("method")

        # capture sessionId from Target.attachedToTarget (needed for page-level domains)
        if method == "Target.attachedToTarget":
            params = msg.get("params", {})
            captured_session_id = params.get("sessionId")
            target_info = params.get("targetInfo", {})
            target_type = target_info.get("type", "unknown")
            if captured_session_id and target_type in ["page", "iframe"]:
                self.page_session_id = captured_session_id
                self._session_id_event.set()  # signal that sessionId is available
                logger.info("🎯 Captured page sessionId: %s (target type: %s)", self.page_session_id, target_type)

        # handle network events via AsyncNetworkMonitor
        handled_network = await self.network_monitor.handle_network_message(msg, self)
        if handled_network:
            return

        # handle storage events via AsyncStorageMonitor
        handled_storage = await self.storage_monitor.handle_storage_message(msg, self)
        if handled_storage:
            return

        # handle window property events via AsyncWindowPropertyMonitor
        handled_window_property = await self.window_property_monitor.handle_window_property_message(msg, self)
        if handled_window_property:
            return

        # handle interaction events via AsyncInteractionMonitor
        handled_interaction = await self.interaction_monitor.handle_interaction_message(msg, self)
        if handled_interaction:
            return

        # handle command replies
        if "id" in msg:
            #logger.debug("📥 Processing as COMMAND REPLY (id=%s)", msg.get("id"))
            await self._handle_command_reply(msg)
            return

        # any event that reached this point is unhandled

    async def run(self) -> None:
        """Main message processing loop."""
        logger.info("🔌 Connecting to CDP: %s", self.ws_url)
        async with connect(uri=self.ws_url, max_size=None) as ws:
            self.ws = ws
            logger.info("✅ WebSocket connected")

            # start message receiver task BEFORE setup_cdp so responses can be received
            message_count = 0
            message_receiver_done = asyncio.Event()

            async def message_receiver() -> None:
                """Receive and process WebSocket messages."""
                nonlocal message_count
                try:
                    async for message in ws:
                        message_count += 1
                        if message_count % 500 == 0:
                            # log total message count once every 500 messages
                            logger.info("📊📊📊 Processed %d messages total", message_count)
                        try:
                            msg = json.loads(message)
                            await self.handle_message(msg)
                        except Exception as e:
                            logger.error("❌ Error handling message #%d: %s", message_count, e, exc_info=True)
                            logger.error("❌ Message was: %s", message[:250])
                except asyncio.CancelledError:
                    logger.info("🛑 Message receiver cancelled (processed %d messages)", message_count)
                    raise
                except Exception as e:
                    logger.error("❌ Error in message receiver: %s", e, exc_info=True)
                finally:
                    message_receiver_done.set()

            # start message receiver as background task
            receiver_task = asyncio.create_task(coro=message_receiver())

            # give message receiver a moment to start
            await asyncio.sleep(0.1)

            # setup CDP (message receiver is running, so responses can be received)
            # setup_cdp() will get the sessionId first, then enable domains
            await self.setup_cdp()
            logger.info("✅ CDP setup complete, message loop running")

            try:
                # wait for message receiver to complete
                await receiver_task
            except asyncio.CancelledError:
                logger.info("🛑 Session cancelled (processed %d messages)", message_count)
                receiver_task.cancel()
                try:
                    await receiver_task
                except asyncio.CancelledError:
                    pass

                await self.storage_monitor.monitor_cookie_changes(self)
                logger.info("✅ Cookies synced")
                raise
            except Exception as e:
                logger.error("❌ Connection error: %s", e, exc_info=True)

    async def finalize(self) -> None:
        """
        Finalize the session by syncing cookies, collecting window properties,
        and consolidating network data.

        Call this after run() completes or is cancelled.
        Requires self.paths to be set with appropriate file paths.
        """
        logger.info("🔧 Finalizing session...")

        # Final cookie sync
        try:
            await self.storage_monitor.monitor_cookie_changes(self)
            logger.info("✅ Cookies synced")
        except Exception as e:
            logger.warning("⚠️ Could not sync cookies: %s", e)

        # Force final window property collection
        try:
            await self.window_property_monitor.force_collect(self)
            logger.info("✅ Window properties collected")
        except Exception as e:
            logger.warning("⚠️ Could not collect window properties: %s", e)

        # Consolidate network transactions (if paths provided)
        network_events_path = self.paths.get("network_events_path")
        if network_events_path:
            consolidated_path = self.paths.get("consolidated_transactions_json_path")
            if consolidated_path:
                try:
                    AsyncNetworkMonitor.consolidate_transactions(
                        network_events_path=network_events_path,
                        output_path=consolidated_path,
                    )
                    logger.info("✅ Transactions consolidated")
                except Exception as e:
                    logger.error("❌ Failed to consolidate transactions: %s", e)

            # Generate HAR file
            har_path = self.paths.get("network_har_path")
            if har_path:
                try:
                    AsyncNetworkMonitor.generate_har_from_transactions(
                        network_events_path=network_events_path,
                        har_path=har_path,
                        title="Web Hacker Session",
                    )
                    logger.info("✅ HAR file generated")
                except Exception as e:
                    logger.error("❌ Failed to generate HAR file: %s", e)

        logger.info("✅ Session finalization complete")

    def get_monitoring_summary(self) -> dict[str, Any]:
        """
        Get summary of all monitoring activities.
        Returns:
            Dictionary with summaries from all monitors.
        """
        return {
            "network": self.network_monitor.get_network_summary(),
            "storage": self.storage_monitor.get_storage_summary(),
            "window_properties": self.window_property_monitor.get_window_property_summary(),
            "interactions": self.interaction_monitor.get_interaction_summary(),
        }
