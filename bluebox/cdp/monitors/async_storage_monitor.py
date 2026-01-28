"""
bluebox/cdp/monitors/async_storage_monitor.py

Async storage monitor for CDP.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from bluebox.cdp.monitors.abstract_async_monitor import AbstractAsyncMonitor
from bluebox.data_models.cdp import StorageEvent
from bluebox.utils.logger import get_logger

if TYPE_CHECKING:
    from bluebox.cdp.async_cdp_session import AsyncCDPSession

logger = get_logger(name=__name__)


class AsyncStorageMonitor(AbstractAsyncMonitor):
    """
    Monitors browser storage changes using native CDP events (no JavaScript injection).
    """

    # Abstract method implementations ______________________________________________________________________________________

    @classmethod
    def get_ws_event_summary(cls, detail: dict[str, Any]) -> dict[str, Any]:
        """
        Extract a lightweight summary of a storage event for WebSocket streaming.
        Args:
            detail: The full event detail dict emitted by the monitor.
        Returns:
            A simplified dict with fields relevant for real-time storage monitoring.
        """
        event_type = detail.get("type", "unknown")
        summary: dict[str, Any] = {
            "type": cls.get_monitor_category(),
            "event_type": event_type,
        }

        # add relevant fields based on event type
        if "cookie" in event_type.lower():
            summary["cookie_count"] = detail.get("total_count")
            # for changes, include counts
            if event_type == "cookieChange":
                summary["added_count"] = len(detail.get("added", []))
                summary["modified_count"] = len(detail.get("modified", []))
                summary["removed_count"] = len(detail.get("removed", []))
        elif "storage" in event_type.lower():
            summary["origin"] = detail.get("origin")
            summary["key"] = detail.get("key")

        return summary


    # Magic methods ________________________________________________________________________________________________________

    def __init__(
        self,
        event_callback_fn: Callable[[str, dict], Awaitable[None]]
    ) -> None:
        """
        Initialize AsyncStorageMonitor.
        Args:
            event_callback_fn: Async callback function that takes (category: str, detail: BaseCDPEvent).
                Called when storage events are captured.
        """
        self.event_callback_fn = event_callback_fn

        # storage state tracking
        self.cookies_state: dict[str, dict[str, Any]] = {}  # "domain:name" -> cookie dict
        self.local_storage_state: dict[str, dict[str, str]] = {}  # origin -> {key -> value}
        self.session_storage_state: dict[str, dict[str, str]] = {}  # origin -> {key -> value}
        self.indexed_db_state: dict[str, Any] = {}  # database_id -> database info
        self.cache_storage_state: dict[str, Any] = {}  # cache_id -> cache info

        # storage command tracking
        self.pending_storage_commands: dict[int, dict[str, Any]] = {}  # command_id -> command details

        # debouncing for native cookie checks
        self._last_native_cookie_check: float = 0.0  # current time in seconds


    # Private methods ______________________________________________________________________________________________________

    async def _get_initial_cookies(self, cdp_session: AsyncCDPSession) -> None:
        """Get initial cookie state using native CDP."""
        logger.debug("🍪 Getting initial cookie state...")
        # use Network.getAllCookies as primary method since Storage.getCookies may not exist
        try:
            cmd_id = await cdp_session.send(method="Network.getAllCookies")
            logger.debug("🍪 Sent initial Network.getAllCookies command, cmd_id=%s", cmd_id)
            self.pending_storage_commands[cmd_id] = {
                "type": "getAllCookies",
                "initial": True
            }
        except Exception as e:
            logger.error("❌ Failed to get cookies via Network.getAllCookies: %s", e, exc_info=True)

    async def _get_initial_dom_storage(self) -> None:
        """Get initial DOM storage state."""
        # we'll discover storage IDs through events or by scanning the page;
        # for now, we'll set up listeners for storage events
        pass

    async def _trigger_native_cookie_check(self, cdp_session: AsyncCDPSession) -> None:
        """
        Trigger immediate cookie check using native CDP (with debouncing).
        Args:
            cdp_session: The CDP session to use.
        """
        current_time = time.time()
        # debounce to prevent spam (max once per 500ms)
        if (current_time - self._last_native_cookie_check) > 0.5:
            self._last_native_cookie_check = current_time
            # use Network.getAllCookies as primary method
            try:
                cmd_id = await cdp_session.send(method="Network.getAllCookies")
                self.pending_storage_commands[cmd_id] = {
                    "type": "getAllCookies",
                    "triggered_by": "native_event"
                }
            except Exception as e:
                logger.error("❌ Failed to get cookies via Network.getAllCookies: %s", e, exc_info=True)
        else:
            logger.debug(
                "🍪 Cookie check skipped (debounced, last check was %.2fs ago)", 
                current_time - self._last_native_cookie_check
            )

    async def _handle_fetch_request_paused_for_cookies(self, msg: dict, cdp_session: AsyncCDPSession) -> None:
        """Handle Fetch.requestPaused for Set-Cookie headers (NATIVE)."""
        params = msg.get("params", {})
        # only check response stage (when responseStatusCode is present)
        if params.get("responseStatusCode") is None:
            return  # this is a request stage, not a response stage

        response_headers = params.get("responseHeaders", [])
        logger.debug(
            "🔍 Fetch.requestPaused response headers: type=%s, count=%d",
            type(response_headers), len(response_headers) if isinstance(response_headers, list) else 0
        )

        # headers in Fetch.requestPaused are always a list of {name, value} objects
        set_cookie_found = False
        if isinstance(response_headers, list):
            for header in response_headers:
                if isinstance(header, dict) and header.get("name", "").lower() == "set-cookie":
                    logger.debug("🍪 Found Set-Cookie header in Fetch.requestPaused")
                    set_cookie_found = True
                    await self._trigger_native_cookie_check(cdp_session)
                    break

        if not set_cookie_found:
            logger.debug("🔍 No Set-Cookie header found in Fetch.requestPaused")

    async def _handle_network_response_for_cookies(self, msg: dict, cdp_session: AsyncCDPSession) -> None:
        """Handle Network.responseReceived for Set-Cookie headers (NATIVE)."""
        params = msg.get("params", {})
        response = params.get("response", {})
        headers = response.get("headers", {})

        # check for Set-Cookie headers (case-insensitive);
        # headers might be a dict or a list of {name, value} objects
        set_cookie_found = False
        if isinstance(headers, dict):
            for header_name, header_value in headers.items():
                if header_name.lower() == "set-cookie":
                    logger.debug("🍪 Found Set-Cookie header in Network.responseReceived")
                    set_cookie_found = True
                    await self._trigger_native_cookie_check(cdp_session)
                    break
        elif isinstance(headers, list):
            for header in headers:
                if isinstance(header, dict) and header.get("name", "").lower() == "set-cookie":
                    logger.debug("🍪 Found Set-Cookie header in Network.responseReceived (list format)")
                    set_cookie_found = True
                    await self._trigger_native_cookie_check(cdp_session)
                    break

        if not set_cookie_found:
            pass

    async def _handle_network_response_extra_info_for_cookies(self, msg: dict, cdp_session: AsyncCDPSession) -> None:
        """Handle Network.responseReceivedExtraInfo for cookie headers (NATIVE)."""
        params = msg.get("params", {})
        headers = params.get("headers", {})

        # check for Set-Cookie headers in extra info
        for header_name, header_value in headers.items():
            if header_name.lower() == "set-cookie":
                await self._trigger_native_cookie_check(cdp_session)
                break

    async def _handle_console_for_cookie_operations(self, msg: dict, cdp_session: AsyncCDPSession) -> None:
        """Optional: Handle Runtime console events for document.cookie operations (NATIVE)."""
        params = msg.get("params", {})
        args = params.get("args", [])

        # look for cookie-related console messages
        for arg in args:
            if isinstance(arg, dict) and "value" in arg:
                value = str(arg["value"]).lower()
                if any(keyword in value for keyword in ["cookie", "document.cookie", "set-cookie"]):
                    await self._trigger_native_cookie_check(cdp_session)
                    break

    async def _handle_get_cookies_reply(self, msg: dict, command_info: dict[str, Any]) -> None:
        """Handle Storage.getCookies or Network.getAllCookies reply (NATIVE)."""
        result = msg.get("result", {})
        cookies = result.get("cookies", [])
        is_initial = command_info.get("initial", False)
        triggered_by = command_info.get("triggered_by", "unknown")

        # compare with previous state
        current_cookies = {
            f"{cookie.get('domain', '')}:{cookie.get('name', '')}": cookie
            for cookie in cookies
        }

        if is_initial:
            self.cookies_state = current_cookies
            logger.info("🍪 Initial cookie state populated: %d cookies", len(cookies))
        else:
            # check for changes
            added_cookies = []
            modified_cookies = []
            removed_cookies = []

            # check for added/modified cookies
            for key, cookie in current_cookies.items():
                if key not in self.cookies_state:
                    added_cookies.append(cookie)
                elif self.cookies_state[key] != cookie:
                    modified_cookies.append({
                        "old": self.cookies_state[key],
                        "new": cookie
                    })

            # check for removed cookies
            for key, cookie in self.cookies_state.items():
                if key not in current_cookies:
                    removed_cookies.append(cookie)

            # update state
            self.cookies_state = current_cookies

            # emit changes if any
            if added_cookies or modified_cookies or removed_cookies:
                event = StorageEvent(
                    type="cookieChange",
                    source="native_cdp",
                    triggered_by=triggered_by,
                    added=added_cookies,
                    modified=modified_cookies,
                    removed=removed_cookies,
                    total_count=len(cookies)
                )
                try:
                    await self.event_callback_fn(self.get_monitor_category(), event)
                    logger.debug(
                        "📊 Emitted cookie change event: added=%d, modified=%d, removed=%d", 
                        len(added_cookies), len(modified_cookies), len(removed_cookies)
                    )
                except Exception as e:
                    logger.error("❌ Error calling event_callback for cookie change: %s", e, exc_info=True)

    async def _handle_dom_storage_cleared(self, msg: dict) -> None:
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

        event = StorageEvent(
            type=f"{storage_type}Cleared",
            origin=origin,
        )
        try:
            await self.event_callback_fn(self.get_monitor_category(), event)
            logger.info("📊 Emitted %s cleared event for origin: %s", storage_type, origin)
        except Exception as e:
            logger.error("❌ Error calling event_callback for %s cleared: %s", storage_type, e, exc_info=True)

    async def _handle_dom_storage_removed(self, msg: dict) -> None:
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

        event = StorageEvent(
            type=f"{storage_type}ItemRemoved",
            origin=origin,
            key=key,
        )
        try:
            await self.event_callback_fn(self.get_monitor_category(), event)
            logger.info("📊 Emitted %s item removed event: origin=%s, key=%s", storage_type, origin, key)
        except Exception as e:
            logger.error("❌ Error calling event_callback for %s removed: %s", storage_type, e, exc_info=True)

    async def _handle_dom_storage_added(self, msg: dict) -> None:
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

        event = StorageEvent(
            type=f"{storage_type}ItemAdded",
            origin=origin,
            key=key,
            value=new_value,
        )
        try:
            await self.event_callback_fn(self.get_monitor_category(), event)
            logger.info("📊 Emitted %s item added event: origin=%s, key=%s", storage_type, origin, key)
        except Exception as e:
            logger.error("❌ Error calling event_callback for %s added: %s", storage_type, e, exc_info=True)

    async def _handle_dom_storage_updated(self, msg: dict) -> None:
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

        event = StorageEvent(
            type=f"{storage_type}ItemUpdated",
            origin=origin,
            key=key,
            old_value=old_value,
            new_value=new_value,
        )
        try:
            await self.event_callback_fn(self.get_monitor_category(), event)
            logger.info("📊 Emitted %s item updated event: origin=%s, key=%s", storage_type, origin, key)
        except Exception as e:
            logger.error("❌ Error calling event_callback for %s updated: %s", storage_type, e, exc_info=True)

    async def _handle_get_dom_storage_reply(self, msg: dict, command_info: dict[str, Any]) -> None:
        """Handle DOMStorage.getDOMStorageItems reply."""
        result = msg.get("result", {})
        entries = result.get("entries", [])
        storage_id = command_info.get("storageId", {})
        origin = storage_id.get("securityOrigin", "")
        is_local = storage_id.get("isLocalStorage", True)

        # convert entries to dictionary
        storage_data = {entry[0]: entry[1] for entry in entries}

        if is_local:
            self.local_storage_state[origin] = storage_data
        else:
            self.session_storage_state[origin] = storage_data

    async def _handle_indexeddb_added(self, msg: dict) -> None:
        """Handle IndexedDB events."""
        params = msg.get("params", {})
        event = StorageEvent(
            type="indexedDBEvent",
            params=params,
        )
        try:
            await self.event_callback_fn(self.get_monitor_category(), event)
            logger.info("📊 Emitted IndexedDB event")
        except Exception as e:
            logger.error("❌ Error calling event_callback for IndexedDB event: %s", e, exc_info=True)


    # Public methods _______________________________________________________________________________________________________

    async def setup_storage_monitoring(self, cdp_session: AsyncCDPSession) -> None:
        """
        Setup storage monitoring via CDP session using NATIVE events only.
        """
        logger.info("🔧 Setting up storage monitoring...")

        # enable Network domain for Set-Cookie header detection
        await cdp_session.enable_domain(
            domain="Network",
            params={
                "maxTotalBufferSize": 10_000_000,
                "maxResourceBufferSize": 5_000_000,
                "maxPostDataSize": 65_536,
            },
            wait_for_response=True,
        )

        # enable Runtime domain for console events (optional - may not be supported by all CDP servers)
        await cdp_session.enable_domain(
            domain="Runtime",
            wait_for_response=True,
        )

        # enable Page domain for navigation events (optional - may not be supported by all CDP servers)
        await cdp_session.enable_domain(
            domain="Page",
            wait_for_response=True,
        )

        # enable DOM storage tracking
        await cdp_session.enable_domain(
            domain="DOMStorage",
            wait_for_response=True,
        )

        # enable IndexedDB tracking
        await cdp_session.enable_domain(
            domain="IndexedDB",
            wait_for_response=True,
        )

        # don't get initial cookies here - wait for Page.loadEventFired instead
        # cookies may not be set yet during setup, and we'll check on page load
        logger.info("🍪 Initial cookie check will be triggered on Page.loadEventFired")

        # get initial DOM storage state
        await self._get_initial_dom_storage()

        logger.info("✅ Storage monitoring setup complete")

    async def handle_storage_message(self, msg: dict, cdp_session: AsyncCDPSession) -> bool:
        """
        Handle storage-related CDP messages using NATIVE events only.
        Returns True if handled, False otherwise.
        """
        method = msg.get("method")
        if not method:
            return False

        # handle Fetch.requestPaused for Set-Cookie headers (when using Fetch interception)
        if method == "Fetch.requestPaused":
            await self._handle_fetch_request_paused_for_cookies(msg, cdp_session)
            return False  # don't swallow, let network monitor handle it too

        # handle Network.responseReceived for Set-Cookie headers; Network.responseReceivedExtraInfo for cookie headers
        if method == "Network.responseReceived":
            await self._handle_network_response_for_cookies(msg, cdp_session)
            return True
        if method == "Network.responseReceivedExtraInfo":
            logger.debug("🍪 Storage monitor handling Network.responseReceivedExtraInfo")
            await self._handle_network_response_extra_info_for_cookies(msg, cdp_session)
            return True

        # handle Page.frameNavigated for cookie changes on navigation, Page.loadEventFired for cookie changes on load
        if method == "Page.frameNavigated":
            # trigger cookie check after a short delay to allow cookies to be set
            await self._trigger_native_cookie_check(cdp_session)
            return False  # don't swallow this event
        if method == "Page.loadEventFired":
            # trigger cookie check after page load - this is when cookies are most likely to be set
            logger.info("🍪 Page loaded, triggering cookie check")
            await self._trigger_native_cookie_check(cdp_session)
            return False  # don't swallow this event

        # handle Runtime.consoleAPICalled for cookie changes on console API called
        if method == "Runtime.consoleAPICalled":
            await self._handle_console_for_cookie_operations(msg, cdp_session)
            return True

        # handle DOM storage events
        if method == "DOMStorage.domStorageItemsCleared":
            await self._handle_dom_storage_cleared(msg)
            return True
        if method == "DOMStorage.domStorageItemRemoved":
            await self._handle_dom_storage_removed(msg)
            return True
        if method == "DOMStorage.domStorageItemAdded":
            await self._handle_dom_storage_added(msg)
            return True
        if method == "DOMStorage.domStorageItemUpdated":
            await self._handle_dom_storage_updated(msg)
            return True

        # handle IndexedDB events
        if method == "IndexedDB.databaseCreated":
            await self._handle_indexeddb_added(msg)
            return True
        if method == "IndexedDB.databaseDeleted":
            await self._handle_indexeddb_added(msg)  # reuse same handler for now
            return True

        return False  # message not handled

    async def handle_storage_command_reply(self, msg: dict) -> bool:
        """
        Handle CDP command replies for storage-related commands.
        Returns True if handled, False otherwise.
        """
        cmd_id = msg.get("id")
        if cmd_id is None:
            return False

        # handle pending storage commands
        if cmd_id in self.pending_storage_commands:
            command_info = self.pending_storage_commands.pop(cmd_id)
            command_type = command_info.get("type")
            logger.info("🍪 Handling storage command reply: cmd_id=%s, type=%s", cmd_id, command_type)

            if command_type in ["getAllCookies", "getCookies"]:
                await self._handle_get_cookies_reply(msg, command_info)
                return True
            if command_type == "getDOMStorageItems":
                await self._handle_get_dom_storage_reply(msg, command_info)
                return True
        else:
            #logger.debug("🔍 Command reply cmd_id=%s not in pending_storage_commands", cmd_id)
            pass

        return False  # command not handled

    async def monitor_cookie_changes(self, cdp_session: AsyncCDPSession) -> None:
        """
        Trigger native cookie monitoring (used by external callers).
        Args:
            cdp_session: The CDP session to use.
        """
        await self._trigger_native_cookie_check(cdp_session)

    def get_storage_summary(self) -> dict[str, Any]:
        """
        Get summary of current storage state.
        Returns:
            Dictionary with storage state summary.
        """
        return {
            "cookies_count": len(self.cookies_state),
            "local_storage_origins": list(self.local_storage_state.keys()),
            "session_storage_origins": list(self.session_storage_state.keys()),
            "local_storage_items": sum(len(items) for items in self.local_storage_state.values()),
            "session_storage_items": sum(len(items) for items in self.session_storage_state.values()),
        }
