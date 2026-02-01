"""
bluebox/cdp/monitors/async_window_property_monitor.py

Async window property monitor for CDP.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from bluebox.cdp.monitors.abstract_async_monitor import AbstractAsyncMonitor
from bluebox.data_models.cdp import WindowPropertyChange, WindowPropertyEvent
from bluebox.utils.logger import get_logger

if TYPE_CHECKING:  # avoid circular import
    from bluebox.cdp.async_cdp_session import AsyncCDPSession
    from bluebox.data_models.cdp import BaseCDPEvent

logger = get_logger(name=__name__)


class AsyncWindowPropertyMonitor(AbstractAsyncMonitor):
    """
    Async Window property monitor for CDP.
    Monitors window properties and emits events.
    """

    # Class attributes _____________________________________________________________________________________________________

    # native browser API prefixes, used to identify native vs application objects
    NATIVE_PREFIXES: frozenset[str] = frozenset({
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
        "Geolocation", "Notification", "PushManager", "Cache", "IndexedDB",
    })
    # native name prefixes, used to identify native APIs by name
    NATIVE_NAME_PREFIXES: tuple[str, ...] = (
        "HTML", "SVG", "RTC", "IDB", "WebGL", "Media", "Audio", "Video"
    )
    # common native browser globals, used to filter out native objects
    NATIVE_GLOBALS: frozenset[str] = frozenset({
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
    })


    # Abstract method implementations ______________________________________________________________________________________

    @classmethod
    def get_ws_event_summary(cls, detail: dict[str, Any]) -> dict[str, Any]:
        """
        Extract a lightweight summary of a window property event for WebSocket streaming.
        Args:
            detail: The full event detail dict emitted by the monitor.
        Returns:
            A simplified dict with fields relevant for real-time window property monitoring.
        """
        changes = detail.get("changes", [])
        return {
            "type": cls.get_monitor_category(),
            "url": detail.get("url"),
            "change_count": len(changes),
            "total_keys": detail.get("total_keys"),
            # include first few changed paths for visibility
            "changed_paths": [c.get("path") for c in changes[:5]],
            "change_types": list(set(c.get("change_type") for c in changes)),
        }


    # Magic methods ________________________________________________________________________________________________________

    def __init__(
        self,
        event_callback_fn: Callable[[str, BaseCDPEvent], Awaitable[None]]
    ) -> None:
        """
        Initialize AsyncWindowPropertyMonitor.
        Args:
            event_callback_fn: Async callback function that takes (category: str, detail: BaseCDPEvent).
                Called when window property events are captured.
        """
        self.event_callback_fn = event_callback_fn

        # window properties history: dict[property_path, window_property_data]
        # where window_property_data is dict with:
        #   - "path": str (property path)
        #   - "values": list[dict] (list of value entries, each with value and url)
        self.history_db: dict[str, dict[str, Any]] = {}

        self.last_seen_keys: set[str] = set()  # track keys from previous collection to detect deletions

        # collection state
        self.collection_interval = 10.0  # seconds
        self.last_collection_time = 0.0
        self.navigation_detected = False
        self.page_ready = False  # track if page is ready for collection
        self.collection_task: asyncio.Task | None = None
        self.pending_navigation = False  # track if navigation happened during collection
        self.abort_collection = False  # flag to abort ongoing collection on navigation


    # Static methods _______________________________________________________________________________________________________
  
    @staticmethod
    def _is_application_object(className: str | None, name: str | None) -> bool:
        """Heuristically determine if an object is an application object."""
        if not name:
            return False

        # first, check if className matches native patterns
        if className:
            for prefix in AsyncWindowPropertyMonitor.NATIVE_PREFIXES:
                if className.startswith(prefix):
                    return False

        # if name looks like a native API, it's native
        if name.startswith(AsyncWindowPropertyMonitor.NATIVE_NAME_PREFIXES):
            return False

        # skip common native browser globals
        if name in AsyncWindowPropertyMonitor.NATIVE_GLOBALS:
            return False

        # if className is "Object" or empty, and it passed the blacklist checks above, it is likely an application object
        if className == "Object" or not className:
            return True

        return True  


    # Private methods ______________________________________________________________________________________________________

    async def _fully_resolve_object_flat(
        self,
        cdp_session: AsyncCDPSession,
        object_id: str,
        base_path: str,
        flat_dict: dict[str, Any],
        visited: set[str] | None = None,
        depth: int = 0,
        max_depth: int = 10
    ) -> None:
        """Recursively resolve an object and add all properties to a flat dictionary with dot paths. Non-blocking, fail-fast."""
        # check abort flag at start
        if self.abort_collection:
            return

        if visited is None:
            visited = set()

        if depth > max_depth or object_id in visited:
            return

        visited.add(object_id)
        try:
            # very short timeout - if page changed, object IDs are invalid, just skip
            props_result = await cdp_session.send_and_wait(
                method="Runtime.getProperties",
                params={
                    "objectId": object_id,
                    "ownProperties": True
                },
                timeout=0.5  # fail fast - if page changed, too bad so sad
            )

            # check abort flag after CDP call
            if self.abort_collection:
                return

            props_list = props_result.get("result", [])

            for prop in props_list:
                # check abort flag periodically during processing
                if self.abort_collection:
                    return
                name = prop["name"]
                value = prop.get("value", {})
                value_type = value.get("type", "unknown")
                className = value.get("className", "")

                # skip native APIs at deeper levels
                is_app_obj = AsyncWindowPropertyMonitor._is_application_object(className, name)
                if depth > 0 and not is_app_obj:
                    continue

                prop_path = f"{base_path}.{name}" if base_path else name

                # only store actual values, no metadata
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
                            await self._fully_resolve_object_flat(
                                cdp_session, nested_obj_id, prop_path, flat_dict, visited.copy(), depth + 1, max_depth
                            )
                elif value_type == "function":
                    pass  # skip functions
                else:
                    flat_dict[prop_path] = value.get("value")

        except asyncio.TimeoutError:
            # during navigation, object IDs become invalid - this is expected
            # too bad, so sad - just skip it, don't wait, don't log
            return
        except Exception as e:
            # during navigation, object IDs become invalid - this is expected
            # too bad, so sad - just skip it, don't wait, don't log
            error_str = str(e)
            if "-32000" in error_str or "Cannot find context" in error_str or "context" in error_str.lower():
                # silently skip - object ID became invalid due to navigation
                return
            # only log truly unexpected errors (not timeouts or navigation errors)
            logger.debug("Error resolving object %s: %s", base_path, e)

    async def _get_current_url(self, cdp_session: AsyncCDPSession) -> str:
        """Get current page URL using CDP. Non-blocking, fail-fast."""
        logger.debug("🔧 Getting current page URL...")
        # check abort flag first
        if self.abort_collection:
            return "unknown"

        try:
            # try Page.getFrameTree first; this works even if JavaScript isn't ready
            # very short timeout - fail fast if page changed
            frame_tree = await cdp_session.send_and_wait(
                method="Page.getFrameTree",
                params={},
                timeout=0.5,
            )
            if frame_tree and "frameTree" in frame_tree:
                current_url = frame_tree.get("frameTree", {}).get("frame", {}).get("url")
                if current_url:
                    return current_url
        except Exception:
            # skip fallbacks if first attempt fails
            return "unknown"

        # only try one fallback with very short timeout
        if self.abort_collection:
            return "unknown"

        try:
            result = await cdp_session.send_and_wait(
                method="Runtime.evaluate",
                params={
                    "expression": "window.location.href",
                    "returnByValue": True
                },
                timeout=0.5  # very short timeout
            )
            if result and "result" in result:
                current_url = result["result"].get("value")
                if current_url:
                    return current_url
        except Exception:
            # can't get URL
            pass

        return "unknown"

    async def _collect_window_properties(self, cdp_session: AsyncCDPSession) -> None:
        """Collect all window properties into a flat dictionary. Fully non-blocking, fail-fast."""
        logger.info("🔧 Collecting window properties...")

        # reset abort flag at start of collection
        self.abort_collection = False
        try:
            # check if Runtime context is ready (very short timeout - fail fast)
            if self.abort_collection:
                return

            try:
                test_result = await cdp_session.send_and_wait(
                    method="Runtime.evaluate",
                    params={
                        "expression": "1+1",
                        "returnByValue": True
                    },
                    timeout=0.5  # very short timeout
                )
                if not test_result:
                    return
                if isinstance(test_result, dict):
                    if "error" in test_result or "result" not in test_result:
                        return
            except (asyncio.TimeoutError, Exception):
                # Runtime not ready, skip collection
                return

            # check abort flag before continuing
            if self.abort_collection:
                return

            current_url = await self._get_current_url(cdp_session)

            # check abort flag
            if self.abort_collection:
                return

            # get window object (very short timeout)
            try:
                result = await cdp_session.send_and_wait(
                    method="Runtime.evaluate",
                    params={
                        "expression": "window",
                        "returnByValue": False
                    },
                    timeout=0.5  # very short timeout - fail fast
                )
            except (asyncio.TimeoutError, Exception):
                # can't get window object, skip
                return

            if not result or not result.get("result", {}).get("objectId"):
                return

            # check abort flag
            if self.abort_collection:
                return

            window_obj = result["result"]["objectId"]

            # get all properties of window (short timeout - this is the biggest operation)
            if self.abort_collection:
                return

            try:
                props_result = await cdp_session.send_and_wait(
                    method="Runtime.getProperties",
                    params={
                        "objectId": window_obj,
                        "ownProperties": True
                    },
                    timeout=1.0  # short timeout - if page changed, skip
                )
            except (asyncio.TimeoutError, Exception) as e:
                # if navigation happens during collection, object IDs become invalid
                # just abort collection silently
                error_str = str(e)
                if "-32000" in error_str or "Cannot find context" in error_str:
                    return  # silently abort collection
                # only log truly unexpected errors
                logger.debug("Error getting window properties: %s", e)
                return

            # check abort flag after getting properties
            if self.abort_collection:
                return

            flat_dict: dict[str, Any] = {}
            all_props = props_result.get("result", [])

            total_props = len(all_props)

            skipped_count = 0
            processed_count = 0

            for prop in all_props:
                # check abort flag frequently during processing
                if self.abort_collection:
                    return
                name = prop["name"]
                value = prop.get("value", {})
                value_type = value.get("type", "unknown")
                className = value.get("className", "")

                is_app_object = AsyncWindowPropertyMonitor._is_application_object(className, name)
                if not is_app_object:
                    skipped_count += 1
                    continue

                # only store actual values, no metadata
                if value_type == "string":
                    flat_dict[name] = value.get("value")
                elif value_type in ["number", "boolean"]:
                    flat_dict[name] = value.get("value")
                elif value_type == "object" and value.get("objectId"):
                    # check abort before recursive call
                    if self.abort_collection:
                        return
                    obj_id = value.get("objectId")
                    # recursive resolution with fail-fast timeout (handled inside)
                    await self._fully_resolve_object_flat(cdp_session, obj_id, name, flat_dict, max_depth=10)
                    # check abort after recursive call
                    if self.abort_collection:
                        return
                elif value_type == "function":
                    pass  # skip functions
                else:
                    flat_dict[name] = value.get("value")

                processed_count += 1

            logger.info(
                "📊 Collected window properties: total=%d, processed=%d, skipped=%d",
                total_props, processed_count, skipped_count
            )

            # update history and emit events
            changes: list[WindowPropertyChange] = []

            # update history with new/changed values
            current_keys = set()
            for key, value in flat_dict.items():
                current_keys.add(key)
                if key not in self.history_db:
                    # new key - create entry with first value
                    self.history_db[key] = {
                        "path": key,
                        "values": [
                            {
                                "value": value,
                                "url": current_url
                            }
                        ]
                    }
                    changes.append(
                        WindowPropertyChange(
                            path=key,
                            value=value,
                            change_type="added"
                        )
                    )
                else:
                    # existing key, check if value changed
                    window_property = self.history_db[key]
                    last_entry = window_property["values"][-1]
                    if last_entry["value"] != value:
                        # value changed, add new entry
                        window_property["values"].append({
                            "value": value,
                            "url": current_url
                        })
                        changes.append(
                            WindowPropertyChange(
                                path=key,
                                value=value,
                                change_type="changed"
                            )
                        )

            # check for deleted keys (only check keys from previous collection, not all history!)
            for key in self.last_seen_keys:
                if key not in current_keys:
                    # key was deleted since last collection
                    if key in self.history_db:
                        window_property = self.history_db[key]
                        last_entry = window_property["values"][-1]
                        if last_entry["value"] is not None:
                            # add deletion marker (None value)
                            window_property["values"].append({
                                "value": None,
                                "url": current_url
                            })
                            changes.append(
                                WindowPropertyChange(
                                    path=key,
                                    value=None,
                                    change_type="deleted"
                                )
                            )

            # update last_seen_keys for next collection
            self.last_seen_keys = current_keys

            # emit events for all changes
            if changes:
                try:
                    event = WindowPropertyEvent(
                        url=current_url,
                        changes=changes,
                        total_keys=len(self.history_db),
                    )
                    logger.info("📞 Calling event_callback with category='window_property' (%d changes)", len(changes))
                    await self.event_callback_fn(self.get_monitor_category(), event)
                    logger.info("✅ Successfully called event_callback for window_property")
                except Exception as e:
                    logger.error("❌ Error calling event_callback: %s", e, exc_info=True)

        except Exception as e:
            logger.error("❌ Error collecting window properties: %s", e, exc_info=True)
        finally:
            # clear abort flag and task reference since collection is done
            self.abort_collection = False
            self.collection_task = None

            # after collection finishes, check if navigation is pending
            # if so, trigger a new collection for the new page
            if self.pending_navigation:
                self.pending_navigation = False
                # small delay to let new page settle
                await asyncio.sleep(0.5)
                # reset navigation flag and trigger new collection
                self.navigation_detected = True
                asyncio.create_task(
                    self._collect_window_properties(cdp_session)
                )

    async def _trigger_collection_task(self, cdp_session: AsyncCDPSession) -> None:
        """
        Trigger collection in a separate asyncio task.
        Args:
            cdp_session: Async CDP session.
        """
        logger.info("🔧 Triggering collection task...")
        if (self.collection_task and not self.collection_task.done()):
            return
        self.collection_task = asyncio.create_task(
            self._collect_window_properties(cdp_session)
        )


    # Public methods _______________________________________________________________________________________________________

    async def setup_window_property_monitoring(self, cdp_session: AsyncCDPSession) -> None:
        """
        Setup window property monitoring via CDP session.
        Args:
            cdp_session: Async CDP session.
        """
        logger.info("🔧 Setting up window property monitoring...")

        # enable Page domain for navigation events
        await cdp_session.enable_domain(
            domain="Page",
            wait_for_response=True,
        )

        # enable Runtime domain for property access
        await cdp_session.enable_domain(
            domain="Runtime",
            wait_for_response=True,
        )

        # check if page is already loaded (non-blocking, fail-fast)
        try:
            result = await cdp_session.send_and_wait(
                method="Runtime.evaluate",
                params={
                    "expression": "document.readyState",
                    "returnByValue": True,
                },
                timeout=0.5,  # very short timeout; fail fast
            )
            if result and result.get("result", {}).get("value") == "complete":
                self.page_ready = True
        except Exception:
            # page not ready yet, will check later
            pass

    async def handle_window_property_message(
        self,
        msg: dict,
        cdp_session: AsyncCDPSession,
    ) -> bool:
        """Handle window property-related CDP messages."""
        method = msg.get("method")
        if not method:
            return False

        if method == "Runtime.executionContextsCleared":
            # detect navigation events
            self.page_ready = False
            self.navigation_detected = True
            # if collection is running, signal it to abort (don't block the event loop!)
            if self.collection_task and not self.collection_task.done():
                self.abort_collection = True
                self.pending_navigation = True
            return True

        elif method == "Page.frameNavigated":
            self.navigation_detected = True
            self.page_ready = True
            # only trigger if no collection is running
            if not (self.collection_task and not self.collection_task.done()):
                await self._trigger_collection_task(cdp_session)
            else:
                # collection is running, mark navigation as pending
                self.pending_navigation = True
            return False  # allow other handlers (e.g., DOM monitor) to process this event

        elif method == "Page.domContentEventFired":
            self.page_ready = True
            self.navigation_detected = True
            # only trigger if no collection is running
            if not (self.collection_task and not self.collection_task.done()):
                await self._trigger_collection_task(cdp_session)
            else:
                # collection is running, mark navigation as pending
                self.pending_navigation = True
            return False  # allow other handlers to process this event

        elif method == "Page.loadEventFired":
            self.page_ready = True
            self.navigation_detected = True
            # only trigger if no collection is running
            if not (self.collection_task and not self.collection_task.done()):
                await self._trigger_collection_task(cdp_session)
            else:
                # collection is running, mark navigation as pending
                self.pending_navigation = True
            return False  # allow other handlers (e.g., DOM monitor) to process this event

        return False

    async def check_and_collect(self, cdp_session: AsyncCDPSession) -> None:
        """Check if it's time to collect and collect if needed (runs in background task)."""
        logger.debug("🔧 Checking and collecting window properties...")
        # don't collect until page is ready (after first navigation)
        if not self.page_ready:
            return


        # check if a collection is already running
        if self.collection_task and not self.collection_task.done():
            return

        # collect on navigation or if interval has passed
        current_time = time.time()
        should_collect = (
            self.navigation_detected or
            (current_time - self.last_collection_time) >= self.collection_interval
        )

        if should_collect:
            self.navigation_detected = False
            self.last_collection_time = current_time
            await self._trigger_collection_task(cdp_session)

    async def force_collect(self, cdp_session: AsyncCDPSession) -> None:
        """Force immediate collection of window properties (non-blocking)."""
        # just trigger the task. if it's running, great. if not, start it.
        # we do NOT wait for it to complete.
        await self._trigger_collection_task(cdp_session)

    def get_window_property_summary(self) -> dict[str, Any]:
        """
        Get summary of window property monitoring.
        Returns:
            Dictionary with window property monitoring statistics.
        """
        total_keys = len(self.history_db)
        total_entries = sum(
            len(window_prop.get("values", []))
            for window_prop in self.history_db.values()
        )
        return {
            "total_keys": total_keys,
            "total_history_entries": total_entries,
        }
