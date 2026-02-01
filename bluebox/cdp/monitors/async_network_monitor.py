"""
bluebox/cdp/monitors/async_network_monitor.py

Async network monitor for CDP.
"""

from __future__ import annotations

import base64
import json
import re
from typing import TYPE_CHECKING, Any, Awaitable, Callable, ClassVar

from bluebox.cdp.monitors.abstract_async_monitor import AbstractAsyncMonitor
from bluebox.constants.network import THIRD_PARTY_TRACKING_ANALYTICS_DOMAINS
from bluebox.data_models.cdp import NetworkTransactionEvent
from bluebox.data_models.routine.endpoint import ResourceType
from bluebox.utils.data_utils import get_text_from_html
from bluebox.utils.logger import get_logger

if TYPE_CHECKING:  # avoid circular import
    from bluebox.cdp.async_cdp_session import AsyncCDPSession
    from bluebox.data_models.cdp import BaseCDPEvent

logger = get_logger(name=__name__)


class AsyncNetworkMonitor(AbstractAsyncMonitor):
    """
    Async Network monitor for CDP.
    Intercepts requests/responses and captures bodies.
    Uses callback pattern to emit events (no file saving).
    """

    # Class attributes _____________________________________________________________________________________________________

    CAPTURE_RESOURCES: ClassVar[frozenset[ResourceType]] = frozenset({
        ResourceType.DOCUMENT,
        ResourceType.FETCH,
        ResourceType.SCRIPT,
        ResourceType.XHR,
    })
    STATIC_ASSET_HINTS: ClassVar[tuple[str, ...]] = (
        ".css", ".woff", ".woff2", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico"
    )
    NOISY_NETWORK_EVENTS: ClassVar[frozenset[str]] = frozenset({})

    # for streaming/storage limits
    URL_MAX_CHARS: ClassVar[int] = 150
    RESPONSE_BODY_MAX_CHARS: ClassVar[int] = 250_000


    # Abstract method implementations ______________________________________________________________________________________

    @classmethod
    def get_ws_event_summary(cls, detail: dict[str, Any]) -> dict[str, Any]:
        """
        Extract a lightweight summary of a network event for WebSocket streaming.
        Args:
            detail: The full event detail dict emitted by the monitor.
        Returns:
            A simplified dict with fields relevant for real-time network monitoring.
        """
        return {
            "type": cls.get_monitor_category(),
            "method": detail.get("method"),
            "url": (detail.get("url") or "")[:cls.URL_MAX_CHARS],  # truncate long URLs
            "status": detail.get("status"),
            "resource_type": detail.get("type"),
            "failed": detail.get("failed", False),
        }


    # Magic methods ________________________________________________________________________________________________________

    def __init__(
        self,
        event_callback_fn: Callable[[str, BaseCDPEvent], Awaitable[None]]
    ) -> None:
        """
        Initialize AsyncNetworkMonitor.
        Args:
            event_callback_fn: Async callback function that takes (category: str, detail: BaseCDPEvent).
                Called when network transactions are captured.
        """
        self.event_callback_fn = event_callback_fn

        # network request tracking
        self.req_meta: dict[str, dict[str, Any]] = {}  # request_id -> metadata
        self.fetch_get_body_wait: dict[int, dict[str, Any]] = {}  # cmd_id -> context
        self.completed_transactions: int = 0  # counter for emitted transactions


    # Static methods _______________________________________________________________________________________________________

    @staticmethod
    def _is_internal_url(url: str | None) -> bool:
        if not url:
            return False
        return url.startswith("chrome://")

    @staticmethod
    def _is_static_asset(url: str | None) -> bool:
        if not url:
            return False
        lower = url.lower()
        return any(lower.endswith(ext) for ext in AsyncNetworkMonitor.STATIC_ASSET_HINTS)

    @staticmethod
    def _should_skip_logging(url: str) -> bool:
        """
        Check if a URL should be skipped from logging based on THIRD_PARTY_TRACKING_ANALYTICS_DOMAINS.
        Args:
            url: The URL to check.
        Returns:
            True if the URL should be skipped, False otherwise.
        """
        if not url:
            return False
        url_lower = url.lower()
        if AsyncNetworkMonitor._is_internal_url(url_lower):
            return True
        return any(domain in url_lower for domain in THIRD_PARTY_TRACKING_ANALYTICS_DOMAINS)

    @staticmethod
    def _get_set_cookie_values(headers: dict) -> list:
        """
        Extract Set-Cookie values from headers.
        Args:
            headers (dict): The headers to extract Set-Cookie values from.
        Returns:
            list: The list of Set-Cookie values.
        """
        values = []
        if not headers:
            return values
        try:
            for k, v in headers.items():
                if str(k).lower() == "set-cookie":
                    if isinstance(v, str):
                        parts = v.split("\n") if "\n" in v else [v]
                        for line in parts:
                            line = line.strip()
                            if line:
                                values.append(line)
                    elif isinstance(v, (list, tuple)):
                        for line in v:
                            if line:
                                values.append(str(line))
        except Exception:
            pass
        return values

    @staticmethod
    def _parse_json_if_applicable(data: str | None, content_type: str | None) -> str | dict | list | None:
        """
        Parse JSON string if content-type indicates JSON ("application/json"), otherwise return as-is.
        Args:
            data: The raw string data (postData or responseBody).
            content_type: The content-type header value.
        Returns:
            Parsed JSON dict/list if content-type indicates JSON and parsing succeeds,
            otherwise the original string or None.
        """
        if not data:
            return None

        # check if content-type indicates JSON
        if content_type and "application/json" in content_type.lower():
            try:
                parsed = json.loads(data)
                logger.debug("📦 Parsed JSON data: %d chars -> dict", len(data))
                return parsed
            except (json.JSONDecodeError, TypeError) as e:
                logger.debug("⚠️ Failed to parse data as JSON: %s", e)
                # return original string if parsing fails
                return data

        # not JSON or no content-type, return original string
        return data

    @staticmethod
    def _is_html(response_body: str | bytes | None, content_type: str | None = None) -> bool:
        """
        Lightweight heuristic to detect HTML responses.
        Checks content-type first, then looks for common HTML markers in the body.
        """
        if not response_body:
            logger.debug("🧪 _is_html: empty/none body -> False")
            return False

        if content_type and "html" in content_type.lower():
            logger.debug("🧪 _is_html: content-type indicates html (%s) -> True", content_type)
            return True

        if isinstance(response_body, bytes):
            try:
                response_body = response_body.decode(encoding="utf-8", errors="ignore")
                logger.debug("🧪 _is_html: decoded bytes body (%d chars)", len(response_body))
            except Exception as decode_err:
                logger.debug("🧪 _is_html: failed to decode bytes (%s) -> False", decode_err)
                return False

        # strip leading whitespace/comments and inspect a small snippet
        snippet = response_body.lstrip()[:512].lower()

        if snippet.startswith("<!doctype html") or snippet.startswith("<html"):
            logger.debug("🧪 _is_html: snippet starts with doctype/html -> True")
            return True

        # look for multiple common tags to avoid false positives from single angle brackets
        tag_hits = len(re.findall(r"<(html|head|body|script|style|div|span|p|a|meta|link)\b", snippet))
        logger.debug("🧪 _is_html: tag_hits=%d -> %s", tag_hits, tag_hits >= 2)
        return tag_hits >= 2

    @staticmethod
    def _clean_response_body(response_body: str | bytes | dict | list, content_type: str | None = None) -> str:
        """
        Normalize and truncate the response body with awareness of JSON and HTML.
        Args:
            response_body: The response body to truncate.
        Returns:
            The truncated response body.
        """
        logger.debug(
            "🧪 _clean_response_body: called with type=%s, len=%d, content_type=%s",
            type(response_body).__name__ if response_body is not None else "None",
            len(str(response_body)) if response_body else 0,
            content_type
        )
        # handle None or empty string
        if response_body is None or response_body == "" or response_body == b"":
            logger.debug("🧪 _clean_response_body: empty body -> return empty string")
            return ""
        try:
            # if already parsed JSON (dict/list), pretty-print then truncate
            if isinstance(response_body, (dict, list)):
                try:
                    serialized = json.dumps(obj=response_body, ensure_ascii=False)
                except Exception as e:
                    logger.debug("🧪 _clean_response_body: failed to serialize pre-parsed JSON, using str: %s", e)
                    serialized = str(response_body)
                truncated_response_body = serialized[:AsyncNetworkMonitor.RESPONSE_BODY_MAX_CHARS]
                logger.debug("🧪 _clean_response_body: pre-parsed JSON len=%d", len(truncated_response_body))
                return truncated_response_body

            # if bytes, decode to string
            if isinstance(response_body, bytes):
                response_body = response_body.decode(encoding="utf-8", errors="ignore")
                logger.debug("🧪 _clean_response_body: decoded bytes (%d chars)", len(response_body))

            # attempt JSON parse based on content-type hint
            parsed = AsyncNetworkMonitor._parse_json_if_applicable(response_body, content_type)
            if isinstance(parsed, (dict, list)):
                try:
                    serialized = json.dumps(obj=parsed, ensure_ascii=False)
                except Exception as e:
                    logger.debug("🧪 _clean_response_body: failed to serialize parsed JSON, using str: %s", e)
                    serialized = str(parsed)
                    truncated_parsed = serialized[:AsyncNetworkMonitor.RESPONSE_BODY_MAX_CHARS]
                    logger.debug("🧪 _clean_response_body: parsed JSON len=%d", len(truncated_parsed))
                    return truncated_parsed
            # parsed is str (original or same value) - continue with response_body as-is

            # attempt HTML parse and cleanup
            if AsyncNetworkMonitor._is_html(response_body=response_body, content_type=content_type):
                html = get_text_from_html(html=response_body)  # safe to call with str
                cleaned = html[:AsyncNetworkMonitor.RESPONSE_BODY_MAX_CHARS]
                logger.debug("🧪 _clean_response_body: html cleaned len=%d", len(cleaned))
                return cleaned

            # truncate the response body to the maximum allowed characters
            truncated = response_body[:AsyncNetworkMonitor.RESPONSE_BODY_MAX_CHARS]
            logger.debug("🧪 _clean_response_body: non-html truncated len=%d", len(truncated))
            return truncated
        except Exception as e:
            logger.exception("❌ Failed to truncate response body: %s", e)
            fallback = str(response_body)[:AsyncNetworkMonitor.RESPONSE_BODY_MAX_CHARS]
            return fallback


    # Private methods ______________________________________________________________________________________________________

    async def _on_fetch_request_paused(self, msg: dict, cdp_session: AsyncCDPSession) -> bool:
        """
        Handle Fetch.requestPaused event.
        Args:
            msg: The CDP message to handle.
            cdp_session: The CDP session to use.
        Returns:
            True if message was handled, False otherwise.
        """
        p = msg["params"]
        rid = p["requestId"]
        response_status = p.get("responseStatusCode")
        request = p.get("request", {})
        url = request.get("url", "unknown")
        method = request.get("method", "unknown")
        resource_type = p.get("resourceType")

        # check if URL should be blocked; if so, continue and skip all processing
        if AsyncNetworkMonitor._should_skip_logging(url):
            # clean up any metadata that might have been stored in request stage
            self.req_meta.pop(rid, None)
            if response_status is not None:
                await self._safe_continue_response(rid, cdp_session)
            else:
                await self._safe_continue_request(rid, cdp_session)
            return True

        # check if URL is a static asset; if so, continue and skip all processing
        if AsyncNetworkMonitor._is_static_asset(url):
            logger.debug("⏭️ Static asset (skipping): %s", url)
            # clean up any metadata that might have been stored in request stage
            self.req_meta.pop(rid, None)
            if response_status is not None:
                await self._safe_continue_response(rid, cdp_session)
            else:
                await self._safe_continue_request(rid, cdp_session)
            return True

        # Response stage
        if response_status is not None:
            logger.info("🔄 Fetch.requestPaused (RESPONSE STAGE)")
            # Use requestId as the key since networkId might not be set
            fetch_id = rid  # Use Fetch requestId as the key

            # Get or create metadata
            if fetch_id not in self.req_meta:
                logger.info("🔄 Creating new metadata for fetch_id=%s", fetch_id)
                self.req_meta[fetch_id] = {
                    "requestId": fetch_id,
                    "url": url,
                    "method": method,
                    "type": resource_type,
                }

            req_meta = self.req_meta[fetch_id]

            # Store response headers and status
            response_headers_list = p.get("responseHeaders", [])
            response_headers = {}
            for h in response_headers_list:
                response_headers[h.get("name", "")] = h.get("value", "")

            req_meta.update({
                "status": response_status,
                "statusText": p.get("responseStatusText", ""),
                "responseHeaders": response_headers,
                "mimeType": response_headers.get("content-type", ""),
            })
            logger.debug(
                "🔄 RESPONSE: fetch_id=%s, type=%s, status=%s, tracked_requests=%d", 
                fetch_id, resource_type, response_status, len(self.req_meta)
            )

            # Request response body for resources we want to capture
            if resource_type in AsyncNetworkMonitor.CAPTURE_RESOURCES:
                try:
                    logger.info("📥 Requesting response body for fetch_id=%s", fetch_id)
                    rb_id = await cdp_session.send("Fetch.getResponseBody", {"requestId": rid})
                    self.fetch_get_body_wait[rb_id] = {
                        "rid": rid,
                        "fetch_id": fetch_id,
                    }
                    logger.info("⏳ Waiting for response body (cmd_id=%s)", rb_id)
                    return True
                except Exception as e:
                    logger.warning("❌ Failed to get response body: %s", e)
                    # Emit without body
                    await self._emit_transaction(fetch_id)
                    await self._safe_continue_response(rid, cdp_session)
                    return True
            else:
                # Not capturing, but still emit the transaction
                logger.info("🔄 Resource type %s not in capture list, emitting without body", resource_type)
                await self._emit_transaction(fetch_id)

            await self._safe_continue_response(rid, cdp_session)
            return True

        # Request stage
        else:
            logger.debug("🔄 REQUEST: requestId=%s, url=%s, method=%s, type=%s", rid, url, method, resource_type)

            # Store request metadata
            request_headers_raw = request.get("headers", {})
            request_headers = {}
            if isinstance(request_headers_raw, dict):
                # Headers are already a dict
                request_headers = request_headers_raw
            elif isinstance(request_headers_raw, list):
                # Headers are a list of {name, value} objects
                for h in request_headers_raw:
                    if isinstance(h, dict):
                        request_headers[h.get("name", "")] = h.get("value", "")

            # parse postData if it is JSON
            raw_post_data = request.get("postData")
            content_type = request_headers.get("content-type", "")
            parsed_post_data = AsyncNetworkMonitor._parse_json_if_applicable(raw_post_data, content_type)

            self.req_meta[rid] = {
                "requestId": rid,
                "url": url,
                "method": method,
                "type": resource_type,
                "requestHeaders": request_headers,
                "postData": parsed_post_data,
            }
            logger.debug("💾 Stored request metadata for fetch_id=%s (total tracked: %d)", rid, len(self.req_meta))

            await self._safe_continue_request(rid, cdp_session)
            return True

    async def _on_request_will_be_sent(self, msg: dict) -> bool:
        """Handle Network.requestWillBeSent event."""
        p = msg["params"]
        request_id = p["requestId"]
        url = p["request"]["url"]
        method = p["request"]["method"]
        resource_type = p.get("type")

        # check if URL should be blocked; if so, skip tracking
        if AsyncNetworkMonitor._should_skip_logging(url):
            return True

        # check if URL is a static asset; if so, skip tracking
        if AsyncNetworkMonitor._is_static_asset(url):
            logger.debug("⏭️ Static asset (skipping): %s", url)
            return True

        # parse postData if it is JSON
        request_headers = p["request"].get("headers", {})
        raw_post_data = p["request"].get("postData")
        content_type = request_headers.get("content-type", "")
        parsed_post_data = AsyncNetworkMonitor._parse_json_if_applicable(raw_post_data, content_type)
        
        self.req_meta[request_id] = {
            "requestId": request_id,
            "url": url,
            "method": method,
            "type": resource_type,
            "ts": p.get("timestamp"),
            "requestHeaders": request_headers,
            "postData": parsed_post_data,
        }
        logger.debug("💾 Stored request metadata for id=%s (total tracked: %d)", request_id, len(self.req_meta))
        return True

    async def _on_response_received(self, msg: dict) -> bool:
        """Handle Network.responseReceived event."""
        p = msg["params"]
        request_id = p["requestId"]
        resp = p["response"]
        status = resp.get("status")
        url = resp.get("url", "")

        # check if URL should be blocked; if so, skip tracking
        if AsyncNetworkMonitor._should_skip_logging(url):
            return True

        # check if URL is a static asset; if so, skip tracking
        if AsyncNetworkMonitor._is_static_asset(url):
            logger.debug("⏭️ Static asset (skipping): %s", url)
            return True

        logger.info("📥 Network.responseReceived: status=%s (id=%s)", status, request_id)

        meta = self.req_meta.get(request_id)
        if meta:
            meta.update({
                "status": status,
                "statusText": resp.get("statusText"),
                "responseHeaders": resp.get("headers", {}),
                "mimeType": resp.get("mimeType"),
            })
        else:
            logger.warning("⚠️ Response received for unknown request_id=%s", request_id)
        return True

    async def _on_loading_finished(self, msg: dict) -> bool:
        """Handle Network.loadingFinished event."""
        p = msg["params"]
        request_id = p["requestId"]
        meta = self.req_meta.get(request_id)

        # check if this request should be blocked (check URL from metadata if available)
        if meta:
            url = meta.get("url", "")
            if AsyncNetworkMonitor._should_skip_logging(url):
                # cleanup metadata but don't emit to callback function
                self.req_meta.pop(request_id, None)
                return True
            if AsyncNetworkMonitor._is_static_asset(url):
                # cleanup metadata but don't emit to callback function
                self.req_meta.pop(request_id, None)
                return True

        if meta:
            url = meta.get("url", "unknown")
            method = meta.get("method", "unknown")
            status = meta.get("status", "unknown")

            # ensure responseBody is cleaned before emission
            raw_body = meta.get("responseBody")
            content_type = meta.get("mimeType", "")
            cleaned_body = AsyncNetworkMonitor._clean_response_body(raw_body, content_type)

            event = NetworkTransactionEvent(
                request_id=request_id,
                url=url,
                method=method,
                type=meta.get("type"),
                status=status,
                status_text=meta.get("statusText"),
                request_headers=meta.get("requestHeaders", {}),
                response_headers=meta.get("responseHeaders", {}),
                post_data=meta.get("postData"),
                response_body=cleaned_body,
                response_body_base64=False,  # always False after cleaning
                mime_type=meta.get("mimeType"),
            )
            try:
                logger.info(
                    "📞 _on_loading_finished: emitting transaction with responseBody len=%d",
                    len(cleaned_body) if cleaned_body else 0
                )
                await self.event_callback_fn(self.get_monitor_category(), event)
            except Exception as e:
                logger.error("❌ Error calling event_callback: %s", e, exc_info=True)

            # cleanup
            self.req_meta.pop(request_id, None)
        else:
            pass

        return True

    async def _on_loading_failed(self, msg: dict) -> bool:
        """Handle Network.loadingFailed event."""
        p = msg["params"]
        request_id = p["requestId"]
        error_text = p.get("errorText")
        meta = self.req_meta.get(request_id)

        # check if this request should be blocked (check URL from metadata if available)
        if meta:
            url = meta.get("url", "")
            if AsyncNetworkMonitor._should_skip_logging(url):
                # cleanup metadata but don't emit
                self.req_meta.pop(request_id, None)
                return True
            if AsyncNetworkMonitor._is_static_asset(url):
                # cleanup metadata but don't emit
                self.req_meta.pop(request_id, None)
                return True

        logger.warning("❌ Network.loadingFailed: request_id=%s, error=%s", request_id, error_text)

        if meta:
            url = meta.get("url", "unknown")
            method = meta.get("method", "unknown")
            logger.info("📊 Emitting failed network_transaction: %s %s (id=%s)", method, url, request_id)

            event = NetworkTransactionEvent(
                request_id=request_id,
                url=url,
                method=method,
                type=meta.get("type"),
                request_headers=meta.get("requestHeaders", {}),
                errorText=error_text,
                failed=True
            )

            try:
                logger.debug("📞 Calling event_callback with category='network_transaction' (failed)")
                await self.event_callback_fn(self.get_monitor_category(), event)
            except Exception as e:
                logger.error("❌ Error calling event_callback: %s", e, exc_info=True)

            # cleanup
            self.req_meta.pop(request_id, None)
        else:
            logger.debug("⚠️ Loading failed for unknown request_id=%s (likely already emitted via Fetch or blocked)", request_id)

        return True

    async def _on_fetch_get_body_reply(self, cmd_id: int, msg: dict, cdp_session: AsyncCDPSession) -> bool:
        """Handle Fetch.getResponseBody reply."""
        ctx = self.fetch_get_body_wait.pop(cmd_id, None)
        if ctx is None:
            logger.warning("⚠️ No context found for cmd_id=%s", cmd_id)
            return False
        rid = ctx.get("rid")
        fetch_id = ctx.get("fetch_id")
        if not rid:
            logger.warning("⚠️ No requestId (rid) found in context for cmd_id=%s", cmd_id)
            return False
        if not fetch_id:
            logger.warning("⚠️ No fetch_id found in context for cmd_id=%s", cmd_id)
            return False
        logger.info("📦 Received response body for fetch_id=%s (cmd_id=%s)", fetch_id, cmd_id)

        if "error" in msg:
            logger.warning("❌ Error getting response body: %s", msg.get("error"))
            # Emit without body
            await self._emit_transaction(fetch_id)
            await self._safe_continue_response(rid, cdp_session)
            return True

        body_info = msg.get("result", {})
        body = body_info.get("body", "")
        is_b64 = body_info.get("base64Encoded", False)
        body_size = len(body)

        logger.info("📦 Response body size: %d bytes (base64=%s)", body_size, is_b64)

        # decode base64 if needed
        if is_b64 and body:
            try:
                decoded_body = base64.b64decode(body).decode('utf-8', errors='replace')
                logger.info("📦 Decoded base64 body: %d bytes -> %d chars", body_size, len(decoded_body))
                body = decoded_body
                is_b64 = False  # already decoded
            except Exception as e:
                logger.warning("❌ Failed to decode base64 body: %s", e)
                # keep original if decoding fails

        # store body in metadata
        meta = self.req_meta.get(fetch_id)
        if meta:
            content_type = meta.get("mimeType", "")
            cleaned_body = AsyncNetworkMonitor._clean_response_body(body, content_type)
            meta["responseBody"] = cleaned_body
            meta["responseBodyBase64"] = False  # always false after decoding and cleaning
            logger.info(
                "💾 Stored response body in metadata for fetch_id=%s (was_base64=%s, cleaned_len=%d)", 
                fetch_id, is_b64, len(cleaned_body)
            )

            # emit transaction with body
            await self._emit_transaction(fetch_id)
        else:
            logger.warning("⚠️ No metadata found for fetch_id=%s when storing response body", fetch_id)

        # continue intercepted response
        await self._safe_continue_response(rid, cdp_session)
        return True

    async def _on_response_received_extra_info(self, msg: dict) -> bool:
        """
        Handle Network.responseReceivedExtraInfo event.
        Args:
            msg: The CDP message to handle.
        Returns:
            True if message was handled, False otherwise.
        """
        p = msg.get("params", {})
        request_id = p.get("requestId")
        meta = self.req_meta.setdefault(request_id, {})
        set_cookie_values = self._get_set_cookie_values(headers=p.get("headers", {}))
        if (set_cookie_values and not meta.get("cookiesLogged")):
            meta["setCookies"] = set_cookie_values
            meta["cookiesLogged"] = True
        return True

    async def _emit_transaction(self, fetch_id: str) -> None:
        """Emit a network transaction from metadata."""
        meta = self.req_meta.get(fetch_id)
        if not meta:
            logger.warning("⚠️ No metadata found for fetch_id=%s when emitting transaction", fetch_id)
            return
        
        url = meta.get("url", "unknown")
        # check if URL is a static asset; if so, skip emitting
        if AsyncNetworkMonitor._is_static_asset(url):
            # cleanup metadata but don't emit
            self.req_meta.pop(fetch_id, None)
            return
        
        # ensure responseBody is cleaned before emission
        raw_body = meta.get("responseBody")
        content_type = meta.get("mimeType", "")
        cleaned_body = AsyncNetworkMonitor._clean_response_body(raw_body, content_type)

        event = NetworkTransactionEvent(
            request_id=fetch_id,
            url=url,
            method=meta.get("method", "unknown"),
            type=meta.get("type"),
            status=meta.get("status", "unknown"),
            status_text=meta.get("statusText"),
            request_headers=meta.get("requestHeaders", {}),
            response_headers=meta.get("responseHeaders", {}),
            post_data=meta.get("postData"),
            response_body=cleaned_body,
            response_body_base64=False,  # always False after cleaning
            mime_type=meta.get("mimeType"),
        )

        try:
            logger.info(
                "📞 _emit_transaction: emitting transaction with responseBody len=%d",
                len(cleaned_body) if cleaned_body else 0
            )
            await self.event_callback_fn(self.get_monitor_category(), event)
            self.completed_transactions += 1
        except Exception as e:
            logger.error("❌ Error calling event_callback_fn: %s", e, exc_info=True)

        # cleanup
        self.req_meta.pop(fetch_id, None)

    async def _safe_continue_request(self, rid: str, cdp_session: AsyncCDPSession) -> None:
        """Safely continue a paused Fetch request."""
        try:
            await cdp_session.send("Fetch.continueRequest", {"requestId": rid})
            logger.debug("✅ Continued Fetch request: %s", rid)
        except Exception as e:
            logger.warning("⚠️ Failed to continue Fetch request %s: %s", rid, e)

    async def _safe_continue_response(self, rid: str, cdp_session: AsyncCDPSession) -> None:
        """Safely continue a paused Fetch response."""
        try:
            await cdp_session.send("Fetch.continueResponse", {"requestId": rid})
            logger.debug("✅ Continued Fetch response: %s", rid)
        except Exception as e:
            logger.warning("⚠️ Failed to continue Fetch response %s: %s", rid, e)


    # Public methods _______________________________________________________________________________________________________

    async def setup_network_monitoring(self, cdp_session: AsyncCDPSession) -> None:
        """Setup network monitoring CDP domains."""
        logger.info("🔧 Setting up network monitoring...")

        await cdp_session.enable_domain(
            domain="Network",
            params={
                "includeExtraInfo": True,
                "maxTotalBufferSize": 512_000_000,
                "maxResourceBufferSize": 256_000_000,
            },
            wait_for_response=True,
        )
        logger.debug("✅ Network.enable sent")

        await cdp_session.send(
            method="Network.setCacheDisabled",
            params={"cacheDisabled": True},
        )
        await cdp_session.send(
            method="Network.setBypassServiceWorker",
            params={"bypass": True},
        )
        logger.debug("✅ Network cache and service worker settings configured")

        # enable Fetch interception
        await cdp_session.enable_domain(
            domain="Fetch",
            params={
                "patterns": [
                    {"urlPattern": "*", "requestStage": "Request"},
                    {"urlPattern": "*", "requestStage": "Response"},
                ],
            },
            wait_for_response=True,
        )
        logger.debug("✅ Fetch.enable sent with patterns for REQUEST and RESPONSE stages")

        logger.info("✅ Network monitoring setup complete")

    async def handle_network_message(self, msg: dict, cdp_session: AsyncCDPSession) -> bool:
        """
        Handle incoming network-related CDP message.
        Args:
            msg: The CDP message to handle.
            cdp_session: The CDP session to use.
        Returns:
            True if message was handled, False otherwise.
        """
        method = msg.get("method")
        if not method:
            return False
        if method in self.NOISY_NETWORK_EVENTS:
            return False
        if method == "Fetch.requestPaused":
            await self._on_fetch_request_paused(msg, cdp_session)
            return False  # don't swallow event, allow AsyncStorageMonitor handle it downstream
        if method == "Network.requestWillBeSent":
            return await self._on_request_will_be_sent(msg)
        if method == "Network.responseReceived":
            #TODO::return await self._on_response_received(msg)
            await self._on_response_received(msg)
            return False  # don't swallow event, allow AsyncStorageMonitor handle it downstream
        if method == "Network.responseReceivedExtraInfo":
            #TODO::return await self._on_response_received_extra_info(msg)
            await self._on_response_received_extra_info(msg)
            return False  # don't swallow event, allow AsyncStorageMonitor handle it downstream
        if method == "Network.loadingFinished":
            return await self._on_loading_finished(msg)
        if method == "Network.loadingFailed":
            return await self._on_loading_failed(msg)
        return False

    async def handle_network_command_reply(self, msg: dict, cdp_session: AsyncCDPSession) -> bool:
        """Handle network-related CDP command replies. Returns True if handled."""
        cmd_id = msg.get("id")
        if cmd_id is None:
            return False

        # Handle Fetch.getResponseBody replies
        if cmd_id in self.fetch_get_body_wait:
            return await self._on_fetch_get_body_reply(cmd_id, msg, cdp_session)

        return False

    def get_network_summary(self) -> dict[str, Any]:
        """
        Get summary of current network monitoring state.
        Returns:
            Dictionary with network monitoring statistics.
        """
        return {
            "completed_transactions": self.completed_transactions,
            "requests_tracked": len(self.req_meta),
            "pending_bodies": len(self.fetch_get_body_wait),
        }
