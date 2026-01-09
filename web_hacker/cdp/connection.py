"""
web_hacker/cdp/connection.py

CDP connection and tab management utilities.

This module provides the core functionality for:
- WebSocket connection to Chrome DevTools Protocol
- Tab/context creation and disposal
- CDP command/response helpers
"""

import json
import random
import time
from json import JSONDecodeError
from typing import Callable
from urllib.parse import urlparse, urlunparse

import requests
import websocket
from websocket import WebSocket

from web_hacker.utils.logger import get_logger

logger = get_logger(name=__name__)


# WebSocket URL helpers ___________________________________________________________________________


def get_browser_websocket_url(remote_debugging_address: str) -> str:
    """Get the normalized WebSocket URL for browser connection.

    Args:
        remote_debugging_address: The Chrome debugging server address (e.g., 'http://127.0.0.1:9222').

    Returns:
        The WebSocket URL for connecting to the browser.

    Raises:
        RuntimeError: If unable to get the WebSocket URL from the browser.
    """
    base = remote_debugging_address.rstrip("/")
    try:
        ver = requests.get(f"{base}/json/version", timeout=5)
        ver.raise_for_status()
        data = ver.json()
        raw_ws = data.get("webSocketDebuggerUrl")
        if not raw_ws:
            raise RuntimeError("/json/version missing webSocketDebuggerUrl")

        # Normalize netloc to our reachable hostname:port
        parsed = urlparse(raw_ws)
        base_parsed = urlparse(base)
        fixed_netloc = f"{base_parsed.hostname}:{base_parsed.port}"
        ws_url = urlunparse(parsed._replace(netloc=fixed_netloc))

        logger.debug(f"Raw WebSocket URL: {raw_ws}")
        logger.debug(f"Base URL: {base}")
        logger.debug(f"Fixed netloc: {fixed_netloc}")
        logger.debug(f"Normalized WebSocket URL: {ws_url}")

        return ws_url
    except Exception as e:
        raise RuntimeError(f"Failed to get browser WebSocket URL: {e}")


# CDP command helpers _____________________________________________________________________________


def create_cdp_helpers(
    ws: WebSocket,
) -> tuple[Callable, Callable, Callable]:
    """Create helper functions for CDP communication.

    Args:
        ws: WebSocket connection to Chrome.

    Returns:
        Tuple of (send_cmd, recv_json, recv_until) functions.
    """
    _id_counter = [1]

    def send_cmd(
        method: str,
        params: dict | None = None,
        session_id: str | None = None,
    ) -> int:
        """Send a CDP command and return its ID."""
        msg: dict = {"id": _id_counter[0], "method": method}
        if params:
            msg["params"] = params
        if session_id:
            msg["sessionId"] = session_id
        _id_counter[0] += 1
        ws.send(json.dumps(msg))
        return msg["id"]

    def recv_json(ws_conn: WebSocket, deadline: float) -> dict:
        """Read a single JSON message from WebSocket, skipping empty/non-JSON frames."""
        while time.time() < deadline:
            raw = ws_conn.recv()
            if not raw:
                continue
            try:
                return json.loads(raw)
            except JSONDecodeError:
                continue
        raise TimeoutError("Timed out waiting for a JSON CDP message")

    def recv_until(predicate: Callable[[dict], bool], deadline: float) -> dict:
        """Read messages until predicate matches or timeout."""
        while time.time() < deadline:
            msg = recv_json(ws, deadline)
            if predicate(msg):
                return msg
        raise TimeoutError("Timed out waiting for expected CDP message")

    return send_cmd, recv_json, recv_until


# Tab/context management __________________________________________________________________________


def get_existing_tabs(remote_debugging_address: str) -> list[dict]:
    """Get list of existing browser tabs/targets.

    Args:
        remote_debugging_address: Chrome debugging server address.

    Returns:
        List of target info dicts with keys: id, title, url, type, etc.
    """
    base = remote_debugging_address.rstrip("/")
    try:
        response = requests.get(f"{base}/json/list", timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        raise RuntimeError(f"Failed to get existing tabs: {e}")


def cdp_attach_to_existing_tab(
    remote_debugging_address: str = "http://127.0.0.1:9222",
    target_id: str | None = None,
) -> tuple[str, None, WebSocket]:
    """
    Attach to an existing browser tab instead of creating a new one.

    Args:
        remote_debugging_address: Chrome debugging server address.
        target_id: Specific target ID to attach to. If None, attaches to first available page.

    Returns:
        Tuple of (target_id, None, browser_ws) - browser_context_id is None since we're
        reusing an existing tab.

    Raises:
        RuntimeError: If no suitable tab found or failed to attach.
    """
    # Find a target to attach to
    if target_id is None:
        tabs = get_existing_tabs(remote_debugging_address)
        # Filter for page targets (not devtools, extensions, etc.)
        page_tabs = [t for t in tabs if t.get("type") == "page"]
        if not page_tabs:
            raise RuntimeError("No existing page tabs found to attach to")
        # Use the first available page tab
        target_id = page_tabs[0]["id"]
        logger.debug(f"Auto-selected tab: {page_tabs[0].get('url', 'unknown')}")

    ws_url = get_browser_websocket_url(remote_debugging_address)
    logger.debug(f"cdp_attach_to_existing_tab ws_url: {ws_url}")

    browser_ws = None
    try:
        browser_ws = websocket.create_connection(ws_url, timeout=10)
        logger.debug(f"cdp_attach_to_existing_tab browser_ws: {browser_ws}")
        return target_id, None, browser_ws
    except Exception as e:
        if browser_ws:
            try:
                browser_ws.close()
            except Exception:
                pass
        raise RuntimeError(f"Failed to attach to existing tab: {e}")


def cdp_new_tab(
    remote_debugging_address: str = "http://127.0.0.1:9222",
    incognito: bool = True,
    url: str = "about:blank",
) -> tuple[str, str | None, WebSocket]:
    """
    Create a new browser tab and return target info and browser-level WebSocket.

    Args:
        remote_debugging_address: Chrome debugging server address.
        incognito: Whether to create an incognito context.
        url: Initial URL for the new tab.

    Returns:
        Tuple of (target_id, browser_context_id, browser_ws) where browser_ws is the
        BROWSER-LEVEL WebSocket connection (not page-level).

        NOTE: This browser_ws can be used with Target.attachToTarget + session_id
        for command-driven operations (like routine execution). For event-driven
        monitoring (CDPSession), use a page-level WebSocket instead:
        ws://{host}:{port}/devtools/page/{target_id}

    Raises:
        RuntimeError: If failed to create the tab.
    """
    ws_url = get_browser_websocket_url(remote_debugging_address)
    logger.debug(f"cdp_new_tab ws_url: {ws_url}")

    browser_ws = None
    try:
        try:
            browser_ws = websocket.create_connection(ws_url, timeout=10)
        except Exception as e:
            raise RuntimeError(f"Failed to connect to browser WebSocket: {e}")

        logger.debug(f"cdp_new_tab browser_ws: {browser_ws}")

        send_cmd, _, recv_until = create_cdp_helpers(browser_ws)

        # Create incognito context if requested
        browser_context_id = None
        if incognito:
            iid = send_cmd("Target.createBrowserContext")
            reply = recv_until(lambda m: m.get("id") == iid, time.time() + 10)
            if "error" in reply:
                raise RuntimeError(reply["error"])
            browser_context_id = reply["result"]["browserContextId"]

        # Create the target
        params: dict = {"url": url}
        if browser_context_id:
            params["browserContextId"] = browser_context_id
            params["newWindow"] = True  # Make it a visible incognito window

        tid = send_cmd("Target.createTarget", params)
        reply = recv_until(lambda m: m.get("id") == tid, time.time() + 10)
        if "error" in reply:
            raise RuntimeError(reply["error"])
        target_id = reply["result"]["targetId"]

        return target_id, browser_context_id, browser_ws

    except Exception as e:
        # Only close WebSocket on error
        if browser_ws:
            try:
                browser_ws.close()
            except Exception:
                pass
        raise RuntimeError(f"Failed to create target: {e}")


def dispose_context(remote_debugging_address: str, browser_context_id: str) -> None:
    """
    Dispose of a browser context.

    Args:
        remote_debugging_address: Chrome debugging server address.
        browser_context_id: The browser context ID to dispose.
    """
    ws_url = get_browser_websocket_url(remote_debugging_address)

    ws = websocket.create_connection(ws_url, timeout=10)
    try:
        ws.send(
            json.dumps(
                {
                    "id": 1,
                    "method": "Target.disposeBrowserContext",
                    "params": {"browserContextId": browser_context_id},
                }
            )
        )
        # read one reply (best-effort)
        json.loads(ws.recv())
    finally:
        try:
            ws.close()
        except Exception:
            pass
