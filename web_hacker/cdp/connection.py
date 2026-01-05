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


# User agent generation ___________________________________________________________________________


def generate_random_user_agent() -> str:
    """
    Generate a realistic User-Agent string from various browsers and platforms.

    Returns a randomized User-Agent from Chrome/Edge/Firefox/Safari on
    Windows/macOS/iOS/Android with realistic version ranges.

    Returns:
        str: A realistic User-Agent string.
    """

    def chrome_ver() -> str:
        major = random.randint(122, 130)
        build1 = random.randint(0, 9)
        build2 = random.randint(4000, 6999)
        return f"{major}.{build1}.{build2}.{random.randint(50, 199)}"

    def edge_ver() -> str:
        major = random.randint(122, 130)
        build = random.randint(2000, 2999)
        return f"{major}.0.{build}.{random.randint(50, 150)}"

    def firefox_ver() -> str:
        return str(random.randint(123, 130))

    def win_ver() -> str:
        return random.choice(["10.0", "11.0"])

    def mac_ver() -> str:
        major = random.randint(12, 14)
        minor = random.randint(0, 6)
        return f"10_{major}_{minor}"

    patterns: list[str] = []

    # Chrome on Windows
    patterns.append(
        f"Mozilla/5.0 (Windows NT {win_ver()}; Win64; x64) "
        f"AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{chrome_ver()} Safari/537.36"
    )

    # Chrome on macOS
    patterns.append(
        f"Mozilla/5.0 (Macintosh; Intel Mac OS X {mac_ver()}) "
        f"AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{chrome_ver()} Safari/537.36"
    )

    # Edge on Windows
    patterns.append(
        f"Mozilla/5.0 (Windows NT {win_ver()}; Win64; x64) "
        f"AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{chrome_ver()} Safari/537.36 Edg/{edge_ver()}"
    )

    # Firefox on Windows
    patterns.append(
        f"Mozilla/5.0 (Windows NT {win_ver()}; Win64; x64; rv:{firefox_ver()}.0) "
        f"Gecko/20100101 Firefox/{firefox_ver()}.0"
    )

    # Firefox on macOS
    patterns.append(
        f"Mozilla/5.0 (Macintosh; Intel Mac OS X {mac_ver()}; rv:{firefox_ver()}.0) "
        f"Gecko/20100101 Firefox/{firefox_ver()}.0"
    )

    # Chrome on Android
    android_ver = random.randint(12, 14)
    patterns.append(
        f"Mozilla/5.0 (Linux; Android {android_ver}; SM-S908B) "
        f"AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{chrome_ver()} Mobile Safari/537.36"
    )

    # Safari on iOS
    safari_ios_ver = f"605.1.{random.randint(10, 20)}"
    patterns.append(
        f"Mozilla/5.0 (iPhone; CPU iPhone OS {random.randint(16, 18)}_0 like Mac OS X) "
        f"AppleWebKit/{safari_ios_ver} (KHTML, like Gecko) "
        f"Version/{random.randint(16, 18)}.0 Mobile/15E148 Safari/{safari_ios_ver}"
    )

    # Safari on macOS
    safari_mac_webkit = f"605.1.{random.randint(10, 20)}"
    patterns.append(
        f"Mozilla/5.0 (Macintosh; Intel Mac OS X {mac_ver()}) "
        f"AppleWebKit/{safari_mac_webkit} (KHTML, like Gecko) "
        f"Version/{random.randint(16, 18)}.0 Safari/{safari_mac_webkit}"
    )

    return random.choice(patterns)
