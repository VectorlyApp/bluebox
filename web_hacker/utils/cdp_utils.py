"""
web_hacker/utils/cdp_utils.py

CDP (Chrome DevTools Protocol) utility functions.
"""

import json
import time
from urllib.parse import urlparse, urlunparse

import requests
import websocket
from websocket import WebSocket

from web_hacker.data_models.routine.endpoint import Endpoint
from web_hacker.data_models.routine.execution import FetchExecutionResult
from web_hacker.utils.data_utils import apply_params
from web_hacker.utils.js_utils import generate_fetch_js
from web_hacker.utils.logger import get_logger
from web_hacker.utils.web_socket_utils import send_cmd, recv_until

logger = get_logger(name=__name__)


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


def cdp_new_tab(
    remote_debugging_address: str = "http://127.0.0.1:9222",
    incognito: bool = True,
    url: str = "about:blank",
) -> tuple[str, str | None, WebSocket]:
    """
    Create a new browser tab and return target info and WebSocket connection.

    Args:
        remote_debugging_address: Chrome debugging server address.
        incognito: Whether to create an incognito context.
        url: Initial URL for the new tab.

    Returns:
        Tuple of (target_id, browser_context_id, ws) where ws is the WebSocket connection.

    Raises:
        RuntimeError: If failed to create the tab.
    """
    ws_url = get_browser_websocket_url(remote_debugging_address)
    logger.debug(f"cdp_new_tab ws_url: {ws_url}")

    ws = None
    try:
        try:
            ws = websocket.create_connection(ws_url, timeout=10)
        except Exception as e:
            raise RuntimeError(f"Failed to connect to browser WebSocket: {e}")

        logger.debug(f"cdp_new_tab ws: {ws}")

        # Create incognito context if requested
        browser_context_id = None
        if incognito:
            iid = send_cmd(ws, "Target.createBrowserContext")
            reply = recv_until(ws, lambda m: m.get("id") == iid, time.time() + 10)
            if "error" in reply:
                raise RuntimeError(reply["error"])
            browser_context_id = reply["result"]["browserContextId"]

        # Create the target
        params = {"url": url}
        if browser_context_id:
            params["browserContextId"] = browser_context_id
            params["newWindow"] = True  # Make it a visible incognito window

        tid = send_cmd(ws, "Target.createTarget", params)
        reply = recv_until(ws, lambda m: m.get("id") == tid, time.time() + 10)
        if "error" in reply:
            raise RuntimeError(reply["error"])
        target_id = reply["result"]["targetId"]

        # Return the WebSocket connection along with target info
        # The caller needs to keep this connection open for subsequent operations
        return target_id, browser_context_id, ws

    except Exception as e:
        # Only close WebSocket on error
        if ws:
            try:
                ws.close()
            except:
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
        except:
            pass


def execute_fetch_in_session(
    ws: WebSocket,
    endpoint: Endpoint,
    parameters_dict: dict | None,
    session_id: str,
    timeout: float,
    session_storage_key: str | None = None,
) -> FetchExecutionResult:
    """
    Execute a fetch operation within an existing CDP session.

    This avoids creating new tabs and maintains session storage.

    Args:
        ws: WebSocket connection to Chrome.
        endpoint: The endpoint configuration.
        parameters_dict: Parameters for interpolation.
        session_id: CDP session ID.
        timeout: Request timeout.
        session_storage_key: Optional session storage key to store the result.

    Returns:
        FetchExecutionResult: Result of the fetch execution.
    """
    if parameters_dict is None:
        parameters_dict = {}

    # Apply parameters to endpoint
    fetch_url = apply_params(endpoint.url, parameters_dict)
    headers = {}
    if endpoint.headers:
        headers_str = json.dumps(endpoint.headers)  # convert headers from dict to str
        headers_str_interpolated = apply_params(headers_str, parameters_dict)
        headers = json.loads(headers_str_interpolated)  # convert headers from str to dict

    body = None
    if endpoint.body:
        body_str = json.dumps(endpoint.body)  # convert body from dict to str
        body_str_interpolated = apply_params(body_str, parameters_dict)
        body = json.loads(body_str_interpolated)  # convert body from str to dict

    # Prepare headers and body for injection
    hdrs = headers or {}

    # Serialize body to JS string literal (conversion to form-urlencoded happens in JS after interpolation)
    if body is None:
        body_js_literal = "null"
    elif isinstance(body, (dict, list)):
        body_js_literal = json.dumps(body)  # JS object, will be processed in JS after interpolation
    elif isinstance(body, bytes):
        body_js_literal = json.dumps(body.decode("utf-8", errors="ignore"))
    else:
        # body is already a JSON string, just escape it for JS
        body_js_literal = json.dumps(str(body))

    # Build JS using the shared generator to keep behavior consistent
    expr = generate_fetch_js(
        fetch_url=fetch_url,
        headers=hdrs,
        body_js_literal=body_js_literal,
        endpoint_method=endpoint.method,
        endpoint_credentials=endpoint.credentials,
        session_storage_key=session_storage_key,
    )

    # Execute the fetch
    logger.info(f"Sending Runtime.evaluate for fetch with timeout={timeout}s")
    eval_id = send_cmd(ws,
        "Runtime.evaluate",
        {
            "expression": expr,
            "awaitPromise": True,
            "returnByValue": True,
            "timeout": int(timeout * 1000),
        },
        session_id=session_id,
    )

    reply = recv_until(ws, lambda m: m.get("id") == eval_id, time.time() + timeout)

    if "error" in reply:
        logger.error(f"Error in execute_fetch_in_session (CDP error): {reply['error']}")
        return FetchExecutionResult(ok=False, error=reply["error"])

    payload = reply["result"]["result"].get("value")

    if isinstance(payload, dict) and payload.get("__err"):
        logger.error(f"Error in execute_fetch_in_session (JS error): {payload.get('__err')}")
        return FetchExecutionResult(ok=False, error=payload.get("__err"), resolved_values=payload.get("resolvedValues", {}))

    logger.info(f"Payload in execute_fetch_in_session: {str(payload)[:1000]}...")  # Truncate for safety

    return FetchExecutionResult(ok=True, result=payload.get("value"), resolved_values=payload.get("resolvedValues", {}))

