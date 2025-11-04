import json
import logging
import random
import re
import time
from json import JSONDecodeError
from urllib.parse import urlparse, urlunparse

import requests
import websocket

from src.config import Config
from src.data_models.production_routine import (
    Routine,
    Endpoint,
    RoutineFetchOperation,
    RoutineNavigateOperation,
    RoutineReturnOperation,
    RoutineSleepOperation,
    RoutineReturnHTMLOperation,
    RoutineReturnScreenshotOperation,
)

logging.basicConfig(level=Config.LOG_LEVEL)
logger = logging.getLogger(__name__)


def _get_browser_websocket_url(remote_debugging_address: str) -> str:
    """Get the normalized WebSocket URL for browser connection."""
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


def _generate_fetch_js(
    fetch_url: str,
    headers: dict,
    body_js_literal: str,
    endpoint_method: str,
    endpoint_credentials: str,
) -> str:
    """Generate JavaScript code for fetch operation."""
    hdrs_json = json.dumps(
        {str(k): (str(v) if not isinstance(v, str) else v) for k, v in headers.items()}
    )

    # Build JavaScript code as a list of lines to avoid triple quote issues
    js_lines = [
        "(async () => {",
        "  const sleep = ms => new Promise(r => setTimeout(r, ms));",
        "  await sleep(100);",
        f"  const url = {json.dumps(fetch_url)};",
        f"  const rawHeaders = {hdrs_json};",
        f"  const BODY_LITERAL = {body_js_literal};",
        "",
        "  // Simple tokens (computed locally, no source lookup)",
        "  function replaceSimpleTokens(str){",
        "    if (typeof str !== 'string') return str;",
        "    // Handle quoted and unquoted: \"{{epoch_milliseconds}}\" or {{epoch_milliseconds}}",
        "    str = str.replace(/\\\"?\\{\\{\\s*epoch_milliseconds\\s*\\}\\}\\\"?/g, () => String(Date.now()));",
        "    // Handle {{uuid}} - generate UUID using crypto.randomUUID() if available",
        "    str = str.replace(/\\\"?\\{\\{\\s*uuid\\s*\\}\\}\\\"?/g, () => {",
        "      if ('randomUUID' in crypto) {",
        "        return crypto.randomUUID();",
        "      }",
        "      // Fallback for browsers without crypto.randomUUID()",
        "      return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {",
        "        const r = Math.random() * 16 | 0;",
        "        const v = c === 'x' ? r : (r & 0x3 | 0x8);",
        "        return v.toString(16);",
        "      });",
        "    });",
        "    return str;",
        "  }",
        "",
        "  function getCookie(name){",
        "    const m = document.cookie.match('(?:^|; )' + name.replace(/[-/\\\\^$*+?.()|[\\]{}]/g,'\\\\$&') + '=([^;]*)');",
        "    return m ? decodeURIComponent(m[1]) : undefined;",
        "  }",
        "  ",
        "  function getMeta(name){",
        '    return document.querySelector(`meta[name="${name}"]`)?.content;',
        "  }",
        "  ",
        "  function looksLikeJsonObject(s){ return typeof s === 'string' && s.trim().startsWith('{') && s.trim().endsWith('}'); }",
        "  function ensureParsed(val){",
        "    if (looksLikeJsonObject(val)) {",
        "      try { return JSON.parse(val); } catch { return val; }",
        "    }",
        "    return val;",
        "  }",
        "  function readStorage(storage, keyPath){",
        "    const [key, ...rest] = keyPath.split('.');",
        "    const raw = storage.getItem(key);",
        "    if (raw == null) return undefined;",
        "    try {",
        "      let obj = JSON.parse(raw);",
        "      obj = ensureParsed(obj);",
        "      return rest.reduce((o,p)=> {",
        "        if (o == null) return undefined;",
        "        o = ensureParsed(o);",
        "        let v = o[p];",
        "        v = ensureParsed(v);",
        "        return v;",
        "      }, obj);",
        "    } catch {",
        "      return rest.length ? undefined : raw;",
        "    }",
        "  }",
        "",
        "  const PLACEHOLDER = /\\\"?\\{\\{\\s*(sessionStorage|localStorage|cookie|meta)\\s*:\\s*([^}]+?)\\s*\\}\\}\\\"?/g;",
        "  function resolveOne(token){",
        "    const [lhs, rhs] = token.split('||');",
        "    const [kind, path] = lhs.split(':');",
        "    let val;",
        "    switch(kind){",
        "      case 'sessionStorage': val = readStorage(window.sessionStorage, path.trim()); break;",
        "      case 'localStorage':   val = readStorage(window.localStorage, path.trim()); break;",
        "      case 'cookie':         val = getCookie(path.trim()); break;",
        "      case 'meta':           val = getMeta(path.trim()); break;",
        "    }",
        "    if ((val === undefined || val === null || val === '') && rhs){",
        "      if (rhs.trim() === 'uuid' && 'randomUUID' in crypto){",
        "        val = crypto.randomUUID();",
        "      } else {",
        "        val = rhs.trim();",
        "      }",
        "    }",
        "    return val;",
        "  }",
        "  ",
        "  function resolvePlaceholders(str){",
        "    if (typeof str !== 'string') return str;",
        "    str = replaceSimpleTokens(str);",
        "    // Follow test.py pattern: for quoted placeholders, strings use raw value, objects use JSON.stringify",
        "    return str.replace(PLACEHOLDER, (m, _k, inner) => {",
        "      const v = resolveOne(`${_k}:${inner}`);",
        "      if (v === undefined || v === null) return m;",
        "      // Check if match was quoted - could be \"{{...}}\" or \\\"{{...}}\\\"",
        "      // Check for escaped quote \\\" at start/end, or simple quote \"",
        "      const startsWithEscaped = m.startsWith('\\\\\"') || m.startsWith('\"');",
        "      const endsWithEscaped = m.endsWith('\\\\\"') || (m.endsWith('\"') && m.length > 2);",
        "      const isQuoted = startsWithEscaped && endsWithEscaped;",
        "      if (isQuoted) {",
        "        // Quoted: strings use raw value (no quotes), objects use JSON.stringify",
        "        return (typeof v === 'string') ? v : JSON.stringify(v);",
        "      } else {",
        "        // Unquoted: always stringify",
        "        return (typeof v === 'object') ? JSON.stringify(v) : String(v);",
        "      }",
        "    });",
        "  }",
        "",
        "  function deepResolve(val){",
        "    if (typeof val === 'string') return resolvePlaceholders(val);",
        "    if (Array.isArray(val)) return val.map(deepResolve);",
        "    if (val && typeof val === 'object') {",
        "      const out = {};",
        "      for (const [k, v] of Object.entries(val)) out[k] = deepResolve(v);",
        "      return out;",
        "    }",
        "    return val;",
        "  }",
        "",
        "  // Resolve headers",
        "  const headers = {};",
        "  for (const [k, v] of Object.entries(rawHeaders || {})) {",
        "    headers[k] = (typeof v === 'string') ? resolvePlaceholders(v) : v;",
        "  }",
        "",
        "  const opts = {",
        f"    method: {json.dumps(endpoint_method)},",
        "    headers,",
        f"    credentials: {json.dumps(endpoint_credentials)}",
        "  };",
        "",
        "  // Resolve body (if any)",
        "  if (BODY_LITERAL !== null) {",
        "    const bodyVal = deepResolve(BODY_LITERAL);",
        "    if (typeof bodyVal === 'string' && bodyVal.trim().startsWith('{') && bodyVal.trim().endsWith('}')) {",
        "      opts.body = bodyVal;",
        "    } else {",
        "      opts.body = JSON.stringify(bodyVal);",
        "    }",
        "  }",
        "",
        "  try {",
        "    const resp = await fetch(url, opts);",
        "    const status = resp.status;",
        "    const val = await resp.text(); return {status, value: val};",
        "  } catch(e) {",
        "    return { __err: 'fetch failed: ' + String(e) };",
        "  }",
        "})()",
    ]

    return "\n".join(js_lines)


def _create_cdp_helpers(ws):
    """Create helper functions for CDP communication."""

    def send_cmd(method, params=None, session_id=None, _id=[1]):
        msg = {"id": _id[0], "method": method}
        if params:
            msg["params"] = params
        if session_id:
            msg["sessionId"] = session_id
        _id[0] += 1
        ws.send(json.dumps(msg))
        return msg["id"]

    def recv_json(ws, deadline):
        """Robustly read a single JSON message from the WS, skipping empty/non-JSON frames."""
        while time.time() < deadline:
            raw = ws.recv()
            if not raw:
                continue
            try:
                return json.loads(raw)
            except JSONDecodeError:
                continue
        raise TimeoutError("Timed out waiting for a JSON CDP message")

    def recv_until(predicate, deadline):
        while time.time() < deadline:
            msg = recv_json(ws, deadline)
            if predicate(msg):
                return msg
        raise TimeoutError("Timed out waiting for expected CDP message")

    return send_cmd, recv_json, recv_until


def cdp_new_tab(
    remote_debugging_address: str = "http://127.0.0.1:9222",
    incognito: bool = True,
    url: str = "about:blank",
):
    """Create a new browser tab and return target info and WebSocket connection."""
    ws_url = _get_browser_websocket_url(remote_debugging_address)
    logger.debug(f"cdp_new_tab ws_url: {ws_url}")

    ws = None
    try:
        try:
            ws = websocket.create_connection(ws_url, timeout=10)
        except Exception as e:
            raise RuntimeError(f"Failed to connect to browser WebSocket: {e}")

        logger.debug(f"cdp_new_tab ws: {ws}")

        send_cmd, recv_json, recv_until = _create_cdp_helpers(ws)

        # Create incognito context if requested
        browser_context_id = None
        if incognito:
            iid = send_cmd("Target.createBrowserContext")
            reply = recv_until(lambda m: m.get("id") == iid, time.time() + 10)
            if "error" in reply:
                raise RuntimeError(reply["error"])
            browser_context_id = reply["result"]["browserContextId"]

        # Create the target
        params = {"url": url}
        if browser_context_id:
            params["browserContextId"] = browser_context_id
            params["newWindow"] = True  # Make it a visible incognito window

        tid = send_cmd("Target.createTarget", params)
        reply = recv_until(lambda m: m.get("id") == tid, time.time() + 10)
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


def dispose_context(remote_debugging_address: str, browser_context_id: str):
    """Dispose of a browser context."""
    ws_url = _get_browser_websocket_url(remote_debugging_address)

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


def _execute_fetch_in_session(
    endpoint: Endpoint,
    parameters_dict: dict | None,
    session_id: str,
    send_cmd: callable,
    recv_until: callable,
    timeout: float,
):
    """
    Execute a fetch operation within an existing CDP session.

    This avoids creating new tabs and maintains session storage.

    Args:
        endpoint: The endpoint configuration.
        parameters_dict: Parameters for interpolation.
        session_id: CDP session ID.
        send_cmd: Function to send CDP commands.
        recv_until: Function to receive CDP responses.
        timeout: Request timeout.

    Returns:
        dict: Result with "ok" status and "result" data.
    """
    if parameters_dict is None:
        parameters_dict = {}

    # Apply parameters to endpoint
    fetch_url = _apply_params(endpoint.url, parameters_dict)
    headers = {}
    if endpoint.headers:
        headers_str = json.dumps(endpoint.headers)  # convert headers from dict to str
        headers_str_interpolated = _apply_params(headers_str, parameters_dict)
        headers = json.loads(headers_str_interpolated)  # convert headers from str to dict
        logger.debug(f"Headers: {headers}")

    body = None
    if endpoint.body:
        body_str = json.dumps(endpoint.body)  # convert body from dict to str
        body_str_interpolated = _apply_params(body_str, parameters_dict)
        body = json.loads(body_str_interpolated)  # convert body from str to dict
        logger.debug(f"Body: {body}")
        
    # Prepare headers and body for injection
    hdrs = headers or {}

    # Serialize body to JS string literal
    if body is None:
        body_js_literal = "null"
    elif isinstance(body, (dict, list)):
        body_js_literal = json.dumps(body)  # JS object, will be JSON.stringify'd in JS
    elif isinstance(body, bytes):
        body_js_literal = json.dumps(body.decode("utf-8", errors="ignore"))
    else:
        # body is already a JSON string, just escape it for JS
        body_js_literal = json.dumps(str(body))

    hdrs_json = json.dumps(
        {str(k): (str(v) if not isinstance(v, str) else v) for k, v in hdrs.items()}
    )

    # Injected JS (same as cdp_fetch but within existing session)
    # Build JS using the shared generator to keep behavior consistent
    expr = _generate_fetch_js(
        fetch_url=fetch_url,
        headers=hdrs,
        body_js_literal=body_js_literal,
        endpoint_method=endpoint.method,
        endpoint_credentials=endpoint.credentials,
    )

    # Execute the fetch
    eval_id = send_cmd(
        "Runtime.evaluate",
        {
            "expression": expr,
            "awaitPromise": True,
            "returnByValue": True,
            "timeout": int(timeout * 1000),
        },
        session_id=session_id,
    )

    reply = recv_until(lambda m: m.get("id") == eval_id, time.time() + timeout)
    if "error" in reply:
        return {"ok": False, "result": reply["error"]}
    payload = reply["result"]["result"].get("value")

    if isinstance(payload, dict) and payload.get("__err"):
        return {"ok": False, "status": payload.get("status"), "result": payload.get("__err")}

    return {"ok": True, "status": payload.get("status"), "result": payload}


def _apply_params(text: str, parameters_dict: dict | None) -> str:
    """
    Replace parameter placeholders in text with actual values.

    Only replaces {{param}} where 'param' is in parameters_dict.
    Leaves other placeholders like {{sessionStorage:...}} untouched.
    
    Follows the pattern from test.py:
    - For string values in quoted placeholders: insert raw string (no quotes)
    - For non-string values in quoted placeholders: use json.dumps(value)
    - For unquoted placeholders: use str(value)

    Args:
        text: Text containing parameter placeholders.
        parameters_dict: Dictionary of parameter values.

    Returns:
        str: Text with parameters replaced.
    """
    if not text or not parameters_dict:
        return text
    
    for key, value in parameters_dict.items():
        # Compute replacement based on value type (following test.py pattern)
        if isinstance(value, str):
            literal = value  # For strings, insert raw string (no quotes)
        else:
            literal = json.dumps(value)  # For numbers/bools/null, use JSON encoding
        
        escaped_key = re.escape(key)
        
        # Pattern 1: Simple quoted placeholder "{{key}}" in JSON string
        # Matches: "{{key}}" (when the JSON value itself is the string "{{key}}")
        # Use regular string concatenation to avoid f-string brace escaping issues
        simple_quoted = '"' + r'\{\{' + r'\s*' + escaped_key + r'\s*' + r'\}\}' + '"'
        text = re.sub(simple_quoted, literal, text)
        
        # Pattern 2: Escaped quote variant \"{{key}}\"
        # In JSON string this appears as: \\"{{key}}\\" 
        # Use regular string concatenation to build pattern with proper escaping
        double_escaped = r'\\"' + r'\{\{' + r'\s*' + escaped_key + r'\s*' + r'\}\}' + r'\\"'
        text = re.sub(double_escaped, literal, text)
        
        # Pattern 3: Bare placeholder {{key}} (unquoted, for URL params, etc.)
        bare_pattern = r'\{\{' + r'\s*' + escaped_key + r'\s*' + r'\}\}'
        text = re.sub(bare_pattern, str(value), text)

    return text


def _generate_random_user_agent() -> str:
    """
    Generate a realistic User-Agent string from various browsers and platforms.

    Returns a randomized User-Agent from Chrome/Edge/Firefox/Safari on
    Windows/macOS/iOS/Android with realistic version ranges.

    Returns:
        str: A realistic User-Agent string.
    """

    def chrome_ver():
        # Chrome 122–130 (major), randomized build pattern
        major = random.randint(122, 130)
        build1 = random.randint(0, 9)
        build2 = random.randint(4000, 6999)
        return f"{major}.{build1}.{build2}.{random.randint(50, 199)}"

    def firefox_ver():
        # Firefox 115–131
        return f"{random.randint(115, 131)}.0"

    def ios_ver():
        # iOS 15.0–17.6
        major = random.randint(15, 17)
        minor = random.randint(0, 6)
        return f"{major}_{minor}"

    def mac_ver():
        # macOS 10.15.7 (Catalina) through 14.5 (Sonoma)
        options = [
            "10_15_7",  # Catalina
            "11_7_10",  # Big Sur
            "12_7_6",  # Monterey
            "13_6_7",  # Ventura
            "14_5",  # Sonoma
        ]
        return random.choice(options)

    def android_ver():
        # Android 10–14
        return str(random.randint(10, 14))

    def android_device():
        devices = [
            "Pixel 7",
            "Pixel 7 Pro",
            "Pixel 8",
            "Pixel 8 Pro",
            "SM-S911B",
            "SM-S916B",
            "SM-S918B",  # Galaxy S23 line
            "SM-S921B",
            "SM-S926B",
            "SM-S928B",  # Galaxy S24 line
            "M2007J3SG",
            "CPH2413",
            "V2145",
        ]
        return random.choice(devices)

    webkit_ver = "537.36"
    safari_ios_ver = "604.1"  # iOS Safari compat tail
    safari_mac_webkit = "605.1.15"  # macOS Safari WebKit
    edge_tail = lambda: f"Edg/{chrome_ver()}"

    # A small menu of realistic patterns
    patterns = []

    # Chrome on macOS
    patterns.append(
        f"Mozilla/5.0 (Macintosh; Intel Mac OS X {mac_ver()}) "
        f"AppleWebKit/{webkit_ver} (KHTML, like Gecko) "
        f"Chrome/{chrome_ver()} Safari/{webkit_ver}"
    )

    # Chrome on Windows
    patterns.append(
        f"Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        f"AppleWebKit/{webkit_ver} (KHTML, like Gecko) "
        f"Chrome/{chrome_ver()} Safari/{webkit_ver}"
    )

    # Edge on Windows
    patterns.append(
        f"Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        f"AppleWebKit/{webkit_ver} (KHTML, like Gecko) "
        f"Chrome/{chrome_ver()} Safari/{webkit_ver} {edge_tail()}"
    )

    # Chrome on Android (Mobile)
    patterns.append(
        f"Mozilla/5.0 (Linux; Android {android_ver()}; {android_device()}) "
        f"AppleWebKit/{webkit_ver} (KHTML, like Gecko) "
        f"Chrome/{chrome_ver()} Mobile Safari/{webkit_ver}"
    )

    # Chrome on Android (Tablet-ish — omit 'Mobile')
    patterns.append(
        f"Mozilla/5.0 (Linux; Android {android_ver()}; {android_device()}) "
        f"AppleWebKit/{webkit_ver} (KHTML, like Gecko) "
        f"Chrome/{chrome_ver()} Safari/{webkit_ver}"
    )

    # Firefox on Windows
    patterns.append(
        f"Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:{firefox_ver().split('.')[0]}) "
        f"Gecko/20100101 Firefox/{firefox_ver()}"
    )

    # Firefox on macOS
    patterns.append(
        f"Mozilla/5.0 (Macintosh; Intel Mac OS X {mac_ver().replace('.', '_')}; rv:{firefox_ver().split('.')[0]}) "
        f"Gecko/20100101 Firefox/{firefox_ver()}"
    )

    # Safari on iPhone
    patterns.append(
        f"Mozilla/5.0 (iPhone; CPU iPhone OS {ios_ver()} like Mac OS X) "
        f"AppleWebKit/{safari_mac_webkit} (KHTML, like Gecko) "
        f"Version/{random.randint(16, 18)}.0 Mobile/15E148 Safari/{safari_ios_ver}"
    )

    # Safari on iPad
    patterns.append(
        f"Mozilla/5.0 (iPad; CPU OS {ios_ver()} like Mac OS X) "
        f"AppleWebKit/{safari_mac_webkit} (KHTML, like Gecko) "
        f"Version/{random.randint(16, 18)}.0 Mobile/15E148 Safari/{safari_ios_ver}"
    )

    # Safari on macOS
    patterns.append(
        f"Mozilla/5.0 (Macintosh; Intel Mac OS X {mac_ver()}) "
        f"AppleWebKit/{safari_mac_webkit} (KHTML, like Gecko) "
        f"Version/{random.randint(16, 18)}.0 Safari/{safari_mac_webkit}"
    )

    user_agent = random.choice(patterns)

    return user_agent


def execute_routine(
    routine: Routine,
    parameters_dict: dict | None = None,
    remote_debugging_address: str = "http://127.0.0.1:9222",
    timeout: float = 180.0,
    wait_after_navigate_sec: float = 3.0,
    close_tab_when_done: bool = True,
    incognito: bool = False,
):
    """
    Execute a routine using Chrome DevTools Protocol.

    Executes a sequence of operations (navigate, sleep, fetch, return) in a browser
    session, maintaining state between operations.

    Args:
        routine: The routine containing operations to execute.
        parameters_dict: Parameters for URL/header/body interpolation.
        remote_debugging_address: Chrome debugging server address.
        timeout: Operation timeout in seconds.
        wait_after_navigate_sec: Wait time after navigation.
        close_tab_when_done: Whether to close the tab when finished.
        incognito: Whether to use incognito mode.

    Returns:
        dict: Result with "ok" status and "result" data.
    """
    if parameters_dict is None:
        parameters_dict = {}

    # Create a new tab for the routine
    try:
        target_id, browser_context_id, ws = cdp_new_tab(
            remote_debugging_address=remote_debugging_address,
            incognito=incognito or routine.incognito,
            url="about:blank",
        )
    except Exception as e:
        return {"ok": False, "result": f"Failed to create tab: {e}"}

    try:
        # Use the WebSocket connection from cdp_new_tab
        # No need to create a new connection since we already have one

        def send_cmd(method, params=None, session_id=None, _id=[1]):
            msg = {"id": _id[0], "method": method}
            if params:
                msg["params"] = params
            if session_id:
                msg["sessionId"] = session_id
            _id[0] += 1
            ws.send(json.dumps(msg))
            return msg["id"]

        def recv_json(ws, deadline):
            while time.time() < deadline:
                raw = ws.recv()
                if not raw:
                    continue
                try:
                    return json.loads(raw)
                except JSONDecodeError:
                    continue
            raise TimeoutError("Timed out waiting for a JSON CDP message")

        def recv_until(predicate, deadline):
            while time.time() < deadline:
                msg = recv_json(ws, deadline)
                if predicate(msg):
                    return msg
            raise TimeoutError("Timed out waiting for expected CDP message")

        # Attach to target
        attach_id = send_cmd(
            "Target.attachToTarget", {"targetId": target_id, "flatten": True}
        )
        reply = recv_until(lambda m: m.get("id") == attach_id, time.time() + timeout)
        session_id = reply["result"]["sessionId"]

        # Enable domains
        send_cmd("Page.enable", session_id=session_id)
        send_cmd("Runtime.enable", session_id=session_id)
        send_cmd("Network.enable", session_id=session_id)

        # Execute operations
        result = None
        current_url = None

        logger.info(f"Executing routine with {len(routine.operations)} operations")
        for i, operation in enumerate(routine.operations):
            logger.info(
                f"Executing operation {i+1}/{len(routine.operations)}: {type(operation).__name__}"
            )

            if isinstance(operation, RoutineNavigateOperation):
                # Navigate to URL
                url = _apply_params(operation.url, parameters_dict)
                send_cmd("Page.navigate", {"url": url}, session_id=session_id)
                current_url = url
                if wait_after_navigate_sec > 0:
                    time.sleep(wait_after_navigate_sec)

            elif isinstance(operation, RoutineSleepOperation):
                # Sleep
                time.sleep(operation.timeout_seconds)

            elif isinstance(operation, RoutineFetchOperation):
                # Navigate to origin URL if we haven't already
                if not current_url:

                    endpoint_url = _apply_params(
                        operation.endpoint.url, parameters_dict
                    )
                    parsed = urlparse(endpoint_url)
                    origin_url = f"{parsed.scheme}://{parsed.netloc}"
                    send_cmd(
                        "Page.navigate", {"url": origin_url}, session_id=session_id
                    )
                    current_url = origin_url
                    if wait_after_navigate_sec > 0:
                        time.sleep(wait_after_navigate_sec)

                fetch_result = _execute_fetch_in_session(
                    endpoint=operation.endpoint,
                    parameters_dict=parameters_dict,
                    session_id=session_id,
                    send_cmd=send_cmd,
                    recv_until=recv_until,
                    timeout=timeout,
                )

                # Store result in session storage if key provided
                if operation.session_storage_key and fetch_result.get("ok"):
                    result_data = fetch_result.get("result", {}).get("value", {})
                    js = f"window.sessionStorage.setItem('{operation.session_storage_key}', JSON.stringify({json.dumps(result_data)}));"
                    send_cmd(
                        "Runtime.evaluate", {"expression": js}, session_id=session_id
                    )

            elif isinstance(operation, RoutineReturnOperation):
                # Get result from session storage
                js = f"window.sessionStorage.getItem('{operation.session_storage_key}')"
                eval_id = send_cmd(
                    "Runtime.evaluate",
                    {"expression": js, "returnByValue": True},
                    session_id=session_id,
                )
                reply = recv_until(
                    lambda m: m.get("id") == eval_id, time.time() + timeout
                )
                stored_value = reply["result"]["result"].get("value")

                if stored_value:
                    try:
                        result = json.loads(stored_value)
                    except Exception as e:
                        result = stored_value
                else:
                    result = None
            
            elif isinstance(operation, RoutineReturnHTMLOperation):
                # Get HTML from page or element
                if operation.scope == "page":
                    # Get full page HTML
                    js = "document.documentElement.outerHTML"
                else:
                    # Get element HTML with selector
                    if not operation.selector:
                        return {"ok": False, "result": "Selector is required for element scope"}

                    # Build JS to find element and get HTML
                    js = f"""
                    (() => {{
                        const element = document.querySelector({json.dumps(operation.selector)});
                        if (!element) {{
                            throw new Error('Element not found with selector: {operation.selector}');
                        }}
                        return element.outerHTML;
                    }})()
                    """

                # Execute JavaScript to get HTML
                eval_id = send_cmd(
                    "Runtime.evaluate",
                    {
                        "expression": js,
                        "returnByValue": True,
                        "timeout": operation.timeout_ms,
                    },
                    session_id=session_id,
                )

                reply = recv_until(
                    predicate=lambda m: m.get("id") == eval_id, 
                    deadline=time.time() + (operation.timeout_ms / 1000),
                )

                if "error" in reply:
                    return {"ok": False, "result": reply["error"]}

                # Check if there was an exception in the JavaScript execution
                if reply.get("result", {}).get("exceptionDetails"):
                    exception = reply["result"]["exceptionDetails"]
                    error_msg = exception.get("exception", {}).get("description", "Unknown error")
                    return {"ok": False, "result": f"JavaScript error: {error_msg}"}

                html_content = reply["result"]["result"].get("value")
                result = html_content

            elif isinstance(operation, RoutineReturnScreenshotOperation):
                # Capture screenshot using CDP Page.captureScreenshot
                screenshot_params = {
                    "format": "png",
                    "captureBeyondViewport": operation.full_page,
                }

                screenshot_id = send_cmd(
                    "Page.captureScreenshot",
                    screenshot_params,
                    session_id=session_id,
                )

                reply = recv_until(
                    predicate=lambda m: m.get("id") == screenshot_id,
                    deadline=time.time() + (operation.timeout_ms / 1000),
                )

                if "error" in reply:
                    return {"ok": False, "result": reply["error"]}

                # Get base64-encoded screenshot data
                screenshot_data = reply.get("result", {}).get("data")
                if not screenshot_data:
                    return {"ok": False, "result": "Failed to capture screenshot"}
                result = screenshot_data

        return {"ok": True, "result": result}

    except Exception as e:
        return {"ok": False, "result": f"Routine execution failed: {e}"}
    finally:
        try:
            if close_tab_when_done:
                send_cmd("Target.closeTarget", {"targetId": target_id})
                if browser_context_id and incognito:
                    dispose_context(remote_debugging_address, browser_context_id)
        except Exception:
            pass
        try:
            ws.close()
        except Exception:
            pass
