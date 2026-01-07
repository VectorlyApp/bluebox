"""
web_hacker/utils/js_utils.py

JavaScript injection utilities for CDP operations.

All JavaScript code that gets injected into the browser should be generated
through functions in this module for consistency and maintainability.
"""

import json


def generate_fetch_js(
    fetch_url: str,
    headers: dict,
    body_js_literal: str,
    endpoint_method: str,
    endpoint_credentials: str,
    session_storage_key: str | None = None,
) -> str:
    """Generate JavaScript code for fetch operation.

    Args:
        fetch_url: The URL to fetch.
        headers: Dictionary of HTTP headers.
        body_js_literal: JavaScript literal for the request body.
        endpoint_method: HTTP method (GET, POST, etc.).
        endpoint_credentials: Credentials mode (same-origin, include, omit).
        session_storage_key: Optional key to store result in session storage.

    Returns:
        JavaScript code string that performs the fetch operation.
    """
    hdrs_json = json.dumps(
        {str(k): (str(v) if not isinstance(v, str) else v) for k, v in headers.items()}
    )

    js_lines = [
        "(async () => {",
        "  const sleep = ms => new Promise(r => setTimeout(r, ms));",
        "  await sleep(100);",
        f"  const url = {json.dumps(fetch_url)};",
        f"  const rawHeaders = {hdrs_json};",
        f"  const BODY_LITERAL = {body_js_literal};",
        "  const resolvedValues = {};",
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
        "  const PLACEHOLDER = /\\\"?\\{\\{\\s*(sessionStorage|localStorage|cookie|meta|windowProperty)\\s*:\\s*([^}]+?)\\s*\\}\\}\\\"?/g;",
        "  function getWindowProperty(path){",
        "    const parts = path.trim().split('.');",
        "    let obj = window;",
        "    for (const part of parts) {",
        "      if (obj == null) return undefined;",
        "      // Check if this part is a method call (with or without arguments)",
        "      const methodMatch = part.match(/^(\\w+)\\((.*)\\)$/);",
        "      if (methodMatch) {",
        "        const methodName = methodMatch[1];",
        "        const argsStr = methodMatch[2].trim();",
        "        if (typeof obj[methodName] !== 'function') {",
        "          obj = undefined;",
        "        } else if (argsStr === '') {",
        "          // No arguments: getToken()",
        "          obj = obj[methodName]();",
        "        } else {",
        "          // Parse simple arguments: getToken(\"key\") or getToken(123)",
        "          try {",
        "            const args = JSON.parse('[' + argsStr + ']');",
        "            obj = obj[methodName](...args);",
        "          } catch {",
        "            obj = undefined;",
        "          }",
        "        }",
        "      } else {",
        "        obj = obj[part];",
        "      }",
        "    }",
        "    return obj;",
        "  }",
        "  function resolveOne(token){",
        "    const [lhs, rhs] = token.split('||');",
        "    const [kind, path] = lhs.split(':');",
        "    let val;",
        "    switch(kind.trim()){",
        "      case 'sessionStorage': val = readStorage(window.sessionStorage, path.trim()); break;",
        "      case 'localStorage':   val = readStorage(window.localStorage, path.trim()); break;",
        "      case 'cookie':         val = getCookie(path.trim()); break;",
        "      case 'meta':           val = getMeta(path.trim()); break;",
        "      case 'windowProperty': val = getWindowProperty(path.trim()); break;",
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
        "      resolvedValues[`${_k}:${inner}`] = (v === undefined) ? null : (typeof v === 'object' ? JSON.stringify(v) : String(v));",
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
        "  // Resolve URL (interpolate placeholders like {{sessionStorage:crumb}})",
        "  const resolvedUrl = resolvePlaceholders(url);",
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
        "    ",
        "    // Check if content-type is application/x-www-form-urlencoded (after interpolation)",
        "    const contentType = headers['content-type'] || headers['Content-Type'] || '';",
        "    const isFormUrlEncoded = contentType.toLowerCase().includes('application/x-www-form-urlencoded');",
        "    ",
        "    if (isFormUrlEncoded && bodyVal && typeof bodyVal === 'object' && !Array.isArray(bodyVal)) {",
        "      // Convert object to URL-encoded string",
        "      const formData = Object.entries(bodyVal).map(([key, value]) => {",
        "        const encodedKey = encodeURIComponent(String(key));",
        "        const encodedValue = encodeURIComponent(String(value === null || value === undefined ? '' : value));",
        "        return `${encodedKey}=${encodedValue}`;",
        "      }).join('&');",
        "      opts.body = formData;",
        "    } else if (typeof bodyVal === 'string' && bodyVal.trim().startsWith('{') && bodyVal.trim().endsWith('}')) {",
        "      opts.body = bodyVal;",
        "    } else {",
        "      opts.body = JSON.stringify(bodyVal);",
        "    }",
        "  }",
        "",
        "  try {",
        "    const resp = await fetch(resolvedUrl, opts);",
        "    const status = resp.status;",
        "    const val = await resp.text();",
        f"    if ({'true' if session_storage_key else 'false'}) {{ try {{ window.sessionStorage.setItem({json.dumps(session_storage_key) if session_storage_key else 'null'}, JSON.stringify(val)); }} catch(e) {{ return {{ __err: 'SessionStorage Error: ' + String(e), resolvedValues }}; }} }}",
        "    return {status, value: 'success', resolvedValues};",
        "  } catch(e) {",
        "    return { __err: 'fetch failed: ' + String(e), resolvedValues };",
        "  }",
        "})()",
    ]

    return "\n".join(js_lines)


def generate_download_js(
    download_url: str,
    headers: dict,
    body_js_literal: str,
    endpoint_method: str,
    endpoint_credentials: str,
    filename: str,
) -> str:
    """Generate JavaScript code for downloading a file as base64.

    Args:
        download_url: The URL to download from.
        headers: Dictionary of headers.
        body_js_literal: JavaScript literal for the request body.
        endpoint_method: HTTP method (GET, POST, etc.).
        endpoint_credentials: Credentials mode (same-origin, include, omit).
        filename: Filename for the downloaded file.

    Returns:
        JavaScript code that fetches the URL, converts response to base64,
        stores in window.__downloadData, and returns metadata for chunked retrieval.
    """
    headers_json = json.dumps(headers)

    js_lines = [
        "(async () => {",
        f"  const url = {json.dumps(download_url)};",
        f"  const headers = {headers_json};",
        f"  const body = {body_js_literal};",
        f"  const filename = {json.dumps(filename)};",
        "",
        "  const opts = {",
        f"    method: {json.dumps(endpoint_method)},",
        "    headers: headers,",
        f"    credentials: {json.dumps(endpoint_credentials)}",
        "  };",
        "",
        "  if (body !== null) {",
        "    opts.body = typeof body === 'string' ? body : JSON.stringify(body);",
        "  }",
        "",
        "  try {",
        "    const resp = await fetch(url, opts);",
        "",
        "    if (!resp.ok) {",
        "      return { __err: 'Download failed with status ' + resp.status };",
        "    }",
        "",
        "    const contentType = resp.headers.get('content-type') || 'application/octet-stream';",
        "    const buffer = await resp.arrayBuffer();",
        "",
        "    // Convert to base64 using Blob + FileReader",
        "    const blob = new Blob([buffer]);",
        "    const base64Data = await new Promise((resolve, reject) => {",
        "      const reader = new FileReader();",
        "      reader.onload = () => {",
        "        const dataUrl = reader.result;",
        "        resolve(dataUrl.split(',')[1]);",
        "      };",
        "      reader.onerror = () => reject(reader.error);",
        "      reader.readAsDataURL(blob);",
        "    });",
        "",
        "    // Store base64 data in window for chunked retrieval",
        "    window.__downloadData = base64Data;",
        "",
        "    return {",
        "      ok: true,",
        "      contentType: contentType,",
        "      filename: filename,",
        "      size: buffer.byteLength,",
        "      base64Length: base64Data.length",
        "    };",
        "  } catch(e) {",
        "    return { __err: 'Download failed: ' + String(e) };",
        "  }",
        "})()",
    ]

    return "\n".join(js_lines)


def generate_click_js(selector: str, ensure_visible: bool) -> str:
    """Generate JavaScript to find element and get click coordinates.

    Args:
        selector: CSS selector for the element.
        ensure_visible: Whether to scroll element into view.

    Returns:
        JavaScript code that returns element coordinates or error info.
    """
    return f"""
(function() {{
    const selector = {json.dumps(selector)};
    const element = document.querySelector(selector);

    if (!element) {{
        const allInputs = Array.from(document.querySelectorAll('input')).map(el => {{
            return {{
                name: el.getAttribute('name'),
                id: el.id,
                type: el.type,
                class: el.className
            }};
        }}).slice(0, 10);

        return {{ 
            error: 'Element not found: ' + selector,
            debug: {{
                pageTitle: document.title,
                url: window.location.href,
                readyState: document.readyState,
                bodyLength: document.body ? document.body.innerHTML.length : 0,
                bodyHTML: document.body ? document.body.innerHTML : '',
                allInputs: allInputs
            }}
        }};
    }}

    const style = window.getComputedStyle(element);
    if (style.display === 'none' || style.visibility === 'hidden' || style.opacity === '0') {{
        return {{ error: 'Element is hidden: ' + selector }};
    }}

    if ({json.dumps(ensure_visible)}) {{
        element.scrollIntoView({{ behavior: 'auto', block: 'center', inline: 'center' }});
    }}

    const rect = element.getBoundingClientRect();

    if (rect.width === 0 || rect.height === 0) {{
        return {{ error: 'Element has no dimensions: ' + selector }};
    }}

    return {{
        x: rect.left + rect.width / 2,
        y: rect.top + rect.height / 2,
        width: rect.width,
        height: rect.height
    }};
}})()
"""


def generate_type_js(selector: str, clear: bool) -> str:
    """Generate JavaScript to find and focus an input element.

    Args:
        selector: CSS selector for the element.
        clear: Whether to clear existing text.

    Returns:
        JavaScript code that focuses the element or returns error.
    """
    return f"""
(function() {{
    const selector = {json.dumps(selector)};
    const element = document.querySelector(selector);

    if (!element) {{
        return {{ error: 'Element not found: ' + selector }};
    }}

    const style = window.getComputedStyle(element);
    if (style.display === 'none' || style.visibility === 'hidden' || style.opacity === '0') {{
        return {{ error: 'Element is hidden: ' + selector }};
    }}

    const tagName = element.tagName.toLowerCase();
    const isInput = tagName === 'input' || tagName === 'textarea';
    const isContentEditable = element.isContentEditable;

    if (!isInput && !isContentEditable) {{
        return {{ error: 'Element is not an input, textarea, or contenteditable: ' + selector }};
    }}

    if ({json.dumps(clear)}) {{
        if (isInput) {{
            element.value = '';
        }} else if (isContentEditable) {{
            element.textContent = '';
        }}
    }}

    element.focus();

    return {{ success: true }};
}})()
"""


def generate_scroll_element_js(selector: str, delta_x: int, delta_y: int, behavior: str) -> str:
    """Generate JavaScript to scroll a specific element.

    Args:
        selector: CSS selector for the element.
        delta_x: Horizontal scroll amount.
        delta_y: Vertical scroll amount.
        behavior: Scroll behavior ('auto' or 'smooth').

    Returns:
        JavaScript code that scrolls the element.
    """
    return f"""
(function() {{
    const selector = {json.dumps(selector)};
    const element = document.querySelector(selector);

    if (!element) {{
        return {{ error: 'Element not found: ' + selector }};
    }}

    const deltaX = {json.dumps(delta_x)};
    const deltaY = {json.dumps(delta_y)};

    if (deltaX !== 0 || deltaY !== 0) {{
        element.scrollBy({{
            left: deltaX,
            top: deltaY,
            behavior: {json.dumps(behavior)}
        }});
    }}

    return {{ success: true }};
}})()
"""


def generate_scroll_window_js(
    x: int | None,
    y: int | None,
    delta_x: int,
    delta_y: int,
    behavior: str,
) -> str:
    """Generate JavaScript to scroll the window.

    Args:
        x: Absolute X position (or None for relative).
        y: Absolute Y position (or None for relative).
        delta_x: Relative horizontal scroll amount.
        delta_y: Relative vertical scroll amount.
        behavior: Scroll behavior ('auto' or 'smooth').

    Returns:
        JavaScript code that scrolls the window.
    """
    return f"""
(function() {{
    const x = {json.dumps(x)};
    const y = {json.dumps(y)};
    const deltaX = {json.dumps(delta_x)};
    const deltaY = {json.dumps(delta_y)};
    const behavior = {json.dumps(behavior)};

    if (x !== null || y !== null) {{
        window.scrollTo({{
            left: x !== null ? x : window.scrollX,
            top: y !== null ? y : window.scrollY,
            behavior: behavior
        }});
    }}
    else if (deltaX !== 0 || deltaY !== 0) {{
        window.scrollBy({{
            left: deltaX,
            top: deltaY,
            behavior: behavior
        }});
    }}

    return {{ success: true }};
}})()
"""


def generate_wait_for_url_js(url_regex: str) -> str:
    """Generate JavaScript to check if current URL matches a regex.

    Args:
        url_regex: Regular expression pattern to match.

    Returns:
        JavaScript code that checks URL match.
    """
    return f"""
(function() {{
    const urlRegex = new RegExp({json.dumps(url_regex)});
    const currentUrl = window.location.href;
    const matches = urlRegex.test(currentUrl);

    return {{
        matches: matches,
        currentUrl: currentUrl,
        pattern: {json.dumps(url_regex)}
    }};
}})()
"""


def generate_store_in_session_storage_js(key: str, value_json: str) -> str:
    """Generate JavaScript to store a value in session storage.

    Args:
        key: Session storage key.
        value_json: JSON string to store.

    Returns:
        JavaScript code that stores the value.
    """
    return f"""
(function() {{
    try {{
        window.sessionStorage.setItem({json.dumps(key)}, {json.dumps(value_json)});
        return {{ ok: true }};
    }} catch(e) {{
        return {{ ok: false, error: String(e) }};
    }}
}})()
"""


def generate_get_session_storage_length_js(key: str) -> str:
    """Generate JavaScript to get length of a session storage value.

    Args:
        key: Session storage key.

    Returns:
        JavaScript expression that returns the length.
    """
    return f"window.sessionStorage.getItem({json.dumps(key)})?.length || 0"


def generate_get_session_storage_chunk_js(key: str, offset: int, end: int) -> str:
    """Generate JavaScript to get a chunk of a session storage value.

    Args:
        key: Session storage key.
        offset: Start offset.
        end: End offset.

    Returns:
        JavaScript code that returns the substring.
    """
    return f"""
(function() {{
    const val = window.sessionStorage.getItem({json.dumps(key)});
    return val.substring({offset}, {end});
}})()
"""


def generate_get_download_chunk_js(offset: int, end: int) -> str:
    """Generate JavaScript to get a chunk of download data.

    Args:
        offset: Start offset.
        end: End offset.

    Returns:
        JavaScript expression that returns the substring.
    """
    return f"window.__downloadData.substring({offset}, {end})"


def generate_get_html_js(selector: str | None = None) -> str:
    """Generate JavaScript to get HTML content.

    Args:
        selector: CSS selector for element, or None for full page.

    Returns:
        JavaScript expression that returns HTML.
    """
    if selector is None:
        return "document.documentElement.outerHTML"
    return f"document.querySelector({json.dumps(selector)})?.outerHTML || ''"


def generate_js_evaluate_with_storage_js(
    iife: str,
    session_storage_key: str,
) -> str:
    """
    Wrap IIFE in an outer async IIFE that executes it and stores result in session storage.
    
    This allows executing JavaScript AND storing the result in session storage
    with a SINGLE CDP Runtime.evaluate call.
    
    Args:
        iife: The IIFE code (already validated to be in IIFE format).
        session_storage_key: Key to store result in session storage.
    
    Returns:
        JavaScript code that executes the IIFE and stores the result.
    """
    return f"""(async () => {{
    // Execute IIFE and await if it returns a promise
    const __result = await Promise.resolve({iife});
    if (__result !== undefined) {{
        try {{
            window.sessionStorage.setItem({json.dumps(session_storage_key)}, JSON.stringify(__result));
        }} catch(e) {{
            return {{ __err: 'SessionStorage Error: ' + String(e) }};
        }}
    }}
    return true;
}})()"""


