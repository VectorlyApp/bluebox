"""
bluebox/endpoint_discovery/extractors.py

Deterministic regex-based extractors for HTTP call patterns in JavaScript code.

Extracts call sites from:
- fetch() calls
- XMLHttpRequest .open() calls
- axios.get/post/put/delete/patch/request calls
- jQuery $.ajax / $.get / $.post / $.getJSON calls
- Raw URL string literals that look like API endpoints

Each extractor returns a list of EndpointCallSite objects with as much
information as can be statically extracted from the surrounding code context.
"""

import re
from typing import Any

from bluebox.endpoint_discovery.models import CallType, EndpointCallSite, VariableBinding
from bluebox.utils.logger import get_logger

logger = get_logger(name=__name__)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

# How many chars of context to grab around a match for analysis
_CONTEXT_BEFORE = 500
_CONTEXT_AFTER = 800

_HTTP_METHODS = {"GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"}

# Matches a JS string literal: "...", '...', or `...` (no nesting)
_STRING_LITERAL_RE = re.compile(
    r"""(?:"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'|`(?:[^`\\]|\\.)*`)"""
)

# Matches a template literal with ${...} expressions
_TEMPLATE_LITERAL_RE = re.compile(r'`([^`]*)`')

# Matches URL-like paths: /v1/..., /api/..., https://...
_URL_PATH_RE = re.compile(
    r"""(?:https?://[^\s"'`),;]+|/(?:v[0-9]+|api|content|config|football|broadcasting|search|graphql|auth|oauth|login|users?|data)[^\s"'`),;]*)"""
)

# Matches response consumption patterns like .json(), .text(), .blob()
_RESPONSE_CONSUMPTION_RE = re.compile(
    r'\.(?:json|text|blob|arrayBuffer|formData|bytes)\s*\(\s*\)'
)


def _get_context(source: str, pos: int, before: int = _CONTEXT_BEFORE, after: int = _CONTEXT_AFTER) -> str:
    """Extract a window of code around a position."""
    start = max(0, pos - before)
    end = min(len(source), pos + after)
    return source[start:end]


def _extract_string_value(token: str) -> str | None:
    """
    Extract the string value from a JS string literal token.
    Handles "...", '...', and `...` (template literals returned with ${} intact).
    Returns None if not a recognizable string literal.
    """
    token = token.strip()
    if len(token) < 2:
        return None
    if (token[0] == '"' and token[-1] == '"') or (token[0] == "'" and token[-1] == "'"):
        return token[1:-1]
    if token[0] == '`' and token[-1] == '`':
        return token[1:-1]
    return None


def _find_balanced_parens(source: str, open_pos: int) -> str:
    """
    Given the position of an opening '(', find the matching ')' and return
    the content between them. Respects string literals and nested parens.
    Returns content between parens, or empty string if unbalanced.
    """
    depth = 0
    in_string: str | None = None
    escape_next = False
    i = open_pos

    while i < len(source):
        ch = source[i]

        if escape_next:
            escape_next = False
            i += 1
            continue

        if ch == '\\':
            escape_next = True
            i += 1
            continue

        if in_string:
            if ch == in_string:
                in_string = None
            i += 1
            continue

        if ch in ('"', "'", '`'):
            in_string = ch
            i += 1
            continue

        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
            if depth == 0:
                return source[open_pos + 1:i]

        i += 1

    # Unbalanced — return what we can
    return source[open_pos + 1:min(open_pos + 1000, len(source))]


def _find_balanced_braces(source: str, open_pos: int) -> str:
    """
    Given the position of an opening '{', find the matching '}' and return
    the content between them. Respects string literals and nested braces.
    """
    depth = 0
    in_string: str | None = None
    escape_next = False
    i = open_pos

    while i < len(source):
        ch = source[i]

        if escape_next:
            escape_next = False
            i += 1
            continue

        if ch == '\\':
            escape_next = True
            i += 1
            continue

        if in_string:
            if ch == in_string:
                in_string = None
            i += 1
            continue

        if ch in ('"', "'", '`'):
            in_string = ch
            i += 1
            continue

        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return source[open_pos + 1:i]

        i += 1

    return source[open_pos + 1:min(open_pos + 2000, len(source))]


def _extract_object_keys(obj_src: str) -> dict[str, str]:
    """
    Extract top-level key:value pairs from a JS object literal string.
    Returns a dict of key -> raw value expression.
    Only captures the first level of nesting.
    """
    result: dict[str, str] = {}
    # Match patterns like: key: value, "key": value, 'key': value
    key_value_re = re.compile(
        r'''(?:["'](\w[\w-]*)["']|(\w[\w-]*))\s*:\s*'''
    )
    for m in key_value_re.finditer(obj_src):
        key = m.group(1) or m.group(2)
        val_start = m.end()
        # Find the end of the value (next comma at same depth, or end of string)
        depth = 0
        in_str: str | None = None
        esc = False
        val_end = val_start
        for j in range(val_start, len(obj_src)):
            ch = obj_src[j]
            if esc:
                esc = False
                continue
            if ch == '\\':
                esc = True
                continue
            if in_str:
                if ch == in_str:
                    in_str = None
                continue
            if ch in ('"', "'", '`'):
                in_str = ch
                continue
            if ch in ('(', '{', '['):
                depth += 1
            elif ch in (')', '}', ']'):
                if depth == 0:
                    val_end = j
                    break
                depth -= 1
            elif ch == ',' and depth == 0:
                val_end = j
                break
        else:
            val_end = len(obj_src)

        raw_val = obj_src[val_start:val_end].strip()
        if raw_val:
            result[key] = raw_val

    return result


def _extract_method_from_options(options_str: str) -> str | None:
    """Extract HTTP method from a fetch/ajax options object string."""
    m = re.search(r'method\s*:\s*["\'](\w+)["\']', options_str, re.IGNORECASE)
    if m:
        method = m.group(1).upper()
        if method in _HTTP_METHODS:
            return method
    return None


def _extract_headers_from_options(options_str: str) -> dict[str, str] | None:
    """Extract headers dict from an options object string."""
    m = re.search(r'headers\s*:\s*\{', options_str)
    if not m:
        return None
    brace_pos = options_str.index('{', m.start())
    headers_content = _find_balanced_braces(options_str, brace_pos)
    if not headers_content:
        return None
    return _extract_object_keys(headers_content) or None


def _extract_body_from_options(options_str: str) -> tuple[str | None, dict[str, Any] | None]:
    """Extract body raw string and inferred schema from options object."""
    # Look for body: ... or data: ...
    m = re.search(r'(?:body|data)\s*:\s*', options_str)
    if not m:
        return None, None

    rest = options_str[m.end():]

    # Check if body is JSON.stringify({...})
    stringify_m = re.match(r'JSON\.stringify\s*\(', rest)
    if stringify_m:
        paren_content = _find_balanced_parens(rest, stringify_m.end() - 1)
        # Try to extract object keys from stringified content
        brace_m = re.search(r'\{', paren_content)
        if brace_m:
            obj_content = _find_balanced_braces(paren_content, brace_m.start())
            keys = _extract_object_keys(obj_content)
            return f"JSON.stringify({paren_content[:200]})", {k: "<dynamic>" for k in keys} if keys else None
        return f"JSON.stringify({paren_content[:200]})", None

    # Check if body starts with {
    if rest.lstrip().startswith('{'):
        brace_pos = rest.index('{')
        obj_content = _find_balanced_braces(rest, brace_pos)
        keys = _extract_object_keys(obj_content)
        return rest[:brace_pos + len(obj_content) + 2][:200], {k: "<dynamic>" for k in keys} if keys else None

    # Raw body value - grab until next comma at depth 0
    depth = 0
    end = 0
    for j, ch in enumerate(rest):
        if ch in ('(', '{', '['):
            depth += 1
        elif ch in (')', '}', ']'):
            if depth == 0:
                end = j
                break
            depth -= 1
        elif ch == ',' and depth == 0:
            end = j
            break
    else:
        end = min(200, len(rest))
    return rest[:end].strip()[:200], None


def _extract_credentials_from_options(options_str: str) -> str | None:
    """Extract credentials mode from options."""
    m = re.search(r'credentials\s*:\s*["\'](\w[\w-]*)["\']', options_str)
    return m.group(1) if m else None


def _extract_response_handling(context_after: str) -> str | None:
    """Detect how the response is consumed from code following the fetch call."""
    m = _RESPONSE_CONSUMPTION_RE.search(context_after[:300])
    return m.group(0) if m else None


def _classify_url(url: str) -> list[str]:
    """Classify a URL into tags based on its pattern."""
    tags: list[str] = []
    url_lower = url.lower()

    if any(seg in url_lower for seg in ('/api/', '/v1/', '/v2/', '/v3/', '/graphql')):
        tags.append('api')
    if any(seg in url_lower for seg in ('/config/', '.json', '/metadata')):
        tags.append('config')
    if any(seg in url_lower for seg in ('analytics', 'tracking', 'pixel', 'beacon', 'telemetry')):
        tags.append('tracking')
    if any(seg in url_lower for seg in ('cdn', 'static', 'resources', 'assets', '/img/', '/images/')):
        tags.append('cdn')
    if any(seg in url_lower for seg in ('/auth', '/oauth', '/login', '/token', '/session')):
        tags.append('auth')
    if any(seg in url_lower for seg in ('/search', '/query', '/find')):
        tags.append('search')
    if any(seg in url_lower for seg in ('/content/', '/article', '/post')):
        tags.append('content')

    return tags or ['other']


def _templatize_url(url: str) -> str:
    """
    Convert a URL with ${...} template expressions into {param} placeholders.
    e.g. /v1/competitions/${e}/seasons/${t} -> /v1/competitions/{competitionId}/seasons/{seasonId}

    Also handles:
    - Trailing ${...} expressions appended to the last path segment
    - Complex expressions like ${Kd(n)} which are query param helpers
    - Mixed segments like "matches${Kd(t)}" -> "matches"
    """
    # First, handle function calls that append query params (e.g. ${Kd(n)})
    # These are typically query param builder calls — strip them
    url = re.sub(r'\$\{\w+\([^)]*\)\}', '', url)

    # Strip any trailing ${...} that looks like a suffix rather than a path segment
    # e.g. "/seasons/${t}" is a param, but "stats${Kd(r)}" was a func call (already handled above)

    # Split into path and query parts
    query_part = ''
    if '?' in url:
        url, query_part = url.split('?', 1)

    parts = url.split('/')
    result_parts: list[str] = []
    param_counter: dict[str, int] = {}

    for i, part in enumerate(parts):
        # Entire segment is a ${...} expression
        if re.fullmatch(r'\$\{[^}]+\}', part):
            prev_segment = parts[i - 1] if i > 0 else ''
            param_name = _infer_param_name(prev_segment, param_counter)
            result_parts.append('{' + param_name + '}')
        # Segment contains a ${...} mixed with static text
        elif '${' in part and '}' in part:
            # Replace each ${...} within the segment
            def _replace_inline(m: re.Match, idx: int = i) -> str:
                prev_segment = parts[idx - 1] if idx > 0 else ''
                pname = _infer_param_name(prev_segment, param_counter)
                return '{' + pname + '}'
            cleaned = re.sub(r'\$\{[^}]+\}', _replace_inline, part)
            result_parts.append(cleaned)
        elif part.startswith('$') and len(part) <= 3:
            # Single variable reference like just a minified var
            prev_segment = parts[i - 1] if i > 0 else ''
            param_name = _infer_param_name(prev_segment, param_counter)
            result_parts.append('{' + param_name + '}')
        else:
            result_parts.append(part)

    result = '/'.join(result_parts)

    # Re-attach query part if present (templatize query params too)
    if query_part:
        query_part = re.sub(r'\$\{[^}]+\}', '{param}', query_part)
        result = result + '?' + query_part

    # Clean up any empty trailing segments from stripped expressions
    result = re.sub(r'/+$', '', result)

    return result


def _infer_param_name(prev_segment: str, counter: dict[str, int]) -> str:
    """Infer a meaningful parameter name from the previous path segment."""
    # Common path-to-param mappings
    mappings: dict[str, str] = {
        'competitions': 'competitionId',
        'seasons': 'seasonId',
        'teams': 'teamId',
        'players': 'playerId',
        'matches': 'matchId',
        'users': 'userId',
        'articles': 'articleId',
        'matchweeks': 'matchweekId',
        'phases': 'phaseId',
        'meetings': 'meetingId',
        'awards': 'awardId',
        'events': 'eventId',
        'regions': 'regionId',
        'categories': 'categoryId',
        'clubs': 'clubId',
        'countries': 'countryId',
    }

    param_name = mappings.get(prev_segment, prev_segment.rstrip('s') + 'Id' if prev_segment else 'param')

    # Ensure uniqueness
    if param_name in counter:
        counter[param_name] += 1
        return f"{param_name}_{counter[param_name]}"
    else:
        counter[param_name] = 0
        return param_name


def _compute_confidence(call_site: EndpointCallSite) -> float:
    """Compute a confidence score for a discovered call site."""
    score = 0.5  # baseline

    # Boost for resolved URL
    if call_site.url_resolved:
        score += 0.2
    # Boost for URL template
    if call_site.url_template:
        score += 0.1
    # Boost for known method
    if call_site.method != "GET":
        score += 0.05  # explicit method is more informative
    # Boost for headers
    if call_site.headers:
        score += 0.05
    # Boost for body schema
    if call_site.body_schema:
        score += 0.1

    return min(score, 1.0)


# ---------------------------------------------------------------------------
# Fetch extractor
# ---------------------------------------------------------------------------

# Pattern: fetch(url) or fetch(url, options)
_FETCH_RE = re.compile(r'\bfetch\s*\(')


def extract_fetch_calls(
    source: str,
    source_url: str,
    source_request_id: str,
) -> list[EndpointCallSite]:
    """
    Extract all fetch() call sites from JavaScript source code.

    Handles patterns:
    - fetch("https://example.com/api")
    - fetch(`${base}/path/${id}`)
    - fetch(variable)
    - fetch(url, { method: "POST", headers: {...}, body: ... })
    """
    results: list[EndpointCallSite] = []

    for match in _FETCH_RE.finditer(source):
        pos = match.start()
        paren_pos = match.end() - 1  # position of '('

        # Extract full arguments
        args_content = _find_balanced_parens(source, paren_pos)
        if not args_content:
            continue

        # Get context for snippet
        context = _get_context(source, pos)

        # Split arguments (respecting nesting)
        url_arg, options_arg = _split_fetch_args(args_content)

        # Extract URL
        url_raw = url_arg.strip()
        url_resolved = _extract_string_value(url_raw)
        url_template = None
        if url_resolved:
            url_template = _templatize_url(url_resolved)
        elif '`' in url_raw:
            # Template literal — extract and templatize
            tpl_m = _TEMPLATE_LITERAL_RE.search(url_raw)
            if tpl_m:
                url_resolved = tpl_m.group(1)
                url_template = _templatize_url(url_resolved)

        # Extract options
        method = "GET"
        headers = None
        body_raw = None
        body_schema = None
        credentials = None

        if options_arg:
            method = _extract_method_from_options(options_arg) or "GET"
            headers = _extract_headers_from_options(options_arg)
            body_raw, body_schema = _extract_body_from_options(options_arg)
            credentials = _extract_credentials_from_options(options_arg)

        # Detect response handling
        after_context = source[match.end():min(match.end() + 500, len(source))]
        response_handling = _extract_response_handling(after_context)

        call_site = EndpointCallSite(
            call_type=CallType.FETCH,
            raw_snippet=context[:600],
            source_url=source_url,
            source_request_id=source_request_id,
            position=pos,
            url_raw=url_raw[:500],
            url_resolved=url_resolved,
            url_template=url_template,
            method=method,
            headers=headers,
            body_raw=body_raw,
            body_schema=body_schema,
            credentials=credentials,
            response_handling=response_handling,
        )
        results.append(call_site)

    return results


def _split_fetch_args(args_content: str) -> tuple[str, str | None]:
    """
    Split fetch(url, options) arguments into url and options parts.
    Handles nested expressions in URL argument.
    """
    # Walk through looking for comma at depth 0
    depth = 0
    in_str: str | None = None
    esc = False

    for i, ch in enumerate(args_content):
        if esc:
            esc = False
            continue
        if ch == '\\':
            esc = True
            continue
        if in_str:
            if ch == in_str:
                in_str = None
            continue
        if ch in ('"', "'", '`'):
            in_str = ch
            continue
        if ch in ('(', '{', '['):
            depth += 1
        elif ch in (')', '}', ']'):
            depth -= 1
        elif ch == ',' and depth == 0:
            return args_content[:i].strip(), args_content[i + 1:].strip()

    return args_content.strip(), None


# ---------------------------------------------------------------------------
# XMLHttpRequest extractor
# ---------------------------------------------------------------------------

_XHR_OPEN_RE = re.compile(r'\.open\s*\(\s*["\'](\w+)["\']\s*,\s*')


def extract_xhr_calls(
    source: str,
    source_url: str,
    source_request_id: str,
) -> list[EndpointCallSite]:
    """
    Extract XMLHttpRequest .open() call sites.

    Handles patterns:
    - xhr.open("GET", "https://api.example.com/data")
    - req.open("POST", url)
    """
    results: list[EndpointCallSite] = []

    # Only process if XMLHttpRequest is actually used
    if 'XMLHttpRequest' not in source:
        return results

    for match in _XHR_OPEN_RE.finditer(source):
        pos = match.start()
        method = match.group(1).upper()
        if method not in _HTTP_METHODS:
            continue

        # Get the URL argument after the method
        after_method = source[match.end():]
        url_raw = ""
        url_resolved = None

        # Try to extract string literal URL
        str_match = _STRING_LITERAL_RE.match(after_method)
        if str_match:
            url_raw = str_match.group()
            url_resolved = _extract_string_value(url_raw)
        else:
            # Variable reference — grab identifier
            var_match = re.match(r'(\w+(?:\.\w+)*)', after_method)
            if var_match:
                url_raw = var_match.group(1)

        context = _get_context(source, pos)

        # Look for setRequestHeader calls nearby
        headers = _extract_xhr_headers(context)

        # Look for .send(body) nearby
        body_raw, body_schema = _extract_xhr_body(context)

        url_template = _templatize_url(url_resolved) if url_resolved else None

        call_site = EndpointCallSite(
            call_type=CallType.XHR,
            raw_snippet=context[:600],
            source_url=source_url,
            source_request_id=source_request_id,
            position=pos,
            url_raw=url_raw[:500],
            url_resolved=url_resolved,
            url_template=url_template,
            method=method,
            headers=headers,
            body_raw=body_raw,
            body_schema=body_schema,
        )
        results.append(call_site)

    return results


def _extract_xhr_headers(context: str) -> dict[str, str] | None:
    """Extract headers from setRequestHeader calls in context."""
    headers: dict[str, str] = {}
    for m in re.finditer(r'\.setRequestHeader\s*\(\s*["\']([\w-]+)["\']\s*,\s*([^)]+)\)', context):
        key = m.group(1)
        val = m.group(2).strip()
        str_val = _extract_string_value(val)
        headers[key] = str_val if str_val else val[:100]
    return headers or None


def _extract_xhr_body(context: str) -> tuple[str | None, dict[str, Any] | None]:
    """Extract body from .send() calls in context."""
    m = re.search(r'\.send\s*\(', context)
    if not m:
        return None, None
    paren_content = _find_balanced_parens(context, m.end() - 1)
    if not paren_content or paren_content.strip() in ('', 'null', 'undefined'):
        return None, None

    body_raw = paren_content.strip()[:200]

    # Try to extract JSON.stringify schema
    stringify_m = re.match(r'JSON\.stringify\s*\(', body_raw)
    if stringify_m:
        inner = _find_balanced_parens(body_raw, stringify_m.end() - 1)
        brace_m = re.search(r'\{', inner)
        if brace_m:
            obj_content = _find_balanced_braces(inner, brace_m.start())
            keys = _extract_object_keys(obj_content)
            if keys:
                return body_raw, {k: "<dynamic>" for k in keys}

    return body_raw, None


# ---------------------------------------------------------------------------
# Axios extractor
# ---------------------------------------------------------------------------

_AXIOS_RE = re.compile(
    r'\b(?:axios|http|client|api|instance|request)\s*\.\s*(get|post|put|delete|patch|head|options|request)\s*\(',
    re.IGNORECASE
)


def extract_axios_calls(
    source: str,
    source_url: str,
    source_request_id: str,
) -> list[EndpointCallSite]:
    """
    Extract axios-style HTTP client call sites.

    Handles patterns:
    - axios.get("/api/users")
    - axios.post("/api/users", data, config)
    - http.get(url)
    - client.request({ method: "POST", url: "/api" })
    """
    results: list[EndpointCallSite] = []

    # Quick check for axios-like patterns
    if not re.search(r'axios|\.get\s*\(|\.post\s*\(|\.put\s*\(|\.delete\s*\(|\.patch\s*\(', source):
        return results

    for match in _AXIOS_RE.finditer(source):
        pos = match.start()
        method_name = match.group(1).upper()

        # Find the arguments
        paren_start = source.index('(', match.end() - 1)
        args_content = _find_balanced_parens(source, paren_start)
        if not args_content:
            continue

        context = _get_context(source, pos)

        if method_name == 'REQUEST':
            # axios.request({ url: ..., method: ..., ... })
            brace_m = re.search(r'\{', args_content)
            if not brace_m:
                continue
            obj_content = _find_balanced_braces(args_content, brace_m.start())
            keys = _extract_object_keys(obj_content)
            url_raw = keys.get('url', '')
            method = _extract_method_from_options(obj_content) or "GET"
            headers = _extract_headers_from_options(obj_content)
            body_raw, body_schema = _extract_body_from_options(obj_content)
        else:
            method = method_name if method_name in _HTTP_METHODS else "GET"
            # First arg is URL
            url_arg, rest = _split_fetch_args(args_content)
            url_raw = url_arg.strip()
            headers = None
            body_raw = None
            body_schema = None

            if rest and method in ("POST", "PUT", "PATCH"):
                # Second arg is data, third is config
                data_arg, config_arg = _split_fetch_args(rest)
                body_raw = data_arg[:200] if data_arg else None
                if config_arg:
                    headers = _extract_headers_from_options(config_arg)

        url_resolved = _extract_string_value(url_raw)
        url_template = None
        if url_resolved:
            url_template = _templatize_url(url_resolved)
        elif '`' in url_raw:
            tpl_m = _TEMPLATE_LITERAL_RE.search(url_raw)
            if tpl_m:
                url_resolved = tpl_m.group(1)
                url_template = _templatize_url(url_resolved)

        after_context = source[match.end():min(match.end() + 500, len(source))]
        response_handling = _extract_response_handling(after_context)

        call_site = EndpointCallSite(
            call_type=CallType.AXIOS,
            raw_snippet=context[:600],
            source_url=source_url,
            source_request_id=source_request_id,
            position=pos,
            url_raw=url_raw[:500],
            url_resolved=url_resolved,
            url_template=url_template,
            method=method,
            headers=headers,
            body_raw=body_raw,
            body_schema=body_schema,
            response_handling=response_handling,
        )
        results.append(call_site)

    return results


# ---------------------------------------------------------------------------
# jQuery AJAX extractor
# ---------------------------------------------------------------------------

_JQUERY_AJAX_RE = re.compile(r'\$\s*\.\s*(ajax|get|post|getJSON|getScript)\s*\(')


def extract_jquery_calls(
    source: str,
    source_url: str,
    source_request_id: str,
) -> list[EndpointCallSite]:
    """
    Extract jQuery AJAX call sites.

    Handles patterns:
    - $.ajax({ url: ..., method: ..., ... })
    - $.get("/api/data")
    - $.post("/api/data", payload)
    - $.getJSON("/api/data.json")
    """
    results: list[EndpointCallSite] = []

    # Quick check
    if '$.' not in source:
        return results

    for match in _JQUERY_AJAX_RE.finditer(source):
        pos = match.start()
        jquery_method = match.group(1).lower()

        paren_start = source.index('(', match.end() - 1)
        args_content = _find_balanced_parens(source, paren_start)
        if not args_content:
            continue

        context = _get_context(source, pos)

        if jquery_method == 'ajax':
            # $.ajax({ url: ..., type/method: ..., data: ... })
            brace_m = re.search(r'\{', args_content)
            if not brace_m:
                continue
            obj_content = _find_balanced_braces(args_content, brace_m.start())
            keys = _extract_object_keys(obj_content)
            url_raw = keys.get('url', '')
            # jQuery uses both 'type' and 'method'
            method = _extract_method_from_options(obj_content)
            if not method:
                type_m = re.search(r'type\s*:\s*["\'](\w+)["\']', obj_content, re.IGNORECASE)
                method = type_m.group(1).upper() if type_m else "GET"
            headers = _extract_headers_from_options(obj_content)
            body_raw, body_schema = _extract_body_from_options(obj_content)
        else:
            # $.get(url), $.post(url, data), $.getJSON(url)
            method_map = {'get': 'GET', 'post': 'POST', 'getjson': 'GET', 'getscript': 'GET'}
            method = method_map.get(jquery_method, 'GET')
            url_arg, rest = _split_fetch_args(args_content)
            url_raw = url_arg.strip()
            headers = None
            body_raw = None
            body_schema = None

            if rest and jquery_method == 'post':
                body_raw = rest[:200]

        url_resolved = _extract_string_value(url_raw)
        url_template = None
        if url_resolved:
            url_template = _templatize_url(url_resolved)

        call_site = EndpointCallSite(
            call_type=CallType.JQUERY_AJAX,
            raw_snippet=context[:600],
            source_url=source_url,
            source_request_id=source_request_id,
            position=pos,
            url_raw=url_raw[:500],
            url_resolved=url_resolved,
            url_template=url_template,
            method=method,
            headers=headers,
            body_raw=body_raw,
            body_schema=body_schema,
        )
        results.append(call_site)

    return results


# ---------------------------------------------------------------------------
# URL string extractor (catches endpoints not associated with explicit calls)
# ---------------------------------------------------------------------------

_API_URL_RE = re.compile(
    r"""(?:["'`])"""
    r"""("""
    r"""https?://[^\s"'`]{10,300}"""  # full URL
    r"""|"""
    r"""/(?:v[0-9]+|api|graphql)/[^\s"'`]{5,300}"""  # API path
    r""")"""
    r"""(?:["'`])""",
)

_TEMPLATE_API_URL_RE = re.compile(
    r"""`("""
    r"""\$\{[^}]+\}(?:/[^\s`]{5,300})"""  # template with path
    r"""|"""
    r"""(?:https?://[^\s`]*\$\{[^}]+\}[^\s`]*)"""  # full URL template
    r""")`"""
)


def extract_url_strings(
    source: str,
    source_url: str,
    source_request_id: str,
) -> list[EndpointCallSite]:
    """
    Extract URL strings that look like API endpoints but may not be
    associated with an explicit fetch/XHR/axios call.

    These are lower-confidence discoveries, but valuable for finding
    endpoint patterns in config objects, route definitions, etc.
    """
    results: list[EndpointCallSite] = []
    seen_urls: set[str] = set()

    # Static string URLs
    for match in _API_URL_RE.finditer(source):
        url = match.group(1)
        if url in seen_urls:
            continue
        seen_urls.add(url)

        # Skip non-API URLs
        if _is_asset_url(url) or _is_noise_url(url):
            continue

        pos = match.start()
        context = _get_context(source, pos, before=200, after=300)

        call_site = EndpointCallSite(
            call_type=CallType.URL_STRING,
            raw_snippet=context[:400],
            source_url=source_url,
            source_request_id=source_request_id,
            position=pos,
            url_raw=url[:500],
            url_resolved=url,
            url_template=_templatize_url(url),
            method=_guess_method_from_context(context),
        )
        results.append(call_site)

    # Template literal URLs
    for match in _TEMPLATE_API_URL_RE.finditer(source):
        url = match.group(1)
        if url in seen_urls:
            continue
        seen_urls.add(url)

        if _is_asset_url(url) or _is_noise_url(url):
            continue

        pos = match.start()
        context = _get_context(source, pos, before=200, after=300)

        call_site = EndpointCallSite(
            call_type=CallType.URL_STRING,
            raw_snippet=context[:400],
            source_url=source_url,
            source_request_id=source_request_id,
            position=pos,
            url_raw=url[:500],
            url_resolved=url,
            url_template=_templatize_url(url),
            method=_guess_method_from_context(context),
        )
        results.append(call_site)

    return results


def _is_asset_url(url: str) -> bool:
    """Check if a URL is likely a static asset, not an API endpoint."""
    asset_extensions = (
        '.js', '.css', '.png', '.jpg', '.jpeg', '.gif', '.svg',
        '.woff', '.woff2', '.ttf', '.otf', '.eot', '.ico',
        '.mp3', '.mp4', '.webm', '.ogg', '.wav',
        '.map', '.min.js', '.min.css',
    )
    url_lower = url.lower().split('?')[0]  # strip query params
    if url_lower.endswith(asset_extensions):
        return True
    return False


def _is_noise_url(url: str) -> bool:
    """Check if a URL is likely noise (namespaces, social links, specs, etc.)."""
    url_lower = url.lower()

    # XML/HTML namespace URIs
    if 'w3.org/' in url_lower:
        return True

    # Social media bare domains (no path beyond /)
    noise_domains = (
        'facebook.com', 'instagram.com', 'twitter.com', 'x.com',
        'youtube.com', 'tiktok.com', 'snapchat.com', 'linkedin.com',
        'whatsapp.com', 'pinterest.com', 'reddit.com',
        'github.com', 'example.com', 'localhost',
    )
    from urllib.parse import urlparse
    try:
        parsed = urlparse(url)
        host = parsed.netloc.lower()
        path = parsed.path.rstrip('/')
        # Domain-only or single-segment social links
        for domain in noise_domains:
            if host.endswith(domain) and (not path or path.count('/') <= 1):
                return True
    except ValueError:
        pass

    # Spec/documentation URLs
    if any(seg in url_lower for seg in ('aomedia.org', 'developers.google.com/ad-manager')):
        return True

    # Too short to be a real API endpoint (after removing scheme)
    path_part = url.split('://')[-1] if '://' in url else url
    if len(path_part) < 5:
        return True

    return False


def _guess_method_from_context(context: str) -> str:
    """Guess HTTP method from surrounding code context."""
    # Look for nearby method hints
    method_hints = re.findall(r'(?:method|type)\s*:\s*["\'](\w+)["\']', context, re.IGNORECASE)
    for hint in method_hints:
        if hint.upper() in _HTTP_METHODS:
            return hint.upper()
    return "GET"


# ---------------------------------------------------------------------------
# Master extraction function
# ---------------------------------------------------------------------------

def extract_all_call_sites(
    source: str,
    source_url: str,
    source_request_id: str,
) -> list[EndpointCallSite]:
    """
    Run all extractors on a JS source file and return all discovered call sites.

    This deduplicates overlapping matches (e.g. a fetch() that also matches
    the URL string extractor).
    """
    all_sites: list[EndpointCallSite] = []

    # Run extractors in priority order
    fetch_sites = extract_fetch_calls(source, source_url, source_request_id)
    xhr_sites = extract_xhr_calls(source, source_url, source_request_id)
    axios_sites = extract_axios_calls(source, source_url, source_request_id)
    jquery_sites = extract_jquery_calls(source, source_url, source_request_id)
    url_sites = extract_url_strings(source, source_url, source_request_id)

    all_sites.extend(fetch_sites)
    all_sites.extend(xhr_sites)
    all_sites.extend(axios_sites)
    all_sites.extend(jquery_sites)

    # For URL string sites, skip any that overlap with explicit call sites
    explicit_positions: set[int] = set()
    for site in fetch_sites + xhr_sites + axios_sites + jquery_sites:
        # Mark a range around each explicit call site
        for offset in range(-200, 200):
            explicit_positions.add(site.position + offset)

    for site in url_sites:
        if site.position not in explicit_positions:
            all_sites.append(site)

    return all_sites
