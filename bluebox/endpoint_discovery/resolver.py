"""
bluebox/endpoint_discovery/resolver.py

Variable resolver for JavaScript endpoint URL construction.

Performs deterministic backward analysis from call sites to resolve:
- Variable assignments (const/let/var x = "value")
- Object property assignments (config.baseUrl = "https://...")
- String concatenation (base + "/path/" + id)
- Template literal construction (`${base}/path/${id}`)
- Constant propagation through simple assignments
"""

import re
from typing import Any

from bluebox.endpoint_discovery.models import VariableBinding
from bluebox.utils.logger import get_logger

logger = get_logger(name=__name__)

# ---------------------------------------------------------------------------
# Patterns for variable resolution
# ---------------------------------------------------------------------------

# Matches: const/let/var identifier = "string"
_VAR_ASSIGN_STR_RE = re.compile(
    r'''(?:const|let|var)\s+(\w+)\s*=\s*(["'`](?:[^"'`\\]|\\.)*["'`])'''
)

# Matches: identifier = "string" (assignment without declaration)
_ASSIGN_STR_RE = re.compile(
    r'''(\w+)\s*=\s*(["'`](?:[^"'`\\]|\\.)*["'`])'''
)

# Matches: identifier.property = "string"
_PROP_ASSIGN_STR_RE = re.compile(
    r'''(\w+\.\w+(?:\.\w+)*)\s*=\s*(["'`](?:[^"'`\\]|\\.)*["'`])'''
)

# Matches: { key: "string" } or { "key": "string" }
_OBJ_KEY_STR_RE = re.compile(
    r'''(?:["']?(\w[\w.-]*)["']?)\s*:\s*(["'`](?:[^"'`\\]|\\.)*["'`])'''
)

# Matches URL-like config: BASE_URL, apiUrl, baseUrl, api, etc.
_URL_CONFIG_KEY_RE = re.compile(
    r'''(?:BASE_URL|baseUrl|baseURL|apiUrl|apiURL|API_URL|apiBase|API_BASE|api|endpoint|host|origin|root|server|BASE|DOMAIN)''',
    re.IGNORECASE
)


def _extract_string_value(token: str) -> str | None:
    """Extract the actual string from a JS string literal token."""
    token = token.strip()
    if len(token) < 2:
        return None
    if (token[0] == '"' and token[-1] == '"') or (token[0] == "'" and token[-1] == "'"):
        return token[1:-1]
    if token[0] == '`' and token[-1] == '`':
        return token[1:-1]
    return None


class VariableResolver:
    """
    Resolves variable values by scanning backward from a call site.

    Operates on a single JS source file. Builds a scope of known variable
    bindings and can resolve references found in URL expressions.
    """

    def __init__(self, source: str) -> None:
        self._source = source
        self._bindings: dict[str, VariableBinding] = {}
        self._config_objects: dict[str, dict[str, str]] = {}
        self._scanned = False

    def scan(self) -> None:
        """
        Scan the entire source to build a map of variable bindings.
        This is an O(n) pass through the source that captures all
        string-valued assignments.
        """
        if self._scanned:
            return

        # Pass 1: const/let/var declarations with string values
        for m in _VAR_ASSIGN_STR_RE.finditer(self._source):
            name = m.group(1)
            value = _extract_string_value(m.group(2))
            if value is not None:
                self._bindings[name] = VariableBinding(
                    name=name,
                    value=value,
                    source_position=m.start(),
                )

        # Pass 2: property assignments with string values
        for m in _PROP_ASSIGN_STR_RE.finditer(self._source):
            name = m.group(1)
            value = _extract_string_value(m.group(2))
            if value is not None:
                self._bindings[name] = VariableBinding(
                    name=name,
                    value=value,
                    source_position=m.start(),
                )

        # Pass 3: object literals with URL-config-like keys
        self._scan_config_objects()

        self._scanned = True
        logger.debug("VariableResolver scanned %d bindings", len(self._bindings))

    def _scan_config_objects(self) -> None:
        """Scan for config/settings objects that contain URL-like values."""
        # Look for objects assigned to URL-config-like names
        obj_assign_re = re.compile(
            r'(?:const|let|var)\s+(\w+)\s*=\s*\{',
        )
        for m in obj_assign_re.finditer(self._source):
            var_name = m.group(1)
            brace_pos = self._source.index('{', m.end() - 1)

            # Quick check: is the object content within reach?
            end_search = min(brace_pos + 5000, len(self._source))
            obj_snippet = self._source[brace_pos:end_search]

            # Simple extraction of key-value pairs
            config: dict[str, str] = {}
            for kv in _OBJ_KEY_STR_RE.finditer(obj_snippet[:3000]):
                key = kv.group(1)
                val = _extract_string_value(kv.group(2))
                if val:
                    config[key] = val
                    # Also store as var_name.key for property access resolution
                    full_name = f"{var_name}.{key}"
                    self._bindings[full_name] = VariableBinding(
                        name=full_name,
                        value=val,
                        source_position=kv.start() + brace_pos,
                    )

            if config:
                self._config_objects[var_name] = config

    def resolve_variable(self, name: str) -> VariableBinding | None:
        """
        Resolve a variable name to its binding, if known.
        Handles both simple names ("x") and property paths ("config.baseUrl").
        """
        self.scan()
        return self._bindings.get(name)

    def resolve_expression(self, expr: str, search_start: int = 0) -> str | None:
        """
        Attempt to resolve a JS expression to a concrete string value.

        Handles:
        - String literals: "value", 'value'
        - Template literals: `${var}/path`
        - Concatenation: base + "/path"
        - Simple variable references: varName
        - Property access: obj.prop

        Args:
            expr: The JS expression to resolve.
            search_start: Position in source to search backward from for local bindings.

        Returns:
            Resolved string value, or None if cannot be fully resolved.
        """
        self.scan()
        expr = expr.strip()

        # Direct string literal
        if (expr.startswith('"') and expr.endswith('"')) or \
           (expr.startswith("'") and expr.endswith("'")):
            return _extract_string_value(expr)

        # Template literal
        if expr.startswith('`') and expr.endswith('`'):
            return self._resolve_template_literal(expr[1:-1])

        # String concatenation
        if '+' in expr:
            return self._resolve_concatenation(expr)

        # Variable reference
        binding = self.resolve_variable(expr)
        if binding and binding.value is not None:
            return binding.value

        return None

    def _resolve_template_literal(self, template: str) -> str | None:
        """
        Resolve a template literal by substituting ${...} expressions.
        Returns the resolved string, or the template with unresolved ${} intact.
        """
        def replace_expr(m: re.Match) -> str:
            inner_expr = m.group(1)
            resolved = self._resolve_simple_expr(inner_expr)
            if resolved is not None:
                return resolved
            return m.group(0)  # keep original ${...}

        result = re.sub(r'\$\{([^}]+)\}', replace_expr, template)
        return result

    def _resolve_concatenation(self, expr: str) -> str | None:
        """
        Resolve a string concatenation expression like: base + "/path/" + id
        """
        # Split on + at depth 0
        parts: list[str] = []
        current: list[str] = []
        depth = 0
        in_str: str | None = None
        esc = False

        for ch in expr:
            if esc:
                esc = False
                current.append(ch)
                continue
            if ch == '\\':
                esc = True
                current.append(ch)
                continue
            if in_str:
                current.append(ch)
                if ch == in_str:
                    in_str = None
                continue
            if ch in ('"', "'", '`'):
                in_str = ch
                current.append(ch)
                continue
            if ch in ('(', '{', '['):
                depth += 1
                current.append(ch)
            elif ch in (')', '}', ']'):
                depth -= 1
                current.append(ch)
            elif ch == '+' and depth == 0:
                parts.append(''.join(current).strip())
                current = []
            else:
                current.append(ch)

        if current:
            parts.append(''.join(current).strip())

        resolved_parts: list[str] = []
        all_resolved = True
        for part in parts:
            part = part.strip()
            if not part:
                continue
            resolved = self.resolve_expression(part)
            if resolved is not None:
                resolved_parts.append(resolved)
            else:
                resolved_parts.append(f"${{{part}}}")
                all_resolved = False

        if not resolved_parts:
            return None

        return ''.join(resolved_parts)

    def _resolve_simple_expr(self, expr: str) -> str | None:
        """Resolve a simple expression (variable name, property access, or literal)."""
        expr = expr.strip()

        # String literal
        val = _extract_string_value(expr)
        if val is not None:
            return val

        # Variable/property reference
        binding = self.resolve_variable(expr)
        if binding and binding.value is not None:
            return binding.value

        # Property access chain: try partial resolution
        if '.' in expr:
            parts = expr.split('.')
            # Try progressively shorter prefixes
            for i in range(len(parts), 0, -1):
                prefix = '.'.join(parts[:i])
                binding = self.resolve_variable(prefix)
                if binding and binding.value is not None:
                    return binding.value

        return None

    def resolve_url_at_callsite(self, url_raw: str, position: int) -> tuple[str | None, list[VariableBinding]]:
        """
        Resolve a URL expression found at a call site.

        Also looks in the local scope (backward from position) for
        variable assignments that might be relevant.

        Args:
            url_raw: The raw URL expression from the call site.
            position: Character position of the call site.

        Returns:
            Tuple of (resolved_url, list_of_resolved_variables).
        """
        self.scan()

        # First try the local scope: look backward for nearby assignments
        local_bindings = self._scan_local_scope(position)

        # Merge local bindings (they take precedence)
        merged = dict(self._bindings)
        for b in local_bindings:
            merged[b.name] = b

        # Attempt resolution with merged scope
        old_bindings = self._bindings
        self._bindings = merged
        try:
            resolved = self.resolve_expression(url_raw, position)
        finally:
            self._bindings = old_bindings

        used_vars = [b for b in local_bindings if b.name in url_raw]
        return resolved, used_vars

    def _scan_local_scope(self, position: int, window: int = 2000) -> list[VariableBinding]:
        """
        Scan backward from a position to find local variable assignments.
        """
        start = max(0, position - window)
        local_src = self._source[start:position]
        bindings: list[VariableBinding] = []

        for m in _VAR_ASSIGN_STR_RE.finditer(local_src):
            name = m.group(1)
            value = _extract_string_value(m.group(2))
            if value is not None:
                bindings.append(VariableBinding(
                    name=name,
                    value=value,
                    source_position=start + m.start(),
                ))

        for m in _ASSIGN_STR_RE.finditer(local_src):
            name = m.group(1)
            value = _extract_string_value(m.group(2))
            if value is not None:
                bindings.append(VariableBinding(
                    name=name,
                    value=value,
                    source_position=start + m.start(),
                ))

        return bindings

    def get_all_url_config_values(self) -> dict[str, str]:
        """
        Return all bindings that look like URL/API configuration values.
        Useful for discovering base URLs and API origins.
        """
        self.scan()
        result: dict[str, str] = {}
        for name, binding in self._bindings.items():
            if binding.value is None:
                continue
            if _URL_CONFIG_KEY_RE.search(name):
                result[name] = binding.value
            elif binding.value.startswith(('http://', 'https://', '//')):
                result[name] = binding.value
        return result

    def get_config_objects(self) -> dict[str, dict[str, str]]:
        """Return all config-like objects discovered during scan."""
        self.scan()
        return dict(self._config_objects)
