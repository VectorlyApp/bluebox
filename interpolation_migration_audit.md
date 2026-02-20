# Interpolation Migration Audit

Migration from old format (`\"{{param}}\"` with escaped quotes) to new format (`"{{param}}"` uniform strings, `Parameter.type` drives coercion).

## Stale References Found

### 1. `routine_discovery_agent_beta.py:181` — "quoting errors" in PROMPT_CONSTRUCTING

```python
3. Use `validate_placeholders` on your fetch bodies/headers to catch quoting errors early
```

**Problem:** "quoting errors" is a leftover from the old escaped-quote format. The `_validate_placeholders` function doesn't check quoting — it validates placeholder prefixes and parameter name definitions.

**Fix:** Change to `"catch placeholder syntax errors early"` or similar.

---

### 2. `routine_discovery_agent_beta.py:1782` — "Checks quoting" in `_validate_placeholders` tool description

```python
"Checks quoting, prefixes, and parameter type compatibility. "
```

**Problem:** "Checks quoting" is misleading — the function checks prefixes (`cookie:`, `sessionStorage:`, etc.) and parameter name validity, not quoting.

**Fix:** Change to `"Checks prefixes and parameter type compatibility."`.

---

## Parameter.type Coercion Documentation — Coverage Check

The new coercion model (standalone `{{param}}` → typed value based on `Parameter.type`, substring → always string) is documented in these places:

| Location | Status | Notes |
|---|---|---|
| `agent_docs/core/placeholders.md:17-32` | Good | Full table, standalone vs substring, CDP-matching warning |
| `agent_docs/core/parameters.md:30-45` | Good | Same coverage, type table |
| `agent_docs/core/routines.md:53,95-103` | Good | Standalone vs substring distinction |
| `routine_discovery_agent.py:85-101` (PLACEHOLDER_INSTRUCTIONS) | Good | Examples showing coercion, CDP-matching warning |
| `routine_discovery_agent_beta.py:205-221` (PLACEHOLDER_INSTRUCTIONS) | Good | Same as above |
| `routine_discovery.md:437` | Good | One-liner summary |
| `llms/tools/routine_discovery_tools.py:289-294` (construct_routine desc) | Good | "Parameter.type drives coercion at runtime" |

**No gaps found** — all agent-facing docs correctly describe the new uniform format and `Parameter.type`-driven coercion.

## Clean Areas (No Action Needed)

- `agent_docs/` — All placeholder/parameter docs use new format exclusively
- `routine_discovery_agent.py` — PLACEHOLDER_INSTRUCTIONS are correct
- `routine_discovery_agent_beta.py` — PLACEHOLDER_INSTRUCTIONS are correct (only the tool description and phase prompt have stale "quoting" language)
- `llms/tools/routine_discovery_tools.py` — construct_routine description is correct
- Example routines in `example_data/` — All use new format
- `CLAUDE.md` — References new uniform format
