# BlueBoxAgent: Filter Routine Execution Data

## Problem

When `BlueBoxAgent` executes routines via the Vectorly API (`/routines/{id}/execute`), it saves the **full** `RoutineExecutionResult` JSON to `raw/`. This data is then loaded as `routine_results` in the Python sandbox (`run_python_code`).

The full result includes internal fields the agent should never see:

- **`placeholder_resolution`** — raw dict of resolved placeholder values (tokens, session IDs, auth headers). Sensitive and not useful for the agent's task.
- **`operations_metadata`** — per-operation timing, detail dicts, and full error traces. Internal execution mechanics.

The agent only needs the business data and enough diagnostic info to detect and report failures.

## Current Flow

```
API response (full RoutineExecutionResult)
    │
    ▼
save_result()  ← saves entire response.json() to raw/
    │
    ▼
raw/routine_result_*.json  ← contains placeholder_resolution, operations_metadata
    │
    ▼
workspace.load_raw_json()  → routine_results in Python sandbox
```

**Location:** `bluebox/agents/bluebox_agent.py`, lines ~266-292 (`save_result()` closure inside `_execute_routines_in_parallel`).

## Proposed Solution

Transform the API response **before saving** to strip internal fields and replace them with derived diagnostic summaries.

### What the agent SHOULD see

| Field | Type | Purpose |
|-------|------|---------|
| `ok` | `bool` | Did the routine succeed? |
| `data` | `dict \| list \| str \| None` | The actual result payload |
| `error` | `str \| None` | Error message if failed |
| `warnings` | `list[str]` | Useful context about the execution |
| `is_base64` | `bool` | Whether data is base64-encoded binary |
| `content_type` | `str \| None` | MIME type of the data |
| `filename` | `str \| None` | Suggested filename for downloads |
| `has_failed_placeholders` | `bool` | Were there unresolved variables? |
| `failed_placeholder_count` | `int` | How many placeholders failed |
| `has_operation_errors` | `bool` | Did any operations error? |
| `operation_error_summary` | `list[str]` | e.g. `["fetch: 403 Forbidden", "js_evaluate: timeout"]` |

### What the agent should NOT see

| Field | Reason |
|-------|--------|
| `placeholder_resolution` | Contains raw token/session values; sensitive; not useful for the agent |
| `operations_metadata` | Internal per-operation timing and detail dicts; too granular |

### Detecting "no data" and other issues

With the derived fields, the agent can reason about failures:

- **No data:** `ok=True` but `data` is `None` or empty → routine succeeded but returned nothing
- **Placeholder failures:** `has_failed_placeholders=True` → some variables couldn't be resolved at runtime
- **Operation errors:** `has_operation_errors=True` + `operation_error_summary` → specific operations failed with reasons
- **Partial success:** `ok=True` + `warnings` non-empty → succeeded with caveats

### Prior art: `RoutineDiscoveryAgent`

The discovery agent (`bluebox/agents/routine_discovery_agent.py:659-724`) already does this pattern. On failed validation, it derives:

```python
failed_placeholders = [
    key for key, value in exec_result.placeholder_resolution.items()
    if value is None
]
operation_errors = [
    (op.type, op.error)
    for op in exec_result.operations_metadata
    if op.error is not None
]
```

We apply the same logic but at the save boundary rather than in-line.

## Implementation

### Step 1: Add a sanitization function

Add a helper (either in `bluebox_agent.py` or in `execution.py`) that transforms the raw API response dict:

```python
def _sanitize_execution_result(result_data: dict) -> dict:
    """Strip internal fields from execution result, add diagnostic summaries."""
    sanitized = {
        "ok": result_data.get("ok"),
        "data": result_data.get("data"),
        "error": result_data.get("error"),
        "warnings": result_data.get("warnings", []),
        "is_base64": result_data.get("is_base64", False),
        "content_type": result_data.get("content_type"),
        "filename": result_data.get("filename"),
    }

    # Derive placeholder diagnostics
    placeholder_resolution = result_data.get("placeholder_resolution", {})
    failed = [k for k, v in placeholder_resolution.items() if v is None]
    sanitized["has_failed_placeholders"] = len(failed) > 0
    sanitized["failed_placeholder_count"] = len(failed)

    # Derive operation error diagnostics
    ops_metadata = result_data.get("operations_metadata", [])
    op_errors = [
        f"{op.get('type', 'unknown')}: {op['error']}"
        for op in ops_metadata
        if op.get("error") is not None
    ]
    sanitized["has_operation_errors"] = len(op_errors) > 0
    sanitized["operation_error_summary"] = op_errors

    return sanitized
```

### Step 2: Call it in `save_result()`

In `_execute_routines_in_parallel`, modify `execute_one()`:

```python
def execute_one(req: RoutineExecutionRequest) -> dict[str, Any]:
    url = f"{Config.VECTORLY_API_BASE}/routines/{req.routine_id}/execute"
    try:
        response = requests.post(url, headers=headers, json={"parameters": req.parameters}, timeout=300)
        response.raise_for_status()
        api_result = response.json()
        sanitized = _sanitize_execution_result(api_result)
        return save_result({"success": True, "routine_id": req.routine_id, "data": sanitized})
    except requests.RequestException as e:
        return save_result({"success": False, "routine_id": req.routine_id, "error": str(e)})
```

### Step 3: Update system prompt (optional)

Update the agent's system prompt to mention the new diagnostic fields so the LLM knows to check them:

> The `routine_results` list contains execution results with diagnostic fields: `has_failed_placeholders`, `failed_placeholder_count`, `has_operation_errors`, `operation_error_summary`. Use these to detect and report issues.

## Files to Change

1. **`bluebox/agents/bluebox_agent.py`** — Add `_sanitize_execution_result()`, call it in `execute_one()`, optionally update system prompt
2. **`tests/unit/test_bluebox_agent.py`** (or new test file) — Test the sanitization logic

## Edge Cases

- API returns unexpected shape (no `placeholder_resolution` key) → handled by `.get()` with defaults
- `operations_metadata` entries missing `type` or `error` → handled with `.get("type", "unknown")` and `if op.get("error") is not None`
- Agent already saved results before this change (existing `raw/` files) → old files will still have the raw fields; `load_raw_json()` doesn't break, the agent just sees extra keys. No migration needed.
