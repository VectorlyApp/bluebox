# Implementation Plan: JSSpecialist & InteractionSpecialist

## Overview

Implement two specialist agents extending `AbstractSpecialist`, plus one data store. Follow `NetworkSpyAgent` patterns throughout.

---

## File 1: `bluebox/llms/infra/interactions_data_store.py` (existing stub → flesh out)

Flesh out the existing empty `InteractionsDataStore` class. Follow `NetworkDataStore` pattern.

**Constructor**: `__init__(self, events: list[UIInteractionEvent])`

**Class method**: `from_jsonl(path: str) -> InteractionsDataStore` — parse JSONL of interaction events

**Stats model** (`InteractionStats` dataclass):
- `total_events`, `unique_urls`, `events_by_type: dict[str, int]`, `unique_elements: int`

**Query methods**:
- `filter_by_type(types: list[str])` → filtered events
- `filter_by_element(tag_name?, element_id?, class_name?, type_attr?)` → filtered events
- `get_form_inputs()` → list of input/change dicts with value, element info
- `get_unique_elements()` → deduplicated elements with interaction counts/types
- `get_event_detail(index: int)` → full event dict or None

---

## File 2: `bluebox/agents/specialists/js_specialist.py` (existing stub → implement)

**Purpose**: Two roles — (1) interpret JS files served by the web server, (2) write IIFE JavaScript for `RoutineJsEvaluateOperation` execution in routines.

**Extra constructor args** (beyond base class):
- `js_data_store: NetworkDataStore | None` — loaded from `network/javascript_events.jsonl` (JS files served by the server)
- `dom_snapshots: list[DOMSnapshotEvent] | None` — loaded from `dom/events.jsonl`

**Result models** (in same file):
```
JSCodeResult(BaseModel):
    js_code: str                    # IIFE-wrapped JavaScript
    session_storage_key: str | None # key for sessionStorage result
    timeout_seconds: float          # max execution time (default 5.0)
    description: str                # what the code does

JSCodeFailureResult(BaseModel):
    reason: str
    attempted_approaches: list[str]
```

**Tools** (registered in `_register_tools`):

| Tool | Params | Purpose |
|------|--------|---------|
| `submit_js_code` | `js_code`, `description`, `session_storage_key?`, `timeout_seconds?` | Submit + validate JS. Uses `RoutineJsEvaluateOperation.validate_js_code()` logic. |
| `validate_js_code` | `js_code` | Dry-run validation only. |
| `search_js_files` | `terms: list[str]` | Search JS file response bodies by terms (delegates to `js_data_store.search_entries_by_terms`). |
| `get_js_file_detail` | `request_id: str` | Get full JS file content (delegates to `js_data_store.get_entry`). |
| `get_dom_snapshot` | `index: int?` | Get DOM snapshot (latest by default). Returns truncated strings table + document structure. |

**Finalize tools**: `finalize_result` → validates + stores `JSCodeResult`; `finalize_failure` → stores `JSCodeFailureResult`

**System prompt highlights**:
- You are a JavaScript expert for browser DOM manipulation
- Two capabilities: analyzing served JS files and writing new IIFE JS
- List blocked patterns (eval, fetch, XMLHttpRequest, etc.)
- Explain IIFE requirement and sessionStorage pattern
- Autonomous variant adds iteration urgency notices

---

## File 3: `bluebox/agents/specialists/interaction_specialist.py` (existing stub → implement)

**Purpose**: Analyze UI interaction recordings to infer routine parameters.

**Extra constructor args**: `interaction_data_store: InteractionsDataStore`

**Result models** (in same file):
```
DiscoveredParameter(BaseModel):
    name: str                          # snake_case parameter name
    type: str                          # ParameterType value (string, date, etc.)
    description: str
    examples: list[str]
    source_element_css_path: str | None
    source_element_tag: str | None
    source_element_name: str | None

ParameterDiscoveryResult(BaseModel):
    parameters: list[DiscoveredParameter]

ParameterDiscoveryFailureResult(BaseModel):
    reason: str
    interaction_summary: str
```

**Tools**:

| Tool | Params | Purpose |
|------|--------|---------|
| `get_interaction_summary` | (none) | Stats overview. |
| `search_interactions_by_type` | `types: list[str]` | Filter by InteractionType. |
| `search_interactions_by_element` | `tag_name?`, `element_id?`, `class_name?`, `type_attr?` | Filter by element attrs. |
| `get_interaction_detail` | `index: int` | Full event detail. |
| `get_form_inputs` | (none) | All input/change events with values. |
| `get_unique_elements` | (none) | Deduplicated elements with counts. |

**Finalize tools**: `finalize_result` → validates + stores `ParameterDiscoveryResult`; `finalize_failure` → stores `ParameterDiscoveryFailureResult`

**System prompt highlights**:
- You analyze recorded UI interactions to discover routine parameters
- Focus on form inputs, typed values, dropdown selections, date pickers
- Ignore navigational clicks and non-parameterizable interactions
- Each parameter needs name, type, description, examples from observed values
- Autonomous variant adds iteration urgency notices

---

## File 4: `bluebox/agents/specialists/__init__.py` (update)

Add exports for `JSSpecialist` and `InteractionSpecialist`.

---

## Implementation Order

1. `InteractionsDataStore` — needed by InteractionSpecialist
2. `JSSpecialist` — independent of data store
3. `InteractionSpecialist` — depends on step 1
4. `__init__.py` exports

## Shared Patterns (from NetworkSpyAgent)

- All tool handlers use `@token_optimized` decorator
- Tool registration via `self.llm_client.register_tool(name, description, parameters={JSON Schema})`
- `_execute_tool` dispatches by `if tool_name == "..."` chain
- `_check_autonomous_completion` → True when finalize sets result
- `_reset_autonomous_state` clears `_*_result` and `_*_failure` fields
- Autonomous system prompt includes dynamic urgency based on `self._autonomous_iteration`

## Verification

```bash
pytest tests/ -v
python -c "from bluebox.agents.specialists.js_specialist import JSSpecialist"
python -c "from bluebox.agents.specialists.interaction_specialist import InteractionSpecialist"
python -c "from bluebox.llms.infra.interactions_data_store import InteractionsDataStore"
```
