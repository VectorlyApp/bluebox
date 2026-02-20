# Routine Discovery Agent — Comprehensive Architecture Reference

> **Purpose**: This document is a complete reference for the `RoutineDiscoveryAgent` system, intended to guide a future refactor. It covers scope, data flow, every component, and how they connect.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Scope & Responsibilities](#2-scope--responsibilities)
3. [Entry Points](#3-entry-points)
4. [Agent Core](#4-agent-core-routinediscoveryagent)
5. [The Agentic Loop](#5-the-agentic-loop)
6. [Phase Workflow](#6-phase-workflow)
7. [Tool System](#7-tool-system)
8. [State Management](#8-state-management)
9. [Data Store (CDP Captures)](#9-data-store-cdp-captures)
10. [LLM Client](#10-llm-client)
11. [Data Models](#11-data-models)
12. [Routine Construction Pipeline](#12-routine-construction-pipeline)
13. [Validation & Execution](#13-validation--execution)
14. [Message / Progress System](#14-message--progress-system)
15. [File Map](#15-file-map)
16. [Data Flow Diagram](#16-data-flow-diagram)
17. [Key Architectural Patterns](#17-key-architectural-patterns)
18. [Known Limitations & Gotchas](#18-known-limitations--gotchas)
19. [Test Coverage](#19-test-coverage)

---

## 1. System Overview

The Routine Discovery Agent is an **LLM-driven agentic loop** that analyzes captured browser network traffic (via Chrome DevTools Protocol) to automatically generate reusable browser automation "routines."

**In plain terms**: A user performs actions in a browser → CDP captures the network traffic → the agent figures out which API calls matter, what variables they use, where dynamic tokens come from, and produces a self-contained JSON routine that can replay the same action with different inputs.

### Core Pipeline

```
Browser Actions → CDP Captures → Agent Analysis → DevRoutine → Production Routine → (Optional) Validation
```

---

## 2. Scope & Responsibilities

### What the agent does:
- Identifies the **root network transaction** that accomplishes the user's task
- Classifies variables in the request as **PARAMETER** (user input), **DYNAMIC_TOKEN** (auth/session), or **STATIC_VALUE** (constants)
- Uses **BFS** to resolve dynamic token sources — scanning storage, window properties, and prior transaction responses
- Constructs a **DevRoutine** (simplified LLM-friendly format)
- Productionizes it into a **Routine** (production execution format) via a secondary LLM call
- Optionally **validates** the routine by executing it against a live browser

### What the agent does NOT do:
- Does not capture CDP data (that's `bluebox-monitor`)
- Does not handle UI interactions (clicks, typing) in the discovery phase — only network traffic
- Does not manage browser lifecycle
- Does not handle multi-step workflows with multiple root transactions (it finds ONE root transaction and its dependencies)

### Information the agent has access to:
- **Network transactions**: Full request/response pairs from CDP captures (URL, method, headers, body, response body, timestamps)
- **Storage snapshots**: sessionStorage, localStorage, cookies at capture time
- **Window properties**: JavaScript window property values captured during the session
- **OpenAI vectorstores**: Semantic search over the above data (uploaded as files)
- **Documentation files**: Optional markdown/code files for additional context

---

## 3. Entry Points

### CLI: `bluebox-discover`
**File**: `bluebox/scripts/discover_routine.py`

```bash
bluebox-discover \
  --task "search for flights from LAX to JFK" \
  --cdp-captures-dir ./cdp_captures \
  --output-dir ./output \
  --llm-model gpt-5.1 \
  --remote-debugging-address http://127.0.0.1:9222
```

Flow:
1. Creates `LocalDiscoveryDataStore` from CDP captures directory
2. Builds vectorstores (CDP captures + optional documentation)
3. Instantiates `RoutineDiscoveryAgent` with `LLMClient`
4. Calls `agent.run()` → returns `Routine`
5. Saves `routine.json` and `test_parameters.json`
6. Cleans up vectorstores

### SDK: `RoutineDiscovery`
**File**: `bluebox/sdk/discovery.py`

```python
from bluebox.sdk.discovery import RoutineDiscovery
discovery = RoutineDiscovery(client, task="...", cdp_captures_dir="...")
result = discovery.run()  # -> RoutineDiscoveryResult(routine, test_parameters)
```

Same internal flow as CLI, wrapped in a Python API.

### HTTP Adapter: `bluebox-agent-adapter`
**File**: `bluebox/scripts/agent_http_adapter.py`

Exposes the agent as `POST /discover` and `POST /chat` endpoints.

---

## 4. Agent Core: `RoutineDiscoveryAgent`

**File**: `bluebox/agents/routine_discovery_agent.py`

### Constructor Fields

| Field | Type | Default | Purpose |
|---|---|---|---|
| `llm_client` | `LLMClient` | required | Facade over OpenAI/Anthropic APIs |
| `data_store` | `DiscoveryDataStore` | required | Access to CDP captures, storage, window properties |
| `task` | `str` | required | User's task description (e.g., "search for flights") |
| `emit_message_callable` | `Callable` | required | Callback for progress events |
| `message_history` | `list[dict]` | `[]` | Full LLM conversation history |
| `output_dir` | `str \| None` | `None` | Where to save artifacts |
| `last_response_id` | `str \| None` | `None` | OpenAI Responses API chaining ID |
| `n_transaction_identification_attempts` | `int` | `3` | Max retries for finding root transaction |
| `max_iterations` | `int` | `50` | Max agent loop iterations |
| `timeout` | `float` | `600` | Execution timeout (seconds) |
| `remote_debugging_address` | `str \| None` | `None` | Chrome debug address for validation |

### Private State

| Field | Type | Purpose |
|---|---|---|
| `_state` | `RoutineDiscoveryState` | Tracks phases, queues, extracted data, built routines |

### Key Prompts (class-level constants)

- **`SYSTEM_PROMPT`**: The main system prompt describing the 4-phase workflow, variable classification rules, and important notes (prefer network sources over storage, minimal parameters, etc.)
- **`PLACEHOLDER_INSTRUCTIONS`**: Rules for `{{param_name}}` syntax, source prefixes (`sessionStorage:`, `cookie:`, etc.), type matching
- **`DATA_STORE_PROMPT`**: Template for injecting available data context

The system prompt is **dynamically rebuilt** each iteration via `_get_system_prompt()`, which appends current phase and queue status.

---

## 5. The Agentic Loop

### `run()` method
1. Asserts vectorstore ID exists
2. Creates fresh `RoutineDiscoveryState()`
3. Emits `INITIATED` progress message
4. Registers all 11 tools with `LLMClient`
5. Configures vectorstores for file search
6. Seeds message history:
   - System prompt
   - `"Task: {self.task}"`
   - All available transaction IDs (toon-encoded for token efficiency)
7. Calls `_run_agent_loop()`
8. Emits `FINISHED` on success

### `_run_agent_loop()` method (the core loop)

```
for iteration in range(max_iterations):
    1. Check if phase == COMPLETE → return routine
    2. Build messages (use previous_response_id for incremental context)
    3. Call llm_client.call_sync(messages, system_prompt, tool_choice="auto")
    4. Record assistant response in message_history
    5. If tool_calls present:
       - Execute each tool via _execute_tool()
       - Record results in message_history
    6. If NO tool_calls and NOT complete:
       - Inject phase-specific "ACTION REQUIRED" nudge as system message
       - This prevents the LLM from stalling
```

**Important**: The loop uses OpenAI's `previous_response_id` for context chaining. When available, only messages after the last assistant message are sent (incremental). Otherwise, the full history is sent.

---

## 6. Phase Workflow

### Phase 1: `IDENTIFY_TRANSACTION`

**Goal**: Find the single network transaction that accomplishes the user's task.

**LLM tools used**: `list_transactions` → `get_transaction` → `record_identified_transaction`

**State changes**:
- Sets `state.root_transaction`
- Adds root transaction to BFS queue
- Transitions to `PROCESS_QUEUE`

**Error handling**: After `n_transaction_identification_attempts` failures, raises `TransactionIdentificationFailedError`.

### Phase 2: `PROCESS_QUEUE` (BFS)

**Goal**: For each transaction in the queue, extract and resolve all variables.

**Per-transaction flow**:
1. `get_transaction` — examine full request/response
2. `record_extracted_variables` — classify each variable as PARAMETER, DYNAMIC_TOKEN, or STATIC_VALUE
3. For each DYNAMIC_TOKEN: `scan_for_value` → `record_resolved_variable`
   - If source is another transaction → auto-added to queue (BFS expansion)
4. `mark_transaction_complete` → pops next from queue

**State changes**:
- `state.transaction_data[tx_id]` accumulates request, extracted_variables, resolved_variables
- Queue grows as dependencies are discovered
- When queue empties → transitions to `CONSTRUCT_ROUTINE`

### Phase 3: `CONSTRUCT_ROUTINE`

**Goal**: Build the complete routine from all processed data.

**LLM tools used**: `construct_routine` (with name, description, parameters, operations)

**Internal sub-steps** (inside `_tool_construct_routine`):
1. Create `DevRoutine` from LLM's arguments
2. Validate DevRoutine (structure, parameter usage, placeholder prefixes)
3. If invalid → return errors, LLM retries
4. If valid → **separate LLM call** to productionize:
   - Sends DevRoutine + Routine JSON schema
   - Uses `manual_llm_parse_text_to_model()` with up to 5 retries
   - Converts string headers/body to dict headers/body
5. Stores both `dev_routine` and `production_routine` in state

**Transition**:
- If `remote_debugging_address` set → `VALIDATE_ROUTINE`
- Otherwise → `COMPLETE`

### Phase 4: `VALIDATE_ROUTINE` (Optional)

**Goal**: Execute the routine against a live browser to verify it works.

**LLM tools used**: `execute_routine` (with test parameters from observed values)

**On failure**:
- Returns diagnostics: `failed_placeholders`, `operation_errors`, `warnings`, `placeholder_resolution`
- LLM can use `scan_for_value` to re-verify sources, then `construct_routine` again
- Multiple validation attempts allowed

**On success** → transitions to `COMPLETE`

---

## 7. Tool System

### Tool Definitions
**File**: `bluebox/llms/tools/routine_discovery_tools.py`

`TOOL_DEFINITIONS` is a list of 11 tool definitions (name, description, JSON Schema parameters):

| # | Tool Name | Required Params | Purpose |
|---|---|---|---|
| 1 | `list_transactions` | none | List all available transaction IDs |
| 2 | `get_transaction` | `transaction_id` | Get full request/response details |
| 3 | `scan_for_value` | `value`, opt `before_transaction_id` | Search storage, window props, and prior transactions for a value |
| 4 | `add_transaction_to_queue` | `transaction_id`, `reason` | Manually add a transaction to BFS queue |
| 5 | `get_queue_status` | none | Get queue counts and current transaction |
| 6 | `mark_transaction_complete` | `transaction_id` | Mark done, pop next |
| 7 | `record_identified_transaction` | `transaction_id`, `description`, `url`, `method` | Set root transaction |
| 8 | `record_extracted_variables` | `transaction_id`, `variables[]` | Log variables found in request |
| 9 | `record_resolved_variable` | `variable_name`, `transaction_id`, `source_type`, opt source objects | Record where a dynamic token comes from |
| 10 | `construct_routine` | `name`, `description`, `parameters[]`, `operations[]` | Build the routine |
| 11 | `execute_routine` | `parameters{}` | Validate by executing |

Helpers: `get_tool_by_name(name)`, `get_all_tool_names()`

### Tool Registration

In `_register_tools()`, each tool definition is registered with `llm_client.register_tool(name, description, parameters)`.

### Tool Dispatch

`_execute_tool(tool_name, tool_arguments)` routes to internal `_tool_*` methods. All return `dict`. Exceptions become `{"error": str(e)}`.

---

## 8. State Management

**File**: `bluebox/data_models/routine_discovery/state.py`

### `DiscoveryPhase` (StrEnum)

Used by the legacy agent:
- `IDENTIFY_TRANSACTION`
- `PROCESS_QUEUE`
- `CONSTRUCT_ROUTINE`
- `VALIDATE_ROUTINE`

Also defines beta agent phases: `PLANNING`, `DISCOVERING`, `CONSTRUCTING`, `VALIDATING`, `COMPLETE`, `FAILED`

### `RoutineDiscoveryState` (BaseModel)

| Field | Type | Purpose |
|---|---|---|
| `root_transaction` | `TransactionIdentificationResponse \| None` | The identified root |
| `transaction_queue` | `list[str]` | Pending BFS queue |
| `processed_transactions` | `list[str]` | Completed (in BFS order) |
| `current_transaction` | `str \| None` | Currently being processed |
| `transaction_data` | `dict[str, dict]` | Per-tx: `request`, `extracted_variables`, `resolved_variables` list |
| `all_resolved_variables` | `list[ResolvedVariableResponse]` | All resolved variables across all transactions |
| `dev_routine` | `DevRoutine \| None` | Intermediate format |
| `production_routine` | `Routine \| None` | Final format |
| `test_parameters` | `dict[str, str]` | Test values |
| `phase` | `DiscoveryPhase` | Current phase |
| `identification_attempts` | `int` | Retry counter |
| `construction_attempts` | `int` | Retry counter |
| `validation_attempts` | `int` | Retry counter |

### Key Methods

- **`add_to_queue(tx_id)`** → `(bool added, int position)` — deduplicates against current, processed, and queued
- **`pop_next_transaction()`** → `str | None` — FIFO pop, sets `current_transaction`
- **`mark_transaction_complete(tx_id)`** — moves to `processed_transactions`
- **`store_transaction_data(tx_id, ...)`** — stores request, extracted_variables, or appends a resolved_variable
- **`get_ordered_transactions()`** — returns **reversed** BFS order (dependencies first, root last) — critical for execution ordering
- **`get_queue_status()`** — summary dict with counts

---

## 9. Data Store (CDP Captures)

**File**: `bluebox/llms/infra/data_store.py`

### `DiscoveryDataStore` (ABC)

Abstract interface. Key methods:

| Method | Returns | Purpose |
|---|---|---|
| `get_all_transaction_ids()` | `list[str]` | All transaction IDs |
| `get_transaction_by_id(id)` | `dict` | Full request/response |
| `get_transaction_timestamp(id)` | `float` | Timestamp for ordering |
| `scan_transaction_responses(value, max_timestamp)` | `list[str]` | Find value in response bodies |
| `scan_storage_for_value(value)` | `list[str]` | Find value in cookies/localStorage/sessionStorage |
| `scan_window_properties_for_value(value)` | `list[dict]` | Find value in window properties |
| `make_cdp_captures_vectorstore()` | void | Upload captures to OpenAI vectorstore |
| `get_vectorstore_ids()` | `list[str]` | Get vectorstore IDs for file_search |
| `generate_data_store_prompt()` | `str` | Text summary of available data |

### `LocalDiscoveryDataStore` (concrete implementation)

**CDP capture processing**:
- Reads `events.jsonl` files from `network/`, `storage/`, `window_properties/` subdirectories
- Generates transaction IDs in `{timestamp}_{safe_url}` format
- Groups network events into request/response structures
- Saves individual transaction JSON files + consolidated JSON
- Response bodies >1000 chars are truncated in consolidated file (full versions in individual files)

**Vectorstore management**:
- Creates OpenAI vectorstores with 1-day expiry
- Uploads files in parallel (4 threads)
- Supports filtering by transaction UUID

**Scanning**:
- `scan_transaction_responses()` — iterates consolidated transactions in timestamp order, checks if value appears in response body (substring match)
- `scan_storage_for_value()` — checks consolidated storage items
- `scan_window_properties_for_value()` — checks consolidated window properties

---

## 10. LLM Client

**File**: `bluebox/llms/llm_client.py`

### `LLMClient` (BaseModel)

A facade over vendor-specific clients (OpenAI, Anthropic). Key features:

- **Model selection**: Constructor takes `LLMModel` (union of `OpenAIModel | AnthropicModel`), auto-discovers the right vendor client
- **Tool registration**: `register_tool(name, description, parameters)`, `clear_tools()`
- **Vectorstore support**: `set_file_search_vectorstores(ids, filters)` — OpenAI-only, enables semantic search over uploaded files
- **API calls**: `call_sync(messages, system_prompt, tool_choice, previous_response_id, response_model, ...)`

### `LLMChatResponse`
**File**: `bluebox/data_models/llms/interaction.py`

| Field | Type | Purpose |
|---|---|---|
| `content` | `str \| None` | Text response |
| `tool_calls` | `list[LLMToolCall]` | Tool invocations (each: `tool_name`, `tool_arguments`, `call_id`) |
| `response_id` | `str \| None` | For chaining with `previous_response_id` |
| `parsed` | `Any \| None` | Structured output result |

---

## 11. Data Models

### LLM Response Models
**File**: `bluebox/data_models/routine_discovery/llm_responses.py`

#### Variable Classification

```
VariableType (StrEnum):
  PARAMETER      — user input (search_query, item_id)
  DYNAMIC_TOKEN  — auth tokens, CSRF, session values
  STATIC_VALUE   — hardcoded constants (app version, User-Agent)
```

#### `Variable`
- `type`: VariableType
- `requires_dynamic_resolution`: bool
- `name`: str
- `observed_value`: str (actual value from capture)
- `values_to_scan_for`: list[str]

#### Source Types

| Source | Fields | Example |
|---|---|---|
| `SessionStorageSource` | `type` (COOKIE/LOCAL_STORAGE/SESSION_STORAGE), `dot_path` | `sessionStorage:auth.access_token` |
| `WindowPropertySource` | `dot_path` | `window.ytcfg.data_.INNERTUBE_API_KEY` |
| `TransactionSource` | `transaction_id`, `dot_path` | Response field from a prior API call |

#### `ResolvedVariableResponse`
Links a `Variable` to exactly one source (session_storage, window_property, or transaction).

#### `TransactionIdentificationResponse`
Records the root transaction: `transaction_id`, `description`, `url`, `method`, `short_explanation`.

### Routine Models

#### `DevRoutine` — Simplified LLM-generation format
**File**: `bluebox/data_models/routine/dev_routine.py`

Only 4 operation types:
- `DevNavigateOperation` (url)
- `DevSleepOperation` (timeout_seconds)
- `DevFetchOperation` (endpoint with **string** headers/body, session_storage_key)
- `DevReturnOperation` (session_storage_key)

**Validation rules** (`validate()` → `(bool, list[str], Exception | None)`):
- Must have 3+ operations (navigate + fetch + return minimum)
- First op = navigate, last op = return (or download), second-to-last = fetch
- All defined parameters must be used in operations
- All remaining placeholders must have valid prefixes
- Session storage keys must be consistent
- Last fetch's session_storage_key must match return's session_storage_key

#### `Routine` — Production execution format
**File**: `bluebox/data_models/routine/routine.py`

- `name`, `description`
- `operations`: list[RoutineOperationUnion] (13 operation types)
- `parameters`: list[Parameter]

**Validation** (`model_validator`):
- 2+ operations (or single download)
- Last operation must be `return`, `return_html`, or `download`
- Return's `session_storage_key` must reference a prior fetch/js_evaluate that sets it
- All defined parameters must be used; no undefined parameters allowed

**Execution** (`execute()`):
- Creates a Chrome tab, enables CDP domains
- Builds `RoutineExecutionContext`
- Iterates operations calling `operation.execute(context)`
- Returns `RoutineExecutionResultWithMetadata`

### Operation Types (13 total)
**File**: `bluebox/data_models/routine/operation.py`

| Type | Key Fields | Description |
|---|---|---|
| `navigate` | `url`, `sleep_after_navigation_seconds` | Page navigation |
| `sleep` | `timeout_seconds` | Wait |
| `fetch` | `endpoint`, `session_storage_key` | HTTP fetch, store result |
| `return` | `session_storage_key` | Read stored result (chunked 256KB) |
| `get_cookies` | `session_storage_key`, `domain_filter` | CDP cookie extraction |
| `click` | `selector`, `button`, `click_count` | Mouse click |
| `type` | `selector`, `text`, `clear` | Keyboard typing |
| `press` | `key` | Single key press |
| `wait_for_url` | `url_regex`, `timeout_ms` | Poll URL until match |
| `scroll` | `selector`, x/y/delta | Page/element scroll |
| `return_html` | `scope`, `selector` | Return page/element HTML |
| `download` | `endpoint`, `filename` | Binary fetch → base64 |
| `js_evaluate` | `js`, `timeout_seconds`, `session_storage_key` | Execute JavaScript |

`RoutineOperationUnion` = discriminated union on `type` field.

### Parameters
**File**: `bluebox/data_models/routine/parameter.py`

```
ParameterType: STRING, INTEGER, NUMBER, BOOLEAN, DATE, DATETIME, EMAIL, URL, ENUM
```

`Parameter` fields: `name`, `type`, `required`, `description`, `default`, `examples`, `observed_value`, validation constraints (min/max, pattern, enum_values).

**Builtin parameters** (auto-resolved, never defined by user): `uuid`, `epoch_milliseconds`.

**Valid placeholder prefixes**: `sessionStorage`, `localStorage`, `cookie`, `meta`, `windowProperty`.

### Placeholder Extraction
**File**: `bluebox/data_models/routine/placeholder.py`

`extract_placeholders_from_json_str(json_string) -> list[str]`

Regex: `\{\{\s*([^}]+?)\s*\}\}` — finds all `{{...}}` patterns, returns deduplicated list.

### Endpoint
**File**: `bluebox/data_models/routine/endpoint.py`

`Endpoint`: `url`, `method` (HTTPMethod enum), `headers` (dict), `body` (dict), `credentials` (SAME_ORIGIN/INCLUDE/OMIT), `description`.

---

## 12. Routine Construction Pipeline

This is the most complex part. Inside `_tool_construct_routine()`:

```
LLM provides: name, description, parameters[], operations[]
                    |
                    v
            DevRoutine(...)
                    |
                    v
           dev_routine.validate()
           ├── FAIL → return errors to LLM, retry
           └── PASS ↓
                    |
                    v
        Separate LLM call: "Productionize this"
        Input: DevRoutine JSON + Routine JSON schema + PLACEHOLDER_INSTRUCTIONS
        Output format: json_object (enforced)
                    |
                    v
        manual_llm_parse_text_to_model()
        - Strips markdown code blocks
        - Parses with Pydantic (Routine)
        - On failure: appends error, retries (up to 5x)
        - On exhaustion: raises LLMStructuredOutputError
                    |
                    v
          production_routine (Routine)
```

### `manual_llm_parse_text_to_model()`
**File**: `bluebox/utils/llm_utils.py`

Critical for converting free-form LLM text into a validated Pydantic model. Uses `response_format={"type": "json_object"}` for guaranteed JSON output. Retries with error context on validation failures.

---

## 13. Validation & Execution

### Execute Routine Tool
**File**: `bluebox/llms/tools/execute_routine_tool.py`

`execute_routine(routine, parameters, remote_debugging_address, timeout, close_tab_when_done, tab_id)`:
1. Parses routine dict into `Routine` object
2. Creates `RoutineExecutor`
3. Calls `executor.execute()`
4. Returns `{"success": True, "result": result}` or `{"success": False, "error": str}`

### Test Parameter Generation

`_get_test_parameters_for_validation()`:
- Extracts `observed_value` from each parameter in the production routine
- Falls back to type-based defaults (`"1"` for integer, `"false"` for boolean, etc.)
- Returns `dict[str, str]`

### Execution Context
**File**: `bluebox/data_models/routine/execution.py`

`RoutineExecutionContext` carries mutable state through all operations:
- CDP session and WebSocket connection
- User parameters + type map for coercion
- Current URL tracking
- `RoutineExecutionResultWithMetadata` (operations modify directly)

### Execution Result
`RoutineExecutionResultWithMetadata` extends `RoutineExecutionResult` with:
- `warnings`: list[str]
- `operations_metadata`: per-operation timing and errors
- `placeholder_resolution`: dict[str, str | None] — shows what each placeholder resolved to (or None if failed)

---

## 14. Message / Progress System

**File**: `bluebox/data_models/routine_discovery/message.py`

### Message Types

| Type | When Emitted |
|---|---|
| `INITIATED` | Discovery started |
| `PROGRESS_THINKING` | Agent is processing (default for `_emit_progress`) |
| `PROGRESS_RESULT` | Milestone reached (e.g., "Identified target transaction") |
| `FINISHED` | Discovery complete |
| `ERROR` | Failure |

### `RoutineDiscoveryMessage`
Fields: `type`, `timestamp` (UTC), `content` (str).

Delivered via `emit_message_callable` — a callback set by the caller (CLI prints to stdout, SDK stores for programmatic access).

---

## 15. File Map

```
bluebox/
├── agents/
│   └── routine_discovery_agent.py          ← THE AGENT (this doc)
├── data_models/
│   ├── routine_discovery/
│   │   ├── state.py                        ← DiscoveryPhase, RoutineDiscoveryState
│   │   ├── message.py                      ← RoutineDiscoveryMessage, MessageType
│   │   └── llm_responses.py                ← Variable, VariableType, sources, etc.
│   ├── routine/
│   │   ├── routine.py                      ← Routine (production model + execution)
│   │   ├── dev_routine.py                  ← DevRoutine (simplified for LLM)
│   │   ├── operation.py                    ← 13 operation types + union
│   │   ├── parameter.py                    ← Parameter, ParameterType, builtins
│   │   ├── placeholder.py                  ← extract_placeholders_from_json_str()
│   │   ├── endpoint.py                     ← Endpoint, HTTPMethod, CREDENTIALS
│   │   └── execution.py                    ← RoutineExecutionContext, results
│   └── llms/
│       └── interaction.py                  ← LLMChatResponse, LLMToolCall
├── llms/
│   ├── llm_client.py                       ← LLMClient facade
│   ├── infra/
│   │   └── data_store.py                   ← DiscoveryDataStore, LocalDiscoveryDataStore
│   └── tools/
│       ├── routine_discovery_tools.py      ← TOOL_DEFINITIONS (11 tools)
│       └── execute_routine_tool.py         ← execute_routine() function
├── scripts/
│   ├── discover_routine.py                 ← CLI entry point (bluebox-discover)
│   └── agent_http_adapter.py              ← HTTP adapter
├── sdk/
│   └── discovery.py                        ← RoutineDiscovery SDK class
└── utils/
    ├── exceptions.py                       ← TransactionIdentificationFailedError, etc.
    ├── llm_utils.py                        ← manual_llm_parse_text_to_model()
    └── logger.py                           ← get_logger()

tests/
└── unit/
    ├── agents/
    │   └── test_routine_discovery_agent.py ← 621 lines, comprehensive unit tests
    └── llms/tools/
        └── test_routine_discovery_tools.py ← 385 lines, tool definition tests
```

---

## 16. Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        CDP Captures Dir                         │
│  network/events.jsonl  storage/events.jsonl  window_props/...   │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                   LocalDiscoveryDataStore                        │
│  • Parses events.jsonl → transaction dicts                      │
│  • Creates consolidated JSON files                              │
│  • Uploads to OpenAI vectorstores                               │
│  • Provides scan_* methods for value lookup                     │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                  RoutineDiscoveryAgent.run()                     │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Phase 1: IDENTIFY_TRANSACTION                            │   │
│  │  LLM → list_transactions → get_transaction               │   │
│  │      → record_identified_transaction                      │   │
│  │  Output: root_transaction, added to BFS queue             │   │
│  └────────────────────────┬─────────────────────────────────┘   │
│                           ▼                                     │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Phase 2: PROCESS_QUEUE (BFS)                             │   │
│  │  For each transaction:                                    │   │
│  │    LLM → get_transaction                                  │   │
│  │        → record_extracted_variables (PARAM/TOKEN/STATIC)  │   │
│  │    For each DYNAMIC_TOKEN:                                │   │
│  │        → scan_for_value (storage, window, transactions)   │   │
│  │        → record_resolved_variable                         │   │
│  │          (if source=transaction → auto-add to queue)      │   │
│  │        → mark_transaction_complete → pop next             │   │
│  │  Output: transaction_data{} with all variables resolved   │   │
│  └────────────────────────┬─────────────────────────────────┘   │
│                           ▼                                     │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Phase 3: CONSTRUCT_ROUTINE                               │   │
│  │  LLM → construct_routine(name, desc, params, ops)         │   │
│  │    1. Create DevRoutine → validate()                      │   │
│  │    2. Separate LLM call → productionize to Routine        │   │
│  │       (manual_llm_parse_text_to_model, up to 5 retries)  │   │
│  │  Output: dev_routine + production_routine                 │   │
│  └────────────────────────┬─────────────────────────────────┘   │
│                           ▼                                     │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Phase 4: VALIDATE_ROUTINE (if browser available)         │   │
│  │  LLM → execute_routine(test_parameters)                   │   │
│  │    • Runs against live Chrome via RoutineExecutor          │   │
│  │    • On failure: diagnostics → LLM fixes → retry          │   │
│  │    • On success: → COMPLETE                               │   │
│  └────────────────────────┬─────────────────────────────────┘   │
│                           ▼                                     │
│                    Return Routine                                │
└─────────────────────────────────────────────────────────────────┘
```

---

## 17. Key Architectural Patterns

### 1. LLM-Driven Tool-Calling Loop
The LLM decides the workflow by choosing which tools to call. The agent provides phase-specific "nudge" prompts when the LLM stalls (no tool calls). This is a standard ReAct-style pattern.

### 2. BFS Dependency Resolution
When a dynamic token's source is another transaction, that transaction is automatically queued. The reversed BFS order (`get_ordered_transactions()`) ensures dependencies execute before dependents in the final routine.

### 3. Two-Stage Routine Construction
`DevRoutine` (simple string-based) → `Routine` (production dict-based). This reduces LLM errors by first validating a simpler structure, then converting via a separate focused LLM call.

### 4. Observed Values for Testing
Parameters store `observed_value` (the actual value from CDP capture) enabling automated validation without user input.

### 5. Placeholder Syntax Unification
All placeholders use `{{name}}` format. `Parameter.type` drives coercion at runtime (e.g., `{{limit}}` with type=integer → `50` not `"50"`).

### 6. Vectorstore-Backed Context
CDP captures and documentation are uploaded to OpenAI vectorstores, giving the LLM semantic search over the data. This supplements the explicit tool-based data access.

### 7. Incremental Context via `previous_response_id`
Uses OpenAI's Responses API chaining to avoid resending the full conversation each iteration — only new messages since the last assistant response are sent.

---

## 18. Known Limitations & Gotchas

1. **Single root transaction**: The agent finds ONE root API call and its dependencies. Multi-step workflows with independent root transactions are not supported.

2. **Productionize call is fragile**: The separate LLM call to convert DevRoutine → Routine can fail. It uses `manual_llm_parse_text_to_model` with 5 retries, but complex routines may still fail to parse.

3. **Vectorstore dependency**: Requires OpenAI vectorstores (API key required). The `LocalDiscoveryDataStore` creates vectorstores with 1-day expiry and must clean them up.

4. **Response body truncation**: Consolidated transaction files truncate response bodies >1000 chars. Full versions exist in individual files but the scan methods use the consolidated file.

5. **Storage vs Network preference**: The system prompt instructs the LLM to prefer network (transaction) sources over storage sources, since storage may be empty in a fresh session. This is a heuristic, not enforced programmatically.

6. **Phase nudge messages**: When the LLM doesn't make tool calls, the agent injects system-role messages. These accumulate in the conversation and can confuse the LLM if it stalls repeatedly.

7. **DevRoutine validation is stricter than Routine**: DevRoutine requires navigate→fetch→return sequence. Routine allows more flexibility (13 operation types, download as last op, etc.). This means the DevRoutine acts as a bottleneck for what the agent can produce.

8. **No parallel tool execution**: Tool calls from a single LLM response are executed sequentially, even if they could be parallelized.

9. **Message history grows unbounded**: The full conversation is stored in `message_history`. For complex routines with many transactions, this can become very large.

10. **Type coercion gotcha**: The `PLACEHOLDER_INSTRUCTIONS` emphasize matching types to the raw CDP request (e.g., if `"adults": "5"` is a string in the request, use type=string, not integer). Getting this wrong can break APIs.

---

## 19. Test Coverage

### Agent Tests
**File**: `tests/unit/agents/test_routine_discovery_agent.py` (621 lines)

| Test Class | Coverage |
|---|---|
| `TestToolListTransactions` | Returns IDs, handles empty |
| `TestToolGetTransaction` | Returns details, error on invalid |
| `TestToolScanForValue` | All source types, timestamp filter, result limits |
| `TestToolAddToQueue` | Valid add, invalid TX, duplicates |
| `TestToolGetQueueStatus` | Returns status dict |
| `TestToolMarkComplete` | Marks complete, phase transition when empty |
| `TestToolRecordIdentifiedTransaction` | Valid recording, error on invalid, max attempts exception |
| `TestToolRecordExtractedVariables` | Records variables, stores in state |
| `TestToolRecordResolvedVariable` | Storage source, transaction source with auto-dependency, unknown variable error |
| `TestToolExecute` | Dispatch routing, unknown tool error, exception handling |
| `TestAgentInitialization` | Tool registration, system prompt includes state |
| `TestMessageHistory` | Add messages, add tool results |
| `TestGetTestParameters` | Observed values, type-based defaults |
| `TestConstructRoutineTool` | Routine construction validation |
| `TestExecuteRoutineTool` | Errors without routine or browser |

### Tool Definition Tests
**File**: `tests/unit/llms/tools/test_routine_discovery_tools.py` (385 lines)

- All tools have required keys (name, description, parameters)
- JSON Schema validity for all parameter definitions
- Unique tool names
- Per-tool tests for required params, types, enums
- Helper function tests (`get_tool_by_name`, `get_all_tool_names`)
