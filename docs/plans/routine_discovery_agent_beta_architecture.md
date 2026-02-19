# RoutineDiscoveryAgentBeta — Architecture & Component Reference

> **Purpose:** This document is a deep-dive reference for the `RoutineDiscoveryAgentBeta` system. It maps every component, its responsibilities, file location, and how the pieces connect. Use this as context for refactoring.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [How Discovery Works (End-to-End)](#how-discovery-works-end-to-end)
3. [Phase Machine](#phase-machine)
4. [Agent Hierarchy](#agent-hierarchy)
5. [Component Reference](#component-reference)
   - [Orchestrator Agent](#1-orchestrator-agent)
   - [Abstract Agent Infrastructure](#2-abstract-agent-infrastructure)
   - [Specialist Agents](#3-specialist-agents)
   - [Data Loaders](#4-data-loaders)
   - [State Models](#5-state-models)
   - [LLM Response Models](#6-llm-response-models)
   - [Routine Models](#7-routine-models)
   - [Utility Modules](#8-utility-modules)
   - [TUI Layer](#9-tui-layer)
6. [Data Flow Diagram](#data-flow-diagram)
7. [Tool Inventory](#tool-inventory)
8. [Key Design Patterns](#key-design-patterns)
9. [Known Architectural Issues](#known-architectural-issues)

---

## System Overview

`RoutineDiscoveryAgentBeta` is an **LLM-powered orchestrator** that analyzes captured browser network traffic (CDP captures) and produces a reusable `Routine` — a JSON workflow that can replay an API interaction programmatically.

**The core idea:** A user performs an action in a browser (e.g., searching for trains). The system captures all network traffic, storage events, and interactions. The orchestrator then coordinates specialist AI agents to reverse-engineer which API call accomplished the user's task, what parameters it needs, where dynamic tokens come from, and assembles it all into an executable routine.

```
Browser Capture (CDP)
        │
        ▼
┌─────────────────────────────────────────┐
│   RoutineDiscoveryAgentBeta (Orchestrator)   │
│                                          │
│  Phase: PLANNING → DISCOVERING →         │
│         CONSTRUCTING → VALIDATING →      │
│         COMPLETE                         │
│                                          │
│  Delegates to:                           │
│  ┌──────────────┐ ┌──────────────────┐   │
│  │ Network      │ │ ValueTrace       │   │
│  │ Specialist   │ │ Resolver         │   │
│  └──────────────┘ └──────────────────┘   │
│  ┌──────────────┐ ┌──────────────────┐   │
│  │ JS           │ │ Interaction      │   │
│  │ Specialist   │ │ Specialist       │   │
│  └──────────────┘ └──────────────────┘   │
└─────────────────────────────────────────┘
        │
        ▼
   Routine (JSON)
```

---

## How Discovery Works (End-to-End)

### 1. **User provides a task and CDP captures**

```bash
bluebox-routine-discovery-agent-beta \
  --cdp-captures-dir ./cdp_captures \
  --task "Search for trains from NYC to Boston"
```

The TUI script (`run_routine_discovery_agent_beta.py`) loads data from JSONL files into data loaders, then creates the orchestrator agent.

### 2. **PLANNING: Find the right API endpoint**

The orchestrator creates a `Task` for `NetworkSpecialist`:
> "Find the API endpoint that accomplishes: Search for trains from NYC to Boston"

The `NetworkSpecialist` runs autonomously — it searches response bodies by terms (e.g., "train", "NYC", "Boston"), ranks transactions by relevance, and returns the best-matching transaction ID. The orchestrator records this as the **root transaction** via `record_identified_endpoint`.

### 3. **DISCOVERING: Extract and resolve variables (BFS)**

The root transaction is added to a BFS queue. For each transaction:

1. The orchestrator inspects the request (URL, headers, body)
2. Calls `record_extracted_variable` for each piece of the request:
   - **PARAMETER** — user-provided values (e.g., `origin_city=NYC`)
   - **DYNAMIC_TOKEN** — session tokens, CSRF tokens, auth headers
   - **STATIC_VALUE** — constants (app version, client name)
3. For each DYNAMIC_TOKEN, creates a `Task` for `ValueTraceResolverSpecialist`:
   > "Trace the origin of value 'eyJhbG...' (variable: auth_token)"
4. The specialist searches across network responses, storage events, and window properties to find where the token originated
5. The orchestrator records the resolution via `record_resolved_variable`:
   - If source is another transaction → that transaction is auto-added to the BFS queue
   - If source is storage → recorded as `{{sessionStorage:path}}` or `{{cookie:name}}`
   - If source is window property → recorded as `{{windowProperty:path}}`
6. Calls `mark_transaction_processed` when done

This continues until the queue is empty — all dependency chains are fully resolved.

### 4. **CONSTRUCTING: Build the routine**

The orchestrator calls `get_discovery_context` to assemble all discoveries, then calls `construct_routine` with:
- **Parameters:** User-provided inputs with types and observed values
- **Operations:** Ordered sequence (navigate → fetch dependencies → fetch main → return)
- **Placeholders:** `{{param_name}}` for user inputs, `{{sessionStorage:path}}` for runtime values

The routine is validated against `Routine.model_validate()` and structural warnings are checked.

### 5. **VALIDATING: Test it (if browser connected)**

If a Chrome instance is available (`--remote-debugging-address`), the orchestrator:
1. Calls `validate_routine` with observed test parameters
2. The routine executes against the live browser via CDP
3. Calls `analyze_validation` to assess results
4. If data matches the task → `done()`
5. If not → `construct_routine` again with fixes

### 6. **Output**

The TUI saves `routine.json` and `test_parameters.json` to the output directory. The routine can be executed later via `bluebox-execute`.

---

## Phase Machine

```
PLANNING ──────► DISCOVERING ──────► CONSTRUCTING ──────► VALIDATING ──────► COMPLETE
    │                 │                    │                    │
    │                 │                    │                    │
    ▼                 ▼                    ▼                    ▼
  FAILED           FAILED              FAILED              FAILED
```

| Phase | Entry Condition | Exit Condition | Key Tools |
|-------|----------------|----------------|-----------|
| **PLANNING** | Initial state | `record_identified_endpoint` called | `create_task`, `run_pending_tasks`, `record_identified_endpoint` |
| **DISCOVERING** | Root transaction set | BFS queue empty + all tokens resolved | `get_transaction`, `record_extracted_variable`, `record_resolved_variable`, `mark_transaction_processed` |
| **CONSTRUCTING** | Auto-transition from DISCOVERING or manual | `construct_routine` succeeds | `get_discovery_context`, `validate_placeholders`, `construct_routine` |
| **VALIDATING** | `validate_routine` called | `analyze_validation` with `data_matches_task=True` | `validate_routine`, `analyze_validation` |
| **COMPLETE** | `done()` called | Terminal | `done` |
| **FAILED** | Error or `fail()` called | Terminal | `fail` |

---

## Agent Hierarchy

```
AbstractAgent                          # Base: LLM calls, tools, chat history
├── AbstractSpecialist                 # Adds: autonomous mode, finalize gating
│   ├── NetworkSpecialist              # Searches network traffic for endpoints
│   ├── ValueTraceResolverSpecialist   # Traces token origins across data sources
│   ├── JSSpecialist                   # Writes/validates browser JavaScript
│   └── InteractionSpecialist          # Analyzes recorded UI interactions
│
└── RoutineDiscoveryAgentBeta          # Orchestrator (NOT a specialist)
        Uses: AgentOrchestrationState  # Task queue management
        Uses: RoutineDiscoveryState    # Discovery-specific state (phases, BFS, variables)
        Creates: Task → delegates to specialists
```

**Important distinction:** `RoutineDiscoveryAgentBeta` extends `AbstractAgent` directly — it is NOT a specialist. It runs its own main loop (`run()` method) and delegates to specialists via the task system.

---

## Component Reference

### 1. Orchestrator Agent

| File | Class | Purpose |
|------|-------|---------|
| `bluebox/agents/routine_discovery_agent_beta.py` | `RoutineDiscoveryAgentBeta` | Main orchestrator. Coordinates specialists, manages discovery phases, constructs routines. |

**Key attributes:**
- `_network_data_loader`, `_storage_data_loader`, `_window_property_data_loader`, `_js_data_loader`, `_interaction_data_loader`, `_documentation_data_loader` — Data sources
- `_orchestration_state: AgentOrchestrationState` — Task/subagent tracking
- `_discovery_state: RoutineDiscoveryState` — Phase, BFS queue, variables, routine
- `_agent_instances: dict[str, AbstractSpecialist]` — Live specialist instances (keyed by subagent ID)
- `_final_routine: Routine | None` — Result
- `_remote_debugging_address: str | None` — Chrome CDP address for validation

**Main loop (`run()`):**
1. Seeds conversation with task description
2. Iterates up to `max_iterations` (default 50)
3. Each iteration: build system prompt → call LLM → process tool calls
4. LLM is forced to use tools (`tool_choice="required"`)
5. If LLM responds without tools, a phase-specific guidance message is injected
6. Exits when phase reaches `COMPLETE` or `FAILED`

**System prompt construction (`_get_system_prompt()`):**
- Always includes `PROMPT_CORE` (identity + delegation rules)
- Always includes specialist descriptions from `AgentCard.description`
- Includes only the **current phase's** instructions (PLANNING, DISCOVERING, etc.)
- Appends data source summaries, current state counts, discovery progress
- During CONSTRUCTING/VALIDATING, appends placeholder instructions and routine schema

---

### 2. Abstract Agent Infrastructure

| File | Class/Symbol | Purpose |
|------|-------------|---------|
| `bluebox/agents/abstract_agent.py` | `AbstractAgent` | Base class: LLM calling, tool registration, chat history, message streaming |
| `bluebox/agents/abstract_agent.py` | `AgentCard` | Frozen dataclass with `description` — self-describing metadata for each agent |
| `bluebox/agents/abstract_agent.py` | `@agent_tool` | Decorator that marks methods as LLM-callable tools. Auto-generates JSON Schema from type hints. Supports dynamic `availability` gating. |
| `bluebox/agents/abstract_agent.py` | `_ToolMeta` | Internal frozen dataclass: `name`, `description`, `parameters`, `availability` |

**AbstractAgent key methods:**
- `_get_system_prompt() -> str` — Abstract, must be overridden
- `_call_llm(messages, system_prompt, tool_choice) -> LLMChatResponse` — Calls vendor API with streaming
- `_process_tool_calls(tool_calls)` — Executes tools in parallel via ThreadPoolExecutor, stores results as TOOL chat messages
- `_sync_tools()` — Re-evaluates `availability` lambdas to register/unregister tools dynamically
- `_collect_tools()` — Discovers all `@agent_tool`-decorated methods via class introspection
- `_add_chat(role, content, ...)` — Creates Chat, updates ChatThread, calls persist callback
- `_build_messages_for_llm()` — Converts Chat history to LLM message format

**Documentation tools (auto-available if `documentation_data_loader` provided):**
- `_search_docs(query)` — Exact string search
- `_get_doc_file(path)` — Read file content
- `_search_docs_by_terms(terms)` — Relevance-ranked search
- `_search_docs_by_regex(pattern)` — Regex pattern search

| File | Class/Symbol | Purpose |
|------|-------------|---------|
| `bluebox/agents/specialists/abstract_specialist.py` | `AbstractSpecialist` | Extends AbstractAgent with autonomous mode, iteration tracking, finalize gating |
| `bluebox/agents/specialists/abstract_specialist.py` | `RunMode` | StrEnum: `CONVERSATIONAL` or `AUTONOMOUS` |
| `bluebox/agents/specialists/abstract_specialist.py` | `AutonomousConfig` | NamedTuple: `min_iterations` (default 3), `max_iterations` (default 10) |

**AbstractSpecialist key additions:**
- `run_autonomous(task, config, output_schema, output_description) -> BaseModel | None` — Public API to run specialist to completion
- `can_finalize: bool` — Property: True after `min_iterations` reached in autonomous mode
- Finalize tools (available when `can_finalize=True`):
  - `_finalize_with_output(output)` — Validate against schema and return
  - `_finalize_result(output)` — Return without schema
  - `_finalize_with_failure(reason)` / `_finalize_failure(reason)` — Failure paths
  - `add_note(note)` — Attach warning/note to result
- `_get_urgency_notice()` — Iteration-aware nudge toward finalization
- `_run_autonomous_loop()` — Forces `tool_choice="required"`, gates finalize tools

---

### 3. Specialist Agents

| File | Class | What It Does | Data Loaders Used |
|------|-------|-------------|-------------------|
| `bluebox/agents/specialists/network_specialist.py` | `NetworkSpecialist` | Finds API endpoints in captured traffic. Searches response bodies by relevance-ranked terms, inspects headers/schemas. | `NetworkDataLoader` |
| `bluebox/agents/specialists/value_trace_resolver_specialist.py` | `ValueTraceResolverSpecialist` | Traces where a dynamic token value originated. Searches across network responses, browser storage, and window properties. | `NetworkDataLoader`, `StorageDataLoader`, `WindowPropertyDataLoader` |
| `bluebox/agents/specialists/js_specialist.py` | `JSSpecialist` | Writes and validates IIFE JavaScript for browser execution. Can test code against live websites. | `NetworkDataLoader`, `JSDataLoader`, DOM snapshots |
| `bluebox/agents/specialists/interaction_specialist.py` | `InteractionSpecialist` | Analyzes recorded UI interactions (clicks, inputs, form fills) to discover user-provided parameters. | `InteractionsDataLoader` |

**How the orchestrator creates specialists:**

```python
# In _create_specialist() — factory method
if agent_type == SpecialistAgentType.NETWORK_SPECIALIST:
    return NetworkSpecialist(
        emit_message_callable=self._emit_message_callable,
        llm_model=self._subagent_llm_model,
        network_data_loader=self._network_data_loader,
        documentation_data_loader=self._documentation_data_loader,
        run_mode=RunMode.AUTONOMOUS,  # Always autonomous when delegated
    )
```

**How tasks flow:**
1. Orchestrator calls `create_task(agent_type, prompt)` → creates `Task` in `AgentOrchestrationState`
2. Orchestrator calls `run_pending_tasks()` → for each pending task:
   - `_get_or_create_agent(task)` — reuses existing agent or creates new one
   - `_execute_task(task)` → calls `agent.run_autonomous(task.prompt, config)`
   - Multiple pending tasks run in parallel via `ThreadPoolExecutor`
3. Orchestrator calls `get_task_result(task_id)` to read results

---

### 4. Data Loaders

All data loaders inherit from `AbstractDataLoader[EventType, StatsType]` and share:
- Constructor takes a JSONL file path
- `.entries` property for raw access
- `.stats` property for aggregate info
- Various search/filter methods

| File | Class | Data Source | Key Methods |
|------|-------|-----------|-------------|
| `bluebox/llms/data_loaders/network_data_loader.py` | `NetworkDataLoader` | Network JSONL (HTTP transactions) | `search_entries_by_terms()`, `get_entry(request_id)`, `search_response_bodies()`, `get_response_body_schema()`, `api_urls`, `url_counts` |
| `bluebox/llms/data_loaders/storage_data_loader.py` | `StorageDataLoader` | Storage JSONL (cookies, localStorage, sessionStorage, IndexedDB) | `search_values()`, `get_entries_by_key()`, `get_entries_by_origin()` |
| `bluebox/llms/data_loaders/window_property_data_loader.py` | `WindowPropertyDataLoader` | Window properties JSONL | `search_values()`, `get_changes_by_path()` |
| `bluebox/llms/data_loaders/js_data_loader.py` | `JSDataLoader` | JavaScript files JSONL | `search_by_terms()`, `search_by_regex()`, `get_file_content()` |
| `bluebox/llms/data_loaders/interactions_data_loader.py` | `InteractionsDataLoader` | UI interaction JSONL | `filter_by_type()`, `get_form_inputs()`, `get_unique_elements()` |
| `bluebox/llms/data_loaders/documentation_data_loader.py` | `DocumentationDataLoader` | Markdown docs + code files | `search_content()`, `get_file_content()`, `get_documentation_index()` |

**JSONL file structure (from `--cdp-captures-dir`):**
```
cdp_captures/
├── network/
│   ├── events.jsonl          → NetworkDataLoader
│   └── javascript_events.jsonl → JSDataLoader
├── storage/
│   └── events.jsonl          → StorageDataLoader
├── window_properties/
│   └── events.jsonl          → WindowPropertyDataLoader
└── interaction/
    └── events.jsonl          → InteractionsDataLoader
```

---

### 5. State Models

| File | Class | Purpose |
|------|-------|---------|
| `bluebox/data_models/orchestration/state.py` | `AgentOrchestrationState` | Tracks all delegated tasks and subagent instances. Methods: `add_task()`, `get_pending_tasks()`, `get_completed_tasks()`, `get_queue_status()`, etc. |
| `bluebox/data_models/orchestration/task.py` | `Task` | Unit of work: agent_type, prompt, status, result, loops tracking. Auto-generated 6-char ID. |
| `bluebox/data_models/orchestration/task.py` | `SubAgent` | Metadata about a specialist instance: type, LLM model, task IDs. |
| `bluebox/data_models/orchestration/task.py` | `TaskStatus` | StrEnum: `PENDING`, `IN_PROGRESS`, `PAUSED`, `COMPLETED`, `FAILED` |
| `bluebox/data_models/orchestration/task.py` | `SpecialistAgentType` | StrEnum: `js_specialist`, `network_specialist`, `value_trace_resolver`, `interaction_specialist` |
| `bluebox/data_models/routine_discovery/state.py` | `RoutineDiscoveryState` | Discovery-specific state: phase, root_transaction, BFS queue, transaction_data (per-tx extracted/resolved variables), production_routine, test_parameters, validation tracking. |
| `bluebox/data_models/routine_discovery/state.py` | `DiscoveryPhase` | StrEnum: `PLANNING`, `DISCOVERING`, `CONSTRUCTING`, `VALIDATING`, `COMPLETE`, `FAILED` (+ legacy phases) |

**RoutineDiscoveryState key attributes:**
- `root_transaction: TransactionIdentificationResponse | None` — The main API endpoint
- `transaction_queue: list[str]` — BFS queue of transaction IDs to process
- `processed_transactions: list[str]` — Completed transaction IDs
- `transaction_data: dict[str, dict]` — Per-transaction: `{request, extracted_variables, resolved_variables}`
- `all_resolved_variables: list[ResolvedVariableResponse]` — Flat list across all transactions
- `production_routine: Routine | None` — The constructed routine
- `test_parameters: dict[str, str]` — Observed values for validation
- `construction_attempts: int`, `validation_attempts: int` — Retry counters

**RoutineDiscoveryState key methods:**
- `add_to_queue(tx_id) -> (added: bool, position: int)` — Adds if not already processed/queued
- `mark_transaction_complete(tx_id)` — Moves from queue to processed
- `get_queue_status()` — Counts for system prompt
- `get_ordered_transactions()` — Returns transactions in execution order (dependencies first)

---

### 6. LLM Response Models

| File | Class | Purpose |
|------|-------|---------|
| `bluebox/data_models/routine_discovery/llm_responses.py` | `TransactionIdentificationResponse` | Root transaction: `transaction_id`, `url`, `method`, `description` |
| | `VariableType` | StrEnum: `PARAMETER`, `DYNAMIC_TOKEN`, `STATIC_VALUE` |
| | `Variable` | Extracted variable: `type`, `name`, `observed_value`, `requires_dynamic_resolution`, `values_to_scan_for` |
| | `ExtractedVariableResponse` | Container: `transaction_id` + `list[Variable]` |
| | `ResolvedVariableResponse` | Resolution: `variable` + source (`SessionStorageSource` or `TransactionSource` or `WindowPropertySource`) |
| | `SessionStorageSource` | Storage resolution: `type` (cookie/localStorage/sessionStorage) + `dot_path` |
| | `TransactionSource` | Network resolution: `transaction_id` + `dot_path` into response body |
| | `WindowPropertySource` | Window resolution: `dot_path` into `window` object |
| | `SessionStorageType` | StrEnum: `cookie`, `localStorage`, `sessionStorage` |

---

### 7. Routine Models

| File | Class/Symbol | Purpose |
|------|-------------|---------|
| `bluebox/data_models/routine/routine.py` | `Routine` | Core model: `name`, `description`, `parameters`, `operations`. Has `model_validate()`, `model_schema_markdown()`, `get_structure_warnings()`, `execute()`. |
| `bluebox/data_models/routine/operation.py` | `RoutineOperationTypes` | StrEnum of all operation types |
| | `RoutineNavigateOperation` | Navigate to URL |
| | `RoutineFetchOperation` | HTTP fetch in browser context, stores result in session storage |
| | `RoutineReturnOperation` | Retrieves data from session storage (chunked for large responses) |
| | `RoutineDownloadOperation` | Downloads binary file as base64 |
| | `RoutineClickOperation` | Click DOM element |
| | `RoutineTypeOperation` | Type text into input |
| | `RoutinePressOperation` | Press keyboard key |
| | `RoutineWaitForUrlOperation` | Wait for URL pattern match |
| | `RoutineScrollOperation` | Scroll page |
| | `RoutineReturnHTMLOperation` | Return page/element HTML |
| | `RoutineJsEvaluateOperation` | Execute custom IIFE JavaScript |
| | `RoutineOperationUnion` | Discriminated union for deserialization |
| `bluebox/data_models/routine/parameter.py` | `Parameter` | User input: `name`, `type` (string/integer/number/boolean/date/enum), `required`, `description`, `observed_value`, validation constraints |
| | `ParameterType` | StrEnum: `STRING`, `INTEGER`, `NUMBER`, `BOOLEAN`, `DATE`, `DATETIME`, `EMAIL`, `URL`, `ENUM` |
| | `VALID_PLACEHOLDER_PREFIXES` | frozenset: `{sessionStorage, localStorage, cookie, meta, windowProperty}` |
| | `BUILTIN_PARAMETERS` | Auto-generated: `uuid`, `epoch_milliseconds` |
| `bluebox/data_models/routine/placeholder.py` | `extract_placeholders_from_json_str()` | Regex extraction of `{{...}}` patterns from JSON strings |
| `bluebox/data_models/routine/endpoint.py` | `HTTPMethod` | StrEnum: GET, POST, PUT, DELETE, PATCH, etc. |
| | `Endpoint` | Fetch target: `url`, `method`, `headers`, `body`, `credentials` |

---

### 8. Utility Modules

| File | Function/Class | Purpose |
|------|---------------|---------|
| `bluebox/utils/data_utils.py` | `resolve_dotted_path(logger, obj, path)` | Traverses nested JSON structures via dot notation (e.g., `"data.user.token"`). Used to validate `dot_path` in `record_resolved_variable`. |
| | `apply_params_to_str(text, params)` | Simple `{{param}}` substitution in strings (URLs, selectors) |
| | `apply_params_to_json(d, params, type_map)` | Recursive placeholder resolution with type coercion |
| `bluebox/llms/tools/execute_routine_tool.py` | `execute_routine(routine, parameters, remote_debugging_address, ...)` | Unified entry point: parses routine dict/JSON, creates executor, runs against browser, returns `{success, result/error}` |
| `bluebox/utils/cli_utils.py` | `add_model_argument(parser)` | Adds `--model` arg to argparse |
| | `resolve_model(model_str, console)` | Validates and returns `LLMModel` enum |
| `bluebox/data_models/llms/interaction.py` | `Chat`, `ChatThread` | Persistent message storage (role, content, tool_calls) |
| | `EmittedMessage` types | Agent-to-host communication: `ChatResponseEmittedMessage`, `ToolInvocationResultEmittedMessage`, `ErrorEmittedMessage`, `StatusUpdateEmittedMessage` |
| `bluebox/data_models/llms/vendors.py` | `LLMModel`, `OpenAIModel`, `AnthropicModel` | Model enum definitions and resolution |

---

### 9. TUI Layer

| File | Class | Purpose |
|------|-------|---------|
| `bluebox/scripts/run_routine_discovery_agent_beta.py` | `RotutineDiscoveryBetaTUI` | Textual multi-pane TUI. Handles `/discover`, `/execute`, `/routine`, `/save` commands. Manages agent lifecycle. |
| `bluebox/utils/tui_base.py` | `AbstractAgentTUI` | Base Textual app: two-column layout (chat + tools/status), message routing, streaming, slash commands, context tracking. |

**TUI responsibilities:**
- **CLI parsing** — Resolves JSONL paths from `--cdp-captures-dir` or individual `--*-jsonl` flags
- **Data loading** — Creates all data loaders, including `DocumentationDataLoader` from package source
- **Agent creation** — Instantiates `RoutineDiscoveryAgentBeta` with all loaders
- **Message routing** — `_handle_message()` dispatches `EmittedMessage` subtypes to chat log, tool pane, status bar
- **State dumping** — On every message, dumps chat threads and state snapshots to `output_dir/` for debugging
- **Auto-save** — Saves `routine.json` and `test_parameters.json` on successful discovery

**Slash commands:**
| Command | Action |
|---------|--------|
| `/discover <task>` | Create new agent, run `agent.run()` in background thread |
| `/execute` | Execute discovered routine against live browser |
| `/routine` | Display routine details in chat |
| `/save <path>` | Save routine JSON to file |
| `/status` | Show compact status summary |
| `/reset` | Clear agent and start fresh |

---

## Data Flow Diagram

```
CDP Captures (JSONL files)
    │
    ▼
Data Loaders (NetworkDataLoader, StorageDataLoader, etc.)
    │
    ├──────────────────────────────────────────┐
    ▼                                          ▼
RoutineDiscoveryAgentBeta              Specialist Agents
(orchestrator)                         (autonomous workers)
    │                                          │
    │  create_task(agent_type, prompt)          │
    │ ──────────────────────────────────►       │
    │                                   run_autonomous()
    │  ◄─────────────────────────────────────── │
    │  get_task_result(task_id)                 │
    │                                          │
    ▼                                          │
RoutineDiscoveryState                          │
    │  root_transaction                        │
    │  transaction_queue (BFS)                 │
    │  extracted_variables                     │
    │  resolved_variables                      │
    │                                          │
    ▼                                          │
construct_routine()                            │
    │                                          │
    ▼                                          │
Routine (model_validate)                       │
    │                                          │
    ▼  (if browser connected)                  │
validate_routine()                             │
    │  → execute_routine()                     │
    │  → analyze_validation()                  │
    │                                          │
    ▼                                          │
done() → Routine JSON output
```

---

## Tool Inventory

### Orchestrator Tools (RoutineDiscoveryAgentBeta)

| Tool | Phase | Purpose |
|------|-------|---------|
| `create_task` | Any | Delegate work to a specialist |
| `run_pending_tasks` | Any | Execute all pending tasks (parallel) |
| `list_tasks` | Any | Show task statuses |
| `get_task_result` | Any | Read completed task result |
| `list_transactions` | Any | List network transaction IDs (prefer delegating to specialist) |
| `get_transaction` | Any | Get full transaction details |
| `scan_for_value` | Any | Basic value search across all data sources |
| `record_identified_endpoint` | PLANNING | Set root transaction |
| `record_extracted_variable` | DISCOVERING | Log a variable from a transaction |
| `record_resolved_variable` | DISCOVERING | Record where a dynamic token comes from |
| `mark_transaction_processed` | DISCOVERING | Mark transaction as fully analyzed |
| `get_discovery_context` | CONSTRUCTING | Get all discovered data for routine building |
| `validate_placeholders` | CONSTRUCTING | Check placeholder syntax before constructing |
| `construct_routine` | CONSTRUCTING | Build and validate a Routine |
| `validate_routine` | VALIDATING | Execute routine against live browser |
| `analyze_validation` | VALIDATING | Reflect on validation results |
| `done` | COMPLETE | Mark discovery as successful |
| `fail` | FAILED | Mark discovery as failed |

### Tool Availability Gating

Many tools use dynamic availability via lambdas:
- `record_extracted_variable` — only after root_transaction is set
- `construct_routine` — only when root_transaction set AND transaction_queue is empty
- `validate_routine` — only when routine exists AND browser connected
- `analyze_validation` — only when validation result exists AND not yet analyzed
- `done` — only when `_can_complete()` returns True (routine exists + validated if browser)
- `fail` — only before root_transaction set OR after 5+ construction attempts

---

## Key Design Patterns

### 1. Orchestrator-Specialist Delegation
The orchestrator never does deep analysis itself. It creates tasks, runs them, and synthesizes results. This keeps the orchestrator's context window focused on high-level reasoning.

### 2. Dynamic Tool Availability
Tools appear/disappear based on agent state. The LLM only sees tools relevant to the current phase, reducing confusion and invalid tool calls.

### 3. BFS Transaction Processing
When a dynamic token traces back to another transaction, that transaction is auto-added to the processing queue. This discovers the full dependency chain (e.g., auth flow → token endpoint → main API call).

### 4. Phase-Scoped System Prompts
The system prompt changes per phase — only the current phase's instructions are included. This focuses the LLM on the immediate task rather than overwhelming it with all possible actions.

### 5. Forced Tool Use
`tool_choice="required"` ensures the LLM always calls a tool, preventing it from generating text-only responses that stall progress.

### 6. State Snapshotting (TUI)
The TUI dumps state to disk on every message, creating a timeline of discovery progress for debugging and analysis.

---

## Known Architectural Issues

These are observations for the future refactor, not bugs:

1. **RoutineDiscoveryAgentBeta extends AbstractAgent, not AbstractSpecialist** — It has its own `run()` loop instead of using `run_autonomous()`. This means it doesn't benefit from the specialist's iteration tracking, urgency notices, or finalize gating.

2. **Phase transitions are scattered** — Phase changes happen in multiple places: `_create_task` sets DISCOVERING, `_record_identified_endpoint` sets DISCOVERING, `_run_pending_tasks` auto-transitions to CONSTRUCTING, `_construct_routine` sets CONSTRUCTING, `_validate_routine` sets VALIDATING. There's no centralized state machine.

3. **Two parallel state objects** — `AgentOrchestrationState` (tasks/subagents) and `RoutineDiscoveryState` (discovery-specific) are separate but tightly coupled. The orchestrator must manually keep them in sync.

4. **Specialist lifecycle is implicit** — Agents are created lazily in `_get_or_create_agent` and stored in `_agent_instances`. There's no explicit lifecycle management (cleanup, context limits, rotation).

5. **The no-tool-call fallback** — When the LLM responds without tools, the orchestrator injects a guidance SYSTEM message. This workaround suggests the system prompts could be more directive.

6. **Validation is optional** — If no browser is connected, the routine is returned without any execution test. The `_can_complete()` logic has different paths for browser vs no-browser.

7. **TUI class name typo** — `RotutineDiscoveryBetaTUI` (missing an 'i' in "Routine"). Same typo in log messages: "RotutineDiscoveryBeta".

8. **Direct state access in TUI** — The TUI directly accesses `agent._discovery_state`, `agent._orchestration_state`, `agent._agent_instances`, etc. These are internal attributes, not a public API.

9. **Task max_loops default mismatch** — `Task.max_loops` defaults to 5, but `create_task` passes `max_loops=15` by default. The actual default depends on the call path.

10. **Legacy phases in DiscoveryPhase** — The enum contains both beta phases (`PLANNING`, `DISCOVERING`, etc.) and legacy phases (`IDENTIFY_TRANSACTION`, `PROCESS_QUEUE`, etc.) from the original `RoutineDiscoveryAgent`.
