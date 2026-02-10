# SuperDiscoveryAgent Audit — Deferred Items

Items excised from the active audit (`super_discovery_agent_audit.md`) for later triage.
Sourced from the original audit (`og_super_discovery_agent_audit.md`, 2026-02-09).

---

## 1. System Prompt Issues

### 1.2. Dynamic state appended to system prompt on every call **[CODE]**

`_get_system_prompt()` (lines 315-370) rebuilds the full system prompt each iteration, appending data source summaries, orchestration state, discovery progress, and browser status. This is correct architecturally, but combined with the verbose base prompt, it creates a large system message that grows over time.

### 1.3. Tool descriptions say "PREFER specialist" but the tools still exist **[CODE]**

Several tools have discouraging prefixes:

- `list_transactions`: `"[PREFER network_specialist] List transaction IDs..."` (line 985)
- `scan_for_value`: `"[PREFER value_trace_resolver SPECIALIST] Basic value search..."` (line 1051)

These are soft warnings embedded in tool descriptions. The LLM may still use them directly, and the wording is awkward — tool descriptions should describe what the tool does, not nag the LLM about preferred alternatives.

### 1.4. JSSpecialist system prompt references a nonexistent tool **[CODE]**

`js_specialist.py` line 91: the `SYSTEM_PROMPT` lists `submit_js_code` as a tool ("submit final validated code"), but **no such tool exists** in the class. The LLM will try to call it and get an error. The actual finalization tools are the generic `finalize_result`/`finalize_with_output` from `AbstractSpecialist`.

---

## 2. Tool Gaps and Limitations

### 2.1. JSSpecialist created without any data loaders **[CODE]**

When `SuperDiscoveryAgent._create_specialist()` creates a `JSSpecialist` (lines 523-529), it only passes `emit_message_callable`, `llm_model`, `remote_debugging_address`, and `run_mode`. It does **not** pass:

- `network_data_store` — all network search tools disabled
- `js_data_store` — all JS file search tools disabled
- `dom_snapshots` — DOM snapshot tool disabled

This means a JSSpecialist spawned by the orchestrator is essentially crippled — it can validate JS syntax and execute in browser (if connected), but can't search network traffic, inspect JS files, or see DOM snapshots. It's a code formatter, not a specialist.

---

## 3. Orchestration and Coordination Problems

### 3.3. No mechanism to pass context between specialists **[CODE]**

When the orchestrator creates tasks for different specialists, each specialist starts fresh with only its data loaders and the task prompt. There's no way to pass the NetworkSpecialist's findings directly to the ValueTraceResolver. The orchestrator must:

1. Get NetworkSpecialist results
2. Parse them itself
3. Formulate a new prompt for ValueTraceResolver including the relevant details

This is error-prone and depends on the orchestrator LLM correctly relaying information between specialists.

### 3.4. `tool_choice="required"` forces unnecessary tool calls **[CODE]**

Both the SuperDiscoveryAgent (line 409) and the autonomous specialist loop (line 558) use `tool_choice="required"`, which forces the LLM to call a tool every iteration. When the LLM wants to reason or explain before acting, it's forced to make a tool call anyway. The SuperDiscoveryAgent has a fallback (lines 434-481) that injects guidance when no tool calls happen, but this should be rare with `tool_choice="required"`.

---

## 4. Data Access and Visibility

### 4.1. NetworkSpecialist lacks access to storage/window data **[BOTH]**

The NetworkSpecialist only receives `network_data_store`. When it finds an endpoint and a value appears in a request header, it can't check whether that value comes from a cookie or localStorage — it must punt back to the orchestrator for that. The agent itself noted: "I don't have browser console logs, DOM snapshots, screenshots, or local storage / cookies (unless they appear in headers)."

### 4.2. No temporal ordering exposed to the orchestrator **[AGENT]**

The SuperDiscoveryAgent told us that timing/ordering of transactions is approximated. The `NetworkDataLoader` has entries in capture order, but the orchestrator's `list_transactions` and `get_transaction` tools don't surface timestamps. The ValueTraceResolver's `get_network_entry` does include `entry.timestamp`, but the orchestrator can't easily determine "which request came first" without delegating.

### 4.3. No visibility into request initiator (what triggered the request) **[AGENT]**

Both agents noted this gap. There's no data mapping UI actions to network requests. The NetworkSpecialist said: "I wish I had an event log: `(timestamp, UI action, request_id)`." The interaction data loader captures UI events separately, but there's no correlation between interaction events and network requests.

---

## 5. Variable Resolution and Placeholder Handling

### 5.4. Single observed value makes classification unreliable **[AGENT]**

Both agents highlighted this: with only one captured example of each value, distinguishing PARAMETER from DYNAMIC_TOKEN from STATIC_VALUE relies entirely on naming heuristics. The agent said: "With only one observed value, I can't see variability, so I rely on naming and format heuristics."
