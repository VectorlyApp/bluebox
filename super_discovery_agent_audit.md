# SuperDiscoveryAgent Audit Report

**Date:** 2026-02-09

**Methodology:** Source code review + live agent interviews via HTTP adapter

---

## 1. System Prompt Issues

### 1.1. Prompt is excessively long and repetitive **[CODE]**

The `SYSTEM_PROMPT` constant (lines 102-236) is ~135 lines of dense instructions, plus `PLACEHOLDER_INSTRUCTIONS` (lines 81-100) — all injected into every single LLM call alongside dynamic state sections. Key problems:

- **Variable classification rules** are stated three times: in the main workflow text (Phase 2, step 2), in the dedicated "Variable Classification Rules" section (lines 182-198), and again in the "Important Notes" section (lines 226-234 re: "prefer network sources").
- The **placeholder syntax** section (lines 81-100) is 20 lines of dense examples. It's critical for routine construction but irrelevant during earlier phases (PLANNING, DISCOVERING). Every iteration pays the token cost regardless.
- **Phase-specific instructions** are all presented upfront even though only one phase is active at a time. The LLM in PLANNING phase has to mentally filter out CONSTRUCTING and VALIDATING instructions.

### 1.2. Dynamic state appended to system prompt on every call **[CODE]**

`_get_system_prompt()` (lines 315-370) rebuilds the full system prompt each iteration, appending data source summaries, orchestration state, discovery progress, and browser status. This is correct architecturally, but combined with the verbose base prompt, it creates a large system message that grows over time.

### 1.3. Tool descriptions say "PREFER specialist" but the tools still exist **[CODE]**

Several tools have discouraging prefixes:

- `list_transactions`: `"[PREFER network_specialist] List transaction IDs..."` (line 985)
- `scan_for_value`: `"[PREFER value_trace_resolver SPECIALIST] Basic value search..."` (line 1051)

These are soft warnings embedded in tool descriptions. The LLM may still use them directly, and the wording is awkward — tool descriptions should describe what the tool does, not nag the LLM about preferred alternatives.

### 1.4. JSSpecialist system prompt references a nonexistent tool **[CODE]**

`js_specialist.py` line 91: the `SYSTEM_PROMPT` lists `submit_js_code` as a tool ("submit final validated code"), but **no such tool exists** in the class. The LLM will try to call it and get an error. The actual finalization tools are the generic `finalize_result`/`finalize_with_output` from `AbstractSpecialist`.

### 1.5. `add_note()` referenced in prompt but not exposed as a tool **[CODE]**

`abstract_specialist.py` line 321: The output schema prompt section tells the LLM to "Use `add_note()` before finalizing to record any notes." But `add_note()` is a Python method (line 286), not an `@agent_tool`. The LLM cannot call it. Notes can only be added programmatically, making this prompt instruction misleading.

### 1.6. `construct_routine` tool schema is too sparse **[CODE]**

The tool's JSON Schema for `operations` (lines 1683-1694) uses `"items": {"type": "object"}` with no property constraints. The description crams the entire operation schema into a single string: `"navigate|fetch|return|sleep|click|input_text|press|..."`. This means the LLM must reconstruct the full schema from a terse description string, increasing the chance of malformed routines. Compare this to the documentation files which contain detailed schemas — but the LLM doesn't know to consult those unless it happens to call `search_docs`.

---

## 2. Tool Gaps and Limitations

### 2.1. JSSpecialist created without any data loaders **[CODE]**

When `SuperDiscoveryAgent._create_specialist()` creates a `JSSpecialist` (lines 523-529), it only passes `emit_message_callable`, `llm_model`, `remote_debugging_address`, and `run_mode`. It does **not** pass:

- `network_data_store` — all network search tools disabled
- `js_data_store` — all JS file search tools disabled
- `dom_snapshots` — DOM snapshot tool disabled

This means a JSSpecialist spawned by the orchestrator is essentially crippled — it can validate JS syntax and execute in browser (if connected), but can't search network traffic, inspect JS files, or see DOM snapshots. It's a code formatter, not a specialist.

### 2.2. No tool to search request headers specifically **[BOTH]**

The `NetworkSpecialist` has `search_responses_by_terms` (searches response bodies) and `search_requests_by_terms` (searches URL + headers + body). But the SuperDiscoveryAgent's direct `scan_for_value` tool (lines 1068-1131) only searches:

- Response bodies
- Response headers
- Storage events
- Window properties

It **does not search request headers or request bodies**. If a dynamic token appears in a request header (like `Authorization: Bearer ...`), `scan_for_value` won't find it in the request side — only in responses and storage. The NetworkSpecialist can search requests, but the orchestrator's direct tools can't.

### 2.3. `search_response_bodies` tool on NetworkSpecialist does exact match only **[CODE]**

`_search_response_bodies` (network_specialist.py) does case-insensitive exact string matching. There's no fuzzy/partial matching, no regex support for response body search. For values that appear in slightly different forms (URL-encoded, base64-encoded, truncated), this won't find them.

### 2.4. No tool for the orchestrator to search request bodies/headers directly **[CODE]**

The orchestrator (`SuperDiscoveryAgent`) has `get_transaction` which returns the full request details, and `scan_for_value` which searches responses and storage. But there's no direct tool for the orchestrator to do a bulk search across all request bodies or headers. It must delegate to `NetworkSpecialist` for that.

### 2.5. Response body truncation at 5000 chars without indication of loss **[CODE]**

`get_transaction` (line 1047) truncates `response_body` to 5000 chars without telling the LLM how much was lost or offering pagination. For large JSON responses (common with API calls), the truncation could cut off critical fields. Compare to `NetworkSpecialist._get_entry_detail` which adds a truncation message.

---

## 3. Orchestration and Coordination Problems

### 3.1. Availability gating creates an invisible tool problem **[BOTH]**

The SuperDiscoveryAgent confirmed in interview that it sees only 14 tools initially. Tools like `record_extracted_variable`, `record_resolved_variable`, `mark_transaction_processed`, `construct_routine`, `validate_routine`, `analyze_validation`, and `done` are **gated behind availability lambdas** that require state changes (e.g., `root_transaction is not None`).

The problem: The system prompt's workflow instructions reference these tools extensively, but the LLM **cannot see them in its tool list** until preconditions are met. This creates confusion — the agent described some tools as "conceptual steps" it couldn't actually invoke. The `_sync_tools()` call before each LLM call does re-register tools, so they appear after state changes, but:

- The LLM doesn't know tools will appear later
- The workflow text assumes all tools exist from the start
- There's no mechanism to tell the LLM "these tools will become available after you do X"

### 3.2. Sequential task execution bottleneck **[CODE]**

`_run_pending_tasks()` (lines 900-980) executes tasks **sequentially** in a for loop. If the orchestrator creates multiple independent tasks (e.g., a NetworkSpecialist search and a ValueTraceResolver trace), they run one after another despite being independent. This doubles latency needlessly.

### 3.3. No mechanism to pass context between specialists **[CODE]**

When the orchestrator creates tasks for different specialists, each specialist starts fresh with only its data loaders and the task prompt. There's no way to pass the NetworkSpecialist's findings directly to the ValueTraceResolver. The orchestrator must:

1. Get NetworkSpecialist results
2. Parse them itself
3. Formulate a new prompt for ValueTraceResolver including the relevant details

This is error-prone and depends on the orchestrator LLM correctly relaying information between specialists.

### 3.4. `tool_choice="required"` forces unnecessary tool calls **[CODE]**

Both the SuperDiscoveryAgent (line 409) and the autonomous specialist loop (line 558) use `tool_choice="required"`, which forces the LLM to call a tool every iteration. When the LLM wants to reason or explain before acting, it's forced to make a tool call anyway. The SuperDiscoveryAgent has a fallback (lines 434-481) that injects guidance when no tool calls happen, but this should be rare with `tool_choice="required"`.

### 3.5. Phase transitions are implicit and fragile **[CODE]**

Phase transitions happen as side effects of tool calls scattered throughout the code:

- `_create_task` sets phase to `DISCOVERING` (line 820)
- `_record_identified_endpoint` sets phase to `DISCOVERING` (line 1224)
- `_run_pending_tasks` may set phase to `CONSTRUCTING` (line 964)
- `_construct_routine` sets phase to `CONSTRUCTING` (line 1719)
- `_validate_routine` sets phase to `VALIDATING` (line 1794)
- `_done` sets phase to `COMPLETE` (line 2006)

There's no central state machine. The phase can be set from multiple code paths, and some transitions are duplicated (both `_create_task` and `_record_identified_endpoint` set DISCOVERING). If a tool is called out of expected order, the phase may not make sense.

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

### 5.1. Resolution preference rule is hard to enforce reliably **[BOTH]**

The system prompt says to prefer `source_type='transaction'` over storage when both match. But the orchestrator delegates value tracing to `ValueTraceResolverSpecialist`, which returns its findings — and the specialist's prompt doesn't contain this preference rule. The specialist may report storage as the source, and the orchestrator must override. There's no mechanism ensuring the preference is applied consistently.

### 5.2. No validation of placeholder syntax before routine construction **[CODE]**

Placeholder syntax errors (missing escaped quotes, wrong prefix) are only caught when `Routine.model_validate()` runs inside `_construct_routine`. There's no tool for the LLM to validate individual placeholder expressions before assembling the full routine. This means the LLM must get it right on the first try or iterate through construction failures.

### 5.3. `record_resolved_variable` doesn't validate the source exists **[CODE]**

When `source_type='transaction'` is used, the tool auto-adds the source transaction to the queue and initializes its data (lines 1459-1475). But it doesn't verify that the specified `dot_path` actually resolves to the expected value in the source transaction's response. The LLM could record an incorrect path and discover the error only during routine execution.

### 5.4. Single observed value makes classification unreliable **[AGENT]**

Both agents highlighted this: with only one captured example of each value, distinguishing PARAMETER from DYNAMIC_TOKEN from STATIC_VALUE relies entirely on naming heuristics. The agent said: "With only one observed value, I can't see variability, so I rely on naming and format heuristics."

---

## 6. Top 5 Concrete Improvements

### 1. Phase-scoped system prompts **[HIGH IMPACT]**

**Problem:** The 135-line system prompt includes instructions for all phases, wasting tokens and confusing the LLM.

**Fix:** Split the system prompt into phase-specific sections. `_get_system_prompt()` should include only the instructions relevant to the current `DiscoveryPhase`. Placeholder syntax instructions should only appear during CONSTRUCTING phase. Variable classification rules only during DISCOVERING. This would reduce prompt size by ~50% in most iterations and make instructions more focused.

**Files:** `super_discovery_agent.py` lines 102-236, 315-370

### 2. Fix JSSpecialist data loader wiring **[HIGH IMPACT]**

**Problem:** JSSpecialist is created without `network_data_store`, `js_data_store`, or `dom_snapshots`, disabling most of its tools.

**Fix:** In `_create_specialist()` (line 523), pass the available data loaders:

```python
return JSSpecialist(
    emit_message_callable=self._emit_message_callable,
    llm_model=self._subagent_llm_model,
    network_data_store=self._network_data_loader,
    js_data_store=self._js_data_loader,
    remote_debugging_address=self._remote_debugging_address,
    run_mode=RunMode.AUTONOMOUS,
)
```

Also fix the phantom `submit_js_code` tool reference in the JSSpecialist system prompt (line 91).

**Files:** `super_discovery_agent.py` lines 523-529, `js_specialist.py` line 91

### 3. Make hidden tools visible via documentation in the prompt **[HIGH IMPACT]**

**Problem:** Availability-gated tools are invisible to the LLM but referenced in workflow instructions.

**Fix:** Add a "Tools by Phase" section to the system prompt that lists all tools and their activation conditions. For example:

```text
## Tools Available by Phase
- PLANNING: create_task, run_pending_tasks, list_transactions, scan_for_value, ...
- DISCOVERING (after record_identified_endpoint): + record_extracted_variable, record_resolved_variable, mark_transaction_processed
- CONSTRUCTING (after all transactions processed): + construct_routine, get_discovery_context
- VALIDATING (after construct_routine): + validate_routine, analyze_validation
- Any time after routine constructed: + done
```

This lets the LLM plan ahead knowing what tools will appear when.

**Files:** `super_discovery_agent.py` lines 102-236

### 4. Parallel task execution **[MEDIUM IMPACT]**

**Problem:** `_run_pending_tasks()` executes tasks sequentially. Two independent specialist tasks take 2x the time they should.

**Fix:** Use `ThreadPoolExecutor` (already imported in `abstract_agent.py`) to execute independent tasks concurrently. The infrastructure for parallel execution already exists in `_process_tool_calls`.

**Files:** `super_discovery_agent.py` lines 900-980

### 5. Add `construct_routine` schema documentation or a schema-check tool **[MEDIUM IMPACT]**

**Problem:** The `construct_routine` tool accepts `"items": {"type": "object"}` for both parameters and operations arrays — essentially no schema validation from the tool definition. The LLM must guess the correct operation structure from a terse description string.

**Fix:** Either (a) expand the JSON Schema in the tool definition to include proper sub-schemas for each operation type (navigate, fetch, return, etc.), or (b) add a `validate_routine_draft` tool that validates a partial routine and returns specific schema errors before the LLM commits to `construct_routine`. The documentation files already contain detailed schemas — automatically including the relevant schema in the CONSTRUCTING phase prompt would also help.

**Files:** `super_discovery_agent.py` lines 1667-1706

---

### Honorable Mentions

- **`add_note()` is not a tool**: Either expose it as `@agent_tool` or remove the prompt reference (`abstract_specialist.py:286,321`)
- **Response body truncation**: `get_transaction` should indicate truncation length like `get_entry_detail` does (`super_discovery_agent.py:1047`)
- **No request-side search on orchestrator**: Add request header/body search to `scan_for_value`, or document that delegation to NetworkSpecialist is required
- **ValueTraceResolver doesn't know the "prefer transaction over storage" rule**: Add it to its system prompt (`value_trace_resolver_specialist.py:AUTONOMOUS_SYSTEM_PROMPT`)
