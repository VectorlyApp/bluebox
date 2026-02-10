# SuperDiscoveryAgent Audit Report

**Date:** 2026-02-09

**Methodology:** Source code review + live agent interviews via HTTP adapter

---

## 1. System Prompt Issues

### 1.6. `construct_routine` tool schema is too sparse **[CODE]**

The tool's JSON Schema for `operations` (lines 1683-1694) uses `"items": {"type": "object"}` with no property constraints. The description crams the entire operation schema into a single string: `"navigate|fetch|return|sleep|click|input_text|press|..."`. This means the LLM must reconstruct the full schema from a terse description string, increasing the chance of malformed routines. Compare this to the documentation files which contain detailed schemas — but the LLM doesn't know to consult those unless it happens to call `search_docs`.

---

## 2. Tool Gaps and Limitations

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

---

## 3. Orchestration and Coordination Problems

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

## 5. Variable Resolution and Placeholder Handling

### 5.1. Resolution preference rule is hard to enforce reliably **[BOTH]**

The system prompt says to prefer `source_type='transaction'` over storage when both match. But the orchestrator delegates value tracing to `ValueTraceResolverSpecialist`, which returns its findings — and the specialist's prompt doesn't contain this preference rule. The specialist may report storage as the source, and the orchestrator must override. There's no mechanism ensuring the preference is applied consistently.

### 5.2. No validation of placeholder syntax before routine construction **[CODE]**

Placeholder syntax errors (missing escaped quotes, wrong prefix) are only caught when `Routine.model_validate()` runs inside `_construct_routine`. There's no tool for the LLM to validate individual placeholder expressions before assembling the full routine. This means the LLM must get it right on the first try or iterate through construction failures.

### 5.3. `record_resolved_variable` doesn't validate the source exists **[CODE]**

When `source_type='transaction'` is used, the tool auto-adds the source transaction to the queue and initializes its data (lines 1459-1475). But it doesn't verify that the specified `dot_path` actually resolves to the expected value in the source transaction's response. The LLM could record an incorrect path and discover the error only during routine execution.

---

## 6. Remaining Improvements

### 1. Fix JSSpecialist data loader wiring **[HIGH IMPACT]**

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

### 2. Add `construct_routine` schema documentation or a schema-check tool **[MEDIUM IMPACT]**

**Problem:** The `construct_routine` tool accepts `"items": {"type": "object"}` for both parameters and operations arrays — essentially no schema validation from the tool definition. The LLM must guess the correct operation structure from a terse description string.

**Fix:** Either (a) expand the JSON Schema in the tool definition to include proper sub-schemas for each operation type (navigate, fetch, return, etc.), or (b) add a `validate_routine_draft` tool that validates a partial routine and returns specific schema errors before the LLM commits to `construct_routine`. The documentation files already contain detailed schemas — automatically including the relevant schema in the CONSTRUCTING phase prompt would also help.

**Files:** `super_discovery_agent.py` lines 1667-1706

---

### Honorable Mentions

- **No request-side search on orchestrator**: Add request header/body search to `scan_for_value`, or document that delegation to NetworkSpecialist is required
- **ValueTraceResolver doesn't know the "prefer transaction over storage" rule**: Add it to its system prompt (`value_trace_resolver_specialist.py:AUTONOMOUS_SYSTEM_PROMPT`)
