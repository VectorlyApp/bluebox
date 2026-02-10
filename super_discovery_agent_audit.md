# SuperDiscoveryAgent Audit Report

**Date:** 2026-02-09

**Methodology:** Source code review + live agent interviews via HTTP adapter

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

### 5.2. No validation of placeholder syntax before routine construction **[CODE]**

Placeholder syntax errors (missing escaped quotes, wrong prefix) are only caught when `Routine.model_validate()` runs inside `_construct_routine`. There's no tool for the LLM to validate individual placeholder expressions before assembling the full routine. This means the LLM must get it right on the first try or iterate through construction failures.

---

### Honorable Mentions

(None remaining)
