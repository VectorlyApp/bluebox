# RoutineInspector

Independent quality gate that judges constructed routines. It has **no browser tools and no capture lookup tools** — it cannot interact with the live site or query raw CDP data. All factual context arrives in a single task prompt. It scores on 6 dimensions and returns a structured `RoutineInspectionResult`.

Think of it as a peer reviewer: reads the routine cold, checks if the claims hold up, and decides — ship, revise, or reject.

## Tools

The inspector has no domain-specific tools (no browser, no capture lookup). Its only tools come from `AbstractSpecialist`:

| Tool | Purpose |
|------|---------|
| `finalize_with_output(output)` | Submit the structured `RoutineInspectionResult`. The output is **validated at runtime against `INSPECTION_OUTPUT_SCHEMA`** using `jsonschema.validate()` — if the output doesn't match the schema, the tool returns a validation error and forces the LLM to retry. This is how structured output is enforced without using vendor-level structured output mode. Available after `min_iterations` (3). |
| `finalize_with_failure(reason)` | Mark inspection as failed. **Blocked** if a previous `finalize_with_output` call failed validation — the LLM must fix and retry instead of giving up. |
| `add_note(note)` | Attach warnings or observations to the result. |
| `search_docs(query)` | *(Optional — only available if `documentation_data_loader` is provided.)* Search agent documentation for remediation patterns (e.g. CORS fixes, auth patterns). Used to provide specific fix recommendations for blocking issues. |
| `get_doc_file(path)` | *(Optional.)* Read a specific documentation file. |

**How structured output works:** The PI dispatches the inspector with `output_schema=INSPECTION_OUTPUT_SCHEMA`. `AbstractSpecialist.run_autonomous()` stores this schema, injects it into the inspector's system prompt (so the LLM sees the exact JSON Schema it must produce), and wires it into the `finalize_with_output` tool's parameter definition (so the LLM sees the required fields in the tool signature). When the inspector calls the tool, `jsonschema.validate()` enforces the schema. This is the same mechanism available to experiment workers via `output_schema` — but the inspector is the only agent that actually receives one (see [potential improvement #12](../api_indexing_spec_v2/potential_improvements.md#12-experiment-workers-return-freeform-output--pi-never-specifies-output-schemas)).

## What It Sees

All factual context arrives in the task prompt — the inspector has no tools to fetch additional data:

| Context | Contents |
|---------|----------|
| Routine JSON | Full routine: name, description, parameters, operations |
| Execution result | HTTP status codes per operation, response data, unresolved placeholders, warnings |
| Test parameters | Values used for the test execution |
| Exploration summaries | Network, DOM, storage summaries for cross-reference |
| Agent docs | *(Optional, via tools)* Markdown docs on operation types, common errors, placeholder syntax — used for actionable remediation advice |

## What It Returns

`RoutineInspectionResult` (schema-validated via `finalize_with_output`):

```json
{
  "overall_pass": false,
  "overall_score": 37,
  "dimensions": {
    "task_completion": {"score": 1, "reasoning": "..."},
    "data_quality": {"score": 1, "reasoning": "..."},
    "parameter_coverage": {"score": 9, "reasoning": "..."},
    "routine_robustness": {"score": 2, "reasoning": "..."},
    "structural_correctness": {"score": 5, "reasoning": "..."},
    "documentation_quality": {"score": 7, "reasoning": "..."}
  },
  "blocking_issues": ["HTTP 401 on availability endpoint..."],
  "recommendations": ["Add auth token acquisition step..."],
  "summary": "The routine fails at runtime due to missing auth..."
}
```

## 6 Scoring Dimensions (0-10 each)

### 1. Task Completion

Does the returned data **ACTUALLY** accomplish what the routine promises?

- Check the REAL execution result, not hypotheticals
- HTTP 4xx/5xx = automatic score <= 2
- Only score above 5 if the response contains ACTUAL meaningful data

### 2. Data Quality

Is the **ACTUAL** response complete and meaningful?

- A 401 error body is not "data" = score 0
- Truncated, empty, or error responses = score 0-3
- Only score above 5 if response contains REAL, COMPLETE data

### 3. Parameter Coverage

Are the right values parameterized?

- User-facing inputs (search terms, dates, IDs) should be parameters
- Site-level constants (API keys, subscription keys) should be hardcoded
- Score based on design, not execution result

### 4. Routine Robustness

Would this work in a fresh session?

- Unresolved placeholders = score <= 4
- Failed auth tokens = score <= 3
- Dynamic tokens must be fetched at runtime, not hardcoded

### 5. Structural Correctness

Operations in the right order?

- Navigate before fetch
- Token acquisition before data fetch
- session_storage_key written before read
- Valid placeholder syntax

### 6. Documentation Quality

Will other agents find and understand this routine in the database?

**Routine name** (0-3 points):
- snake_case, verb_site_noun pattern, >= 3 segments
- MUST include site/service name
- GOOD: `get_premierleague_standings` — BAD: `get_standings`

**Routine description** (0-4 points):
- >= 8 words
- Must explain: what it does, what inputs it takes, what data it returns

**Parameter descriptions** (0-3 points):
- >= 3 words each
- Non-obvious params (IDs, slugs) MUST explain where to get valid values

## Verdict Rules

- `overall_pass = true` if: no blocking_issues AND overall_score >= 60
- `overall_score = round(sum of 6 dimensions / 60 * 100)`
- `documentation_quality <= 4` → automatic blocking issue
- HTTP 4xx/5xx in execution → automatic blocking issue
- Unresolved placeholders → automatic blocking issue

## Critical Rule: Judge Actual Results, Not Hypotheticals

The inspector scores based on **WHAT ACTUALLY HAPPENED**, not what "would work if...":

- If execution returned a 401 → the routine FAILED. Score task_completion and data_quality <= 2.
- "It would return rich data with valid credentials" is speculation, not inspection.
- Test parameters like `"REPLACE_WITH_..."` mean the routine wasn't properly tested.
- A broken routine is WORSE than no routine — it wastes database space and misleads other agents.

## Automatic Failure Signals

Any of these → task_completion <= 2 AND data_quality <= 2:

- HTTP 4xx or 5xx status codes in ANY operation
- Unresolved placeholders in warnings
- Error messages in response body ("Access denied", "Unauthorized", etc.)
- Placeholder test parameters ("REPLACE_WITH_...", "TODO", "YOUR_..._HERE")
- Empty or null response data

## File

`bluebox/agents/routine_inspector.py`
