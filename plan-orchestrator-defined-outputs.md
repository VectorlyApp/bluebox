# Plan: Dynamic Output Schema for Specialist Agents

## Summary

Allow `SuperDiscoveryAgent` (the orchestrator LLM) to **dynamically define** what result structure a specialist should return at task creation time. The orchestrator decides the schema on each call based on what information it needs. Results are wrapped in a universal container with notes.

## Current State

- Each specialist defines its own result models (e.g., `EndpointDiscoveryResult`)
- Results are returned via typed `finalize_result(endpoints: list[dict])` tools
- No way to pass expected schema from orchestrator to specialist
- No universal wrapper for execution metadata

## Design

### 1. Universal Result Wrapper

Create `SpecialistResultWrapper` in a new file:

```python
class SpecialistResultWrapper(BaseModel):
    """Universal wrapper for specialist results."""
    output: dict[str, Any] | None = Field(
        default=None,
        description="The actual result matching the orchestrator's expected schema",
    )
    success: bool = Field(
        default=True,
        description="Whether the specialist completed successfully",
    )
    notes: list[str] = Field(
        default_factory=list,
        description="Notes, complaints, warnings, or errors encountered during execution",
    )
    failure_reason: str | None = Field(
        default=None,
        description="If success=False, explains why the task failed",
    )
```

### 2. Task Model Extension

Add to `Task`:
```python
output_schema: dict[str, Any] | None = None     # JSON Schema for expected result
output_description: str | None = None           # Human-readable description
```

### 3. AbstractSpecialist Changes

- Add `set_output_schema(schema, description)` method
- Add `add_note(note: str)` helper for notes/complaints/warnings/errors
- Add generic `_finalize_with_output(output: dict)` tool (availability: `can_finalize AND output_schema is set`)
- Inject schema into system prompt when set

### 4. SuperDiscoveryAgent Changes

- `_create_task()`: Add `output_schema` and `output_description` params
- `_execute_task()`: Call `agent.set_output_schema()` before running

## Files to Modify

| File | Change |
|------|--------|
| `bluebox/data_models/orchestration/result.py` | **NEW** - `SpecialistResultWrapper` model |
| `bluebox/data_models/orchestration/task.py` | Add `output_schema`, `output_description` to `Task` |
| `bluebox/agents/specialists/abstract_specialist.py` | Add schema injection, generic finalize, `add_note()` |
| `bluebox/agents/super_discovery_agent.py` | Pass schema in create_task/execute_task |
| `bluebox/data_models/orchestration/__init__.py` | Export new model |
| `bluebox/agents/specialists/interaction_specialist.py` | Remove typed result models, use generic finalize |
| `bluebox/agents/specialists/docs_specialist.py` | Remove typed result models, use generic finalize |
| `bluebox/agents/specialists/js_specialist.py` | Remove typed result models, use generic finalize |
| `bluebox/agents/specialists/network_specialist.py` | Remove typed result models, use generic finalize |
| `bluebox/agents/specialists/value_trace_resolver_specialist.py` | Remove typed result models, use generic finalize |

## Implementation Steps

1. Create `SpecialistResultWrapper` in new `result.py`
2. Extend `Task` model with schema fields
3. Update `AbstractSpecialist`:
   - Add instance vars: `_task_output_schema`, `_notes`, `_wrapped_result`
   - Add `set_output_schema()` method
   - Add `add_note()` helper
   - Add `_finalize_with_output()` tool with `availability=lambda self: self.can_finalize and self._task_output_schema`
   - Update `_get_autonomous_system_prompt()` to include schema section
   - Update `_get_autonomous_result()` to return wrapper when set
4. Update `SuperDiscoveryAgent._create_task()` with new params
5. Update `SuperDiscoveryAgent._execute_task()` to pass schema to specialist
6. **Remove old result models** from each specialist:
   - `interaction_specialist.py`: Remove `ParameterDiscoveryResult`, `ParameterDiscoveryFailureResult`, `DiscoveredParameter`
   - `docs_specialist.py`: Remove `DocumentSearchResult`, `DocumentSearchFailureResult`, `DiscoveredDocument`
   - `js_specialist.py`: Remove `JSCodeResult`, `JSCodeFailureResult`
   - `network_specialist.py`: Remove `EndpointDiscoveryResult`, `DiscoveryFailureResult`, `DiscoveredEndpoint`
   - `value_trace_resolver_specialist.py`: Remove `TokenOriginResult`, `TokenOriginFailure`, `TokenOrigin`
7. Update each specialist to use generic `_finalize_with_output()` instead of typed finalize tools

## Breaking Changes (No Backwards Compatibility)

This is a clean break — all specialist-specific result models are removed. All specialists will use:
- `SpecialistResultWrapper` as their return type
- `_finalize_with_output(output: dict)` as their finalize tool
- Orchestrator-defined schemas

## Design Decisions

- **Schema Format**: JSON Schema (dict) — flexible, works with any structure
- **Notes Collection**: Manual via single `add_note()` method (for notes, complaints, warnings, errors)

## Schema Validation Approach

Use `jsonschema` library to validate `output` against `output_schema` in `_finalize_with_output()`. On validation failure, return error to LLM so it can fix the output.

## Example: How the Orchestrator LLM Uses This

The orchestrator LLM (SuperDiscoveryAgent) **dynamically decides** what schema it needs based on the task at hand. The schema is not hardcoded — the orchestrator constructs it when calling `create_task`:

```
Orchestrator LLM thinks: "I need to find the search API endpoint. I want the specialist
to return the URL, HTTP method, and any required headers."

Orchestrator LLM calls create_task:
  agent_type: "network_spy"
  prompt: "Find the API endpoint that handles train searches"
  output_schema: {
    "type": "object",
    "properties": {
      "url": {"type": "string", "description": "Full endpoint URL"},
      "method": {"type": "string", "enum": ["GET", "POST", "PUT", "DELETE"]},
      "required_headers": {"type": "object", "description": "Headers needed for the request"}
    },
    "required": ["url", "method"]
  }
  output_description: "Return the search endpoint details"
```

The specialist sees the schema in its system prompt and uses `finalize_with_output()` to return matching data:

```
Specialist calls finalize_with_output:
  output: {
    "url": "https://api.amtrak.com/v2/search",
    "method": "POST",
    "required_headers": {"x-api-key": "...", "content-type": "application/json"}
  }
```

Result returned to orchestrator:
```python
{
    "output": {"url": "...", "method": "POST", "required_headers": {...}},
    "success": True,
    "notes": ["Found 3 similar endpoints, returned the primary search API"],
    "failure_reason": None,
}
```

The key point: **the orchestrator chooses the schema dynamically based on what it needs**, not based on a predefined model.

## Verification

1. Update/remove existing tests that rely on old result models
2. Create a simple test task with custom schema, verify specialist can finalize with matching output
3. Test schema validation rejects mismatched output
4. Test `add_note()` captures notes correctly in wrapper
