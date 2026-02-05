# Plan: Dynamic Output Schema for Specialist Agents

## Summary

Allow `SuperDiscoveryAgent` to define what result structure a specialist should return at task creation time, wrapping results in a universal container with complaints/errors/warnings.

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
    data: dict[str, Any] | None = None          # The actual result
    success: bool = True
    complaints: list[str] = []                   # Issues encountered
    errors: list[str] = []                       # Non-fatal errors
    warnings: list[str] = []                     # FYIs for orchestrator
    failure_reason: str | None = None            # If success=False
```

### 2. Task Model Extension

Add to `Task`:
```python
output_schema: dict[str, Any] | None = None     # JSON Schema for expected result
output_description: str | None = None           # Human-readable description
```

### 3. AbstractSpecialist Changes

- Add `set_output_schema(schema, description)` method
- Add `add_complaint()`, `add_warning()`, `add_error()` helpers
- Add generic `_finalize_with_data(data: dict)` tool (availability: `can_finalize AND output_schema is set`)
- Inject schema into system prompt when set

### 4. SuperDiscoveryAgent Changes

- `_create_task()`: Add `output_schema` and `output_description` params
- `_execute_task()`: Call `agent.set_output_schema()` before running

## Files to Modify

| File | Change |
|------|--------|
| `bluebox/data_models/orchestration/result.py` | **NEW** - `SpecialistResultWrapper` model |
| `bluebox/data_models/orchestration/task.py` | Add `output_schema`, `output_description` to `Task` |
| `bluebox/agents/specialists/abstract_specialist.py` | Add schema injection, generic finalize, helpers |
| `bluebox/agents/super_discovery_agent.py` | Pass schema in create_task/execute_task |
| `bluebox/data_models/orchestration/__init__.py` | Export new model |

## Implementation Steps

1. Create `SpecialistResultWrapper` in new `result.py`
2. Extend `Task` model with schema fields
3. Update `AbstractSpecialist`:
   - Add instance vars: `_task_output_schema`, `_complaints`, `_warnings`, `_errors`, `_wrapped_result`
   - Add `set_output_schema()` method
   - Add `add_complaint/warning/error()` helpers
   - Add `_finalize_with_data()` tool with `availability=lambda self: self.can_finalize and self._task_output_schema`
   - Update `_get_autonomous_system_prompt()` to include schema section
   - Update `_get_autonomous_result()` to return wrapper when set
4. Update `SuperDiscoveryAgent._create_task()` with new params
5. Update `SuperDiscoveryAgent._execute_task()` to pass schema to specialist

## Backwards Compatibility

- Existing typed finalize tools remain unchanged
- Generic finalize only available when `output_schema` is provided
- Specialists without schema continue to return their typed models

## Design Decisions

- **Schema Format**: JSON Schema (dict) — flexible, works with any structure
- **Complaint/Warning Collection**: Manual via `add_complaint()`, `add_warning()`, `add_error()` methods

## Schema Validation Approach

Use `jsonschema` library to validate `data` against `output_schema` in `_finalize_with_data()`. On validation failure, return error to LLM so it can fix the data.

## Example Usage

```python
# SuperDiscoveryAgent creates task with expected schema
@agent_tool()
def _create_task(self, ...):
    task = Task(
        agent_type="network_spy",
        prompt="Find the search API endpoint",
        output_schema={
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "method": {"type": "string", "enum": ["GET", "POST"]},
                "headers": {"type": "object"},
            },
            "required": ["url", "method"],
        },
        output_description="Return the main API endpoint details",
    )

# Specialist receives schema in system prompt, uses generic finalize:
# "finalize_with_data(data={'url': '...', 'method': 'POST', 'headers': {...}})"

# Result returned to orchestrator:
{
    "data": {"url": "...", "method": "POST", "headers": {...}},
    "success": True,
    "complaints": ["Response body was truncated, may be missing fields"],
    "errors": [],
    "warnings": ["Found 3 similar endpoints, returned most likely match"],
}
```

## Verification

1. Run existing tests to ensure backwards compatibility
2. Create a simple test task with custom schema, verify specialist can finalize with matching data
3. Test schema validation rejects mismatched data
