# Plan: Migrate placeholder interpolation to dict-walking + schema-typed approach

## Summary

Replace the current regex-on-serialized-JSON interpolation (`apply_params`) with a two-part system:
1. **Dict-walking** for structured data (body, headers) — walk the Python dict, find `"{{key}}"` string values, replace with properly-typed Python values based on `Parameter.type`
2. **String interpolation** for flat strings (URLs, selectors, text, JS code, filenames) — regex replace `{{key}}` with `str(value)`

The escaped-quote convention (`\"{{param}}\"`) is eliminated entirely. All placeholders use `{{param}}` uniformly. Type info comes from the `Parameter` schema, not from quoting style.

---

## New `apply_params` behavior

### Signature change
```python
# Old
def apply_params(text: str, parameters_dict: dict | None) -> str

# New — two functions
def apply_params_to_dict(d: dict, parameters_dict: dict, param_type_map: dict[str, ParameterType]) -> dict
def apply_params_to_str(text: str, parameters_dict: dict) -> str
```

### `apply_params_to_dict(d, parameters_dict, param_type_map)`
Recursively walk the dict. For each string value:
- **Standalone placeholder** (`"{{key}}"` is the entire string): replace with typed value based on `param_type_map[key]` — string types → `str(value)`, integer → `int(value)`, number → `float(value)`, boolean → `bool(value)`
- **Substring placeholder** (`"prefix {{key}} suffix"`): regex-replace `{{key}}` with `str(value)`, result stays a string
- **Non-param placeholders** (`{{sessionStorage:...}}`, `{{uuid}}`, etc.): leave untouched

### `apply_params_to_str(text, parameters_dict)`
Simple regex: replace `{{key}}` with `str(value)` for each key in parameters_dict. Used for URLs, selectors, text, JS code, filenames — contexts where the result is always a string.

---

## Files to modify

### 1. `bluebox/utils/data_utils.py`
- Replace `apply_params` with `apply_params_to_dict` and `apply_params_to_str`
- `apply_params_to_dict`: recursive dict walker, handles standalone vs substring, uses `param_type_map` for type coercion
- `apply_params_to_str`: simple `{{key}}` → `str(value)` regex substitution

### 2. `bluebox/data_models/routine/operation.py` (all call sites)
Update each operation to use the appropriate function:

| Operation | Field | Function |
|---|---|---|
| `RoutineFetchOperation._execute_fetch` | `endpoint.url` | `apply_params_to_str` |
| `RoutineFetchOperation._execute_fetch` | `endpoint.headers` | `apply_params_to_dict` (pass dict directly, no serialize/deserialize) |
| `RoutineFetchOperation._execute_fetch` | `endpoint.body` | `apply_params_to_dict` (pass dict directly, no serialize/deserialize) |
| `RoutineNavigateOperation` | `url` | `apply_params_to_str` |
| `RoutineClickOperation` | `selector` | `apply_params_to_str` |
| `RoutineTypeOperation` | `selector`, `text` | `apply_params_to_str` |
| `RoutineScrollOperation` | `selector` | `apply_params_to_str` |
| `RoutineReturnHTMLOperation` | `selector` | `apply_params_to_str` |
| `RoutineDownloadOperation` | `url` | `apply_params_to_str` |
| `RoutineDownloadOperation` | `headers` | `apply_params_to_dict` |
| `RoutineDownloadOperation` | `body` | `apply_params_to_dict` |
| `RoutineDownloadOperation` | `filename` | `apply_params_to_str` |
| `RoutineJsEvaluateOperation` | `js` | `apply_params_to_str` |

**Key win for fetch/download**: eliminate the `json.dumps` → `apply_params` → `json.loads` round-trip. Just pass the dict directly to `apply_params_to_dict`.

Operations need access to `param_type_map`. Add it to `RoutineExecutionContext`.

### 3. `bluebox/data_models/routine/execution.py`
Add `param_type_map: dict[str, ParameterType]` field to `RoutineExecutionContext`.

### 4. `bluebox/data_models/routine/routine.py`
- Populate `param_type_map` in `execute()` when creating `RoutineExecutionContext`
- **Remove** the `validate_parameter_usage` logic that enforces escape-quoted format for strings. Replace with simpler validation: all defined params must appear as `{{param}}` somewhere in the routine JSON, and all `{{param}}` placeholders (that aren't storage/builtin) must be defined.

### 5. `bluebox/data_models/routine/placeholder.py`
- Simplify `extract_placeholders_from_json_str`: no longer need `PlaceholderQuoteType` distinction. Just find all `{{...}}` patterns.
- Can remove `PlaceholderQuoteType` enum and simplify `ExtractedPlaceholder`.

### 6. `bluebox/agents/routine_discovery_agent.py`
- Update `PLACEHOLDER_INSTRUCTIONS` (lines 70-88): remove all escaped-quote rules. New instructions: always use `{{param}}`, type comes from parameter definition, no special quoting needed.

### 7. Agent docs (all reference escaped-quote convention)
- `bluebox/agent_docs/core/placeholders.md`
- `bluebox/agent_docs/core/routines.md`
- `bluebox/agent_docs/core/parameters.md`
- `bluebox/agent_docs/operations/fetch.md`
- `bluebox/agent_docs/common-issues/placeholder-not-resolved.md`

### 8. `CLAUDE.md`
- Remove line about escape-quoted format requirement

### 9. Example routines
- `example_routines/amtrak_one_way_train_search_routine.json` — remove `\"` around placeholders
- `example_routines/download_arxive_paper_routine.json` — same
- `example_routines/massachusetts_corp_search_routine.json` — same
- `example_routines/get_new_polymarket_bets_routine.json` — same

### 10. Tests
- `tests/unit/test_data_utils.py` — rewrite `apply_params` tests for new functions
- `tests/unit/test_routine_validation.py` — update validation tests (no more escape-quoted enforcement)
- `tests/unit/test_operations.py` — update operation parameter interpolation tests
- `tests/data/input/production_routine/routine_escaped_string_params.json` — update fixture

### 11. JS placeholder resolution (`bluebox/utils/js_utils.py`)
- Update the JS regex pattern (line ~121) that matches `\"?{{...}}\"?` — the `\"?` optional escaped-quote matching may need adjustment since templates no longer contain escaped quotes. Storage/builtin placeholders in headers/body will now just be `{{sessionStorage:...}}` without escaped quotes.

---

## Verification

1. `pytest tests/ -v` — all tests pass
2. Manually inspect an example routine to confirm no escaped quotes remain
3. Spot-check that `apply_params_to_dict` correctly produces typed values (int, bool, string) in a body dict
4. Verify JS-resolved placeholders (sessionStorage, etc.) still work by checking the generated JS regex handles unquoted `{{sessionStorage:...}}` patterns
