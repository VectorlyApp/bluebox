# Simplify Placeholder Interpolation — Implementation Plan

## Goal

Eliminate the escape-quoted convention (`\"{{param}}\"`) and use simple `{{param}}` everywhere. `Parameter.type` drives type coercion at resolution time.

**Before:** `"name": "\"{{username}}\""` (string), `"count": "{{limit}}"` (int)
**After:** `"name": "{{username}}"` (string, because Parameter.type=string), `"count": "{{limit}}"` (int, because Parameter.type=integer)

---

## Design Decisions

1. **Remove `apply_params()` entirely** — not deprecate. The new system operates on structured data (dicts/lists) and plain strings differently. Old regex-on-serialized-JSON approach doesn't fit.

2. **`param_type_map` flow:** `Routine.execute()` builds `{name: type_str}` from `self.parameters`, passes it via `RoutineExecutionContext`. Operations forward it to `apply_params_to_json()`.

3. **Type coercion for edge cases:** `_coerce_value("true", "boolean")` → `True`, `_coerce_value("42", "integer")` → `42`. Handles common string representations.

4. **Storage/meta/window/builtin placeholders:** Already use `{{...}}` format in many places. No fundamental change — they're resolved at runtime in browser JS, not Python-side. The JS regex just gets simplified (remove optional `\"?` matching).

5. **Existing production routines:** Will need a migration script (out of scope for this PR, but flagged). Old `\"{{param}}\"` format will break validation since the new system expects `{{param}}`.

---

## Files to Modify (21 files, ordered by dependency)

### Phase 1: Core Data Model Changes

#### 1. `bluebox/data_models/routine/placeholder.py`
- **Delete** `PlaceholderQuoteType` enum
- **Delete** `ExtractedPlaceholder` dataclass
- **Rewrite** `extract_placeholders_from_json_str()` → returns `list[str]` (just content strings)
- Single regex `\{\{([^}]*)\}\}` matches all placeholders uniformly, no quote-type tracking

#### 2. `bluebox/data_models/routine/execution.py`
- **Add** `param_type_map: dict[str, str] = Field(default_factory=dict)` to `RoutineExecutionContext`

#### 3. `bluebox/utils/data_utils.py`
- **Delete** `apply_params()` function
- **Add** `apply_params_to_str(text, parameters_dict)` — simple `{{key}}` → `str(value)` replacement for string contexts (URLs, selectors, filenames, JS code, text)
- **Add** `apply_params_to_json(d, parameters_dict, param_type_map)` — recursively walks dict/list:
  - Standalone `"{{key}}"` (entire value) → typed coercion via `_coerce_value()`
  - Substring `"prefix {{key}} suffix"` → `str()` substitution
  - Runtime placeholders (storage/cookie/meta/window/builtins) left untouched
- **Add** `_coerce_value(value, param_type)` helper:
  - `"string"/"date"/"datetime"/"email"/"url"/"enum"` → `str(value)`
  - `"integer"` → `int(value)`
  - `"number"` → `float(value)`
  - `"boolean"` → handles `True/False`, `"true"/"false"`, `"1"/"0"`
- **Add** private helpers: `_is_runtime_placeholder()`, `_resolve_value()`, `_resolve_string_value()`
- **Update** module docstring to reflect new function names

### Phase 2: Update Operation Execution

#### 4. `bluebox/data_models/routine/operation.py`
- **Change import:** `apply_params` → `apply_params_to_str, apply_params_to_json`
- **RoutineNavigateOperation:** `apply_params(url, ...)` → `apply_params_to_str(url, ...)`
- **RoutineFetchOperation._execute_fetch:**
  - URL: `apply_params_to_str(url, params)`
  - Headers: `apply_params_to_json(headers_dict, params, param_type_map)` (no more json.dumps→apply→json.loads round-trip)
  - Body: `apply_params_to_json(body_dict, params, param_type_map)` (same simplification)
- **RoutineFetchOperation._execute_operation:** URL for CORS navigation: `apply_params_to_str()`
- **RoutineClickOperation:** `apply_params_to_str(selector, ...)`
- **RoutineTypeOperation:** `apply_params_to_str(selector, ...)`, `apply_params_to_str(text, ...)`
- **RoutineScrollOperation:** `apply_params_to_str(selector, ...)`
- **RoutineReturnHTMLOperation:** `apply_params_to_str(selector, ...)`
- **RoutineDownloadOperation:** Same pattern as fetch — `apply_params_to_str` for URL/filename, `apply_params_to_json` for headers/body
- **RoutineJsEvaluateOperation:** `apply_params_to_str(js, ...)`

#### 5. `bluebox/data_models/routine/routine.py`
- **Remove import** of `PlaceholderQuoteType`
- **Rewrite `validate_routine_structure()`:**
  - Remove `param_type_map`, `non_string_types` used for quote validation
  - `placeholders` is now `list[str]` — iterate over strings, not `ExtractedPlaceholder` objects
  - Remove entire quote-type validation block (lines 170-181)
  - Remove "unquoted placeholder" scan in unused-params error message (lines 188-191)
  - Keep: structural checks, prefix validation, used/unused/undefined param checks
- **Update `execute()`:** Build `param_type_map` and pass to `RoutineExecutionContext`:
  ```python
  param_type_map = {p.name: p.type.value for p in self.parameters}
  routine_execution_context = RoutineExecutionContext(..., param_type_map=param_type_map)
  ```

### Phase 3: Simplify JS-Side Resolution

#### 6. `bluebox/utils/js_utils.py`
In `_get_placeholder_resolution_js_helpers()`:
- **`replaceSimpleTokens()`:** Remove `\\\"?` from epoch_milliseconds and uuid regexes
  - Before: `/\\\"?\\{\\{\\s*epoch_milliseconds\\s*\\}\\}\\\"?/g`
  - After: `/\\{\\{\\s*epoch_milliseconds\\s*\\}\\}/g`
- **`PLACEHOLDER` regex:** Remove `\\\"?` from both sides
  - Before: `/\\\"?\\{\\{\\s*(sessionStorage|...)\\s*:\\s*([^}]+?)\\s*\\}\\}\\\"?/g`
  - After: `/\\{\\{\\s*(sessionStorage|...)\\s*:\\s*([^}]+?)\\s*\\}\\}/g`
- **`resolvePlaceholders()`:** Remove quote-detection logic (lines 218-229). Replace with simple:
  ```javascript
  return (typeof v === 'object') ? JSON.stringify(v) : String(v);
  ```
- **Update comments** that reference "quoted" vs "escape-quoted"

### Phase 4: Update Agent Prompts

#### 7. `bluebox/agents/routine_discovery_agent.py`
- **Rewrite `PLACEHOLDER_INSTRUCTIONS`** (lines 85-104):
  - Remove two-quote-set explanation
  - New: "ALL placeholders use `{{param}}` — type comes from parameter definition"
  - New examples showing simple format
  - Emphasize: parameter `type` must match what the API actually expects
  - Remove "MUST ENSURE EACH PLACEHOLDER IS SURROUNDED BY QUOTES OR ESCAPED QUOTES"

#### 8. `bluebox/agents/routine_discovery_agent_beta.py`
- **Rewrite `PLACEHOLDER_INSTRUCTIONS`** (lines 205-224): Same as above
- **Rewrite `_validate_placeholder_syntax()`** (lines ~1820-1919):
  - Remove `PlaceholderQuoteType` import and usage
  - Iterate over `list[str]` from `extract_placeholders_from_json_str()`
  - Remove string-must-be-escape-quoted validation
  - Keep prefix validation and param-is-defined check

#### 9. `bluebox/agents/guide_agent.py`
- **Update `_NOTES_SECTION`** (lines 217-220):
  - Remove: "Quotes or escaped quotes are ESSENTIAL AROUND {{parameter_name}}"
  - New: "ALL placeholders use {{param_name}} format — parameter type drives resolution"

### Phase 5: Update Agent Documentation

#### 10. `bluebox/agent_docs/core/placeholders.md`
- **Major rewrite.** Remove "The One Rule" about escape-quoted strings
- All examples: `"\"{{x}}\""` → `"{{x}}"`
- New type resolution table showing Parameter.type drives output
- Simplify Quick Reference table

#### 11. `bluebox/agent_docs/core/parameters.md`
- Update Parameter Types table: all types use `"{{x}}"` standalone (remove `\"` variants)
- Update "Rule" text: no more escape-quote requirement
- Update Usage examples

#### 12. `bluebox/agent_docs/operations/fetch.md`
- All examples: `"\"{{search_term}}\""` → `"{{search_term}}"`
- Remove tip line 337: "Escape-quote string placeholders"
- Update Placeholder Support table examples

#### 13. `bluebox/agent_docs/operations/ui-operations.md`
- `input_text` examples: `"\"{{username}}\""` → `"{{username}}"`
- Common Patterns section: same updates

#### 14. `bluebox/agent_docs/common-issues/placeholder-not-resolved.md`
- Remove "String not escape-quoted" cause from table
- Update symptom/fix descriptions

### Phase 6: Update Example Routines

#### 15. `example_data/example_routines/amtrak_one_way_train_search_routine.json`
- `searchTerm=\"{{origin}}\"` → `searchTerm={{origin}}`
- `searchTerm=\"{{destination}}\"` → `searchTerm={{destination}}`
- `\"{{departureDate}}\"T00:00:00` → `{{departureDate}}T00:00:00`

#### 16. `example_data/example_routines/download_arxive_paper_routine.json`
- `https://arxiv.org/pdf/\"{{paper_id}}\"` → `https://arxiv.org/pdf/{{paper_id}}`
- `\"{{paper_id}}\".pdf` → `{{paper_id}}.pdf`

#### 17. `example_data/example_routines/massachusetts_corp_search_routine.json`
- `"\"{{entity_name}}\""` → `"{{entity_name}}"`

#### 18. `example_data/example_routines/get_new_polymarket_bets_routine.json`
- `limit=\"{{limit}}\"` → `limit={{limit}}`
- `offset=\"{{offset}}\"` → `offset={{offset}}`

### Phase 7: Rewrite Tests

#### 19. `tests/unit/data_models/routine/test_routine_validation.py`
- **Rewrite `TestExtractPlaceholdersFromJson`:** test new function returning `list[str]`
  - Remove all `PlaceholderQuoteType` / `ExtractedPlaceholder` references
  - Test: uniform detection of `{{param}}` regardless of surrounding quotes
  - Test: unquoted placeholders ARE now detected (not ignored)
- **Delete `TestParameterTypeQuoteValidation`** entirely (no more quote-type validation)
- **Update all test operations** to use `{{param}}` format:
  - `RoutineNavigateOperation(url="https://example.com/\"{{user_id}}\"")` → `url="https://example.com/{{user_id}}"`
  - `body={"name": "\"{{user_name}}\""}` → `body={"name": "{{user_name}}"}`
- **Delete** `test_premier_league_routine_unquoted_placeholder_is_ignored` (unquoted is now the standard)
- **Flip** `test_string_param_in_body_quoted_format_raises_error` → now should PASS (no error)
- **Delete** `test_string_param_escape_quoted_in_body_valid` (no escape-quoted concept)

#### 20. `tests/unit/utils/test_data_utils.py`
- **Delete** all `TestApplyParams*` classes (8 classes)
- **Add `TestApplyParamsToJson`:**
  - Standalone string param `"{{key}}"` → `"value"` (str)
  - Standalone integer param `"{{key}}"` → `42` (int)
  - Standalone boolean param `"{{key}}"` → `True` (bool)
  - Standalone number param `"{{key}}"` → `3.14` (float)
  - Substring `"prefix {{key}} suffix"` → `"prefix value suffix"` (always str)
  - Runtime placeholders untouched
  - Nested dict/list recursion
  - Missing params left as-is
  - None value handling
- **Add `TestApplyParamsToStr`:**
  - URL replacement
  - Multiple params in one string
  - Runtime placeholders untouched
  - Empty/None inputs
- **Add `TestCoerceValue`:**
  - String "true" → bool True
  - String "42" → int 42
  - String "3.14" → float 3.14
  - None passthrough
  - Already-correct types passthrough

#### 21. `tests/unit/data_models/routine/test_operations.py`
- **Update import:** `apply_params` → `apply_params_to_str`
- **Update 3 call sites** (lines 594, 614, 634) to use `apply_params_to_str`

### Phase 8: Project Documentation

#### 22. `CLAUDE.md`
- Remove: "String parameters in JSON bodies need double escaping: `\"{{param}}\"`"
- Remove: "Fix any placeholder escaping issues (string params need `\"{{param}}\"`)"
- Update "Placeholder Resolution" line under Important Patterns
- Update JavaScript Code Generation section if it references escape quotes

---

## Execution Order

```
Phase 1 (Foundation):     placeholder.py → execution.py → data_utils.py
Phase 2 (Operations):     operation.py → routine.py
Phase 3 (JS):             js_utils.py
Phase 4 (Agent prompts):  routine_discovery_agent.py → routine_discovery_agent_beta.py → guide_agent.py
Phase 5 (Agent docs):     placeholders.md → parameters.md → fetch.md → ui-operations.md → placeholder-not-resolved.md
Phase 6 (Examples):       4 example routine JSON files
Phase 7 (Tests):          test_routine_validation.py → test_data_utils.py → test_operations.py
Phase 8 (Project docs):   CLAUDE.md
```

After each phase, run `pytest tests/ -v` to verify no regressions.

---

## Risk Mitigation

- **Test frequently:** Run tests after Phases 2, 3, and 7 at minimum
- **Production migration:** Flag in PR description that existing routines need migration
- **Backward compat for JS:** Storage/meta/window resolution in browser is fundamentally unchanged — just regex simplification
