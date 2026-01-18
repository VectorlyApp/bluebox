# Debugging Routines

> How to diagnose and fix common routine issues.

## First: Check Operation Metadata

**Every operation logs execution details.** After running a routine, inspect `result.operations_metadata`:

```python
result = routine.execute(params)
for op in result.operations_metadata:
    print(f"{op.type}: {op.duration_seconds:.2f}s")
    print(f"  Details: {op.details}")
    if op.error:
        print(f"  ERROR: {op.error}")
```

**What each operation logs:**
| Operation | `details` fields |
|-----------|------------------|
| `fetch` | `request`, `response` (status, headers, url) |
| `click` | `selector`, `element`, `click_coordinates` |
| `input_text` | `selector`, `text_length`, `element` |
| `js_evaluate` | `console_logs`, `execution_error`, `storage_error` |

---

## Common Issues

| Issue | File |
|-------|------|
| Page not loaded, element not found | [page-not-loaded.md](common-issues/page-not-loaded.md) |
| js_evaluate returns undefined, can't parse data, blocked patterns | [js-evaluate-issues.md](common-issues/js-evaluate-issues.md) |
| Fetch returns 401/403 | [unauthenticated.md](common-issues/unauthenticated.md) |
| Placeholder not resolved | [placeholder-not-resolved.md](common-issues/placeholder-not-resolved.md) |
| Click/input fails | [element-not-found.md](common-issues/element-not-found.md) |
| Fetch returns HTML | [fetch-returns-html.md](common-issues/fetch-returns-html.md) |
| Routine execution timeout | [execution-timeout.md](common-issues/execution-timeout.md) |

---

## Debugging Workflow

1. **Run routine, check `result.ok`**
2. **If failed, check `result.error`**
3. **Iterate through `result.operations_metadata`**
   - Which operation failed?
   - What's in `details`?
   - For js_evaluate: check `console_logs`
4. **Add `console.log()` to js_evaluate** - all output captured
5. **Use `return_html`** to see current page state
6. **Check storage** with js_evaluate to verify data flow
