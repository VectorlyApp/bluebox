# js_evaluate Issues

> Common problems when using `js_evaluate` operations: code returns undefined due to missing IIFE wrappers or return statements, stored fetch data needs JSON parsing, and certain APIs like `fetch()` or `eval()` are blocked and must use dedicated operations instead. Related: [js-evaluate.md](../operations/js-evaluate.md), [fetch.md](../operations/fetch.md)

---

## Returns undefined

**Symptom:** `session_storage_key` contains `null` or nothing

**Causes & Fixes:**

| Cause | Fix |
|-------|-----|
| Missing IIFE wrapper | Wrap in `(function() { ... })()` |
| No return statement | Add `return` before the value |
| Async without await | Use `async function` and `await` |

```javascript
// WRONG
document.title

// WRONG - no return
(function() { const x = document.title; })()

// RIGHT
(function() { return document.title; })()
```

---

## Can't Access Fetch Data

**Symptom:** `data.items` is undefined

**Cause:** Fetch stores results as **strings** in sessionStorage. You must parse them.

```javascript
// WRONG - raw is a string!
var raw = sessionStorage.getItem('api_response');
return raw.items;  // undefined

// RIGHT - parse first
var raw = sessionStorage.getItem('api_response');
var data = JSON.parse(raw);
return data.items;
```

**Debug with console.log:**
```javascript
(function() {
  var raw = sessionStorage.getItem('api_response');
  console.log('Raw type:', typeof raw);
  console.log('Raw preview:', raw ? raw.substring(0, 200) : null);

  var data = JSON.parse(raw);
  console.log('Parsed keys:', Object.keys(data));

  return data;
})()
```
Check `console_logs` in operation metadata.

---

## Blocked Pattern Error

**Symptom:** Error about blocked JavaScript pattern

**Cause:** Security restrictions block certain APIs.

**Blocked patterns:**
- `fetch()` → Use `fetch` operation instead (note: `prefetch()`, `refetch()` etc. are allowed)
- `eval()`, `Function()` → Rewrite without dynamic code
- `addEventListener()`, `MutationObserver`, `IntersectionObserver` → Not supported
- `window.close()` → Not supported
