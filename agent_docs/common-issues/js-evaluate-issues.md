# js_evaluate Issues

> Related: [js-evaluate.md](../operations/js-evaluate.md), [fetch.md](../operations/fetch.md)

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

**Symptom:** `JSON.parse()` fails, `data.items` is undefined

**Cause:** Fetch stores results as **strings**. You must parse them.

**Important:** Data may be **doubly stringified**. You may need to call `JSON.parse()` twice!

```javascript
// WRONG - raw is a string!
const raw = sessionStorage.getItem('api_response');
return raw.items;  // undefined

// RIGHT - parse first
const raw = sessionStorage.getItem('api_response');
const data = JSON.parse(raw);
return data.items;

// If still a string, parse again!
const raw = sessionStorage.getItem('api_response');
let data = JSON.parse(raw);
if (typeof data === 'string') {
  data = JSON.parse(data);  // Double parse
}
return data.items;
```

**Always debug with console.log:**
```javascript
(function() {
  const raw = sessionStorage.getItem('api_response');
  console.log('Raw type:', typeof raw);
  console.log('Raw value:', raw?.substring(0, 200));

  let data = JSON.parse(raw);
  console.log('After first parse, type:', typeof data);

  if (typeof data === 'string') {
    console.log('Still a string! Parsing again...');
    data = JSON.parse(data);
  }

  console.log('Final data keys:', Object.keys(data));
  return data;
})()
```
Check `console_logs` in operation metadata.

---

## Blocked Pattern Error

**Symptom:** Error about blocked JavaScript pattern

**Cause:** Security restrictions block certain APIs.

**Blocked patterns:**
- `fetch()` → Use `fetch` operation instead
- `eval()`, `Function()` → Rewrite without dynamic code
- `addEventListener()` → Not supported
- `location`, `history` → Use `navigate` operation
