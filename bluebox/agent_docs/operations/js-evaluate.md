# js_evaluate Operation

> Execute custom JavaScript in the browser. Use for: DOM extraction, data transformation, hidden tokens, reCAPTCHA. **Use console.log() for debugging - logs appear in operation metadata!**

**Code:** [operation.py](bluebox/data_models/routine/operation.py) (`RoutineJsEvaluateOperation`)

## When to Use

- Extract hidden DOM elements (CSRF tokens, nonces, hidden inputs)
- Get reCAPTCHA tokens
- Scrape/format HTML tables into structured data
- Transform stored fetch results (remember: **fetch results are STRINGS - use JSON.parse()!**)
- Complex conditional logic not possible with other operations

## Basic Format

```json
{
  "type": "js_evaluate",
  "js": "(function() { /* your code */ return result; })()",
  "timeout_seconds": 5,
  "session_storage_key": "result_key"
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `js` | string | Yes | - | JavaScript code in IIFE format |
| `timeout_seconds` | float | No | 5.0 | Max execution time (0-10 seconds) |
| `session_storage_key` | string | No | null | Store result for later retrieval |

## IMPORTANT: Return Value Auto-Storage

**When you provide `session_storage_key`, the return value of your IIFE is automatically stored in sessionStorage.** This enables chaining with other operations.

```json
[
  {
    "type": "js_evaluate",
    "js": "(function() { return { title: document.title, url: location.href }; })()",
    "session_storage_key": "page_info"
  },
  {
    "type": "fetch",
    "endpoint": {
      "url": "https://api.example.com/log",
      "method": "POST",
      "body": {"page": "{{sessionStorage:page_info.title}}"}
    }
  }
]
```

The returned object `{ title: ..., url: ... }` is automatically stored under `page_info` in sessionStorage, accessible via `{{sessionStorage:page_info.title}}` in subsequent operations.

---

## CRITICAL: IIFE Format Required

**All JavaScript MUST be wrapped in an IIFE (Immediately Invoked Function Expression).**

```javascript
// Valid formats
(function() { return document.title; })()
(() => { return document.title; })()
(async function() { return await somePromise; })()
(async () => { return await somePromise; })()

// INVALID - will be rejected
document.title                              // No wrapper
function() { return document.title; }()     // Missing outer parens
(function() { return document.title; })     // Not invoked
```

---

## Debugging with console.log()

**Use `console.log()` liberally!** All output is captured in operation metadata (`console_logs` field).

```javascript
(function() {
  const rows = document.querySelectorAll('table tr');
  console.log('Found rows:', rows.length);  // Captured in metadata!

  const data = Array.from(rows).map(row => {
    const cells = row.querySelectorAll('td');
    console.log('Row cells:', cells.length, cells[0]?.textContent);
    return { name: cells[0]?.textContent?.trim() };
  });

  console.log('Extracted:', JSON.stringify(data));
  return data;
})()
```

Access logs in execution result:
```python
result = routine.execute(params)
for op in result.operations_metadata:
    if op.type == "js_evaluate":
        print(op.details.get("console_logs"))
```

---

## Placeholder Support

**User parameters ARE interpolated:**
```json
{
  "js": "(function() { return document.querySelector('{{selector}}').textContent; })()"
}
```

**Storage/cookie/window placeholders are NOT interpolated - access directly:**
```javascript
// WRONG - will error
(function() { return '{{sessionStorage:myKey}}'; })()

// RIGHT - access directly in JS
(function() { return sessionStorage.getItem('myKey'); })()
(function() { return JSON.parse(sessionStorage.getItem('myKey')); })()
```

**Builtins are NOT interpolated - use native JS:**
```javascript
// WRONG
(function() { return '{{uuid}}'; })()

// RIGHT
(function() { return crypto.randomUUID(); })()
(function() { return Date.now(); })()
```

---

## Common Use Cases

### Extract Hidden CSRF Token
```json
{
  "type": "js_evaluate",
  "js": "(function() { return document.querySelector('input[name=\"csrf_token\"]').value; })()",
  "session_storage_key": "csrf_token"
}
```

### Extract Data Attribute
```json
{
  "type": "js_evaluate",
  "js": "(function() { return document.querySelector('[data-config]').dataset.config; })()",
  "session_storage_key": "config"
}
```

### Scrape HTML Table
```json
{
  "type": "js_evaluate",
  "js": "(function() { const rows = document.querySelectorAll('table tbody tr'); return Array.from(rows).map(row => { const cells = row.querySelectorAll('td'); return { name: cells[0]?.textContent?.trim() || '', id: cells[1]?.textContent?.trim() || '', status: cells[2]?.textContent?.trim() || '' }; }); })()",
  "session_storage_key": "table_data"
}
```

### Transform Fetch Result (IMPORTANT!)
**Fetch stores results as JSON strings. You MUST parse them:**
```json
{
  "type": "js_evaluate",
  "js": "(function() { const raw = sessionStorage.getItem('api_response'); const data = JSON.parse(raw); return data.items.filter(i => i.active).map(i => ({ id: i.id, name: i.name })); })()",
  "session_storage_key": "filtered_items"
}
```

### Get reCAPTCHA Token
```json
{
  "type": "js_evaluate",
  "js": "(async function() { return await grecaptcha.execute('site_key', {action: 'submit'}); })()",
  "session_storage_key": "recaptcha_token",
  "timeout_seconds": 10
}
```

### Extract Window Config Object
```json
{
  "type": "js_evaluate",
  "js": "(function() { return window.__INITIAL_STATE__ || window.__CONFIG__ || null; })()",
  "session_storage_key": "page_config"
}
```

### Get All Links on Page
```json
{
  "type": "js_evaluate",
  "js": "(function() { return Array.from(document.querySelectorAll('a[href]')).map(a => ({ text: a.textContent.trim(), href: a.href })); })()",
  "session_storage_key": "links"
}
```

---

## Blocked Patterns (Security)

These patterns are detected and rejected:

| Pattern | Reason |
|---------|--------|
| `eval()` | Dynamic code generation |
| `Function()` constructor | Dynamic code generation |
| `fetch()` | Use `fetch` operation instead (note: `prefetch()`, `refetch()` etc. are allowed) |
| `XMLHttpRequest` | Network requests |
| `WebSocket` | Network requests |
| `sendBeacon` | Exfiltration |
| `addEventListener()` | Persistent event hooks |
| `MutationObserver` | Persistent observers |
| `IntersectionObserver` | Persistent observers |
| `window.close()` | Lifecycle control |

---

## What's Allowed

- Promises and async/await
- Loops (while, for, do) - timeout prevents infinite loops
- DOM manipulation and queries
- sessionStorage/localStorage access
- All standard synchronous JS

---

## Error Handling

Errors are captured in operation metadata:

| Field | Description |
|-------|-------------|
| `console_logs` | All console.log() output |
| `execution_error` | Exception from your JS code |
| `storage_error` | Failed to store result |

If an error occurs, the operation raises `RuntimeError`.

---

## Common Mistakes

1. **Forgetting IIFE wrapper**
   ```javascript
   // WRONG
   document.title

   // RIGHT
   (function() { return document.title; })()
   ```

2. **Not parsing fetch results** - Data from `fetch` is stored as STRING!
   ```javascript
   // WRONG - data is a string!
   const data = sessionStorage.getItem('api_result');
   return data.items;  // ERROR: undefined

   // RIGHT - parse first!
   const data = JSON.parse(sessionStorage.getItem('api_result'));
   return data.items;
   ```

3. **Using storage placeholders in JS**
   ```javascript
   // WRONG - not interpolated
   return '{{sessionStorage:key}}';

   // RIGHT
   return sessionStorage.getItem('key');
   ```

4. **Not returning a value**
   ```javascript
   // WRONG - returns undefined
   (function() { const x = 1 + 1; })()

   // RIGHT
   (function() { return 1 + 1; })()
   ```

---

## When to Use Other Operations Instead

| Use Case | Use This Instead |
|----------|------------------|
| Click a button | `click` |
| Type into input | `input_text` |
| Make HTTP request | `fetch` |
| Navigate to URL | `navigate` |
| Wait for URL change | `wait_for_url` |
| Get raw page HTML | `return_html` |
