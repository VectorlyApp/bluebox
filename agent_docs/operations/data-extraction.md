# Data Extraction Operations

> Operations that extract final data from the browser: `return` and `return_html`.

**Code:** [operation.py](web_hacker/data_models/routine/operation.py) (`RoutineReturnOperation`, `RoutineReturnHTMLOperation`)

These operations set `result.data` and are typically the last operation in a routine.

## return

Retrieves data from sessionStorage (previously stored by `fetch` operations).

```json
{
  "type": "return",
  "session_storage_key": "search_results"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `session_storage_key` | string | Yes | Key to retrieve from sessionStorage |

**How it works:**
1. Reads value from `sessionStorage.getItem(key)`
2. Handles large values via chunked reads (256KB chunks)
3. Sets `context.result.data` with the retrieved value

## return_html

Returns HTML content from the page or a specific element.

```json
{
  "type": "return_html",
  "scope": "page"
}
```

```json
{
  "type": "return_html",
  "scope": "element",
  "selector": "#results-table",
  "timeout_ms": 10000
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `scope` | `"page"` \| `"element"` | No | `"page"` | What HTML to return |
| `selector` | string | If scope=element | - | CSS selector for element |
| `timeout_ms` | int | No | 20000 | Wait time for element |

**How it works:**
- `scope: "page"` → returns `document.documentElement.outerHTML`
- `scope: "element"` → returns `element.outerHTML` for matched selector

## When to Use Each

| Use Case | Operation |
|----------|-----------|
| API response data (from fetch) | `return` |
| Scraped/structured data (from js_evaluate) | `return` |
| Raw page HTML for parsing | `return_html` (page) |
| Specific section HTML | `return_html` (element) |

## Examples

### Return fetch result
```json
[
  {"type": "fetch", "endpoint": {...}, "session_storage_key": "api_data"},
  {"type": "return", "session_storage_key": "api_data"}
]
```

### Return js_evaluate result
```json
[
  {"type": "js_evaluate", "js": "(()=>{...})()", "session_storage_key": "scraped"},
  {"type": "return", "session_storage_key": "scraped"}
]
```

### Return page HTML
```json
[
  {"type": "navigate", "url": "https://example.com"},
  {"type": "return_html", "scope": "page"}
]
```

### Return element HTML
```json
[
  {"type": "navigate", "url": "https://example.com"},
  {"type": "return_html", "scope": "element", "selector": "table.results"}
]
```
