# Navigation Operations

> Operations for page navigation and timing: `navigate`, `sleep`, `wait_for_url`.

**Code:** [operation.py](bluebox/data_models/routine/operation.py)

## navigate

Navigates to a URL and waits for page load.

```json
{
  "type": "navigate",
  "url": "https://example.com"
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `url` | string | Yes | - | URL to navigate to |
| `sleep_after_navigation_seconds` | float | No | 3.0 | Wait time after navigation |

**Placeholder support:**
- User parameters: **Yes** - `"url": "https://example.com/{{page_id}}"`
- Storage/cookie/meta: **No** - not supported in navigate URLs

**Examples:**
```json
{"type": "navigate", "url": "https://example.com"}
{"type": "navigate", "url": "https://api.example.com/page/{{page_id}}"}
{"type": "navigate", "url": "https://example.com", "sleep_after_navigation_seconds": 5.0}
```

---

## sleep

Pauses execution for a specified duration.

```json
{
  "type": "sleep",
  "timeout_seconds": 2.0
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `timeout_seconds` | float | Yes | Seconds to wait |

**When to use:**
- After `navigate` if default 3s isn't enough
- Before clicking elements that load dynamically
- Between UI operations for stability

**Examples:**
```json
{"type": "sleep", "timeout_seconds": 1.5}
{"type": "sleep", "timeout_seconds": 0.5}
```

---

## wait_for_url

Waits for the current URL to match a regex pattern. Useful after clicks that trigger navigation.

```json
{
  "type": "wait_for_url",
  "url_regex": "results\\.aspx",
  "timeout_ms": 10000
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `url_regex` | string | Yes | - | Regex pattern to match |
| `timeout_ms` | int | No | 20000 | Maximum wait time |

**Examples:**
```json
{"type": "wait_for_url", "url_regex": "/dashboard"}
{"type": "wait_for_url", "url_regex": "search\\?q=", "timeout_ms": 15000}
{"type": "wait_for_url", "url_regex": "confirmation|success"}
```

---

## Common Patterns

### Basic page load
```json
[
  {"type": "navigate", "url": "https://example.com"},
  {"type": "sleep", "timeout_seconds": 2.0}
]
```

### Navigate with parameter
```json
[
  {"type": "navigate", "url": "https://example.com/users/{{user_id}}"}
]
```

### Click triggers navigation
```json
[
  {"type": "click", "selector": "#submit-button"},
  {"type": "wait_for_url", "url_regex": "results", "timeout_ms": 10000},
  {"type": "sleep", "timeout_seconds": 1.0}
]
```

### Slow-loading page
```json
[
  {"type": "navigate", "url": "https://slow-site.com", "sleep_after_navigation_seconds": 8.0}
]
```
