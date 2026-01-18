# Fetch Returns 401/403 (Unauthenticated)

> Related: [fetch.md](../operations/fetch.md), [js-evaluate.md](../operations/js-evaluate.md)

**Symptom:** API returns unauthorized error

**Diagnose:** Use `js_evaluate` to inspect what auth exists:
```javascript
(function() {
  return {
    cookies: document.cookie,
    sessionStorage: Object.fromEntries(
      Object.keys(sessionStorage).map(k => [k, sessionStorage.getItem(k)])
    ),
    localStorage: Object.fromEntries(
      Object.keys(localStorage).map(k => [k, localStorage.getItem(k)])
    ),
    windowConfig: window.__CONFIG__ || window.__INITIAL_STATE__ || null
  };
})()
```

**Solutions:**

| Problem | Fix |
|---------|-----|
| Cookies not sent | Set `"credentials": "include"` |
| Wrong origin | Navigate to API origin first |
| Token in JS variable | Extract via `js_evaluate`, use in header |
| HttpOnly cookie needed | Use `get_cookies` operation |

**Example: Navigate first, then fetch with cookies**
```json
[
  {"type": "navigate", "url": "https://example.com"},
  {"type": "sleep", "timeout_seconds": 2.0},
  {
    "type": "fetch",
    "endpoint": {
      "url": "https://example.com/api/data",
      "method": "GET",
      "credentials": "include"
    }
  }
]
```
