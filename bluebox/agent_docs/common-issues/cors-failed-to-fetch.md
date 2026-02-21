# Fetch Fails with TypeError: Failed to fetch (CORS)

> Fetch operations fail with "TypeError: Failed to fetch" when the browser's origin doesn't match the API server's CORS `Access-Control-Allow-Origin` header. Fix by adding a `navigate` operation to the allowed origin before any `fetch`. Related: [fetch.md](../operations/fetch.md), [navigation.md](../operations/navigation.md)

**Symptom:** Fetch operation returns `TypeError: Failed to fetch` or the response data is `null`/empty despite the endpoint working in experiments.

**Root Cause:** The routine executor starts from `about:blank` (origin = `null`). Many APIs restrict CORS to their own website origin. For example, `api.nasdaq.com` only allows requests from origin `https://www.nasdaq.com`. Without a `navigate` operation first, the browser's origin is `null` and every `fetch` is blocked by CORS.

**How to detect:** If an experiment confirmed the API works from the site's origin (e.g. `browser_eval_js(fetch(...))` succeeded after navigating to `www.example.com`) but the routine's `fetch` operation fails with `TypeError: Failed to fetch`, the routine is missing a `navigate` step.

**Solutions:**

| Problem | Fix |
|---------|-----|
| API requires same-origin (e.g. `api.example.com` allows `www.example.com`) | Add `navigate` to the allowed origin before `fetch` |
| API requires `Origin`/`Referer` headers | Add `"Origin"` and `"Referer"` to fetch headers |
| API is on the same domain as the website | Add `navigate` to the website URL first |
| Cloudflare/WAF blocks CORS preflight (OPTIONS → 403) | Set `"credentials": "omit"` on the fetch endpoint — this avoids the preflight OPTIONS request entirely, bypassing the block. Works for public APIs that don't need cookies |
| All else fails | Use `js_evaluate` with `fetch()` instead of a `fetch` operation — JS fetch from the navigated page context has the correct origin |

**RULE:** Every routine that calls an external API SHOULD start with a `navigate` operation to establish the correct browser origin. This is cheap (one page load) and prevents CORS issues.

**Example: Navigate to allowed origin, then fetch from API subdomain**
```json
[
  {"type": "navigate", "url": "https://www.example.com"},
  {
    "type": "fetch",
    "endpoint": {
      "url": "https://api.example.com/api/data?q={{query}}",
      "method": "GET",
      "headers": {
        "Accept": "application/json, text/plain, */*"
      }
    },
    "session_storage_key": "result"
  },
  {"type": "return", "session_storage_key": "result"}
]
```

**Example: Navigate + auth token + data fetch (common pattern)**
```json
[
  {"type": "navigate", "url": "https://www.example.com"},
  {
    "type": "fetch",
    "endpoint": {
      "url": "https://api.example.com/api/token",
      "method": "POST",
      "headers": {"Content-Type": "application/json"},
      "body": {"applicationName": "web"}
    },
    "session_storage_key": "auth_response"
  },
  {
    "type": "js_evaluate",
    "expression": "(function(){ var r = JSON.parse(sessionStorage.getItem('auth_response')); return r.data.token; })()",
    "session_storage_key": "bearer_token"
  },
  {
    "type": "fetch",
    "endpoint": {
      "url": "https://api.example.com/api/data",
      "method": "GET",
      "headers": {
        "Authorization": "Bearer {{sessionStorage.bearer_token}}",
        "Accept": "application/json"
      }
    },
    "session_storage_key": "data_result"
  },
  {"type": "return", "session_storage_key": "data_result"}
]
```

**Cloudflare / WAF Blocking Preflight Requests**

Some APIs behind Cloudflare or other WAFs block CORS preflight (OPTIONS) requests with 403. This happens when `credentials: "include"` triggers a preflight that Cloudflare rejects. The captured network data will show OPTIONS requests returning 403 with `server: cloudflare` and `content-type: text/html`.

**Fix:** If the API does NOT require cookies or session auth, set `"credentials": "omit"` on the fetch endpoint. This tells the browser NOT to send cookies, which often eliminates the preflight OPTIONS request entirely, bypassing the Cloudflare block.

**When to try this:** The experiment shows `TypeError: Failed to fetch` AND the captured network data shows OPTIONS preflight returning 403 from Cloudflare. Try `credentials: "omit"` first — many public search/listing APIs work without cookies.

```json
[
  {"type": "navigate", "url": "https://www.example.com"},
  {
    "type": "fetch",
    "endpoint": {
      "url": "https://api.example.com/search",
      "method": "POST",
      "headers": {"Content-Type": "application/json", "Accept": "application/json"},
      "body": {"query": "{{search_term}}", "page": "{{page}}"},
      "credentials": "omit"
    },
    "session_storage_key": "search_result"
  },
  {"type": "return", "session_storage_key": "search_result"}
]
```
