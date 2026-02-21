# Auth & Token Resolution Strategies

Tokens and API keys are the #1 reason routines fail with 401/403 errors. This guide covers every way to discover, extract, and resolve auth credentials at runtime.

## The Two Categories

| Category | Lifespan | Strategy |
|----------|----------|----------|
| **Static credentials** (API keys, subscription keys, client IDs) | Long-lived or permanent | Hardcode in the routine |
| **Dynamic tokens** (JWT, Bearer, session tokens, CSRF) | Short-lived, expire | Fetch at runtime within the routine |

## Where Tokens Live — Discovery Checklist

When exploring a site's auth, check ALL of these sources. Tokens can come from anywhere.

### 1. Network Requests (Most Common)

The captured session shows exactly which headers and tokens were used.

**How to find them:**
- Use `capture_search_transactions` to search for keywords: "token", "auth", "key", "bearer", "jwt"
- Use `capture_get_transaction` to inspect specific request headers
- Look for `Authorization: Bearer ...` headers
- Look for custom headers: `Ocp-Apim-Subscription-Key`, `X-Api-Key`, `X-Auth-Token`
- Look for POST requests to `/token`, `/auth`, `/login`, `/oauth` endpoints

**What you'll find:**
- The token endpoint URL
- The exact headers and body needed to get a token
- The response shape (where the token lives in the JSON response)
- Any static API keys used alongside the token

### 2. DOM — Inline Scripts and Meta Tags

Sites often embed tokens or config objects directly in the HTML.

**Common patterns:**
```html
<!-- Meta tags -->
<meta name="csrf-token" content="abc123def456">
<meta name="api-key" content="pk_live_xxxx">

<!-- Inline script config -->
<script>
  window.__CONFIG__ = { apiKey: "abc123", authToken: "xyz789" };
  window.__INITIAL_STATE__ = { auth: { token: "..." } };
  window.ENV = { API_KEY: "..." };
</script>

<!-- Data attributes -->
<div data-api-key="abc123" data-csrf="xyz789"></div>
```

**Routine resolution:**
```json
{"type": "navigate", "url": "https://example.com"},
{"type": "js_evaluate", "js": "(function() { return { token: document.querySelector('meta[name=\"csrf-token\"]').content }; })();", "session_storage_key": "csrf_data"}
```

Or use placeholders:
```json
"headers": {
  "X-CSRF-Token": "{{meta:csrf-token}}",
  "X-Api-Key": "{{windowProperty:__CONFIG__.apiKey}}"
}
```

### 3. Browser Storage (localStorage / sessionStorage)

Sites store tokens in browser storage after the user (or the site's JS) authenticates.

**How to discover:**
- Navigate to the site, then use `js_evaluate` to dump storage:
```javascript
(function() {
  var ss = {};
  for (var i = 0; i < sessionStorage.length; i++) {
    var k = sessionStorage.key(i);
    ss[k] = sessionStorage.getItem(k);
  }
  var ls = {};
  for (var i = 0; i < localStorage.length; i++) {
    var k = localStorage.key(i);
    ls[k] = localStorage.getItem(k);
  }
  return { sessionStorage: ss, localStorage: ls };
})()
```

**Common keys to look for:** `token`, `access_token`, `auth`, `jwt`, `session`, `user`

**Routine resolution:**
```json
"headers": {
  "Authorization": "Bearer {{localStorage:auth.access_token}}",
  "X-Session": "{{sessionStorage:session.token}}"
}
```

### 4. Cookies

Some sites use cookie-based auth — the token IS the cookie.

**How to discover:**
- Use `get_cookies` operation to see all cookies including HttpOnly ones
- Look for cookies named: `session`, `token`, `auth`, `sid`, `csrf`, `XSRF-TOKEN`

**Routine resolution — two approaches:**

**a) Let the browser send cookies automatically:**
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

**b) Extract cookie value explicitly:**
```json
"headers": {
  "X-XSRF-TOKEN": "{{cookie:XSRF-TOKEN}}"
}
```

### 5. Window Properties (JavaScript Globals)

Sites set global JS variables with config/auth info.

**How to discover:**
```javascript
(function() {
  var keys = ['__CONFIG__', '__INITIAL_STATE__', 'ENV', '__NEXT_DATA__',
              'config', 'appConfig', '__APP_DATA__', '_env'];
  var found = {};
  keys.forEach(function(k) {
    if (window[k]) found[k] = window[k];
  });
  return found;
})()
```

**Routine resolution:**
```json
"headers": {
  "X-Api-Key": "{{windowProperty:__CONFIG__.apiKey}}"
}
```

### 6. API Token Endpoints (Runtime Fetch)

The most robust approach for dynamic tokens — fetch the token at runtime.

**Pattern: fetch token → extract → use in subsequent requests**
```json
[
  {"type": "navigate", "url": "https://example.com", "sleep_after_navigation_seconds": 2.0},
  {
    "type": "fetch",
    "endpoint": {
      "url": "https://example.com/api/auth/token",
      "method": "POST",
      "headers": {
        "Content-Type": "application/json",
        "X-Api-Key": "HARDCODED_SITE_KEY_FROM_CAPTURES"
      },
      "body": {
        "applicationName": "website",
        "channel": "Web"
      },
      "credentials": "same-origin"
    },
    "session_storage_key": "token_response"
  },
  {
    "type": "fetch",
    "endpoint": {
      "url": "https://example.com/api/data",
      "method": "GET",
      "headers": {
        "Authorization": "Bearer {{sessionStorage:token_response.token}}",
        "X-Api-Key": "HARDCODED_SITE_KEY_FROM_CAPTURES"
      }
    },
    "session_storage_key": "data_result"
  },
  {"type": "return", "session_storage_key": "data_result"}
]
```

### 7. JS Evaluation (Extract from Running Page)

When tokens are generated by the site's JavaScript and aren't in storage or DOM.

**Pattern: navigate → let site JS run → extract token via JS eval**
```json
[
  {"type": "navigate", "url": "https://example.com", "sleep_after_navigation_seconds": 3.0},
  {
    "type": "js_evaluate",
    "js": "(function() { try { var state = JSON.parse(sessionStorage.getItem('persist:root')); var auth = JSON.parse(state.auth); return { token: auth.accessToken }; } catch(e) { return { error: String(e) }; } })();",
    "session_storage_key": "extracted_token"
  },
  {
    "type": "fetch",
    "endpoint": {
      "url": "https://example.com/api/data",
      "headers": {
        "Authorization": "Bearer {{sessionStorage:extracted_token.token}}"
      }
    },
    "session_storage_key": "result"
  },
  {"type": "return", "session_storage_key": "result"}
]
```

## Experiment Strategies for the PI

When a site requires auth, the PI should dispatch experiments that explore MULTIPLE resolution strategies. Don't just try one approach and give up.

### Experiment 1: Discover What Auth Exists

```
"Navigate to {site_url}, wait for page load, then inspect ALL available auth sources:
1. Run JS to dump sessionStorage, localStorage, and window config objects
2. Use get_cookies to list all cookies
3. Check DOM for meta tags with csrf/token/key attributes
4. Use capture_search_transactions to find requests with 'token' or 'auth' in the URL

Report back: what tokens/keys did you find, where did they come from, and what
do they look like (first 20 chars)? We saw '{observed_token_prefix}...' in the
captured session — is it still the same or has it changed?"
```

### Experiment 2: Try Token Endpoint

```
"The captured session shows a token endpoint at {token_url}.
1. Use capture_get_transaction to get the EXACT headers and body from the capture
2. Navigate to {site_url} first to establish cookies
3. Call the token endpoint with the same headers/body
4. If it returns a token, store it and try calling {data_endpoint} with
   Authorization: Bearer {token}
5. If it fails, try variations: different Content-Type, with/without credentials,
   with cookies via credentials:'include'

The captured request had these headers: {captured_headers}
The captured body was: {captured_body}"
```

### Experiment 3: Try Page-Embedded Token

```
"Navigate to {site_url} and wait 3 seconds for JS to execute.
Then try to find auth tokens in the page:
1. Check window.__CONFIG__, window.__INITIAL_STATE__, window.ENV
2. Check sessionStorage and localStorage for 'token', 'auth', 'jwt' keys
3. Check meta tags for csrf-token, api-key
4. If you find a token, try using it to call {data_endpoint}

In the captured session, we saw a token that looked like: '{token_sample}'
It may be the same static value or may have changed."
```

### Experiment 4: Try Cookie-Based Auth

```
"Navigate to {site_url} and wait for page load.
The site may use cookie-based auth (the navigation itself establishes the session).
1. After navigation, call {data_endpoint} with credentials:'include' to send cookies
2. If that works, the routine just needs navigate + fetch with credentials:'include'
3. If it fails, dump cookies with get_cookies to see what cookies exist
4. Try different credential modes: 'same-origin' vs 'include'"
```

## Common Auth Patterns by Site Type

| Site Type | Typical Auth | Strategy |
|-----------|-------------|----------|
| Modern SPA (React/Angular) | JWT via token endpoint | Fetch token → use Bearer header |
| Traditional server-rendered | Session cookie | Navigate → fetch with `credentials: "include"` |
| Public API with key | Static API key in header | Hardcode from captures |
| CSRF-protected forms | CSRF token in meta/cookie | Extract via `{{meta:csrf-token}}` or `{{cookie:XSRF-TOKEN}}` |
| OAuth-protected | Access token via OAuth flow | Fetch token endpoint with client credentials |
| Azure API Management | Subscription key + JWT | Hardcode sub key, fetch JWT at runtime |

## Key Rules

1. **Static keys are HARDCODED** — API keys, subscription keys, client IDs from captures go directly into the routine. Never expose them as user parameters.
2. **Dynamic tokens are FETCHED** — JWT, Bearer, session tokens must be obtained at runtime via a fetch or js_evaluate operation within the routine.
3. **Always navigate first** — Most auth requires being on the site's origin for cookies and CORS to work.
4. **Check multiple sources** — A token might be in storage, DOM, cookies, AND network requests. Find the most reliable source.
5. **Include observed values in experiments** — Tell the worker what the token looked like in the captured session so they know what to look for and can verify if it's static or dynamic.
6. **The PI must try multiple strategies** — If the token endpoint fails, try page-embedded tokens. If those fail, try cookie-based auth. If that fails, try JS evaluation. Don't give up after one approach.
