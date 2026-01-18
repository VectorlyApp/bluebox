# Fetch Operations

> HTTP requests via browser fetch API: `fetch`, `download`, `get_cookies`.

**Code:** [operation.py](web_hacker/data_models/routine/operation.py), [endpoint.py](web_hacker/data_models/routine/endpoint.py)

## fetch

Executes an HTTP request using the browser's fetch API. Supports full placeholder resolution including sessionStorage, localStorage, cookies.

```json
{
  "type": "fetch",
  "endpoint": {
    "url": "https://api.example.com/search",
    "method": "POST",
    "headers": {"Content-Type": "application/json"},
    "body": {"query": "\"{{search_term}}\""},
    "credentials": "same-origin"
  },
  "session_storage_key": "search_results"
}
```

### Fetch Fields

| Field                   | Type     | Required | Description                             |
| ----------------------- | -------- | -------- | --------------------------------------- |
| `endpoint`            | Endpoint | Yes      | Request configuration                   |
| `session_storage_key` | string   | No       | Key to store response in sessionStorage |

### Endpoint Fields

| Field           | Type   | Required | Default           | Description                                                              |
| --------------- | ------ | -------- | ----------------- | ------------------------------------------------------------------------ |
| `url`         | string | Yes      | -                 | Request URL                                                              |
| `method`      | string | Yes      | -                 | `GET`, `POST`, `PUT`, `DELETE`, `PATCH`, `OPTIONS`, `HEAD` |
| `headers`     | object | No       | `{}`            | Request headers                                                          |
| `body`        | object | No       | `null`          | Request body (auto-serialized)                                           |
| `credentials` | string | No       | `"same-origin"` | `"same-origin"`, `"include"`, `"omit"`                             |
| `description` | string | No       | -                 | Human-readable description                                               |

### Placeholder Support

**All placeholders work in fetch** - URL, headers, and body:

| Source         | Example                                                 |
| -------------- | ------------------------------------------------------- |
| User params    | `"query": "\"{{search_term}}\""`                      |
| sessionStorage | `"token": "\"{{sessionStorage:auth.access_token}}\""` |
| localStorage   | `"theme": "\"{{localStorage:user.theme}}\""`          |
| Cookies        | `"session": "\"{{cookie:session_id}}\""`              |
| Window props   | `"key": "\"{{windowProperty:__CONFIG__.apiKey}}\""`   |
| Builtins       | `"id": "\"{{uuid}}\""`                                |

### Credentials Modes

| Mode              | Description                                |
| ----------------- | ------------------------------------------ |
| `"same-origin"` | Send cookies only for same-origin requests |
| `"include"`     | Always send cookies (cross-origin)         |
| `"omit"`        | Never send cookies                         |

### How Credentials Affect Execution

The `credentials` setting determines whether the browser includes cookies in the request. This is critical for authenticated API calls.

**When to use `"include"`:**
- Calling APIs that require the user's session (e.g., after navigating to a logged-in page)
- The site stores auth in cookies (not headers)
- You need HttpOnly cookies that JS can't access directly

**When to use `"same-origin"` (default):**
- Calling APIs on the same domain you navigated to
- Standard authenticated requests where cookies should be sent

**When to use `"omit"`:**
- Public APIs that don't need authentication
- Download operations where cookies aren't needed
- Avoiding sending unnecessary cookies

**Example: Using site's session**
```json
[
  {"type": "navigate", "url": "https://example.com"},
  {"type": "sleep", "timeout_seconds": 3.0},
  {
    "type": "fetch",
    "endpoint": {
      "url": "https://example.com/api/user-data",
      "method": "GET",
      "credentials": "include"
    },
    "session_storage_key": "user_data"
  }
]
```

After navigating, the browser has the site's cookies. Using `credentials: "include"` sends those cookies with the fetch request, allowing access to authenticated endpoints.

### Auto-Navigation for CORS

If the current page is `about:blank`, fetch automatically navigates to the target origin first to avoid CORS issues.

---

## Storing and Chaining Results

Use `session_storage_key` to store the response for use in later operations:

```json
[
  {
    "type": "fetch",
    "endpoint": {"url": "https://api.example.com/auth", "method": "POST", "body": {...}},
    "session_storage_key": "auth_response"
  },
  {
    "type": "fetch",
    "endpoint": {
      "url": "https://api.example.com/data",
      "method": "GET",
      "headers": {
        "Authorization": "Bearer \"{{sessionStorage:auth_response.token}}\""
      }
    },
    "session_storage_key": "data_response"
  },
  {"type": "return", "session_storage_key": "data_response"}
]
```

### Accessing Nested Response Data

Use dot notation for nested paths, numeric indices for arrays:

```json
"{{sessionStorage:response.data.user.name}}"
"{{sessionStorage:results.items.0.id}}"
"{{sessionStorage:auth.tokens.access_token}}"
```

---

## Examples

### GET Request

```json
{
  "type": "fetch",
  "endpoint": {
    "url": "https://api.example.com/users/\"{{user_id}}\"",
    "method": "GET",
    "headers": {"Accept": "application/json"},
    "credentials": "same-origin"
  },
  "session_storage_key": "user_data"
}
```

### POST with JSON Body

```json
{
  "type": "fetch",
  "endpoint": {
    "url": "https://api.example.com/search",
    "method": "POST",
    "headers": {"Content-Type": "application/json"},
    "body": {
      "query": "\"{{search_term}}\"",
      "limit": "{{limit}}",
      "filters": {"active": true}
    },
    "credentials": "same-origin"
  },
  "session_storage_key": "search_results"
}
```

### Using Session Token from Previous Fetch

```json
{
  "type": "fetch",
  "endpoint": {
    "url": "https://api.example.com/protected",
    "method": "GET",
    "headers": {
      "Authorization": "Bearer \"{{sessionStorage:login.token}}\"",
      "X-Request-ID": "\"{{uuid}}\""
    },
    "credentials": "include"
  },
  "session_storage_key": "protected_data"
}
```

### Form URL-Encoded Body

```json
{
  "type": "fetch",
  "endpoint": {
    "url": "https://api.example.com/login",
    "method": "POST",
    "headers": {"Content-Type": "application/x-www-form-urlencoded"},
    "body": {
      "username": "\"{{username}}\"",
      "password": "\"{{password}}\""
    },
    "credentials": "same-origin"
  },
  "session_storage_key": "login_response"
}
```

---

## download

Downloads a binary file (PDF, image, etc.) and returns it as base64. Typically the last operation in a routine.

```json
{
  "type": "download",
  "endpoint": {
    "url": "https://example.com/file.pdf",
    "method": "GET",
    "credentials": "omit"
  },
  "filename": "document.pdf"
}
```

| Field        | Type     | Required | Description                             |
| ------------ | -------- | -------- | --------------------------------------- |
| `endpoint` | Endpoint | Yes      | Request configuration                   |
| `filename` | string   | Yes      | Output filename (supports placeholders) |

**Result:** Sets `result.data` (base64), `result.is_base64 = true`, `result.filename`

### Download Example

```json
{
  "type": "download",
  "endpoint": {
    "url": "https://arxiv.org/pdf/\"{{paper_id}}\"",
    "method": "GET",
    "credentials": "omit"
  },
  "filename": "\"{{paper_id}}\".pdf"
}
```

---

## get_cookies

Gets all cookies for the current page via CDP (including HttpOnly cookies that JS can't access).

```json
{
  "type": "get_cookies",
  "session_storage_key": "all_cookies"
}
```

| Field                   | Type   | Required | Description          |
| ----------------------- | ------ | -------- | -------------------- |
| `session_storage_key` | string | No       | Key to store cookies |

Returns array of cookie objects with `name`, `value`, `domain`, `path`, etc.

---

## Common Patterns

### API with Auth Flow

```json
[
  {"type": "navigate", "url": "https://app.example.com"},
  {
    "type": "fetch",
    "endpoint": {
      "url": "https://api.example.com/auth",
      "method": "POST",
      "headers": {"Content-Type": "application/json"},
      "body": {"api_key": "\"{{api_key}}\""}
    },
    "session_storage_key": "auth"
  },
  {
    "type": "fetch",
    "endpoint": {
      "url": "https://api.example.com/data",
      "method": "GET",
      "headers": {"Authorization": "Bearer \"{{sessionStorage:auth.token}}\""}
    },
    "session_storage_key": "result"
  },
  {"type": "return", "session_storage_key": "result"}
]
```

### Fetch with Site's Session Cookie

```json
[
  {"type": "navigate", "url": "https://example.com"},
  {"type": "sleep", "timeout_seconds": 2.0},
  {
    "type": "fetch",
    "endpoint": {
      "url": "https://example.com/api/user",
      "method": "GET",
      "credentials": "include"
    },
    "session_storage_key": "user"
  },
  {"type": "return", "session_storage_key": "user"}
]
```

---

## Tips

1. **Always navigate first** - Fetch needs browser context; navigate to the target origin
2. **Use `credentials: "include"`** - When you need the site's session cookies
3. **Chain with sessionStorage** - Store intermediate results, access via `{{sessionStorage:key.path}}`
4. **Copy headers from network traffic** - Match the real request headers
5. **Escape-quote string placeholders** - Use `"\"{{param}}\""` for strings in body/headers
