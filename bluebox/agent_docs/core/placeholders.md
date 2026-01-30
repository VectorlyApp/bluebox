# Placeholders

> Dynamic value references resolved at execution time. Understand this before writing routines.

Placeholders inject runtime values into routine operations: user parameters, browser storage, cookies, and builtins.

**Code:** [placeholder.py](bluebox/data_models/routine/placeholder.py) (extraction), [js_utils.py](bluebox/utils/js_utils.py) (resolution)

## Syntax

```
{{source:path}}   or   {{param_name}}
```

All placeholders use the same `{{...}}` format. No escaped quotes needed.

## How Type Resolution Works

The parameter's `type` field in the routine schema determines how standalone placeholders resolve:

| Type    | Standalone               | In String                             |
| ------- | ------------------------ | ------------------------------------- |
| string  | `"{{x}}"` → `"john"` | `"...{{username}}..."` → `"...john..."` |
| integer | `"{{x}}"` → `50`    | `"...{{limit}}..."` → `"...50..."`  |
| number  | `"{{x}}"` → `19.99` | `"...{{price}}..."` → `"...19.99..."` |
| boolean | `"{{x}}"` → `true`  | `"...{{active}}..."` → `"...true..."` |

**Standalone** means the entire JSON value is just the placeholder (`"{{x}}"`). The runtime replaces it with the correctly-typed value based on the parameter's `type` field.

**Substring** means the placeholder is part of a larger string (`"prefix {{x}} suffix"`). The value is always stringified.

> **⚠️ Type must match the actual API expectation, not what the value "looks like."** Many APIs send numeric-looking values as JSON strings (e.g., `"adults": "5"`, `"enabled": "true"`). If the CDP capture shows `"5"` (quoted), use `type: string` so it resolves to `"5"`. Using `integer` would produce `5` (unquoted), potentially breaking the API. Always check the raw network data.

## Placeholder Types

### User Parameters

Direct reference to parameters defined in the routine:

```json
"query": "{{search_term}}",
"limit": "{{max_results}}"
```

### Session Storage

Access values stored by previous operations (supports dot-path for nested objects):

```json
"token": "{{sessionStorage:auth.access_token}}",
"code": "{{sessionStorage:response.data.items.0.stationCode}}"
```

### Local Storage

```json
"theme": "{{localStorage:user.preferences.theme}}"
```

### Cookies

```json
"session": "{{cookie:session_id}}",
"csrf": "{{cookie:csrf_token}}"
```

### Window Properties

Access JavaScript `window` object properties:

```json
"apiKey": "{{windowProperty:__CONFIG__.apiKey}}",
"href": "{{windowProperty:location.href}}"
```

### Meta Tags

Access HTML `<meta>` tag content:

```json
"csrf": "{{meta:csrf-token}}"
```

### Builtins

Auto-generated values (no definition needed in `parameters`):

| Placeholder                | Description                                |
| -------------------------- | ------------------------------------------ |
| `{{uuid}}`               | Random UUID via `crypto.randomUUID()`    |
| `{{epoch_milliseconds}}` | Current timestamp in ms via `Date.now()` |

```json
"requestId": "{{uuid}}",
"timestamp": "{{epoch_milliseconds}}"
```

## Hardcoded Values

Copy hardcoded values directly from network traffic. Only use `{{...}}` for dynamic values:

```json
{
  "origin": "{{origin}}",
  "type": "OW",
  "active": true,
  "limit": 100
}
```

- `"{{origin}}"` - placeholder, resolved at runtime
- `"OW"` - hardcoded string, copy as-is
- `true`, `100` - hardcoded primitives, copy as-is

## Nested Path Access

Use dot notation for nested objects and numeric indices for arrays:

```
{{sessionStorage:response.data.user.name}}
{{sessionStorage:results.items.0.id}}
{{sessionStorage:results.items.0.attributes.code}}
```

## Where Placeholders Work

| Location            | User Params | Storage/Cookie/Meta/Window            |
| ------------------- | ----------- | ------------------------------------- |
| Navigate URL        | Yes         | **No** (not yet supported)      |
| Fetch URL           | Yes         | Yes                                   |
| Headers             | Yes         | Yes                                   |
| Body                | Yes         | Yes                                   |
| `input_text.text` | Yes         | No                                    |

## Chaining Operations

Store a fetch response, then use values from it in the next operation:

**Operation 1** - fetch and store:

```json
{
  "type": "fetch",
  "endpoint": {"url": "https://api.example.com/auth"},
  "session_storage_key": "auth_response"
}
```

**Operation 2** - use stored value:

```json
{
  "type": "fetch",
  "endpoint": {
    "url": "https://api.example.com/data",
    "headers": {
      "Authorization": "Bearer {{sessionStorage:auth_response.token}}"
    }
  }
}
```

## Complete Example

```json
{
  "headers": {
    "Authorization": "Bearer {{sessionStorage:auth.token}}",
    "X-Request-ID": "{{uuid}}",
    "Content-Type": "application/json"
  },
  "body": {
    "username": "{{username}}",
    "limit": "{{limit}}",
    "active": "{{is_active}}",
    "stationCode": "{{sessionStorage:stations.0.code}}"
  }
}
```

## Quick Reference

| Source             | Syntax                           | Example                                    |
| ------------------ | -------------------------------- | ------------------------------------------ |
| Parameter (string) | `"{{name}}"`                   | `"{{username}}"`                         |
| Parameter (number) | `"{{name}}"`                   | `"{{limit}}"`                            |
| Session storage    | `"{{sessionStorage:path}}"`    | `"{{sessionStorage:auth.token}}"`        |
| Local storage      | `"{{localStorage:path}}"`      | `"{{localStorage:theme}}"`               |
| Cookie             | `"{{cookie:name}}"`            | `"{{cookie:session_id}}"`               |
| Window property    | `"{{windowProperty:path}}"`    | `"{{windowProperty:__CONFIG__.key}}"`    |
| Meta tag           | `"{{meta:name}}"`              | `"{{meta:csrf-token}}"`                 |
| UUID (builtin)     | `"{{uuid}}"`                   | generates random UUID                      |
| Epoch ms (builtin) | `"{{epoch_milliseconds}}"`     | generates timestamp                        |
