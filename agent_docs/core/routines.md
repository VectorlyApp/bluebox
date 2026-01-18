# Routines

> A Routine is a JSON workflow that automates browser tasks: navigate to a site, perform actions (fetch APIs, interact with UI, run JS), then return data.

**Code:** [routine.py](web_hacker/data_models/routine/routine.py), [operation.py](web_hacker/data_models/routine/operation.py)

## General Flow

Most routines follow this pattern:

```
1. Navigate    → Go to target website, wait for page load
2. Perform     → Fetch APIs, click/type, run JS, chain data via sessionStorage
3. Return      → Extract final result from sessionStorage or HTML
```

## Routine Data Model

```python
class Routine(BaseModel):
    name: str
    description: str
    operations: list[RoutineOperationUnion]
    incognito: bool = True
    parameters: list[Parameter] = []
```

## Parameter Data Model

```python
class Parameter(BaseModel):
    name: str                    # Must be valid Python identifier
    type: ParameterType          # string, integer, number, boolean, date, datetime, email, url, enum
    required: bool = True
    description: str
    default: str | int | float | bool | None = None
    examples: list[str | int | float | bool] = []
    min_length: int | None = None
    max_length: int | None = None
    min_value: int | float | None = None
    max_value: int | float | None = None
    pattern: str | None = None   # Regex for string validation
    enum_values: list[str] | None = None
    format: str | None = None    # e.g., 'YYYY-MM-DD'
```

## Validation

The `Routine` model validates:

1. **All defined parameters must be used** in operations
2. **No undefined parameters** can appear in placeholders
3. **String parameters must use escape-quoted format** (`\"{{param}}\"`), while int/number/bool can use either format

See [routine.py](web_hacker/data_models/routine/routine.py) for full validation logic.

## Placeholders

Placeholders use `"{{name}}" or \"{{name}}\"` syntax and are resolved at execution time. **See `core/placeholders.md` for complete details.**

### User Parameters

Parameters defined in the routine's `parameters` array:

```json
"query": "\"{{search_term}}\""
"page": "{{page_number}}"
```

### Builtin Parameters

Available without definition:

| Placeholder                | Description                                   |
| -------------------------- | --------------------------------------------- |
| `{{uuid}}`               | Generates a random UUID (crypto.randomUUID()) |
| `{{epoch_milliseconds}}` | Current timestamp in ms (Date.now())          |

### Storage Placeholders

Resolved at runtime from browser context (**only in fetch `headers` and `body`**):

| Prefix              | Description            | Example                             |
| ------------------- | ---------------------- | ----------------------------------- |
| `sessionStorage:` | Browser sessionStorage | `"{{sessionStorage:auth.token}}"` |
| `localStorage:`   | Browser localStorage   | `"{{localStorage:user.id}}"`      |
| `cookie:`         | Cookie value           | `"{{cookie:session_id}}"`         |
| `meta:`           | Meta tag content       | `"{{meta:csrf-token}}"`           |
| `window:`         | Window property        | `"{{window:__CONFIG__.apiKey}}"`  |

**Limitation:** Storage placeholders are NOT interpolated in URLs yet - only in fetch headers and body.

### Escape-Quoted Format (PLACEHOLDERS ONLY!)

**String placeholders MUST use escape-quoted format:**

```json
"name": "\"{{username}}\""
"body": {"query": "\"{{search_term}}\""}
```

**Why?** When the placeholder resolves, the outer quotes become part of the JSON string value.

**Non-string types** (int, number, bool) can use either format:

```json
"count": "{{limit}}"
"count": "\"{{limit}}\""
```

### HARDCODED VALUES: COPY AS-IS

**The `\"` syntax is ONLY for placeholder resolution!** Hardcoded values should be copied exactly from network traffic:

```json
{
  "code": "\"{{origin}}\"",
  "type": "OW",
  "active": true
}
```

## Validation Rules

- All defined parameters MUST be used in operations
- No undefined parameters can appear in placeholders
- Builtin parameters (`uuid`, `epoch_milliseconds`) don't need definition
- String parameters MUST use escape-quoted format: `"\"{{param}}\""`

## Available Operations

Operations execute sequentially. **See `operations/overview.md` for details.**

### Navigation & Timing

| Type             | Description                 |
| ---------------- | --------------------------- |
| `navigate`     | Navigate to URL             |
| `sleep`        | Pause execution             |
| `wait_for_url` | Wait for URL to match regex |

### Data Operations

| Type            | Description                                                              |
| --------------- | ------------------------------------------------------------------------ |
| `fetch`       | Execute HTTP request via browser fetch API (see `operations/fetch.md`) |
| `download`    | Download binary file as base64                                           |
| `get_cookies` | Get all cookies (including HttpOnly) via CDP                             |

### UI Automation

| Type           | Description                   |
| -------------- | ----------------------------- |
| `click`      | Click element by CSS selector |
| `input_text` | Type text into input element  |
| `press`      | Press keyboard key            |
| `scroll`     | Scroll page or element        |

### Code Execution

| Type            | Description                                                                 |
| --------------- | --------------------------------------------------------------------------- |
| `js_evaluate` | Execute custom JavaScript (IIFE format). See `operations/js-evaluate.md`. |

### Data Retrieval

| Type            | Description                         |
| --------------- | ----------------------------------- |
| `return`      | Retrieve result from sessionStorage |
| `return_html` | Get HTML content from page/element  |

## Execution Flow

**See `core/execution.md` for complete details.**

1. Creates or attaches to browser tab
2. Enables CDP domains (Page, Runtime, Network, DOM)
3. Iterates through operations sequentially
4. Each operation interpolates parameters, resolves placeholders, executes via CDP
5. Returns `RoutineExecutionResult`

## Example: Amtrak Train Search

```json
{
  "name": "amtrak_routine",
  "description": "Search for trains on Amtrak",
  "incognito": true,
  "parameters": [
    {"name": "origin", "type": "string", "required": true, "description": "Origin city or station code"},
    {"name": "destination", "type": "string", "required": true, "description": "Destination city or station code"},
    {"name": "departureDate", "type": "string", "required": true, "description": "Departure date (YYYY-MM-DD)"}
  ],
  "operations": [
    {"type": "navigate", "url": "https://www.amtrak.com/"},
    {"type": "sleep", "timeout_seconds": 2.0},
    {
      "type": "fetch",
      "endpoint": {
        "url": "https://www.amtrak.com/services/MapDataService/AutoCompleterArcgis/getResponseList?searchTerm=\"{{origin}}\"",
        "method": "GET",
        "headers": {"Accept": "application/json"},
        "credentials": "same-origin"
      },
      "session_storage_key": "origin_stations"
    },
    {
      "type": "fetch",
      "endpoint": {
        "url": "https://www.amtrak.com/services/MapDataService/AutoCompleterArcgis/getResponseList?searchTerm=\"{{destination}}\"",
        "method": "GET",
        "headers": {"Accept": "application/json"},
        "credentials": "same-origin"
      },
      "session_storage_key": "dest_stations"
    },
    {
      "type": "fetch",
      "endpoint": {
        "url": "https://www.amtrak.com/dotcom/journey-solution-option",
        "method": "POST",
        "headers": {
          "Content-Type": "application/json",
          "x-amtrak-trace-id": "{{sessionStorage:ibsession.sessionid}}"
        },
        "body": {
          "journeyRequest": {
            "type": "OW",
            "journeyLegRequests": [{
              "origin": {
                "code": "{{sessionStorage:origin_stations.autoCompleterResponse.autoCompleteList.0.stationCode}}",
                "schedule": {"departureDateTime": "\"{{departureDate}}\"T00:00:00"}
              },
              "destination": {
                "code": "{{sessionStorage:dest_stations.autoCompleterResponse.autoCompleteList.0.stationCode}}"
              }
            }]
          }
        },
        "credentials": "same-origin"
      },
      "session_storage_key": "search_results"
    },
    {"type": "return", "session_storage_key": "search_results"}
  ]
}
```

**For more examples, see `examples.md`.**
