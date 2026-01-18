# Parameters

> User inputs for routines. Defines types, validation, and naming rules.

Parameters enable dynamic values to be injected at execution time via placeholders.

**Code:** [parameter.py](web_hacker/data_models/routine/parameter.py)

## Data Model

```python
class Parameter(BaseModel):
    name: str                    # Valid Python identifier
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

## Parameter Types

| Type | Description | Validation | Standalone | In String |
|------|-------------|------------|------------|-----------|
| `string` | Text | `min_length`, `max_length`, `pattern` | `"\"{{x}}\""` | `"...\"{{x}}\"..."` |
| `integer` | Whole number | `min_value`, `max_value` | `"{{x}}"` | `"...\"{{x}}\"..."` |
| `number` | Decimal | `min_value`, `max_value` | `"{{x}}"` | `"...\"{{x}}\"..."` |
| `boolean` | true/false | - | `"{{x}}"` | `"...\"{{x}}\"..."` |
| `date` | Date string | `format` | `"\"{{x}}\""` | `"...\"{{x}}\"..."` |
| `datetime` | Date+time | `format` | `"\"{{x}}\""` | `"...\"{{x}}\"..."` |
| `email` | Email address | Pattern validated | `"\"{{x}}\""` | `"...\"{{x}}\"..."` |
| `url` | URL string | Pattern validated | `"\"{{x}}\""` | `"...\"{{x}}\"..."` |
| `enum` | One of allowed | `enum_values` required | `"\"{{x}}\""` | `"...\"{{x}}\"..."` |

**Rule:** String-like types always need `\"`. Primitives (int, number, bool) only need `\"` when embedded in a larger string.

## Naming Rules

Parameter names must:
- Be valid Python identifiers (`[a-zA-Z_][a-zA-Z0-9_]*`)
- NOT start with reserved prefixes: `sessionStorage`, `localStorage`, `cookie`, `meta`, `windowProperty`, `uuid`, `epoch_milliseconds`

## Builtin Parameters

Available without definition in `parameters`:

| Name | Description |
|------|-------------|
| `{{uuid}}` | Random UUID via `crypto.randomUUID()` |
| `{{epoch_milliseconds}}` | Current timestamp in ms |

```json
"requestId": "\"{{uuid}}\"",
"timestamp": "\"{{epoch_milliseconds}}\""
```

## Examples

### String with Validation

```json
{
  "name": "search_query",
  "type": "string",
  "required": true,
  "description": "The search term",
  "min_length": 1,
  "max_length": 200,
  "examples": ["flights to NYC"]
}
```

### Integer with Range

```json
{
  "name": "limit",
  "type": "integer",
  "required": false,
  "default": 20,
  "min_value": 1,
  "max_value": 100,
  "description": "Results per page"
}
```

### Enum

```json
{
  "name": "sort_order",
  "type": "enum",
  "required": false,
  "default": "desc",
  "enum_values": ["asc", "desc"],
  "description": "Sort direction"
}
```

### Date with Format

```json
{
  "name": "departure_date",
  "type": "date",
  "required": true,
  "format": "YYYY-MM-DD",
  "description": "Travel date",
  "examples": ["2026-08-22"]
}
```

### Boolean

```json
{
  "name": "include_metadata",
  "type": "boolean",
  "required": false,
  "default": false,
  "description": "Include metadata in response"
}
```

## Usage in Operations

```json
{
  "body": {
    "username": "\"{{username}}\"",
    "limit": "{{limit}}",
    "active": "{{is_active}}",
    "date": "\"{{departure_date}}\"T00:00:00"
  }
}
```

Resolves to:

```json
{
  "body": {
    "username": "john",
    "limit": 50,
    "active": true,
    "date": "2026-08-22T00:00:00"
  }
}
```

See `core/placeholders.md` for complete placeholder syntax.
