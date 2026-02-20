# UI Operations

> Operations that simulate user input: `click`, `input_text`, `press`, `scroll`.

**Code:** [operation.py](bluebox/data_models/routine/operation.py)

## Finding Selectors

**Before writing UI operations, you need to know what elements exist on the page.**

### Method 1: Use `return_html` to inspect the DOM

```json
[
  {"type": "navigate", "url": "https://example.com"},
  {"type": "return_html", "scope": "page"}
]
```

Run this first, then examine the HTML to find CSS selectors for buttons, inputs, etc.

### Method 2: Use `js_evaluate` to query elements

```json
{
  "type": "js_evaluate",
  "js": "(()=>{ return Array.from(document.querySelectorAll('input, button, select')).map(el => ({ tag: el.tagName, id: el.id, name: el.name, class: el.className })); })()",
  "session_storage_key": "elements"
}
```

This returns a list of interactive elements you can target.

---

## click

Clicks an element by CSS selector. Validates visibility to avoid hidden honeypot traps.

```json
{
  "type": "click",
  "selector": "button[type='submit']"
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `selector` | string | Yes | - | CSS selector for element |
| `button` | `"left"` \| `"right"` \| `"middle"` | No | `"left"` | Mouse button |
| `click_count` | int | No | 1 | Number of clicks |
| `timeout_ms` | int | No | 20000 | Wait time for element |
| `ensure_visible` | bool | No | true | Scroll into view first |

**Examples:**
```json
{"type": "click", "selector": "#search-button"}
{"type": "click", "selector": "input[name='submit']"}
{"type": "click", "selector": ".btn-primary"}
{"type": "click", "selector": "select[name='dropdown']"}
```

---

## input_text

Types text into an input element. Validates visibility to avoid hidden honeypot inputs.

```json
{
  "type": "input_text",
  "selector": "input[name='username']",
  "text": "{{username}}",
  "clear": true
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `selector` | string | Yes | - | CSS selector for input |
| `text` | string | Yes | - | Text to type (supports placeholders) |
| `clear` | bool | No | false | Clear existing text first |
| `timeout_ms` | int | No | 20000 | Wait time for element |

**Examples:**
```json
{"type": "input_text", "selector": "#email", "text": "{{email}}", "clear": true}
{"type": "input_text", "selector": "input[name='q']", "text": "{{search_query}}"}
{"type": "input_text", "selector": "textarea.comment", "text": "{{message}}"}
```

---

## press

Presses a keyboard key. Useful for form submission, dropdown navigation, etc.

```json
{
  "type": "press",
  "key": "enter"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `key` | string | Yes | Key to press |

**Supported keys:**
- Navigation: `enter`, `tab`, `escape`, `backspace`, `delete`
- Arrows: `arrowup`, `arrowdown`, `arrowleft`, `arrowright`
- Page: `home`, `end`, `pageup`, `pagedown`
- Modifiers: `shift`, `control`, `alt`, `meta`
- Other: `space`, any single character

**Examples:**
```json
{"type": "press", "key": "enter"}
{"type": "press", "key": "tab"}
{"type": "press", "key": "arrowdown"}
```

---

## scroll

Scrolls the page or a specific element.

```json
{
  "type": "scroll",
  "delta_y": 500
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `selector` | string | No | - | Element to scroll (window if omitted) |
| `x` | int | No | - | Absolute x position (window only) |
| `y` | int | No | - | Absolute y position (window only) |
| `delta_x` | int | No | 0 | Relative horizontal scroll |
| `delta_y` | int | No | 0 | Relative vertical scroll |
| `behavior` | `"auto"` \| `"smooth"` | No | `"auto"` | Scroll animation |
| `timeout_ms` | int | No | 20000 | Wait time for element |

**Examples:**
```json
{"type": "scroll", "delta_y": 300}
{"type": "scroll", "y": 0}
{"type": "scroll", "selector": ".scrollable-div", "delta_y": 200}
{"type": "scroll", "delta_y": 100, "behavior": "smooth"}
```

---

## Common Patterns

### Form submission
```json
[
  {"type": "click", "selector": "input[name='email']"},
  {"type": "input_text", "selector": "input[name='email']", "text": "{{email}}", "clear": true},
  {"type": "click", "selector": "button[type='submit']"}
]
```

### Dropdown selection
```json
[
  {"type": "click", "selector": "select[name='country']"},
  {"type": "press", "key": "arrowdown"},
  {"type": "press", "key": "arrowdown"},
  {"type": "press", "key": "enter"}
]
```

### Search form
```json
[
  {"type": "input_text", "selector": "#search-input", "text": "{{query}}", "clear": true},
  {"type": "press", "key": "enter"}
]
```

### Scroll then click
```json
[
  {"type": "scroll", "delta_y": 500},
  {"type": "sleep", "timeout_seconds": 0.5},
  {"type": "click", "selector": "#load-more"}
]
```

---

## Tips

1. **Always inspect the DOM first** - Use `return_html` or `js_evaluate` before writing UI operations
2. **Use specific selectors** - Prefer `input[name='email']` over `.input-field`
3. **Add sleep after navigation** - Pages need time to load before elements are available
4. **Use `clear: true`** - When typing into inputs that may have existing values
5. **Scroll before clicking** - Some elements only become visible after scrolling
