# Operations Overview

> All operation types available for routines.

**Code:** [operation.py](web_hacker/data_models/routine/operation.py)

## Navigation

| Type | Purpose | Docs |
|------|---------|------|
| `navigate` | Go to a URL, wait for page load | [navigation.md](navigation.md) |
| `sleep` | Pause execution for N seconds | [navigation.md](navigation.md) |
| `wait_for_url` | Wait until URL matches regex pattern | [navigation.md](navigation.md) |

## HTTP Requests

| Type | Purpose | Docs |
|------|---------|------|
| `fetch` | Make HTTP request via browser fetch API | [fetch.md](fetch.md) |
| `download` | Download binary file, return as base64 | [fetch.md](fetch.md) |
| `get_cookies` | Get all cookies (including HttpOnly) via CDP | [fetch.md](fetch.md) |

## UI Automation

| Type | Purpose | Docs |
|------|---------|------|
| `click` | Click an element by CSS selector | [ui-operations.md](ui-operations.md) |
| `input_text` | Type text into an input field | [ui-operations.md](ui-operations.md) |
| `press` | Press a keyboard key (Enter, Tab, etc.) | [ui-operations.md](ui-operations.md) |
| `scroll` | Scroll page or element | [ui-operations.md](ui-operations.md) |

## Data Extraction

| Type | Purpose | Docs |
|------|---------|------|
| `return` | Return data from sessionStorage as routine result | [data-extraction.md](data-extraction.md) |
| `return_html` | Return current page HTML as routine result | [data-extraction.md](data-extraction.md) |

## Code Execution

| Type | Purpose | Docs |
|------|---------|------|
| `js_evaluate` | Execute JavaScript in browser, store result | [js-evaluate.md](js-evaluate.md) |
