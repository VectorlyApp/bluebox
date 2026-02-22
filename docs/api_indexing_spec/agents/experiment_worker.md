# ExperimentWorker

The hands-on agent that executes experiments in a live browser. It has TWO sources of truth:

1. **Captured session data** (reference) — what happened in the recorded browsing session
2. **Live browser** (reality) — what happens when we try it now

The worker bridges the gap: it looks up the captured session for context, then tests hypotheses in the live browser.

## What It Sees

| Context | Source | How |
|---------|--------|-----|
| Experiment task | PI's experiment prompt | Initial autonomous message |
| Data availability summary | System prompt | Stats only (counts) — e.g. "265 requests, 42 unique URLs" — NOT the full exploration summaries |
| Captured network traffic | `capture_search_transactions`, `capture_get_transaction` | On-demand tool calls |
| Captured storage | `capture_search_storage` | On-demand tool calls |
| Captured DOM | `capture_get_page_structure`, `capture_get_element` | On-demand tool calls |
| Cross-source value tracing | `capture_trace_value` | On-demand tool calls |
| Live browser | `browser_navigate`, `browser_eval_js`, `browser_cdp_command`, `browser_get_dom` | On-demand tool calls |

## What It Returns

Structured output via `finalize_with_output()` — the schema is defined by the PI per experiment. Typically contains:
- Whether the hypothesis was confirmed/refuted
- Specific values discovered (tokens, URLs, response shapes)
- Error details if something failed

## Tools

### Browser Tools (live)

| Tool | What It Does |
|------|-------------|
| `browser_navigate(url)` | Navigate to URL, wait for page load. **TIP:** Navigating to an API URL bypasses CORS. |
| `browser_eval_js(expression)` | Execute JavaScript in page context. Swiss army knife: fetch(), DOM reads, clicks, storage access. |
| `browser_cdp_command(method, params)` | Raw Chrome DevTools Protocol command. Powerful for CORS bypasses (Fetch.enable, Network.setExtraHTTPHeaders). |
| `browser_get_dom(selector?, max_depth?)` | Filtered DOM view as JSON tree. Shows key attributes (id, class, name, type, href, etc.). |

### Capture Lookup Tools (reference)

| Tool | What It Does |
|------|-------------|
| `capture_search_transactions(query)` | Search recorded HTTP traffic by keyword. Returns ranked results with request IDs. |
| `capture_get_transaction(request_id)` | Full request/response for a specific captured transaction. Headers, body, status. |
| `capture_search_storage(query)` | Search recorded cookies, localStorage, sessionStorage events. |
| `capture_trace_value(value)` | Cross-source search — find where a token/value appeared across network, storage, DOM, window properties. |
| `capture_get_page_structure(snapshot_index)` | Forms, inputs, meta tags, scripts from recorded DOM snapshot. |
| `capture_get_element(element_type, snapshot_index?)` | Specific element types from recorded DOM. |

### Analysis Tools

| Tool | What It Does |
|------|-------------|
| `execute_python(code)` | Execute Python in sandboxed environment. Pre-loaded: `network_entries`, `storage_entries`, `window_prop_entries`. No imports. |

### Lifecycle Tools

| Tool | What It Does |
|------|-------------|
| `add_note(note)` | Record notes/warnings attached to the result. |
| `finalize_with_output(output)` | Submit structured result (available after min_iterations). |
| `finalize_with_failure(reason)` | Mark experiment as failed. |

## Browser Lifecycle

- Worker lazily creates a **persistent incognito browser tab** on first browser tool call
- Tab persists across the entire experiment (navigate, eval JS, etc.)
- Tab is cleaned up when the worker finishes or times out
- Workers are independent — each gets its own tab, no shared state

## How the PI Uses Workers

The PI writes experiment prompts that tell the worker exactly what to test:

```
"The captured session shows a token endpoint at POST /api/prod-token/api/v1/token.
1. Use capture_get_transaction to get the EXACT headers and body from the capture
2. Navigate to https://www.spirit.com first to establish cookies
3. Call the token endpoint with the same headers/body using browser_eval_js fetch()
4. If it returns a token, store it and try calling /api/prod-availability/api/availability/v3/search
5. Report: did you get a valid token? What did the availability response look like?"
```

Key rules for the PI when writing experiment prompts:
- Reference worker tools by name
- Include specific URLs, headers, body shapes from exploration summaries
- Include observed values from captures (so the worker knows what to look for)
- Explain multiple fallback strategies (if fetch fails, try CDP, try navigation)
- Workers are stateless — include ALL auth instructions in every prompt

## Timeouts

Workers are wrapped in a `ThreadPoolExecutor` with a 180-second timeout (PI's `WORKER_TIMEOUT_SECONDS`). If a worker hangs (LLM or browser), the PI gets a timeout error and can dispatch a new experiment.

## File

`bluebox/agents/workers/experiment_worker.py`
