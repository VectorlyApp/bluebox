# Exploration Specialists

Phase 1 agents that analyze CDP captures across 4 domains. Each specialist runs autonomously, reads captured data through a domain-specific DataLoader, and produces a structured exploration summary.

All specialists extend `AbstractSpecialist` and share the same lifecycle:
1. Receive a DataLoader and output schema
2. Run autonomous loop (3-10 LLM iterations)
3. Use domain-specific tools to analyze captures
4. Call `finalize_with_output()` with the structured summary

## AbstractSpecialist Base

Every specialist inherits:

| Feature | Description |
|---------|-------------|
| **RunMode** | `CONVERSATIONAL` (interactive) or `AUTONOMOUS` (loop with finalization) |
| **AutonomousConfig** | `min_iterations` (3) before finalize tools unlock, `max_iterations` (10) before forced stop. One iteration = one LLM API call (which may include multiple tool calls). |
| **Output schema injection** | Schema is injected into `finalize_with_output` tool so the LLM sees exact required fields |
| **Urgency notices** | As iterations approach max, LLM gets "Only N iterations left" nudges |
| **Finalize gating** | `finalize_with_output` / `finalize_with_failure` only available after min_iterations |

## NetworkSpecialist

**Data:** `NetworkDataLoader` — HTTP request/response traffic
**Finds:** API endpoints, auth patterns, request shapes, interest levels

### Tools

| Tool | Description |
|------|-------------|
| `search_responses_by_terms(terms, top_n)` | Relevance-ranked search across network entries |
| `get_entry_detail(request_id)` | Full request/response for a specific transaction |
| `get_response_body_schema(request_id)` | JSON schema of a response body |
| `get_unique_urls()` | All captured URLs |
| `filter_by_method(method)` | Filter by HTTP method |
| `filter_by_host(host)` | Filter by hostname |

### Output: `NetworkExplorationSummary`

- `total_requests` — count of captured HTTP transactions
- `endpoints[]` — each with url_pattern, method, category (data/auth/action), interest level, description
- `auth_observations[]` — observed authentication patterns
- `narrative` — free-form observations

## DOMSpecialist

**Data:** `DOMDataLoader` — page DOM snapshots with string interning
**Finds:** Page structure, forms, embedded tokens, framework data blobs, scripts

### Tools

| Tool | Description |
|------|-------------|
| `list_pages()` | All captured pages with URLs, titles, string counts |
| `get_elements(element_type)` | Inputs, buttons, links, headings, meta_tags, hidden_inputs, clickable |
| `get_forms(snapshot_index?)` | Forms with action URLs, methods, child inputs |
| `get_tables(snapshot_index?)` | Data tables with headers and row counts |
| `get_scripts(snapshot_index?)` | Script tags: `__NEXT_DATA__`, `__NUXT__`, inline JSON, ld+json |
| `get_text_content(snapshot_index)` | Visible text content |
| `search_strings(value)` | Search across snapshot string tables |
| `get_snapshot_diff(index_a, index_b)` | What changed between two snapshots |
| `get_navigation_sequence()` | Ordered page visit sequence |

### Output: `DOMExplorationSummary`

- `total_snapshots` — number of page snapshots
- `pages[]` — page descriptions with URLs
- `forms[]` — form descriptions (action, method, inputs)
- `embedded_tokens[]` — tokens in meta tags, hidden inputs
- `data_blobs[]` — data in script tags
- `tables[]` — data table descriptions
- `inferred_framework` — Angular, React, Next.js, etc.
- `narrative` — observations about page structure

## StorageSpecialist

**Data:** `StorageDataLoader` — cookies, localStorage, sessionStorage, IndexedDB events
**Finds:** Auth tokens, session data, cached configuration

### Tools

Tools for searching and filtering storage events by type, origin, key, and value.

### Output: `StorageExplorationSummary`

- `total_events` — total storage mutation events
- `noise_filtered` — events filtered as noise
- `tokens[]` — auth-relevant tokens with storage type, lifecycle, likely use
- `data_blocks[]` — significant stored data (JSON objects, config)
- `narrative` — storage patterns, auth lifecycle

## InteractionSpecialist (UI)

**Data:** `InteractionsDataLoader` — user clicks, inputs, selections, navigation
**Finds:** User intent, interaction flow, form submissions, parameter values

### Tools

Tools for filtering interactions by type and element, extracting form inputs, and deduplicating elements.

### Output: `UIExplorationSummary`

- `total_events` — total interaction events
- `user_inputs[]` — typed values with element info (parameter discovery)
- `clicks[]` — clicked elements with context
- `navigation_flow[]` — page-to-page sequence
- `inferred_intent` — what the user was trying to do
- `narrative` — interaction patterns

## How Summaries Feed the PI

All 4 summaries are serialized to JSON and injected into the PI's system prompt:

```
## Exploration Summaries

### Network
{network_summary_json}

### DOM
{dom_summary_json}

### Storage
{storage_summary_json}

### UI
{ui_summary_json}
```

The PI reads these to understand:
- What APIs exist and how they're structured (network)
- What the site looks like and what tokens are embedded (DOM)
- What auth state lives in the browser (storage)
- What the user actually did and intended (UI)

## Files

- `bluebox/agents/specialists/abstract_specialist.py`
- `bluebox/agents/specialists/network_specialist.py`
- `bluebox/agents/specialists/dom_specialist.py`
- `bluebox/agents/specialists/storage_specialist.py`
- `bluebox/agents/specialists/interaction_specialist.py`
