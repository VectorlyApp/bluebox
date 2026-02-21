# Exploration Phase

Phase 1 of the API Indexing Pipeline. Four specialist agents run in parallel to analyze different domains of the CDP captures. Their outputs — exploration summaries — become the PI's understanding of the target site.

## How It Runs

```python
# run_api_indexing.py → run_explorations()
ThreadPoolExecutor(max_workers=4)
├── run_network_exploration()  → exploration/network.json
├── run_dom_exploration()      → exploration/dom.json
├── run_storage_exploration()  → exploration/storage.json
└── run_ui_exploration()       → exploration/ui.json
```

Each specialist:
1. Receives a **DataLoader** (parsed JSONL captures)
2. Receives an **output schema** (the exploration summary structure)
3. Runs autonomously for 3-10 LLM iterations
4. Calls `finalize_with_output()` with the structured summary
5. Result is saved to `exploration/{domain}.json`

## Four Domains

### Network Exploration

**Specialist:** `NetworkSpecialist`
**Data:** `NetworkDataLoader` (HTTP request/response events)

**What it discovers:**
- API endpoint URLs, methods, status codes
- Request/response shapes (headers, body structure)
- Auth patterns (Bearer tokens, API keys, subscription keys)
- Endpoint categorization (data, auth, action, navigation)
- Interest levels (high/medium/low) per endpoint

**Output: `NetworkExplorationSummary`**
```json
{
  "total_requests": 265,
  "endpoints": [
    {
      "url_pattern": "/api/prod-availability/api/availability/v3/search",
      "method": "POST",
      "category": "data",
      "hit_count": 2,
      "description": "Full flight availability search with trips, journeys, fares",
      "interest": "high",
      "request_ids": ["interception-job-1638.0"]
    }
  ],
  "auth_observations": [
    "Bearer JWT on all /api/ requests, obtained from POST /api/prod-token/api/v1/token",
    "Ocp-Apim-Subscription-Key header required on all prod-* API calls"
  ],
  "narrative": "Free-form observations about the site's API architecture..."
}
```

### DOM Exploration

**Specialist:** `DOMSpecialist`
**Data:** `DOMDataLoader` (full page DOM snapshots with string interning)

**What it discovers:**
- Page structure and navigation sequence
- Forms with action URLs, methods, and input fields
- Embedded tokens (meta tags, hidden inputs, CSRF tokens)
- Data blobs in script tags (`__NEXT_DATA__`, `__NUXT__`, inline JSON, ld+json)
- Framework inference (Angular, React, Next.js, etc.)
- Anti-bot/security scripts (PerimeterX, Akamai, Dynatrace)

**Output: `DOMExplorationSummary`**
```json
{
  "total_snapshots": 2,
  "pages": [
    "[0] https://www.spirit.com/ — interstitial/loading page",
    "[1] https://www.spirit.com/ — main SPA with booking form"
  ],
  "forms": ["Search form POST /api/search — origin, destination, date, passengers"],
  "embedded_tokens": ["meta[name=csrf-token] — 64-char hex, rotates per page load"],
  "data_blobs": ["script#__NEXT_DATA__ — 15kb JSON with station list, feature flags"],
  "tables": [],
  "inferred_framework": "Angular",
  "narrative": "Anti-bot challenge flow, extensive third-party integrations..."
}
```

### Storage Exploration

**Specialist:** `StorageSpecialist`
**Data:** `StorageDataLoader` (cookies, localStorage, sessionStorage, IndexedDB)

**What it discovers:**
- Auth tokens and session data in browser storage
- Cookie lifecycle (what's set, when, by whom)
- Cached API keys and configuration
- Storage-based state management patterns

**Output: `StorageExplorationSummary`**
```json
{
  "total_events": 1050,
  "noise_filtered": 800,
  "tokens": [
    "sessionStorage[auth_token] — JWT (~1.2kb), written on page load, used as Bearer"
  ],
  "data_blocks": [
    "localStorage[user_profile] — JSON with name, email, subscription tier"
  ],
  "narrative": "Auth lifecycle, cross-domain cookie patterns..."
}
```

### UI Exploration

**Specialist:** `InteractionSpecialist`
**Data:** `InteractionsDataLoader` (user clicks, typed inputs, navigation events)

**What it discovers:**
- What the user actually did during the recorded session
- Form inputs with values (search terms, dates, codes)
- Click targets (buttons, links, tabs)
- Navigation flow (page sequence)
- Inferred user intent

**Output: `UIExplorationSummary`**
```json
{
  "total_events": 47,
  "user_inputs": [
    "input#origin (text) — user typed 'BOS', airport code",
    "input#departure (date) — user selected '2026-04-06'"
  ],
  "clicks": [
    "button.search-btn 'Search Flights' — submits search form"
  ],
  "navigation_flow": ["/ → /book/flights → /results"],
  "inferred_intent": "Search one-way flights from BOS to ATL on 2026-04-06",
  "narrative": "Interaction patterns and observed user behavior..."
}
```

## How the PI Uses Exploration Summaries

All four summaries are injected into the PI's **system prompt** as context. The PI never accesses raw captures directly — it reads the summaries to understand:

1. **What endpoints exist** (network) — which APIs to build routines for
2. **What auth is needed** (network + storage) — tokens, keys, cookies
3. **What the pages look like** (DOM) — forms, embedded data, framework
4. **What the user did** (UI) — intent, parameters, workflow

The PI then plans a catalog of routines and dispatches experiments to workers who have both capture lookup tools AND a live browser.

## Skipping Exploration

If exploration was already run (summaries exist on disk), use `--skip-exploration` to jump directly to Phase 2:

```bash
python -m bluebox.scripts.run_api_indexing \
    --cdp-captures-dir ./cdp_captures \
    --task "..." \
    --skip-exploration \
    --output-dir ./api_indexing_output
```

The pipeline loads existing summaries from `output_dir/exploration/`.
