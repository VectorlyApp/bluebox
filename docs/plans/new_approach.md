# New Discovery Approach

## Phase 1: Exploration

### Goal

Before attempting to discover or build anything, understand **what exists** in the captured session. This phase answers: "What did the user do, what happened under the hood, and what's the API surface area?"

### Why This Matters

The current system skips exploration entirely — it jumps straight to "find endpoint X for task Y." This means the orchestrator is searching for a needle without knowing what the haystack looks like. It can't distinguish important endpoints from noise, doesn't know the auth model, and has no sense of the session's overall shape.

Exploration gives the orchestrator a **map** before it starts navigating.

### The Four Domains

All browser activity falls into four observable domains. Each one tells a different part of the story:

| Domain | What It Contains | What It Reveals |
|--------|-----------------|-----------------|
| **Network** | HTTP requests/responses, WebSocket messages | API endpoints, auth flows, data fetches, error patterns |
| **Storage** | Cookies, sessionStorage, localStorage, IndexedDB | Auth tokens, session state, cached data, user preferences |
| **DOM** | Page structure, dynamic changes, rendered content | Forms, interactive elements, data display patterns |
| **Interactions** | Clicks, typing, scrolling, form submissions | User intent, input fields, the task flow as performed |

No single domain tells the full story. A form submission (interactions) triggers a POST (network) that returns a token (stored in storage) which is rendered on the page (DOM). The value of exploration is connecting these pieces.

### How It Works

Each domain gets a dedicated explorer agent. The explorer's job is simple: **look at everything, summarize what's interesting.**

#### Pass 1: Independent Exploration

Each domain agent receives only its own data and produces a structured summary.

**Network Explorer** answers:
- What hosts are contacted? Which serve APIs vs static content?
- What are the distinct API endpoints? (group by URL pattern, not individual requests)
- What HTTP methods and content types are used?
- What auth patterns are visible? (Bearer tokens, cookies, CSRF headers, API keys)
- Which requests returned errors? What kind?
- What's the request sequence? (what depends on what, based on ordering)

**Storage Explorer** answers:
- What's in cookies? Which look like auth/session tokens vs tracking?
- What's in sessionStorage/localStorage? Any JWT-shaped values? API keys?
- What changed during the session? (values that were written/updated)
- What storage keys correlate with each other? (e.g., `auth.token` and `auth.refresh_token`)

**DOM Explorer** answers:
- What pages were visited? What's the navigation sequence?
- What forms exist? What fields do they contain?
- What interactive elements are present? (buttons, dropdowns, search bars)
- What dynamic content appeared? (search results, loaded data, error messages)

**Interactions Explorer** answers:
- What did the user actually do? (click sequence, form fills, navigation)
- What values did the user input? (these are likely routine parameters)
- What was the user's apparent goal? (inferred from the action sequence)
- What elements did the user interact with? (CSS selectors, element types)

#### Pass 1 Output Schema

Each explorer produces a structured summary, not free text:

```python
class DomainSummary(BaseModel):
    domain: str                          # "network", "storage", "dom", "interactions"
    highlights: list[str]                # top 3-5 most important findings
    entities: list[DomainEntity]         # discovered things (endpoints, tokens, forms, etc.)
    patterns: list[str]                  # observed patterns ("all API calls include X-CSRF header")
    open_questions: list[str]            # things this domain can't answer alone
                                         # ("POST /search requires a token — where does it come from?")

class DomainEntity(BaseModel):
    name: str                            # human-readable name
    entity_type: str                     # "api_endpoint", "auth_token", "form", "user_action", etc.
    details: dict                        # domain-specific details
    importance: str                      # "high", "medium", "low"
    related_to: list[str]               # references to other entities by name (even cross-domain guesses)
```

#### Pass 2: Cross-Domain Enrichment

Each domain agent now receives:
- Its own data (same as before)
- The **summaries from all other domains** (not their raw data — just the `DomainSummary` objects)

With this context, each agent re-explores and produces a **refined summary**. The key difference:

**Network Explorer (pass 2)** — now knows the user typed "NYC" into a form (from interactions) → can label the `origin` field in `POST /search` as a user parameter. Now knows there's a JWT in sessionStorage (from storage) → can identify which requests use it.

**Storage Explorer (pass 2)** — now knows which API calls exist (from network) → can map tokens to the endpoints that consume them. Now knows the user filled a search form (from interactions) → can distinguish session state from user data.

**DOM Explorer (pass 2)** — now knows what fetches happened (from network) → can connect page elements to API calls. Now knows what the user clicked (from interactions) → can identify which elements are actionable.

**Interactions Explorer (pass 2)** — now knows the API endpoints (from network) → can map user actions to their backend effects. Now knows what tokens exist (from storage) → can flag which user actions triggered auth flows.

#### Pass 2 Output

Same `DomainSummary` schema, but richer. The `open_questions` should be fewer (cross-domain context answered some). The `related_to` fields should be more precise (guesses become confirmed connections).

### Merged Session Overview

After both passes, the four refined summaries are combined (deterministically, no LLM needed) into a **SessionOverview**:

```python
class SessionOverview(BaseModel):
    """The complete picture of what happened in this capture session."""

    # Site identity
    site: str                                 # primary domain
    hosts: list[str]                          # all contacted hosts

    # API surface
    api_endpoints: list[EndpointSummary]      # discovered API endpoints
    auth_model: AuthModelSummary              # how auth works on this site

    # User journey
    pages_visited: list[str]                  # navigation sequence
    user_actions: list[UserActionSummary]     # what the user did, in order
    user_inputs: dict[str, str]              # values the user typed (likely parameters)

    # Cross-domain connections
    connections: list[CrossDomainConnection]   # "form submit → POST /search → results displayed"

    # Open questions
    unresolved: list[str]                     # things none of the domains could fully explain

class EndpointSummary(BaseModel):
    url_pattern: str                          # "/api/v2/search"
    method: str
    purpose: str                              # "train route search"
    auth_required: bool
    auth_mechanism: str | None                # "bearer_token", "cookie", "csrf_header"
    parameters: list[str]                     # known parameters (from form fields or request body)
    depends_on: list[str]                     # other endpoints this one needs (e.g., auth)

class AuthModelSummary(BaseModel):
    mechanism: str                            # "jwt_in_session_storage", "cookie_based", "api_key", etc.
    token_source: str | None                  # where the token comes from
    token_storage: str | None                 # where it's kept
    refresh_flow: str | None                  # how it gets refreshed, if observed
    affected_endpoints: list[str]             # which endpoints require this auth

class CrossDomainConnection(BaseModel):
    description: str                          # human-readable: "User clicks 'Search' → POST /api/search"
    trigger: str                              # "interaction:click_search_button"
    effect: str                               # "network:post_api_search"
    confidence: float                         # how sure we are about this connection
```

### What the Orchestrator Gets

After Phase 1, the orchestrator's system prompt includes a compact version of the `SessionOverview`:

```
SESSION OVERVIEW (from exploration):
- Site: amtrak.com (3 hosts: www.amtrak.com, api.amtrak.com, auth.amtrak.com)
- API endpoints found: 4
  1. POST /dotcom/api/search — train route search (auth: csrf header)
  2. GET /auth/token — token refresh (auth: cookie)
  3. GET /api/user/preferences — user prefs (auth: bearer)
  4. POST /api/analytics/event — analytics (no auth) [LOW PRIORITY]
- Auth: JWT in sessionStorage, sourced from GET /auth/token, refreshed on page load
- User journey: landed on homepage → navigated to tickets → filled search form → clicked search
- User inputs: origin="NYC", destination="BOS", date="2024-03-15", passengers="1"
- Key connection: form submit on /tickets → POST /dotcom/api/search
```

This is ~150 tokens. The orchestrator now knows the full landscape and can make informed decisions about what to experiment with, what dependencies to expect, and what the user was trying to do.

### When to Skip or Shorten

Not every session needs full two-pass exploration:

| Session complexity | Approach |
|---|---|
| Simple (1 form, 1 API call, basic auth) | Single pass may be enough — or skip straight to experimentation |
| Medium (3-5 endpoints, token auth) | Full two-pass exploration |
| Complex (many endpoints, OAuth, multi-step flows) | Full exploration + potentially a third targeted pass on auth |
| API index mode (map everything) | Full exploration is mandatory — it IS the first step of cataloging |

### Cost Considerations

- **Pass 1**: 4 parallel LLM calls (one per domain). Each processes its domain's data.
- **Pass 2**: 4 parallel LLM calls. Each processes its own data + 3 small summaries.
- **Total**: 8 LLM calls before any experimentation begins.

For simple sessions this is expensive relative to the task. A complexity heuristic (count of unique endpoints, count of storage keys, count of interactions) could decide whether to run the full two-pass or a lightweight single-pass variant.

### Relationship to Other Phases

Phase 1 (Exploration) feeds directly into:
- **Experimentation (Proposal #7)**: The orchestrator knows which endpoints to test, what auth to expect, and where the user's parameters are
- **API Index (Proposal #8)**: The `api_endpoints` list from SessionOverview becomes the candidate list for catalog mapping
- **Evaluator (Proposal #1)**: The SessionOverview provides ground truth for evaluating whether a routine captures the right flow
