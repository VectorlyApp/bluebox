"""
Data models for Phase 1: Exploration.

The exploration phase analyzes raw captured session data across all domains
(network, storage, DOM, interactions) and produces high-level summaries
of what exists. The goal is to filter thousands of raw events down to
the 5-15 endpoints that actually matter, ranked by utility.
"""

from enum import Enum

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Network domain
# ---------------------------------------------------------------------------


class EndpointCategory(str, Enum):
    """How an endpoint is used."""

    ACTION = "action"          # POSTs/PUTs that do something (search, submit, create)
    DATA = "data"              # GETs that return meaningful data (user info, config, prices)
    AUTH = "auth"              # Token endpoints, login, refresh, CSRF fetches
    NAVIGATION = "navigation"  # HTML page loads, redirects


class InterestLevel(str, Enum):
    """How relevant an endpoint is for routine construction."""

    HIGH = "high"      # Core business endpoints the routine will call
    MEDIUM = "medium"  # Supporting endpoints (reference data, config, session mgmt)
    LOW = "low"        # Noise (analytics, tracking, ads, bot detection)


class EndpointCluster(BaseModel):
    """A group of requests to the same logical endpoint."""

    url_pattern: str = Field(
        description="Deduplicated URL pattern, e.g. '/api/v2/search'"
    )
    method: str = Field(
        description="HTTP method, e.g. 'POST'"
    )
    category: EndpointCategory = Field(
        description="What role this endpoint plays"
    )
    hit_count: int = Field(
        description="How many times this endpoint was called in the session"
    )
    description: str = Field(
        description="What this endpoint does, e.g. 'Train route search — returns JSON with schedules and prices'"
    )
    interest: InterestLevel = Field(
        description="How relevant this endpoint is for routine construction"
    )
    request_ids: list[str] = Field(
        default_factory=list,
        description="References to raw captured request IDs for drilling into details later"
    )


class NetworkExplorationSummary(BaseModel):
    """High-level summary of the network domain after exploration."""

    total_requests: int = Field(
        description="Total number of requests in the capture"
    )
    endpoints: list[EndpointCluster] = Field(
        default_factory=list,
        description="Discovered endpoints, sorted by interest level (high first)"
    )
    auth_observations: list[str] = Field(
        default_factory=list,
        description="Observed auth patterns, e.g. 'Bearer JWT on all /api/ requests', 'CSRF header on POSTs'"
    )
    narrative: str = Field(
        default="",
        description="Free-form observations: oddities, patterns, anything that doesn't fit the structured fields"
    )


# ---------------------------------------------------------------------------
# Storage domain
# ---------------------------------------------------------------------------


class StorageExplorationSummary(BaseModel):
    """High-level summary of the storage domain after exploration.

    The storage explorer scans all browser storage (cookies, localStorage,
    sessionStorage, IndexedDB) for two things that matter:
    1. Tokens — auth tokens, CSRF tokens, API keys, session IDs
    2. Data blocks — large structured values cached client-side
    Everything else is noise.
    """

    total_events: int = Field(
        description="Total number of storage events in the capture"
    )
    noise_filtered: int = Field(
        description="Events discarded as noise "
                    "(tracking cookies, analytics IDs, consent flags, etc.)"
    )
    tokens: list[str] = Field(
        default_factory=list,
        description=(
            "Each entry is a freeform description of one discovered token. "
            "Include: where it lives (cookie/localStorage/sessionStorage), "
            "the key name, what kind of token it looks like (JWT, session ID, "
            "CSRF, API key), rough size, and whether it changed during the session. "
            "e.g. sessionStorage[auth_token] -- JWT (~1.2kb), written once on "
            "page load, likely used as Bearer header"
        )
    )
    data_blocks: list[str] = Field(
        default_factory=list,
        description=(
            "Each entry is a freeform description of one meaningful data block. "
            "Include: where it lives, the key name, what it contains, rough size. "
            "e.g. localStorage[user_profile] -- JSON object (~2kb) with name, "
            "email, subscription tier, preferences. Written once."
        )
    )
    narrative: str = Field(
        default="",
        description="Freeform observations: auth patterns, storage lifecycle, "
                    "oddities, cross-domain connections, anything else worth noting"
    )


# ---------------------------------------------------------------------------
# DOM domain
# ---------------------------------------------------------------------------


class DOMExplorationSummary(BaseModel):
    """High-level summary of the DOM domain after exploration.

    The DOM explorer scans all captured browser snapshots to extract:
    1. Pages — what pages were visited, their structure and purpose
    2. Forms — what forms exist, their action URLs, methods, and fields
    3. Embedded tokens — CSRF tokens, API keys, session IDs found in
       meta tags, hidden inputs, or script blocks
    4. Data blobs — server-side data rendered into the DOM
       (__NEXT_DATA__, ld+json, inline JSON config)
    5. Tables — data tables with their columns and what they display
    6. Framework — what frontend stack the site uses (Next.js, Nuxt,
       ASP.NET, Angular, etc.) inferred from DOM signals
    """

    total_snapshots: int = Field(
        description="Total number of DOM snapshots in the capture"
    )
    pages: list[str] = Field(
        default_factory=list,
        description=(
            "Each entry is a freeform description of one visited page. "
            "Include: URL, page title, key sections, overall purpose. "
            "e.g. '/book/flights' (Book Flights) -- flight search page "
            "with search form, promotional banners, and fare calendar."
        )
    )
    forms: list[str] = Field(
        default_factory=list,
        description=(
            "Each entry is a freeform description of one form. "
            "Include: which page it appears on, action URL, HTTP method, "
            "what fields it contains (name, type, placeholder), what it does. "
            "e.g. '/book/flights' search form -- POST to /api/search, "
            "fields: origin (text), destination (text), date (date), "
            "passengers (select 1-9). Submit button: 'Search Flights'."
        )
    )
    embedded_tokens: list[str] = Field(
        default_factory=list,
        description=(
            "Each entry is a freeform description of one token/key found in the DOM. "
            "Include: where it was found (meta tag, hidden input, script block), "
            "the name/key, what it looks like (CSRF, API key, session ID, JWT), "
            "rough size, whether it changes across snapshots. "
            "e.g. meta[name=csrf-token] -- 64-char hex string, rotates per page load. "
            "e.g. hidden input '_RequestVerificationToken' in login form -- ASP.NET anti-forgery token."
        )
    )
    data_blobs: list[str] = Field(
        default_factory=list,
        description=(
            "Each entry is a freeform description of one server-side data blob "
            "embedded in the DOM. Include: what script/element contains it, "
            "what data it holds, rough size, what keys are interesting. "
            "e.g. script#__NEXT_DATA__ -- JSON (~15kb) with props.pageProps "
            "containing station list, feature flags, and user session. "
            "e.g. script[type=application/ld+json] -- structured data with "
            "organization name, logo, social links."
        )
    )
    tables: list[str] = Field(
        default_factory=list,
        description=(
            "Each entry is a freeform description of one data table. "
            "Include: which page it appears on, column headers, row count, "
            "what data it displays. "
            "e.g. '/results' flight results table -- columns: Departure, "
            "Arrival, Duration, Price. 12 rows of flight options."
        )
    )
    inferred_framework: str = Field(
        default="",
        description=(
            "Frontend framework inferred from DOM signals. "
            "e.g. 'Next.js', 'Angular', 'React SPA', 'ASP.NET', 'vanilla/unknown'."
        )
    )
    narrative: str = Field(
        default="",
        description=(
            "Freeform observations: navigation flow, page transitions, "
            "dynamic content patterns, anything else worth noting."
        )
    )


# ---------------------------------------------------------------------------
# UI / Interaction domain
# ---------------------------------------------------------------------------


class UIExplorationSummary(BaseModel):
    """High-level summary of user interactions after exploration.

    The UI explorer analyzes recorded interaction events (clicks, typed
    values, selections) and optionally cross-references with DOM snapshots
    to understand what the user did and what they were trying to accomplish.
    """

    total_events: int = Field(
        description="Total number of interaction events in the capture"
    )
    user_inputs: list[str] = Field(
        default_factory=list,
        description=(
            "Each entry describes one user input action. "
            "Include: element (tag, id/name), value entered/selected, input type. "
            "e.g. input#origin (text) -- user typed 'LAX', airport code parameter."
        )
    )
    clicks: list[str] = Field(
        default_factory=list,
        description=(
            "Each entry describes one meaningful click action. "
            "Include: element (tag, id/class/text), what it does, which page. "
            "e.g. button.search-btn 'Search Flights' on /book/flights -- submits search form."
        )
    )
    navigation_flow: list[str] = Field(
        default_factory=list,
        description=(
            "Ordered list of page transitions the user made. "
            "e.g. '/ (Home) -> /book/flights (Search) -> /results (Results)'."
        )
    )
    inferred_intent: str = Field(
        default="",
        description=(
            "What was the user trying to accomplish? Infer from pages visited, "
            "forms filled, values entered, and navigation flow. "
            "e.g. 'Search for one-way flights from ATL to LAX on 2025-03-15'."
        )
    )
    narrative: str = Field(
        default="",
        description=(
            "Freeform observations: interaction patterns, user behavior, "
            "anything else worth noting about what the user did."
        )
    )
