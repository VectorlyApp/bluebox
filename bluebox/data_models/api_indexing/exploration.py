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
    interest_score: float = Field(
        ge=0.0,
        le=1.0,
        description="How useful this endpoint likely is for routine construction (0-1)"
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
        description="Discovered endpoints, sorted by interest_score descending"
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
# UI domain (DOM + Interactions)
# ---------------------------------------------------------------------------


class UIExplorationSummary(BaseModel):
    """High-level summary of the UI domain after exploration.

    The UI explorer combines DOM snapshots and interaction events to answer:
    1. Pages — what pages were visited, what forms/inputs/buttons exist,
       what tokens or data blobs are embedded in the HTML
    2. User inputs — what the user actually typed, selected, or clicked
       (these become routine parameters)
    Everything else (scroll, hover, framework noise) is noise.
    """

    total_snapshots: int = Field(
        description="Total number of DOM snapshots in the capture"
    )
    total_interaction_events: int = Field(
        description="Total number of UI interaction events in the capture"
    )
    pages: list[str] = Field(
        default_factory=list,
        description=(
            "Each entry is a freeform description of one visited page. "
            "Include: URL, page title, what forms/inputs/buttons exist, "
            "any embedded tokens (CSRF in meta tags, hidden inputs), "
            "any server-side data blobs (__NEXT_DATA__, inline JSON). "
            "e.g. '/search' (Flight Search) -- search form with origin, "
            "destination, date inputs + 'Search' submit button. "
            "Hidden input 'csrf_token' with rotating value."
        )
    )
    user_inputs: list[str] = Field(
        default_factory=list,
        description=(
            "Each entry is a freeform description of one user input action. "
            "Include: what element was interacted with (tag, id/name, label), "
            "what value was entered or selected, what type of input it is. "
            "e.g. input#origin (text) -- user typed 'LAX', likely an "
            "airport/station code parameter. "
            "e.g. select#passengers -- user selected '2', numeric dropdown."
        )
    )
    narrative: str = Field(
        default="",
        description="Freeform observations: user flow/journey, page transitions, "
                    "form submission patterns, anything else worth noting"
    )
