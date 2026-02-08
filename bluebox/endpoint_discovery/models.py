"""
bluebox/endpoint_discovery/models.py

Data models for the endpoint discovery pipeline.

Contains Pydantic models for:
- EndpointCallSite: A single location in JS where an HTTP call is made
- DiscoveredEndpoint: A fully resolved API endpoint with method, headers, body
- VariableBinding: A resolved variable assignment in JS scope
- EndpointDiscoveryResult: Aggregated results from analyzing all JS bundles
"""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class CallType(StrEnum):
    """Type of HTTP call pattern detected in JS code."""
    FETCH = "fetch"
    XHR = "XMLHttpRequest"
    AXIOS = "axios"
    JQUERY_AJAX = "jquery_ajax"
    URL_STRING = "url_string"


class VariableBinding(BaseModel):
    """A resolved variable assignment found during static analysis."""
    name: str = Field(description="Variable name (possibly minified)")
    value: str | None = Field(default=None, description="Resolved string value, if deterministic")
    expression: str | None = Field(default=None, description="Raw JS expression if value couldn't be fully resolved")
    source_position: int = Field(default=-1, description="Character offset in the source file")


class EndpointCallSite(BaseModel):
    """A single location in a JS file where an HTTP request is made."""
    call_type: CallType = Field(description="Type of HTTP call pattern (fetch, XHR, axios, etc.)")
    raw_snippet: str = Field(description="Raw JS code surrounding the call site")
    source_url: str = Field(description="URL of the JS file containing this call")
    source_request_id: str = Field(description="Request ID of the JS file in the JSONL")
    position: int = Field(description="Character offset of the call site in the JS source")

    # Extracted fields (may be partial/unresolved)
    url_raw: str = Field(description="The raw URL expression as found in the JS code")
    url_resolved: str | None = Field(default=None, description="Fully resolved URL if all variables could be traced")
    url_template: str | None = Field(
        default=None,
        description="URL with dynamic segments replaced by named placeholders, e.g. /v1/competitions/{competitionId}/seasons/{seasonId}"
    )

    method: str = Field(default="GET", description="HTTP method (GET, POST, etc.)")
    headers: dict[str, str] | None = Field(default=None, description="Request headers if extractable")
    body_raw: str | None = Field(default=None, description="Raw body expression")
    body_schema: dict[str, Any] | None = Field(default=None, description="Inferred body key structure")
    credentials: str | None = Field(default=None, description="Credentials mode (same-origin, include, omit)")
    response_handling: str | None = Field(
        default=None,
        description="How the response is consumed (.json(), .text(), .blob(), etc.)"
    )
    variables_used: list[VariableBinding] = Field(
        default_factory=list,
        description="Variables referenced in this call site that were resolved"
    )


class DiscoveredEndpoint(BaseModel):
    """
    A fully resolved API endpoint, aggregated from one or more call sites.

    Multiple call sites may reference the same endpoint (e.g. the same URL
    called from different parts of the code). This model deduplicates and
    merges information from all call sites.
    """
    url_pattern: str = Field(
        description="Canonical URL pattern with dynamic parts as {param} placeholders"
    )
    base_url: str | None = Field(
        default=None,
        description="Base URL / origin (e.g. https://api.example.com)"
    )
    path: str = Field(description="URL path portion, with {param} placeholders")
    query_params: dict[str, str] | None = Field(
        default=None,
        description="Static query parameters found in the URL"
    )
    methods: list[str] = Field(
        default_factory=lambda: ["GET"],
        description="HTTP methods used to call this endpoint"
    )
    headers: dict[str, str] | None = Field(default=None, description="Common headers across call sites")
    body_schema: dict[str, Any] | None = Field(default=None, description="Merged body key structure from all call sites")
    credentials: str | None = Field(default=None, description="Credentials mode")
    response_type: str | None = Field(default=None, description="Expected response type (.json, .text, etc.)")

    call_sites: list[EndpointCallSite] = Field(
        default_factory=list,
        description="All call sites that reference this endpoint"
    )
    confidence: float = Field(
        default=1.0,
        description="Confidence score 0-1. Higher means more of the URL was statically resolved."
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Tags for categorization (e.g. 'api', 'config', 'tracking', 'cdn')"
    )


class EndpointDiscoveryResult(BaseModel):
    """Aggregated results from analyzing all JS bundles."""
    endpoints: list[DiscoveredEndpoint] = Field(default_factory=list, description="All discovered endpoints")
    call_sites: list[EndpointCallSite] = Field(default_factory=list, description="All raw call sites before dedup")

    # Metadata
    total_js_files_analyzed: int = Field(default=0, description="Number of JS files processed")
    total_unique_js_files: int = Field(default=0, description="Number of unique JS files (by URL)")
    total_call_sites_found: int = Field(default=0, description="Total HTTP call sites detected")
    total_endpoints_discovered: int = Field(default=0, description="Total unique endpoints after dedup")
    js_hosts_analyzed: list[str] = Field(default_factory=list, description="Unique hosts from JS source URLs")

    # Breakdown by call type
    call_type_breakdown: dict[str, int] = Field(
        default_factory=dict,
        description="Count of call sites per CallType"
    )
