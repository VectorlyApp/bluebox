"""
bluebox/llms/infra/network_data_store.py

Data store for network traffic analysis.

Parses JSONL files with NetworkTransactionEvent entries and provides
structured access to network traffic data.
"""

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from bluebox.data_models.cdp import NetworkTransactionEvent
from bluebox.utils.logger import get_logger


logger = get_logger(name=__name__)


def _get_host(url: str) -> str:
    """Extract host from URL."""
    return urlparse(url).netloc


def _get_path(url: str) -> str:
    """Extract path from URL."""
    return urlparse(url).path


@dataclass
class NetworkStats:
    """Summary statistics for a HAR file."""

    total_requests: int = 0
    total_request_bytes: int = 0
    total_response_bytes: int = 0
    total_time_ms: float = 0.0

    methods: dict[str, int] = field(default_factory=dict)
    status_codes: dict[int, int] = field(default_factory=dict)
    content_types: dict[str, int] = field(default_factory=dict)
    hosts: dict[str, int] = field(default_factory=dict)

    unique_hosts: int = 0
    unique_paths: int = 0
    unique_urls: int = 0

    has_cookies: bool = False
    has_auth_headers: bool = False
    has_json_requests: bool = False
    has_form_data: bool = False

    def to_summary(self) -> str:
        """Generate a human-readable summary."""
        lines = [
            f"Total Requests: {self.total_requests}",
            f"Unique Hosts: {self.unique_hosts}",
            f"Unique Paths: {self.unique_paths}",
            f"Total Request Size: {self._format_bytes(self.total_request_bytes)}",
            f"Total Response Size: {self._format_bytes(self.total_response_bytes)}",
            f"Total Time: {self.total_time_ms:.0f}ms",
            "",
            "Methods:",
        ]
        for method, count in sorted(self.methods.items(), key=lambda x: -x[1]):
            lines.append(f"  {method}: {count}")

        lines.append("")
        lines.append("Status Codes:")
        for status, count in sorted(self.status_codes.items()):
            lines.append(f"  {status}: {count}")

        lines.append("")
        lines.append("Top Hosts:")
        for host, count in sorted(self.hosts.items(), key=lambda x: -x[1])[:10]:
            lines.append(f"  {host}: {count}")

        lines.append("")
        lines.append("Top Content Types:")
        for ctype, count in sorted(self.content_types.items(), key=lambda x: -x[1])[:10]:
            lines.append(f"  {ctype}: {count}")

        lines.append("")
        lines.append("Features Detected:")
        if self.has_cookies:
            lines.append("  - Cookies present")
        if self.has_auth_headers:
            lines.append("  - Authentication headers present")
        if self.has_json_requests:
            lines.append("  - JSON request bodies present")
        if self.has_form_data:
            lines.append("  - Form data present")

        return "\n".join(lines)

    @staticmethod
    def _format_bytes(num_bytes: int) -> str:
        """Format bytes as human-readable string."""
        for unit in ["B", "KB", "MB", "GB"]:
            if abs(num_bytes) < 1024:
                return f"{num_bytes:.1f} {unit}"
            num_bytes /= 1024  # type: ignore
        return f"{num_bytes:.1f} TB"


class NetworkDataStore:
    """
    Data store for HAR file analysis.

    Parses HAR content and provides structured access to network traffic data
    including entries, statistics, and search capabilities.

    Usage:
        har_content = open("network.har").read()
        store = NetworkDataStore(har_content)

        print(store.stats.to_summary())
        api_calls = store.search_entries(path_contains="/api/")
    """

    AUTH_HEADERS = frozenset([
        "authorization",
        "x-api-key",
        "x-auth-token",
        "x-access-token",
        "api-key",
        "bearer",
    ])

    # MIME types to exclude (JS, images, media, fonts, etc.)
    EXCLUDED_MIME_PREFIXES = (
        "application/javascript",
        "application/x-javascript",
        "text/javascript",
        "image/",
        "video/",
        "audio/",
        "font/",
        "application/font",
        "application/octet-stream",
    )

    # MIME types to include (HTML, JSON, XML, text)
    INCLUDED_MIME_PREFIXES = (
        "application/json",
        "text/html",
        "text/xml",
        "application/xml",
        "text/plain",
    )

    # Key terms for identifying important API endpoints (singular form to catch plurals too)
    # These are case-insensitive substring matches
    API_KEY_TERMS = (
        # Core API identifiers
        "api",
        "graphql",
        "rest",
        "rpc",
        # Authentication & User
        "auth",
        "login",
        "logout",
        "oauth",
        "token",
        "session",
        "user",
        "account",
        "profile",
        "register",
        "signup",
        # Data Operations
        "search",
        "query",
        "filter",
        "fetch",
        "create",
        "update",
        "delete",
        "submit",
        "save",
        # E-commerce/Transactions
        "cart",
        "checkout",
        "order",
        "payment",
        "purchase",
        "booking",
        "reserve",
        "quote",
        "price",
        # Content
        "content",
        "data",
        "item",
        "product",
        "result",
        "detail",
        "info",
        "summary",
        "list",
        # Actions
        "action",
        "execute",
        "process",
        "validate",
        "verify",
        "confirm",
        "send",
        # Events & Tracking
        "event",
        "track",
        "analytic",
        "metric",
        "log",
        # Autocomplete & Suggestions
        "autocomplete",
        "typeahead",
        "suggest",
        "complete",
        "hint",
        "predict",
        # Backend Services
        "gateway",
        "service",
        "backend",
        "internal",
        "ajax",
        "xhr",
        "bff",
        # Next.js / frameworks
        "_next/data",
        "__api__",
        "_api",
    )

    # Regex pattern for versioned API endpoints (v1, v2, v3, etc.)
    API_VERSION_PATTERN = re.compile(r"/v\d+/", re.IGNORECASE)

    # Third-party domains to exclude from likely API endpoints (analytics, tracking, consent, ads)
    EXCLUDED_THIRD_PARTY_DOMAINS = (
        # Analytics & Performance Monitoring
        "google-analytics.com",
        "googletagmanager.com",
        "analytics.google.com",
        "go-mpulse.net",
        "akamai.net",
        "newrelic.com",
        "nr-data.net",
        "segment.io",
        "segment.com",
        "mixpanel.com",
        "amplitude.com",
        "heap.io",
        "heapanalytics.com",
        "fullstory.com",
        "hotjar.com",
        "mouseflow.com",
        "clarity.ms",
        "matomo.",
        "piwik.",
        # Consent & Privacy
        "onetrust.com",
        "cookielaw.org",
        "trustarc.com",
        "cookiebot.com",
        "consent.cookiebot.com",
        "privacy-center.",
        "consentmanager.",
        # Advertising & Marketing
        "doubleclick.net",
        "googlesyndication.com",
        "googleadservices.com",
        "facebook.net",
        "fbcdn.net",
        "twitter.com/i/",
        "ads-twitter.com",
        "linkedin.com/li/",
        "adsrvr.org",
        "criteo.com",
        "criteo.net",
        "taboola.com",
        "outbrain.com",
        "adnxs.com",
        "rubiconproject.com",
        "pubmatic.com",
        "openx.net",
        "casalemedia.com",
        "demdex.net",
        "omtrdc.net",
        "2o7.net",
        # Error Tracking
        "sentry.io",
        "bugsnag.com",
        "rollbar.com",
        "logrocket.com",
        "trackjs.com",
        # CDNs (usually static assets, not APIs)
        "cloudflare.com/cdn-cgi/",
        "jsdelivr.net",
        "unpkg.com",
        "cdnjs.cloudflare.com",
        # Social Widgets
        "platform.twitter.com",
        "connect.facebook.net",
        "platform.linkedin.com",
        # Misc Third-party
        "recaptcha.net",
        "gstatic.com",
        "fonts.googleapis.com",
        "fonts.gstatic.com",
    )

    @staticmethod
    def _is_relevant_entry(entry: NetworkTransactionEvent) -> bool:
        """
        Check if an entry should be included in analysis.

        Only includes HTML and JSON responses, excludes JS, images, media, fonts.
        """
        mime = entry.mime_type.lower()

        # Exclude known non-relevant types
        for prefix in NetworkDataStore.EXCLUDED_MIME_PREFIXES:
            if mime.startswith(prefix):
                return False

        # Include known relevant types
        for prefix in NetworkDataStore.INCLUDED_MIME_PREFIXES:
            if mime.startswith(prefix):
                return True

        # Exclude by URL extension as fallback
        url_lower = entry.url.lower().split("?")[0]
        excluded_extensions = (".js", ".css", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico",
                               ".woff", ".woff2", ".ttf", ".eot", ".mp4", ".webm", ".mp3", ".wav")
        if url_lower.endswith(excluded_extensions):
            return False

        # Default: include if it has response body
        return bool(entry.response_body)

    def __init__(self, jsonl_path: str) -> None:
        """
        Initialize the NetworkDataStore from a JSONL file.

        Args:
            jsonl_path: Path to JSONL file containing NetworkTransactionEvent entries.
        """
        self._entries: list[NetworkTransactionEvent] = []
        self._entry_index: dict[str, NetworkTransactionEvent] = {}  # request_id -> event
        self._stats: NetworkStats = NetworkStats()

        path = Path(jsonl_path)
        if not path.exists():
            raise ValueError(f"JSONL file does not exist: {jsonl_path}")

        # Load entries from JSONL
        with open(path, mode="r", encoding="utf-8") as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    event = NetworkTransactionEvent.model_validate(data)
                    self._entries.append(event)
                    self._entry_index[event.request_id] = event
                except (json.JSONDecodeError, ValueError) as e:
                    logger.warning("Failed to parse line %d: %s", line_num + 1, e)
                    continue

        self._compute_stats()

        logger.info(
            "NetworkDataStore initialized with %d entries",
            len(self._entries),
        )

    @classmethod
    def from_jsonl(cls, jsonl_path: str) -> "NetworkDataStore":
        """
        Create a NetworkDataStore from a JSONL file containing NetworkTransactionEvent entries.

        Each line in the JSONL file should be a valid NetworkTransactionEvent JSON object.

        Args:
            jsonl_path: Path to the JSONL file.

        Returns:
            A NetworkDataStore instance populated with entries from the JSONL file.

        Example:
            store = NetworkDataStore.from_jsonl("cdp_captures/network/events.jsonl")
        """
        return cls(jsonl_path)

    @property
    def entries(self) -> list[NetworkTransactionEvent]:
        """Return all network transaction events."""
        return self._entries

    @property
    def stats(self) -> NetworkStats:
        """Return computed statistics."""
        return self._stats

    @property
    def raw_data(self) -> dict[str, Any]:
        """Return entries as a dict for compatibility."""
        return {"entries": [e.model_dump() for e in self._entries]}

    def _compute_stats(self) -> None:
        """Compute aggregate statistics from entries."""
        methods: Counter[str] = Counter()
        status_codes: Counter[int] = Counter()
        content_types: Counter[str] = Counter()
        hosts: Counter[str] = Counter()
        paths: set[str] = set()
        urls: set[str] = set()

        total_resp_bytes = 0

        has_auth = False
        has_json = False
        has_form = False

        for entry in self._entries:
            methods[entry.method] += 1
            if entry.status:
                status_codes[entry.status] += 1
            host = _get_host(entry.url)
            path = _get_path(entry.url)
            hosts[host] += 1
            paths.add(path)
            urls.add(entry.url)

            if entry.mime_type:
                # Simplify mime type
                ctype = entry.mime_type.split(";")[0].strip()
                content_types[ctype] += 1

            total_resp_bytes += len(entry.response_body) if entry.response_body else 0

            # Feature detection
            req_headers = entry.request_headers or {}
            for header in self.AUTH_HEADERS:
                if header in req_headers:
                    has_auth = True
                    break

            if entry.post_data:
                content_type = req_headers.get("content-type", "")
                if "json" in content_type:
                    has_json = True
                if "form" in content_type:
                    has_form = True

        self._stats = NetworkStats(
            total_requests=len(self._entries),
            total_request_bytes=0,
            total_response_bytes=total_resp_bytes,
            total_time_ms=0.0,
            methods=dict(methods),
            status_codes=dict(status_codes),
            content_types=dict(content_types),
            hosts=dict(hosts),
            unique_hosts=len(hosts),
            unique_paths=len(paths),
            unique_urls=len(urls),
            has_cookies=False,
            has_auth_headers=has_auth,
            has_json_requests=has_json,
            has_form_data=has_form,
        )

    def search_entries(
        self,
        method: str | None = None,
        host_contains: str | None = None,
        path_contains: str | None = None,
        status_code: int | None = None,
        content_type_contains: str | None = None,
        has_post_data: bool | None = None,
    ) -> list[NetworkTransactionEvent]:
        """
        Search entries with filters.

        Args:
            method: Filter by HTTP method (GET, POST, etc.)
            host_contains: Filter by host containing substring
            path_contains: Filter by path containing substring
            status_code: Filter by exact status code
            content_type_contains: Filter by content type containing substring
            has_post_data: Filter by presence of POST data

        Returns:
            List of matching NetworkTransactionEvent objects.
        """
        results = []

        for entry in self._entries:
            if method and entry.method.upper() != method.upper():
                continue
            host = _get_host(entry.url)
            if host_contains and host_contains.lower() not in host.lower():
                continue
            path = _get_path(entry.url)
            if path_contains and path_contains.lower() not in path.lower():
                continue
            if status_code is not None and entry.status != status_code:
                continue
            if content_type_contains and content_type_contains.lower() not in entry.mime_type.lower():
                continue
            if has_post_data is not None:
                if has_post_data and not entry.post_data:
                    continue
                if not has_post_data and entry.post_data:
                    continue

            results.append(entry)

        return results

    def get_entry(self, request_id: str) -> NetworkTransactionEvent | None:
        """Get entry by request_id."""
        return self._entry_index.get(request_id)

    def get_unique_urls(self) -> list[str]:
        """
        Get all unique URLs from the HAR file.

        Only includes HTML/JSON entries, excludes JS, images, media.

        Returns:
            List of unique URLs, sorted alphabetically.
        """
        urls: set[str] = set()
        for entry in self._entries:
            if self._is_relevant_entry(entry):
                urls.add(entry.url)
        return sorted(urls)

    def get_entry_ids_by_url(self, url: str) -> list[str]:
        """
        Get all request_ids that match the given URL.

        Args:
            url: The URL to search for (exact match).

        Returns:
            List of request_ids matching the URL.
        """
        return [entry.request_id for entry in self._entries if entry.url == url]

    def get_url_counts(self) -> dict[str, int]:
        """
        Get a mapping of each unique URL to its occurrence count.

        Only includes HTML/JSON entries, excludes JS, images, media.

        Returns:
            Dict mapping URL to number of times it appeared in the HAR.
        """
        url_counts: dict[str, int] = {}
        for entry in self._entries:
            if self._is_relevant_entry(entry):
                url_counts[entry.url] = url_counts.get(entry.url, 0) + 1
        # Sort by count descending
        return dict(sorted(url_counts.items(), key=lambda x: -x[1]))

    def search_entries_by_terms(
        self,
        terms: list[str],
        top_n: int = 20,
    ) -> list[dict[str, Any]]:
        """
        Search entries by a list of terms and rank by relevance.

        For each entry, searches the response body for each term and computes:
        - unique_terms_found: how many different terms were found
        - total_hits: total number of term matches across all terms
        - score: (total_hits / num_terms) * unique_terms_found

        Only searches relevant entries (HTML/JSON), excludes JS, images, media.

        Args:
            terms: List of search terms (case-insensitive).
            top_n: Number of top results to return.

        Returns:
            List of dicts with keys: id, url, unique_terms_found, total_hits, score
            Sorted by score descending, limited to top_n.
        """
        results: list[dict[str, Any]] = []
        terms_lower = [t.lower() for t in terms]
        num_terms = len(terms_lower)

        if num_terms == 0:
            return results

        for entry in self._entries:
            # Skip non-relevant entries (JS, images, media, etc.)
            if not self._is_relevant_entry(entry):
                continue

            # Skip if no response body
            if not entry.response_body:
                continue

            content_lower = entry.response_body.lower()

            # Count hits for each term
            unique_terms_found = 0
            total_hits = 0

            for term in terms_lower:
                count = content_lower.count(term)
                if count > 0:
                    unique_terms_found += 1
                    total_hits += count

            # Skip entries with no hits
            if unique_terms_found == 0:
                continue

            # Calculate score: avg hits per term * unique terms
            avg_hits = total_hits / num_terms
            score = avg_hits * unique_terms_found

            results.append({
                "id": entry.request_id,
                "url": entry.url,
                "unique_terms_found": unique_terms_found,
                "total_hits": total_hits,
                "score": score,
            })

        # Sort by score descending
        results.sort(key=lambda x: x["score"], reverse=True)

        return results[:top_n]

    def _is_excluded_third_party(self, url: str) -> bool:
        """Check if URL belongs to an excluded third-party domain."""
        url_lower = url.lower()
        for domain in self.EXCLUDED_THIRD_PARTY_DOMAINS:
            if domain in url_lower:
                return True
        return False

    def likely_api_urls(self) -> list[str]:
        """
        Get URLs that are likely important API endpoints.

        Scans all entry URLs for common API patterns including:
        - Version patterns (/v1/, /v2/, etc.)
        - Common API key terms (api, auth, search, checkout, etc.)

        Excludes:
        - Non-relevant entries (JS, images, media)
        - Third-party analytics, tracking, consent, and ad services

        Returns:
            List of unique URLs matching API patterns, sorted alphabetically.
        """
        matching_urls: set[str] = set()

        for entry in self._entries:
            # Skip non-relevant entries
            if not self._is_relevant_entry(entry):
                continue

            # Skip excluded third-party domains
            if self._is_excluded_third_party(entry.url):
                continue

            url_lower = entry.url.lower()

            # Check for versioned API pattern (/v1/, /v2/, etc.)
            if self.API_VERSION_PATTERN.search(entry.url):
                matching_urls.add(entry.url)
                continue

            # Check for key terms in URL
            for term in self.API_KEY_TERMS:
                if term in url_lower:
                    matching_urls.add(entry.url)
                    break

        return sorted(matching_urls)

    def get_host_stats(self, host_filter: str | None = None) -> list[dict[str, Any]]:
        """
        Get per-host summary statistics.

        Args:
            host_filter: Optional substring to filter hosts (case-insensitive).

        Returns:
            List of dicts sorted by request count descending, each containing:
            - host: The hostname
            - request_count: Number of requests to this host
            - methods: Dict of HTTP method counts
            - status_codes: Dict of status code counts
        """
        host_data: dict[str, dict[str, Any]] = {}

        for entry in self._entries:
            host = _get_host(entry.url)

            # Apply filter if provided
            if host_filter and host_filter.lower() not in host.lower():
                continue

            if host not in host_data:
                host_data[host] = {
                    "request_count": 0,
                    "methods": Counter(),
                    "status_codes": Counter(),
                }

            host_data[host]["request_count"] += 1
            host_data[host]["methods"][entry.method] += 1
            if entry.status:
                host_data[host]["status_codes"][entry.status] += 1

        results = []
        for host, data in sorted(host_data.items(), key=lambda x: -x[1]["request_count"]):
            results.append({
                "host": host,
                "request_count": data["request_count"],
                "methods": dict(data["methods"]),
                "status_codes": {str(k): v for k, v in data["status_codes"].items()},
            })

        return results

    def search_response_bodies(
        self,
        value: str,
        case_sensitive: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Search response bodies for a given value and return matches with context.

        Args:
            value: The value to search for in response bodies.
            case_sensitive: Whether the search should be case-sensitive.

        Returns:
            List of dicts sorted by ascending id, each containing:
            - id: Entry index
            - url: Request URL
            - count: Number of occurrences in the response body
            - sample: Context string (50 chars before and after first occurrence)
        """
        results: list[dict[str, Any]] = []

        if not value:
            return results

        search_value = value if case_sensitive else value.lower()

        for entry in self._entries:
            if not entry.response_body:
                continue

            content = entry.response_body if case_sensitive else entry.response_body.lower()
            original_content = entry.response_body

            # Count occurrences
            count = content.count(search_value)
            if count == 0:
                continue

            # Find first occurrence and extract context
            pos = content.find(search_value)
            context_start = max(0, pos - 50)
            context_end = min(len(original_content), pos + len(value) + 50)

            sample = original_content[context_start:context_end]

            # Add ellipsis if truncated
            if context_start > 0:
                sample = "..." + sample
            if context_end < len(original_content):
                sample = sample + "..."

            results.append({
                "id": entry.request_id,
                "url": entry.url,
                "count": count,
                "sample": sample,
            })

        return results

    def get_auth_headers(self) -> list[tuple[str, str, str]]:
        """
        Get authentication headers found in requests.

        Returns:
            List of (url, header_name, header_value) tuples.
        """
        auth_headers: list[tuple[str, str, str]] = []

        for entry in self._entries:
            req_headers = entry.request_headers or {}
            for header_name, header_value in req_headers.items():
                if header_name.lower() in self.AUTH_HEADERS:
                    # Truncate value for safety
                    truncated = header_value[:50] + "..." if len(header_value) > 50 else header_value
                    auth_headers.append((entry.url, header_name, truncated))

        return auth_headers

    def format_entry_summary(self, entry: NetworkTransactionEvent) -> str:
        """Format a single entry as a summary string."""
        return (
            f"{entry.method} {entry.url}\n"
            f"    Status: {entry.status or 'N/A'} {entry.status_text or ''}\n"
            f"    Type: {entry.mime_type}\n"
            f"    Size: {len(entry.response_body) if entry.response_body else 0} bytes"
        )

    @staticmethod
    def extract_key_structure(data: Any) -> Any:
        """
        Extract only the key structure from a nested dict.

        Recursively processes a nested dictionary and replaces all leaf values
        with counts (for lists) or None (for single dicts). Useful for understanding
        the shape of large JSON responses.

        For lists of dicts:
        - Iterates through ALL elements
        - Merges all unique keys
        - Tracks count of how many items had each key
        - Adds "_total" field showing total number of dicts

        Args:
            data: Any data structure (dict, list, or primitive).

        Returns:
            The structure with leaf values replaced by None or counts.

        Example:
            >>> data = [{"id": 1, "name": "a"}, {"id": 2}, {"id": 3, "extra": "x"}]
            >>> NetworkDataStore.extract_key_structure(data)
            [{"_total": 3, "id": 3, "name": 1, "extra": 1}]
        """
        if isinstance(data, dict):
            return {k: NetworkDataStore.extract_key_structure(v) for k, v in data.items()}
        elif isinstance(data, list):
            if len(data) == 0:
                return []

            # Collect all dicts from the list
            dicts_in_list = [item for item in data if isinstance(item, dict)]
            if not dicts_in_list:
                # List of primitives
                return []

            total_dicts = len(dicts_in_list)

            # Collect all values for each key across all dicts
            key_values: dict[str, list[Any]] = {}
            for d in dicts_in_list:
                for k, v in d.items():
                    if k not in key_values:
                        key_values[k] = []
                    key_values[k].append(v)

            # Build merged structure with counts
            result: dict[str, Any] = {"_total": total_dicts}

            for k, values in key_values.items():
                count = len(values)

                # Check if all values are dicts - merge recursively
                if all(isinstance(v, dict) for v in values):
                    result[k] = NetworkDataStore.extract_key_structure(values)
                # Check if all values are lists - flatten and recurse
                elif all(isinstance(v, list) for v in values):
                    flattened = []
                    for v in values:
                        flattened.extend(v)
                    result[k] = NetworkDataStore.extract_key_structure(flattened)
                else:
                    # Leaf value - use count
                    result[k] = count

            return [result]
        else:
            # Leaf value - replace with None
            return None

    def get_entry_key_structure(self, request_id: str) -> dict[str, Any] | None:
        """
        Get the key structure of an entry's JSON response content.

        Args:
            request_id: The request_id of the entry.

        Returns:
            The key structure of the response JSON, or None if not found/not JSON.
        """
        entry = self.get_entry(request_id)
        if not entry or not entry.response_body:
            return None

        try:
            data = json.loads(entry.response_body)
            return self.extract_key_structure(data)
        except json.JSONDecodeError:
            return None

    def format_entry_detail(self, entry: NetworkTransactionEvent) -> str:
        """Format a single entry with full details."""
        host = _get_host(entry.url)
        path = _get_path(entry.url)

        lines = [
            f"Method: {entry.method}",
            f"URL: {entry.url}",
            f"Host: {host}",
            f"Path: {path}",
            f"Status: {entry.status or 'N/A'} {entry.status_text or ''}",
            f"Content-Type: {entry.mime_type}",
            f"Response Size: {len(entry.response_body) if entry.response_body else 0} bytes",
            "Request Headers:",
        ]

        req_headers = entry.request_headers or {}
        for k, v in sorted(req_headers.items()):
            lines.append(f"  {k}: {v[:100]}{'...' if len(v) > 100 else ''}")

        if entry.post_data:
            lines.append("")
            lines.append("Post Data:")
            post_data_str = json.dumps(entry.post_data) if isinstance(entry.post_data, (dict, list)) else str(entry.post_data)
            if len(post_data_str) > 1000:
                lines.append(f"  {post_data_str[:1000]}...")
            else:
                lines.append(f"  {post_data_str}")

        lines.append("")
        lines.append("Response Headers:")
        resp_headers = entry.response_headers or {}
        for k, v in sorted(resp_headers.items()):
            lines.append(f"  {k}: {v[:100]}{'...' if len(v) > 100 else ''}")

        return "\n".join(lines)
