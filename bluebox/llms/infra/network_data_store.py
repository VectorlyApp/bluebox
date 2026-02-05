"""
bluebox/llms/infra/network_data_store.py

# NOTE: This data store is coming to production soon!
# NOTE: This replaces the vectorstore for network traffic analysis.

Data store for network traffic analysis.

Parses JSONL files with NetworkTransactionEvent entries and provides
structured access to network traffic data.
"""

import fnmatch
import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

from bluebox.constants.network import (
    API_KEY_TERMS,
    API_VERSION_PATTERN,
    AUTH_HEADERS,
    EXCLUDED_MIME_PREFIXES,
    INCLUDED_MIME_PREFIXES,
    SKIP_FILE_EXTENSIONS,
)
from bluebox.data_models.cdp import NetworkTransactionEvent
from bluebox.llms.infra.abstract_data_store import AbstractDataStore
from bluebox.utils.data_utils import extract_object_schema, format_bytes, read_jsonl
from bluebox.utils.logger import get_logger


logger = get_logger(name=__name__)


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
            f"Total Request Size: {format_bytes(self.total_request_bytes)}",
            f"Total Response Size: {format_bytes(self.total_response_bytes)}",
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


class NetworkDataStore(AbstractDataStore[NetworkTransactionEvent, NetworkStats]):
    """
    Data store for HAR file analysis.

    Parses HAR content and provides structured access to network traffic data
    including entries, statistics, and search capabilities.
    """

    @staticmethod
    def _is_relevant_entry(entry: NetworkTransactionEvent) -> bool:
        """
        Check if an entry should be included in analysis.

        Only includes HTML and JSON responses, excludes JS, images, media, fonts.
        """
        mime = entry.mime_type.lower()

        # Exclude known non-relevant types
        for prefix in EXCLUDED_MIME_PREFIXES:
            if mime.startswith(prefix):
                return False

        # Include known relevant types
        for prefix in INCLUDED_MIME_PREFIXES:
            if mime.startswith(prefix):
                return True

        # Exclude by URL extension as fallback
        url_lower = entry.url.lower().split("?")[0]
        if url_lower.endswith(SKIP_FILE_EXTENSIONS):
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

        # Load entries from JSONL, filtering to only relevant entries
        skipped = 0
        for line_num, data in read_jsonl(jsonl_path):
            try:
                event = NetworkTransactionEvent.model_validate(data)
                if self._is_relevant_entry(event):
                    self._entries.append(event)
                    self._entry_index[event.request_id] = event
                else:
                    skipped += 1
            except ValueError as e:
                logger.warning("Failed to validate line %d: %s", line_num + 1, e)
                continue

        self._compute_stats()

        logger.info(
            "NetworkDataStore initialized with %d entries (skipped %d non-relevant)",
            len(self._entries),
            skipped,
        )

    # Abstract method implementations

    def get_entry_id(self, entry: NetworkTransactionEvent) -> str:
        """Get the request_id as the unique identifier."""
        return entry.request_id

    def get_searchable_content(self, entry: NetworkTransactionEvent) -> str | None:
        """Get the response body as searchable content."""
        return entry.response_body

    def get_entry_url(self, entry: NetworkTransactionEvent) -> str | None:
        """Get the URL from the entry."""
        return entry.url

    # Additional properties

    @property
    def raw_data(self) -> dict[str, Any]:
        """Return entries as a dict for compatibility."""
        return {"entries": [e.model_dump() for e in self._entries]}

    @property
    def url_counts(self) -> dict[str, int]:
        """Mapping of each unique URL to its occurrence count."""
        url_counts: Counter[str] = Counter(entry.url for entry in self._entries)
        return dict(url_counts.most_common())

    @property
    def api_urls(self) -> list[str]:
        """URLs that are likely API endpoints, sorted alphabetically."""
        matching_urls: set[str] = set()

        for entry in self._entries:
            url_lower = entry.url.lower()

            # Check for versioned API pattern (/v1/, /v2/, etc.)
            if API_VERSION_PATTERN.search(entry.url):
                matching_urls.add(entry.url)
                continue

            # Check for key terms in URL
            for term in API_KEY_TERMS:
                if term in url_lower:
                    matching_urls.add(entry.url)
                    break

        return sorted(matching_urls)

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
            parsed_url = urlparse(entry.url)
            host = parsed_url.netloc
            path = parsed_url.path
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
            for header in AUTH_HEADERS:
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
            parsed_url = urlparse(entry.url)
            host = parsed_url.netloc
            if host_contains and host_contains.lower() not in host.lower():
                continue
            path = parsed_url.path
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

    def get_entry_ids_by_url_pattern(self, pattern: str) -> list[str]:
        """
        Get all request_ids whose URLs match the given glob pattern.

        Args:
            pattern: Glob pattern to match URLs (e.g., "*api/v1/*", "*/users*").
                     Supports wildcards: * (any chars), ? (single char), [seq] (char set).

        Returns:
            List of request_ids whose URLs match the pattern.
        """
        return [entry.request_id for entry in self._entries if fnmatch.fnmatch(entry.url, pattern)]

    # Alias for backwards compatibility
    search_entries_by_terms = AbstractDataStore.search_by_terms

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
            host = urlparse(entry.url).netloc

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

    def get_response_body_schema(self, request_id: str) -> dict[str, Any] | None:
        """Get the schema of an entry's JSON response body."""
        entry = self.get_entry(request_id)
        if not entry or not entry.response_body:
            return None

        try:
            data = json.loads(entry.response_body)
            return extract_object_schema(data)
        except json.JSONDecodeError:
            return None
