"""
bluebox/llms/infra/har_data_store.py

Data store for HAR (HTTP Archive) file analysis.

Parses HAR files and provides structured access to network traffic data.
"""

import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

from bluebox.utils.logger import get_logger


logger = get_logger(name=__name__)


@dataclass
class HarEntry:
    """Represents a single HAR entry (request/response pair)."""

    id: int  # Same as index, used as unique identifier
    method: str
    url: str
    host: str
    path: str
    status: int
    status_text: str
    mime_type: str
    request_size: int
    response_size: int
    time_ms: float
    request_headers: dict[str, str]
    response_headers: dict[str, str]
    request_cookies: list[dict[str, Any]]
    response_cookies: list[dict[str, Any]]
    query_params: dict[str, str]
    post_data: str | None
    response_content: str | None


@dataclass
class HarStats:
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


class HarDataStore:
    """
    Data store for HAR file analysis.

    Parses HAR content and provides structured access to network traffic data
    including entries, statistics, and search capabilities.

    Usage:
        har_content = open("network.har").read()
        store = HarDataStore(har_content)

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

    @staticmethod
    def _is_relevant_entry(entry: "HarEntry") -> bool:
        """
        Check if an entry should be included in analysis.

        Only includes HTML and JSON responses, excludes JS, images, media, fonts.
        """
        mime = entry.mime_type.lower()

        # Exclude known non-relevant types
        for prefix in HarDataStore.EXCLUDED_MIME_PREFIXES:
            if mime.startswith(prefix):
                return False

        # Include known relevant types
        for prefix in HarDataStore.INCLUDED_MIME_PREFIXES:
            if mime.startswith(prefix):
                return True

        # Exclude by URL extension as fallback
        url_lower = entry.url.lower().split("?")[0]
        excluded_extensions = (".js", ".css", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico",
                               ".woff", ".woff2", ".ttf", ".eot", ".mp4", ".webm", ".mp3", ".wav")
        if url_lower.endswith(excluded_extensions):
            return False

        # Default: include if it has response content
        return bool(entry.response_content)

    def __init__(self, har_content: str) -> None:
        """
        Initialize the HAR data store.

        Args:
            har_content: Raw HAR file content as JSON string.
        """
        self._raw_content = har_content
        self._har_data: dict[str, Any] = {}
        self._entries: list[HarEntry] = []
        self._stats: HarStats = HarStats()

        self._parse()
        self._compute_stats()

        logger.info(
            "HarDataStore initialized with %d entries",
            len(self._entries),
        )

    @property
    def entries(self) -> list[HarEntry]:
        """Return all parsed HAR entries."""
        return self._entries

    @property
    def stats(self) -> HarStats:
        """Return computed statistics."""
        return self._stats

    @property
    def raw_data(self) -> dict[str, Any]:
        """Return raw parsed HAR data."""
        return self._har_data

    def _parse(self) -> None:
        """Parse HAR content into structured entries."""
        try:
            self._har_data = json.loads(self._raw_content)
        except json.JSONDecodeError as e:
            logger.error("Failed to parse HAR JSON: %s", e)
            raise ValueError(f"Invalid HAR JSON: {e}") from e

        log = self._har_data.get("log", {})
        raw_entries = log.get("entries", [])

        for i, entry in enumerate(raw_entries):
            try:
                parsed = self._parse_entry(i, entry)
                self._entries.append(parsed)
            except Exception as e:
                logger.warning("Failed to parse entry %d: %s", i, e)

    def _parse_entry(self, index: int, entry: dict[str, Any]) -> HarEntry:
        """Parse a single HAR entry."""
        request = entry.get("request", {})
        response = entry.get("response", {})

        url = request.get("url", "")
        parsed_url = urlparse(url)

        # Parse headers into dict
        req_headers = {
            h.get("name", "").lower(): h.get("value", "")
            for h in request.get("headers", [])
        }
        resp_headers = {
            h.get("name", "").lower(): h.get("value", "")
            for h in response.get("headers", [])
        }

        # Parse query params
        query_params = {
            q.get("name", ""): q.get("value", "")
            for q in request.get("queryString", [])
        }

        # Get post data
        post_data_obj = request.get("postData", {})
        post_data = post_data_obj.get("text") if post_data_obj else None

        # Get response content
        content_obj = response.get("content", {})
        response_content = content_obj.get("text") if content_obj else None

        return HarEntry(
            id=index,
            method=request.get("method", "GET"),
            url=url,
            host=parsed_url.netloc,
            path=parsed_url.path,
            status=response.get("status", 0),
            status_text=response.get("statusText", ""),
            mime_type=content_obj.get("mimeType", "") if content_obj else "",
            request_size=request.get("headersSize", 0) + request.get("bodySize", 0),
            response_size=response.get("headersSize", 0) + content_obj.get("size", 0) if content_obj else 0,
            time_ms=entry.get("time", 0),
            request_headers=req_headers,
            response_headers=resp_headers,
            request_cookies=request.get("cookies", []),
            response_cookies=response.get("cookies", []),
            query_params=query_params,
            post_data=post_data,
            response_content=response_content,
        )

    def _compute_stats(self) -> None:
        """Compute aggregate statistics from entries."""
        methods: Counter[str] = Counter()
        status_codes: Counter[int] = Counter()
        content_types: Counter[str] = Counter()
        hosts: Counter[str] = Counter()
        paths: set[str] = set()
        urls: set[str] = set()

        total_req_bytes = 0
        total_resp_bytes = 0
        total_time = 0.0

        has_cookies = False
        has_auth = False
        has_json = False
        has_form = False

        for entry in self._entries:
            methods[entry.method] += 1
            status_codes[entry.status] += 1
            hosts[entry.host] += 1
            paths.add(entry.path)
            urls.add(entry.url)

            if entry.mime_type:
                # Simplify mime type
                ctype = entry.mime_type.split(";")[0].strip()
                content_types[ctype] += 1

            total_req_bytes += entry.request_size
            total_resp_bytes += entry.response_size
            total_time += entry.time_ms

            # Feature detection
            if entry.request_cookies or entry.response_cookies:
                has_cookies = True

            for header in self.AUTH_HEADERS:
                if header in entry.request_headers:
                    has_auth = True
                    break

            if entry.post_data:
                content_type = entry.request_headers.get("content-type", "")
                if "json" in content_type:
                    has_json = True
                if "form" in content_type:
                    has_form = True

        self._stats = HarStats(
            total_requests=len(self._entries),
            total_request_bytes=total_req_bytes,
            total_response_bytes=total_resp_bytes,
            total_time_ms=total_time,
            methods=dict(methods),
            status_codes=dict(status_codes),
            content_types=dict(content_types),
            hosts=dict(hosts),
            unique_hosts=len(hosts),
            unique_paths=len(paths),
            unique_urls=len(urls),
            has_cookies=has_cookies,
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
    ) -> list[HarEntry]:
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
            List of matching HarEntry objects.
        """
        results = []

        for entry in self._entries:
            if method and entry.method.upper() != method.upper():
                continue
            if host_contains and host_contains.lower() not in entry.host.lower():
                continue
            if path_contains and path_contains.lower() not in entry.path.lower():
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

    def get_entry(self, index: int) -> HarEntry | None:
        """Get entry by index."""
        if 0 <= index < len(self._entries):
            return self._entries[index]
        return None

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

    def get_entry_ids_by_url(self, url: str) -> list[int]:
        """
        Get all entry IDs that match the given URL.

        Args:
            url: The URL to search for (exact match).

        Returns:
            List of entry IDs matching the URL.
        """
        return [entry.id for entry in self._entries if entry.url == url]

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
        top_n: int = 10,
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

            # Skip if no response content
            if not entry.response_content:
                continue

            content_lower = entry.response_content.lower()

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

            # Calculate score: (avg hits per term * unique terms) / content length
            # Multiply by 1000 to keep scores in reasonable range
            content_len = len(content_lower)
            avg_hits = total_hits / num_terms
            score = (avg_hits * unique_terms_found / content_len) * 1000

            results.append({
                "id": entry.id,
                "url": entry.url,
                "unique_terms_found": unique_terms_found,
                "total_hits": total_hits,
                "content_length": content_len,
                "score": round(score, 4),
            })

        # Sort by score descending
        results.sort(key=lambda x: x["score"], reverse=True)

        return results[:top_n]

    def get_api_endpoints(self) -> list[tuple[str, str, str]]:
        """
        Get list of likely API endpoints.

        Returns:
            List of (method, host, path) tuples for JSON/API requests.
        """
        endpoints: set[tuple[str, str, str]] = set()

        for entry in self._entries:
            # Check if it looks like an API call
            is_api = (
                "api" in entry.path.lower()
                or "json" in entry.mime_type.lower()
                or entry.path.startswith("/v1/")
                or entry.path.startswith("/v2/")
                or entry.path.startswith("/graphql")
            )

            if is_api:
                endpoints.add((entry.method, entry.host, entry.path.split("?")[0]))

        return sorted(endpoints, key=lambda x: (x[1], x[2], x[0]))

    def get_cookies(self) -> dict[str, list[dict[str, Any]]]:
        """
        Get all cookies grouped by domain.

        Returns:
            Dict mapping domain to list of cookie objects.
        """
        cookies: dict[str, list[dict[str, Any]]] = {}

        for entry in self._entries:
            domain = entry.host

            for cookie in entry.request_cookies + entry.response_cookies:
                if domain not in cookies:
                    cookies[domain] = []
                # Avoid duplicates
                cookie_name = cookie.get("name", "")
                if not any(c.get("name") == cookie_name for c in cookies[domain]):
                    cookies[domain].append(cookie)

        return cookies

    def get_auth_headers(self) -> list[tuple[str, str, str]]:
        """
        Get authentication headers found in requests.

        Returns:
            List of (url, header_name, header_value) tuples.
        """
        auth_headers: list[tuple[str, str, str]] = []

        for entry in self._entries:
            for header_name, header_value in entry.request_headers.items():
                if header_name.lower() in self.AUTH_HEADERS:
                    # Truncate value for safety
                    truncated = header_value[:50] + "..." if len(header_value) > 50 else header_value
                    auth_headers.append((entry.url, header_name, truncated))

        return auth_headers

    def format_entry_summary(self, entry: HarEntry) -> str:
        """Format a single entry as a summary string."""
        return (
            f"[{entry.id}] {entry.method} {entry.url}\n"
            f"    Status: {entry.status} {entry.status_text}\n"
            f"    Type: {entry.mime_type}\n"
            f"    Time: {entry.time_ms:.0f}ms\n"
            f"    Size: {entry.response_size} bytes"
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
            >>> HarDataStore.extract_key_structure(data)
            [{"_total": 3, "id": 3, "name": 1, "extra": 1}]
        """
        if isinstance(data, dict):
            return {k: HarDataStore.extract_key_structure(v) for k, v in data.items()}
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
                    result[k] = HarDataStore.extract_key_structure(values)
                # Check if all values are lists - flatten and recurse
                elif all(isinstance(v, list) for v in values):
                    flattened = []
                    for v in values:
                        flattened.extend(v)
                    result[k] = HarDataStore.extract_key_structure(flattened)
                else:
                    # Leaf value - use count
                    result[k] = count

            return [result]
        else:
            # Leaf value - replace with None
            return None

    def get_entry_key_structure(self, entry_id: int) -> dict[str, Any] | None:
        """
        Get the key structure of an entry's JSON response content.

        Args:
            entry_id: The ID of the entry.

        Returns:
            The key structure of the response JSON, or None if not found/not JSON.
        """
        entry = self.get_entry(entry_id)
        if not entry or not entry.response_content:
            return None

        try:
            data = json.loads(entry.response_content)
            return self.extract_key_structure(data)
        except json.JSONDecodeError:
            return None

    def format_entry_detail(self, entry: HarEntry) -> str:
        """Format a single entry with full details."""
        lines = [
            f"=== Entry {entry.id} ===",
            f"Method: {entry.method}",
            f"URL: {entry.url}",
            f"Host: {entry.host}",
            f"Path: {entry.path}",
            f"Status: {entry.status} {entry.status_text}",
            f"Content-Type: {entry.mime_type}",
            f"Time: {entry.time_ms:.0f}ms",
            f"Request Size: {entry.request_size} bytes",
            f"Response Size: {entry.response_size} bytes",
            "",
            "Request Headers:",
        ]

        for k, v in sorted(entry.request_headers.items()):
            lines.append(f"  {k}: {v[:100]}{'...' if len(v) > 100 else ''}")

        if entry.query_params:
            lines.append("")
            lines.append("Query Parameters:")
            for k, v in entry.query_params.items():
                lines.append(f"  {k}: {v}")

        if entry.post_data:
            lines.append("")
            lines.append("Post Data:")
            # Truncate large post data
            if len(entry.post_data) > 1000:
                lines.append(f"  {entry.post_data[:1000]}...")
            else:
                lines.append(f"  {entry.post_data}")

        lines.append("")
        lines.append("Response Headers:")
        for k, v in sorted(entry.response_headers.items()):
            lines.append(f"  {k}: {v[:100]}{'...' if len(v) > 100 else ''}")

        return "\n".join(lines)
