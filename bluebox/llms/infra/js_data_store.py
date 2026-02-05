"""
bluebox/llms/infra/js_data_store.py

Data store for JavaScript files captured during browser sessions.

Parses the javascript_events.jsonl file (already filtered to JS MIME types
by FileEventWriter) and provides JS-specific query methods.
"""

import fnmatch
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from bluebox.data_models.cdp import NetworkTransactionEvent
from bluebox.llms.infra.abstract_data_store import AbstractDataStore
from bluebox.utils.logger import get_logger

logger = get_logger(name=__name__)


@dataclass
class JSFileStats:
    """Summary statistics for JavaScript files."""

    total_files: int = 0
    unique_urls: int = 0
    total_bytes: int = 0
    hosts: dict[str, int] = field(default_factory=dict)

    def to_summary(self) -> str:
        """Generate a human-readable summary."""
        lines = [
            f"Total JS Files: {self.total_files}",
            f"Unique URLs: {self.unique_urls}",
            f"Total Size: {self._format_bytes(self.total_bytes)}",
            "",
            "Top Hosts:",
        ]
        for host, count in sorted(self.hosts.items(), key=lambda x: -x[1])[:10]:
            lines.append(f"  {host}: {count}")
        return "\n".join(lines)

    @staticmethod
    def _format_bytes(num_bytes: int) -> str:
        """Format bytes as human-readable string."""
        for unit in ["B", "KB", "MB", "GB"]:
            if abs(num_bytes) < 1_024:
                return f"{num_bytes:.1f} {unit}"
            num_bytes = num_bytes / 1_024
        return f"{num_bytes:.1f} TB"


class JSDataStore(AbstractDataStore[NetworkTransactionEvent, JSFileStats]):
    """
    Data store for JavaScript files from browser captures.

    Unlike NetworkDataStore (which excludes JS via _is_relevant_entry),
    this loads all entries from javascript_events.jsonl — a file that
    already contains only JS entries.
    """

    def __init__(self, jsonl_path: str) -> None:
        """
        Initialize the JSDataStore from a JSONL file.

        Args:
            jsonl_path: Path to JSONL file containing JS NetworkTransactionEvent entries.
        """
        self._entries: list[NetworkTransactionEvent] = []
        self._entry_index: dict[str, NetworkTransactionEvent] = {}  # request_id -> event
        self._stats: JSFileStats = JSFileStats()

        path = Path(jsonl_path)
        if not path.exists():
            raise ValueError(f"JSONL file does not exist: {jsonl_path}")

        # load all entries (no filtering; the JS JSONL is already pre-filtered)
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

        logger.debug(
            "JSDataStore initialized with %d JS files",
            len(self._entries),
        )

    def _compute_stats(self) -> None:
        """Compute aggregate statistics."""
        hosts: Counter[str] = Counter()
        urls: set[str] = set()
        total_bytes = 0

        for entry in self._entries:
            host = urlparse(entry.url).netloc
            hosts[host] += 1
            urls.add(entry.url)
            total_bytes += len(entry.response_body) if entry.response_body else 0

        self._stats = JSFileStats(
            total_files=len(self._entries),
            unique_urls=len(urls),
            total_bytes=total_bytes,
            hosts=dict(hosts),
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

    # JS-specific methods

    def get_file(self, request_id: str) -> NetworkTransactionEvent | None:
        """Get a JS file entry by request_id."""
        return self._entry_index.get(request_id)

    def get_file_content(self, request_id: str, max_chars: int = 10_000) -> str | None:
        """
        Get truncated response body for a JS file.

        Args:
            request_id: The request_id of the JS file entry.
            max_chars: Maximum characters to return.

        Returns:
            The response body (truncated if needed), or None if not found.
        """
        entry = self._entry_index.get(request_id)
        if not entry or not entry.response_body:
            return None

        content = entry.response_body
        if len(content) > max_chars:
            return content[:max_chars] + f"\n... (truncated, {len(content)} total chars)"
        return content

    def search_by_url(self, pattern: str) -> list[NetworkTransactionEvent]:
        """
        Search JS files by URL glob pattern.

        Args:
            pattern: Glob pattern to match URLs (e.g., "*bundle*", "*/vendor/*").

        Returns:
            List of matching NetworkTransactionEvent entries.
        """
        return [
            entry for entry in self._entries
            if fnmatch.fnmatch(entry.url, pattern)
        ]

    def list_files(self) -> list[dict[str, Any]]:
        """
        List all JS files with summary info.

        Returns:
            List of dicts with keys: request_id, url, size.
        """
        return [
            {
                "request_id": entry.request_id,
                "url": entry.url,
                "size": len(entry.response_body) if entry.response_body else 0,
            }
            for entry in self._entries
        ]
