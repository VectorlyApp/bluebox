"""
bluebox/llms/infra/js_data_store.py

Data store for JavaScript files captured during browser sessions.

Parses the javascript_events.jsonl file (already filtered to JS MIME types
by FileEventWriter) and provides JS-specific query methods.
"""

import fnmatch
import json
import threading

import regex
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from bluebox.data_models.cdp import NetworkTransactionEvent
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


class JSDataStore:
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

    @property
    def entries(self) -> list[NetworkTransactionEvent]:
        """Return all JS file entries."""
        return self._entries

    @property
    def stats(self) -> JSFileStats:
        """Return computed statistics."""
        return self._stats

    def search_by_terms(
        self,
        terms: list[str],
        top_n: int = 20,
    ) -> list[dict[str, Any]]:
        """
        Search JS file response bodies by terms, ranked by relevance.

        Args:
            terms: List of search terms (case-insensitive).
            top_n: Number of top results to return.

        Returns:
            List of dicts with keys: id, url, unique_terms_found, total_hits, score.
        """
        results: list[dict[str, Any]] = []
        terms_lower = [t.lower() for t in terms]
        num_terms = len(terms_lower)

        if num_terms == 0:
            return results

        for entry in self._entries:
            if not entry.response_body:
                continue

            content_lower = entry.response_body.lower()
            unique_terms_found = 0
            total_hits = 0

            for term in terms_lower:
                count = content_lower.count(term)
                if count > 0:
                    unique_terms_found += 1
                    total_hits += count

            if unique_terms_found == 0:
                continue

            avg_hits = total_hits / num_terms
            score = avg_hits * unique_terms_found

            results.append({
                "id": entry.request_id,
                "url": entry.url,
                "unique_terms_found": unique_terms_found,
                "total_hits": total_hits,
                "score": score,
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_n]

    def search_by_regex(
        self,
        pattern: str,
        top_n: int = 20,
        max_matches_per_file: int = 10,
        context_chars: int = 80,
        timeout_seconds: float = 15.0,
    ) -> dict[str, Any]:
        """
        Search JS file response bodies by regex pattern.

        Args:
            pattern: Regex pattern to search for.
            top_n: Max number of files to return.
            max_matches_per_file: Max matches to return per file.
            context_chars: Characters of context around each match.
            timeout_seconds: Max time to spend searching.

        Returns:
            Dict with keys: files (list of matches), timed_out (bool), error (str | None).
        """
        try:
            compiled = regex.compile(pattern, flags=regex.IGNORECASE)
        except regex.error as e:
            logger.error("Invalid regex: %s", e)
            return {
                "files": [],
                "timed_out": False,
                "error": f"Invalid regex: {e}",
            }

        # Per-file timeout to catch catastrophic backtracking
        per_file_timeout = min(5.0, timeout_seconds / 2)

        results: list[dict[str, Any]] = []
        timed_out = False
        stop_event = threading.Event()

        def _search_worker() -> None:
            nonlocal timed_out
            for entry in self._entries:
                if stop_event.is_set():
                    timed_out = True
                    break

                if not entry.response_body:
                    continue

                content = entry.response_body
                matches: list[dict[str, Any]] = []

                try:
                    for match in compiled.finditer(content, timeout=per_file_timeout):
                        if stop_event.is_set():
                            timed_out = True
                            break
                        if len(matches) >= max_matches_per_file:
                            break

                        start = match.start()
                        end = match.end()

                        # extract context snippet
                        snippet_start = max(0, start - context_chars)
                        snippet_end = min(len(content), end + context_chars)
                        snippet = content[snippet_start:snippet_end]

                        # add ellipsis markers if truncated
                        prefix = "..." if snippet_start > 0 else ""
                        suffix = "..." if snippet_end < len(content) else ""

                        matches.append({
                            "match": match.group(),
                            "position": start,
                            "snippet": f"{prefix}{snippet}{suffix}",
                        })
                except TimeoutError:
                    logger.warning("Regex timed out on file %s, skipping", entry.url)

                if matches:
                    results.append({
                        "request_id": entry.request_id,
                        "url": entry.url,
                        "match_count": len(matches),
                        "matches": matches,
                    })

                    if len(results) >= top_n:
                        break

        # run search in thread with timeout
        thread = threading.Thread(
            target=_search_worker,
            daemon=True,
        )
        thread.start()
        thread.join(timeout=timeout_seconds)

        if thread.is_alive():
            logger.warning("Regex search timed out after %s seconds. Returning partial results.", timeout_seconds)
            stop_event.set()
            thread.join(timeout=1.0)  # give it a moment to stop
            timed_out = True

        return {
            "files": results,
            "timed_out": timed_out,
            "error": None,
        }

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
