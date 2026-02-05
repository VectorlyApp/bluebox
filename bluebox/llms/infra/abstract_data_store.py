"""
bluebox/llms/infra/abstract_data_store.py

Abstract base class for data stores.

Provides common patterns for data stores including:
- Entry storage and indexing
- Stats computation
- Term-based search with relevance scoring
- Regex search with timeout protection
"""

import threading
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

import regex

from bluebox.utils.logger import get_logger


logger = get_logger(name=__name__)


# Type vars for entry and stats types
EntryT = TypeVar("EntryT")
StatsT = TypeVar("StatsT")


class AbstractDataStore(ABC, Generic[EntryT, StatsT]):
    """
    Abstract base class for data stores.

    Subclasses must implement:
    - _compute_stats() -> None
    - get_entry_id(entry) -> str
    - get_searchable_content(entry) -> str | None

    Provides common search functionality:
    - search_by_terms(): Term-based search with relevance scoring
    - search_by_regex(): Regex search with timeout protection
    - search_content(): Simple substring search with context
    """

    _entries: list[EntryT]
    _stats: StatsT

    @property
    def entries(self) -> list[EntryT]:
        """Return all entries."""
        return self._entries

    @property
    def stats(self) -> StatsT:
        """Return computed statistics."""
        return self._stats

    @abstractmethod
    def _compute_stats(self) -> None:
        """Compute aggregate statistics from entries. Must set self._stats."""
        ...

    @abstractmethod
    def get_entry_id(self, entry: EntryT) -> str:
        """
        Get a unique identifier for an entry.

        Args:
            entry: The entry to get the ID for.

        Returns:
            A unique string identifier.
        """
        ...

    @abstractmethod
    def get_searchable_content(self, entry: EntryT) -> str | None:
        """
        Get the searchable text content from an entry.

        Args:
            entry: The entry to extract content from.

        Returns:
            The text content to search, or None if no searchable content.
        """
        ...

    def get_entry_url(self, entry: EntryT) -> str | None:
        """
        Get the URL associated with an entry (optional).

        Override in subclasses where entries have URLs.

        Args:
            entry: The entry to get the URL for.

        Returns:
            The URL string, or None if not applicable.
        """
        return None

    def search_by_terms(
        self,
        terms: list[str],
        top_n: int = 20,
    ) -> list[dict[str, Any]]:
        """
        Search entries by a list of terms and rank by relevance.

        For each entry, searches the searchable content for each term and computes:
        - unique_terms_found: how many different terms were found
        - total_hits: total number of term matches across all terms
        - score: (total_hits / num_terms) * unique_terms_found

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
            content = self.get_searchable_content(entry)
            if not content:
                continue

            content_lower = content.lower()
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
                "id": self.get_entry_id(entry),
                "url": self.get_entry_url(entry),
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
        max_matches_per_entry: int = 10,
        context_chars: int = 80,
        timeout_seconds: float = 15.0,
    ) -> dict[str, Any]:
        """
        Search entry contents by regex pattern.

        Uses threading with timeout to protect against catastrophic backtracking.

        Args:
            pattern: Regex pattern to search for.
            top_n: Max number of entries to return.
            max_matches_per_entry: Max matches to return per entry.
            context_chars: Characters of context around each match.
            timeout_seconds: Max time to spend searching.

        Returns:
            Dict with keys:
            - matches: list of entry matches with snippets
            - timed_out: bool indicating if search timed out
            - error: str | None with error message if pattern is invalid
        """
        try:
            compiled = regex.compile(pattern, flags=regex.IGNORECASE)
        except regex.error as e:
            logger.error("Invalid regex: %s", e)
            return {
                "matches": [],
                "timed_out": False,
                "error": f"Invalid regex: {e}",
            }

        # Per-entry timeout to catch catastrophic backtracking
        per_entry_timeout = min(5.0, timeout_seconds / 2)

        results: list[dict[str, Any]] = []
        timed_out = False
        stop_event = threading.Event()

        def _search_worker() -> None:
            nonlocal timed_out
            for entry in self._entries:
                if stop_event.is_set():
                    timed_out = True
                    break

                content = self.get_searchable_content(entry)
                if not content:
                    continue

                matches: list[dict[str, Any]] = []

                try:
                    for match in compiled.finditer(content, timeout=per_entry_timeout):
                        if stop_event.is_set():
                            timed_out = True
                            break
                        if len(matches) >= max_matches_per_entry:
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
                    entry_id = self.get_entry_id(entry)
                    logger.warning("Regex timed out on entry %s, skipping", entry_id)

                if matches:
                    results.append({
                        "id": self.get_entry_id(entry),
                        "url": self.get_entry_url(entry),
                        "match_count": len(matches),
                        "matches": matches,
                    })

                    if len(results) >= top_n:
                        break

        # run search in thread with timeout
        thread = threading.Thread(target=_search_worker, daemon=True)
        thread.start()
        thread.join(timeout=timeout_seconds)

        if thread.is_alive():
            logger.warning(
                "Regex search timed out after %s seconds. Returning partial results.",
                timeout_seconds,
            )
            stop_event.set()
            thread.join(timeout=1.0)  # give it a moment to stop
            timed_out = True

        return {
            "matches": results,
            "timed_out": timed_out,
            "error": None,
        }

    def search_content(
        self,
        value: str,
        case_sensitive: bool = False,
        context_chars: int = 50,
    ) -> list[dict[str, Any]]:
        """
        Search entry contents for a given value and return matches with context.

        Args:
            value: The value to search for.
            case_sensitive: Whether the search should be case-sensitive.
            context_chars: Characters of context around match.

        Returns:
            List of dicts with keys: id, url, count, sample
        """
        results: list[dict[str, Any]] = []

        if not value:
            return results

        search_value = value if case_sensitive else value.lower()

        for entry in self._entries:
            content = self.get_searchable_content(entry)
            if not content:
                continue

            search_content = content if case_sensitive else content.lower()

            count = search_content.count(search_value)
            if count == 0:
                continue

            # Find first occurrence and extract context
            pos = search_content.find(search_value)
            context_start = max(0, pos - context_chars)
            context_end = min(len(content), pos + len(value) + context_chars)

            sample = content[context_start:context_end]

            if context_start > 0:
                sample = "..." + sample
            if context_end < len(content):
                sample = sample + "..."

            results.append({
                "id": self.get_entry_id(entry),
                "url": self.get_entry_url(entry),
                "count": count,
                "sample": sample,
            })

        return results
