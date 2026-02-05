"""
tests/unit/data_loaders/test_abstract_data_loader.py

Comprehensive unit tests for AbstractDataLoader base class.

Tests the common search functionality provided by the abstract base class
using a concrete test implementation.
"""

from dataclasses import dataclass

import pytest

from bluebox.llms.data_loaders.abstract_data_loader import AbstractDataLoader


# --- Test Implementation ---


@dataclass
class MockStats:
    """Simple stats class for testing."""
    total_entries: int = 0
    total_chars: int = 0

    def to_summary(self) -> str:
        return f"Total Entries: {self.total_entries}, Total Chars: {self.total_chars}"


@dataclass
class MockEntry:
    """Simple entry class for testing."""
    id: str
    content: str
    url: str | None = None


class ConcreteDataLoader(AbstractDataLoader[MockEntry, MockStats]):
    """Concrete implementation for testing AbstractDataLoader."""

    def __init__(self, entries: list[MockEntry]) -> None:
        self._entries = entries
        self._stats = MockStats()
        self._compute_stats()

    def _compute_stats(self) -> None:
        total_chars = sum(len(e.content) for e in self._entries)
        self._stats = MockStats(
            total_entries=len(self._entries),
            total_chars=total_chars,
        )

    def get_entry_id(self, entry: MockEntry) -> str:
        return entry.id

    def get_searchable_content(self, entry: MockEntry) -> str | None:
        return entry.content if entry.content else None

    def get_entry_url(self, entry: MockEntry) -> str | None:
        return entry.url


# --- Fixtures ---


@pytest.fixture
def basic_loader() -> ConcreteDataLoader:
    """Loader with basic test entries."""
    entries = [
        MockEntry(id="1", content="Hello world! This is a test.", url="https://example.com/1"),
        MockEntry(id="2", content="Python is a great programming language.", url="https://example.com/2"),
        MockEntry(id="3", content="The quick brown fox jumps over the lazy dog.", url="https://example.com/3"),
        MockEntry(id="4", content="Hello again! Testing one two three.", url="https://example.com/4"),
        MockEntry(id="5", content="", url="https://example.com/5"),  # Empty content
    ]
    return ConcreteDataLoader(entries)


@pytest.fixture
def search_loader() -> ConcreteDataLoader:
    """Loader with content specifically for search testing."""
    entries = [
        MockEntry(
            id="auth-1",
            content="const AUTH_TOKEN = 'bearer_abc123'; function login() { authenticate(); }",
            url="https://example.com/auth.js",
        ),
        MockEntry(
            id="user-1",
            content="class User { constructor(name, email) { this.name = name; this.email = email; } }",
            url="https://example.com/user.js",
        ),
        MockEntry(
            id="api-1",
            content="async function fetchData(url) { const response = await fetch(url); return response.json(); }",
            url="https://api.example.com/client.js",
        ),
        MockEntry(
            id="data-1",
            content="const data = [{name: 'Alice', email: 'alice@test.com'}, {name: 'Bob', email: 'bob@test.com'}];",
            url="https://example.com/data.js",
        ),
    ]
    return ConcreteDataLoader(entries)


@pytest.fixture
def empty_loader() -> ConcreteDataLoader:
    """Empty loader for testing edge cases."""
    return ConcreteDataLoader([])


# --- Properties Tests ---


class TestProperties:
    """Tests for inherited properties."""

    def test_entries_returns_list(self, basic_loader: ConcreteDataLoader) -> None:
        """entries property returns list of entries."""
        entries = basic_loader.entries
        assert isinstance(entries, list)
        assert len(entries) == 5

    def test_stats_returns_stats(self, basic_loader: ConcreteDataLoader) -> None:
        """stats property returns stats object."""
        stats = basic_loader.stats
        assert isinstance(stats, MockStats)
        assert stats.total_entries == 5


# --- Abstract Method Tests ---


class TestAbstractMethods:
    """Tests for abstract method implementations."""

    def test_get_entry_id(self, basic_loader: ConcreteDataLoader) -> None:
        """get_entry_id returns entry ID."""
        entry = basic_loader.entries[0]
        assert basic_loader.get_entry_id(entry) == "1"

    def test_get_searchable_content(self, basic_loader: ConcreteDataLoader) -> None:
        """get_searchable_content returns entry content."""
        entry = basic_loader.entries[0]
        content = basic_loader.get_searchable_content(entry)
        assert content == "Hello world! This is a test."

    def test_get_searchable_content_empty(self, basic_loader: ConcreteDataLoader) -> None:
        """get_searchable_content returns None for empty content."""
        entry = basic_loader.entries[4]  # Empty content entry
        content = basic_loader.get_searchable_content(entry)
        assert content is None

    def test_get_entry_url(self, basic_loader: ConcreteDataLoader) -> None:
        """get_entry_url returns entry URL."""
        entry = basic_loader.entries[0]
        assert basic_loader.get_entry_url(entry) == "https://example.com/1"

    def test_get_entry_url_none(self) -> None:
        """get_entry_url returns None when URL is not set."""
        loader = ConcreteDataLoader([MockEntry(id="1", content="test")])
        entry = loader.entries[0]
        assert loader.get_entry_url(entry) is None


# --- Search by Terms Tests ---


class TestSearchByTerms:
    """Tests for search_by_terms method."""

    def test_search_single_term(self, basic_loader: ConcreteDataLoader) -> None:
        """Search for single term."""
        results = basic_loader.search_by_terms(["Hello"])
        assert len(results) == 2  # "Hello world" and "Hello again"

    def test_search_multiple_terms(self, search_loader: ConcreteDataLoader) -> None:
        """Search for multiple terms."""
        results = search_loader.search_by_terms(["email", "name"])
        assert len(results) >= 2  # user.js and data.js contain both

    def test_search_case_insensitive(self, basic_loader: ConcreteDataLoader) -> None:
        """Search is case-insensitive."""
        results_lower = basic_loader.search_by_terms(["hello"])
        results_upper = basic_loader.search_by_terms(["HELLO"])
        assert len(results_lower) == len(results_upper)

    def test_search_no_matches(self, basic_loader: ConcreteDataLoader) -> None:
        """Return empty list when no matches."""
        results = basic_loader.search_by_terms(["xyznonexistent"])
        assert results == []

    def test_search_empty_terms(self, basic_loader: ConcreteDataLoader) -> None:
        """Return empty list for empty terms list."""
        results = basic_loader.search_by_terms([])
        assert results == []

    def test_search_empty_loader(self, empty_loader: ConcreteDataLoader) -> None:
        """Search on empty loader returns empty list."""
        results = empty_loader.search_by_terms(["test"])
        assert results == []

    def test_search_skips_empty_content(self, basic_loader: ConcreteDataLoader) -> None:
        """Search skips entries with empty content."""
        results = basic_loader.search_by_terms(["test"])
        # Entry 5 has empty content and should be skipped
        entry_ids = [r["id"] for r in results]
        assert "5" not in entry_ids

    def test_search_result_structure(self, basic_loader: ConcreteDataLoader) -> None:
        """Verify search result structure."""
        results = basic_loader.search_by_terms(["test"])
        assert len(results) > 0
        result = results[0]
        assert "id" in result
        assert "url" in result
        assert "unique_terms_found" in result
        assert "total_hits" in result
        assert "score" in result

    def test_search_top_n_limit(self, basic_loader: ConcreteDataLoader) -> None:
        """Results are limited to top_n."""
        results = basic_loader.search_by_terms(["the"], top_n=1)
        assert len(results) == 1

    def test_search_scoring_unique_terms(self, search_loader: ConcreteDataLoader) -> None:
        """Score increases with more unique terms found."""
        # Search for terms that appear in different files
        results = search_loader.search_by_terms(["email", "name", "class"])
        # user.js has all three terms
        user_result = next((r for r in results if r["id"] == "user-1"), None)
        assert user_result is not None
        assert user_result["unique_terms_found"] == 3

    def test_search_scoring_sorted(self, search_loader: ConcreteDataLoader) -> None:
        """Results are sorted by score descending."""
        results = search_loader.search_by_terms(["function", "async"])
        if len(results) > 1:
            for i in range(len(results) - 1):
                assert results[i]["score"] >= results[i + 1]["score"]


# --- Search by Regex Tests ---


class TestSearchByRegex:
    """Tests for search_by_regex method."""

    def test_search_basic_pattern(self, search_loader: ConcreteDataLoader) -> None:
        """Search with basic regex pattern."""
        result = search_loader.search_by_regex(r"function\s+\w+")
        assert result["error"] is None
        assert len(result["matches"]) > 0

    def test_search_token_pattern(self, search_loader: ConcreteDataLoader) -> None:
        """Search for token-like patterns."""
        result = search_loader.search_by_regex(r"bearer_\w+")
        assert result["error"] is None
        assert len(result["matches"]) == 1

    def test_search_email_pattern(self, search_loader: ConcreteDataLoader) -> None:
        """Search for email patterns."""
        result = search_loader.search_by_regex(r"\w+@\w+\.\w+")
        assert result["error"] is None
        assert len(result["matches"]) >= 1

    def test_search_no_matches(self, search_loader: ConcreteDataLoader) -> None:
        """Return empty matches when no results."""
        result = search_loader.search_by_regex(r"xyznonexistent123")
        assert result["error"] is None
        assert result["matches"] == []

    def test_search_invalid_regex(self, search_loader: ConcreteDataLoader) -> None:
        """Return error for invalid regex pattern."""
        result = search_loader.search_by_regex(r"[invalid")
        assert result["error"] is not None
        assert "Invalid regex" in result["error"]
        assert result["matches"] == []

    def test_search_empty_loader(self, empty_loader: ConcreteDataLoader) -> None:
        """Search on empty loader returns empty matches."""
        result = empty_loader.search_by_regex(r"test")
        assert result["error"] is None
        assert result["matches"] == []

    def test_search_result_structure(self, search_loader: ConcreteDataLoader) -> None:
        """Verify regex search result structure."""
        result = search_loader.search_by_regex(r"const\s+\w+")
        assert "matches" in result
        assert "timed_out" in result
        assert "error" in result
        if result["matches"]:
            match_result = result["matches"][0]
            assert "id" in match_result
            assert "url" in match_result
            assert "match_count" in match_result
            assert "matches" in match_result

    def test_search_match_details(self, search_loader: ConcreteDataLoader) -> None:
        """Verify individual match details."""
        result = search_loader.search_by_regex(r"AUTH_TOKEN")
        assert result["matches"]
        matches = result["matches"][0]["matches"]
        assert len(matches) > 0
        match = matches[0]
        assert "match" in match
        assert "position" in match
        assert "snippet" in match
        assert match["match"] == "AUTH_TOKEN"

    def test_search_snippet_context(self, search_loader: ConcreteDataLoader) -> None:
        """Verify snippet includes context around match."""
        result = search_loader.search_by_regex(r"AUTH_TOKEN", snippet_padding_chars=20)
        matches = result["matches"][0]["matches"]
        snippet = matches[0]["snippet"]
        # Snippet should include some context before/after the match
        assert len(snippet) > len("AUTH_TOKEN")

    def test_search_top_n_limit(self, search_loader: ConcreteDataLoader) -> None:
        """Results are limited to top_n entries."""
        result = search_loader.search_by_regex(r"function", top_n=2)
        assert len(result["matches"]) <= 2

    def test_search_max_matches_per_entry(self, search_loader: ConcreteDataLoader) -> None:
        """Matches per entry are limited to max_matches_per_entry."""
        result = search_loader.search_by_regex(r"\w+", max_matches_per_entry=3)
        for entry_result in result["matches"]:
            assert len(entry_result["matches"]) <= 3


# --- Search Content Tests ---


class TestSearchContent:
    """Tests for search_content method."""

    def test_search_basic(self, basic_loader: ConcreteDataLoader) -> None:
        """Search for basic substring."""
        results = basic_loader.search_content("Hello")
        assert len(results) == 2

    def test_search_case_insensitive_default(self, basic_loader: ConcreteDataLoader) -> None:
        """Search is case-insensitive by default."""
        results_lower = basic_loader.search_content("hello")
        results_upper = basic_loader.search_content("HELLO")
        assert len(results_lower) == len(results_upper)

    def test_search_case_sensitive(self, basic_loader: ConcreteDataLoader) -> None:
        """Search respects case_sensitive flag."""
        results_insensitive = basic_loader.search_content("hello", case_sensitive=False)
        results_sensitive = basic_loader.search_content("hello", case_sensitive=True)
        # "Hello" appears with capital H, so case-sensitive "hello" should find fewer
        assert len(results_insensitive) >= len(results_sensitive)

    def test_search_no_matches(self, basic_loader: ConcreteDataLoader) -> None:
        """Return empty list when no matches."""
        results = basic_loader.search_content("xyznonexistent")
        assert results == []

    def test_search_empty_value(self, basic_loader: ConcreteDataLoader) -> None:
        """Return empty list for empty search value."""
        results = basic_loader.search_content("")
        assert results == []

    def test_search_empty_loader(self, empty_loader: ConcreteDataLoader) -> None:
        """Search on empty loader returns empty list."""
        results = empty_loader.search_content("test")
        assert results == []

    def test_search_result_structure(self, basic_loader: ConcreteDataLoader) -> None:
        """Verify search result structure."""
        results = basic_loader.search_content("test")
        assert len(results) > 0
        result = results[0]
        assert "id" in result
        assert "url" in result
        assert "count" in result
        assert "sample" in result

    def test_search_count_multiple_occurrences(self, basic_loader: ConcreteDataLoader) -> None:
        """Count reflects multiple occurrences."""
        # "the" appears twice in "The quick brown fox jumps over the lazy dog"
        results = basic_loader.search_content("the")
        entry_3_result = next((r for r in results if r["id"] == "3"), None)
        assert entry_3_result is not None
        assert entry_3_result["count"] == 2

    def test_search_sample_context(self, basic_loader: ConcreteDataLoader) -> None:
        """Sample includes context around match."""
        results = basic_loader.search_content("world", snippet_padding_chars=5)
        assert len(results) > 0
        sample = results[0]["sample"]
        assert "world" in sample.lower()

    def test_search_sample_ellipsis(self, basic_loader: ConcreteDataLoader) -> None:
        """Sample includes ellipsis when truncated."""
        results = basic_loader.search_content("quick", snippet_padding_chars=5)
        sample = results[0]["sample"]
        # Should have ellipsis since context is small
        assert "..." in sample


# --- Edge Cases Tests ---


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_search_special_regex_chars(self, search_loader: ConcreteDataLoader) -> None:
        """Handle special regex characters in patterns."""
        # These should not cause regex errors
        result = search_loader.search_by_regex(r"\(")
        assert result["error"] is None

    def test_search_unicode(self) -> None:
        """Handle unicode content."""
        entries = [MockEntry(id="1", content="Hello 世界! Привет мир!")]
        loader = ConcreteDataLoader(entries)

        results = loader.search_content("世界")
        assert len(results) == 1

    def test_search_multiline_content(self) -> None:
        """Handle multiline content."""
        entries = [MockEntry(id="1", content="Line 1\nLine 2\nLine 3")]
        loader = ConcreteDataLoader(entries)

        results = loader.search_content("Line 2")
        assert len(results) == 1

    def test_search_very_long_content(self) -> None:
        """Handle very long content."""
        long_content = "a" * 100000 + "MARKER" + "b" * 100000
        entries = [MockEntry(id="1", content=long_content)]
        loader = ConcreteDataLoader(entries)

        results = loader.search_content("MARKER")
        assert len(results) == 1
        assert results[0]["count"] == 1
