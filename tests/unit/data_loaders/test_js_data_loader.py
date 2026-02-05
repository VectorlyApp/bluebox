"""
tests/unit/data_loaders/test_js_data_loader.py

Comprehensive unit tests for JSDataLoader and related classes.
"""

from pathlib import Path

import pytest

from bluebox.llms.data_loaders.js_data_loader import (
    JSDataLoader,
    JSFileStats,
)


# --- Fixtures ---


@pytest.fixture(scope="module")
def js_events_dir(tests_root: Path) -> Path:
    """Directory containing JS event test JSONL files."""
    return tests_root / "data" / "input" / "js_events"


@pytest.fixture
def basic_loader(js_events_dir: Path) -> JSDataLoader:
    """JSDataLoader loaded from basic test data."""
    return JSDataLoader(str(js_events_dir / "js_basic.jsonl"))


@pytest.fixture
def search_loader(js_events_dir: Path) -> JSDataLoader:
    """JSDataLoader loaded from search test data."""
    return JSDataLoader(str(js_events_dir / "js_search.jsonl"))


@pytest.fixture
def urls_loader(js_events_dir: Path) -> JSDataLoader:
    """JSDataLoader loaded from URL patterns test data."""
    return JSDataLoader(str(js_events_dir / "js_urls.jsonl"))


@pytest.fixture
def malformed_loader(js_events_dir: Path) -> JSDataLoader:
    """JSDataLoader loaded from malformed test data (should skip bad lines)."""
    return JSDataLoader(str(js_events_dir / "js_malformed.jsonl"))


# --- JSFileStats Tests ---


class TestJSFileStats:
    """Tests for JSFileStats dataclass."""

    def test_to_summary_basic(self) -> None:
        """Generate summary from basic stats."""
        stats = JSFileStats(
            total_files=10,
            unique_urls=8,
            total_bytes=50000,
            hosts={"example.com": 5, "cdn.example.com": 3, "api.example.com": 2},
        )
        summary = stats.to_summary()
        assert "Total JS Files: 10" in summary
        assert "Unique URLs: 8" in summary
        assert "Top Hosts:" in summary
        assert "example.com: 5" in summary

    def test_to_summary_zero_values(self) -> None:
        """Generate summary with zero values."""
        stats = JSFileStats()
        summary = stats.to_summary()
        assert "Total JS Files: 0" in summary
        assert "Unique URLs: 0" in summary

    def test_format_bytes_small(self) -> None:
        """Format small byte values."""
        assert "500.0 B" == JSFileStats._format_bytes(500)

    def test_format_bytes_kb(self) -> None:
        """Format KB byte values."""
        result = JSFileStats._format_bytes(2048)
        assert "KB" in result

    def test_format_bytes_mb(self) -> None:
        """Format MB byte values."""
        result = JSFileStats._format_bytes(2 * 1024 * 1024)
        assert "MB" in result


# --- JSDataLoader Initialization Tests ---


class TestJSDataLoaderInit:
    """Tests for JSDataLoader initialization."""

    def test_init_basic_file(self, basic_loader: JSDataLoader) -> None:
        """Initialize from basic JSONL file."""
        assert len(basic_loader.entries) == 5

    def test_init_file_not_found(self, js_events_dir: Path) -> None:
        """Raise ValueError when file doesn't exist."""
        with pytest.raises(ValueError, match="does not exist"):
            JSDataLoader(str(js_events_dir / "nonexistent.jsonl"))

    def test_init_empty_file(self, js_events_dir: Path) -> None:
        """Initialize from empty file produces empty loader."""
        loader = JSDataLoader(str(js_events_dir / "js_empty.jsonl"))
        assert len(loader.entries) == 0

    def test_init_malformed_skips_bad_lines(self, malformed_loader: JSDataLoader) -> None:
        """Malformed lines are skipped, valid entries are loaded."""
        # Should have 3 valid entries (good-001, good-002, good-003)
        assert len(malformed_loader.entries) == 3
        request_ids = [e.request_id for e in malformed_loader.entries]
        assert "good-001" in request_ids
        assert "good-002" in request_ids
        assert "good-003" in request_ids


# --- Properties Tests ---


class TestJSDataLoaderProperties:
    """Tests for JSDataLoader properties."""

    def test_entries_returns_list(self, basic_loader: JSDataLoader) -> None:
        """entries property returns list of NetworkTransactionEvent."""
        entries = basic_loader.entries
        assert isinstance(entries, list)
        assert len(entries) > 0

    def test_stats_returns_js_file_stats(self, basic_loader: JSDataLoader) -> None:
        """stats property returns JSFileStats instance."""
        stats = basic_loader.stats
        assert isinstance(stats, JSFileStats)
        assert stats.total_files == len(basic_loader.entries)

    def test_stats_counts_unique_urls(self, basic_loader: JSDataLoader) -> None:
        """Stats correctly counts unique URLs."""
        stats = basic_loader.stats
        # All 5 URLs are unique in basic test data
        assert stats.unique_urls == 5

    def test_stats_counts_total_bytes(self, basic_loader: JSDataLoader) -> None:
        """Stats correctly counts total bytes from response bodies."""
        stats = basic_loader.stats
        # Sum of all response_body lengths
        assert stats.total_bytes > 0

    def test_stats_counts_hosts(self, basic_loader: JSDataLoader) -> None:
        """Stats correctly counts hosts."""
        stats = basic_loader.stats
        # example.com, cdn.example.com, analytics.example.com
        assert len(stats.hosts) == 3
        assert "example.com" in stats.hosts


# --- Abstract Method Implementations ---


class TestAbstractMethodImplementations:
    """Tests for AbstractDataLoader method implementations."""

    def test_get_entry_id(self, basic_loader: JSDataLoader) -> None:
        """get_entry_id returns request_id."""
        entry = basic_loader.entries[0]
        assert basic_loader.get_entry_id(entry) == entry.request_id

    def test_get_searchable_content(self, basic_loader: JSDataLoader) -> None:
        """get_searchable_content returns response body."""
        entry = basic_loader.entries[0]
        content = basic_loader.get_searchable_content(entry)
        assert content == entry.response_body

    def test_get_entry_url(self, basic_loader: JSDataLoader) -> None:
        """get_entry_url returns URL."""
        entry = basic_loader.entries[0]
        assert basic_loader.get_entry_url(entry) == entry.url


# --- File Retrieval Tests ---


class TestFileRetrieval:
    """Tests for get_file and get_file_content methods."""

    def test_get_file_found(self, basic_loader: JSDataLoader) -> None:
        """Get file by valid request_id."""
        entry = basic_loader.get_file("js-001")
        assert entry is not None
        assert entry.url == "https://example.com/main.js"

    def test_get_file_not_found(self, basic_loader: JSDataLoader) -> None:
        """Return None for non-existent request_id."""
        entry = basic_loader.get_file("nonexistent-id")
        assert entry is None

    def test_get_file_content_found(self, basic_loader: JSDataLoader) -> None:
        """Get file content by valid request_id."""
        content = basic_loader.get_file_content("js-001")
        assert content is not None
        assert "hello" in content.lower()

    def test_get_file_content_not_found(self, basic_loader: JSDataLoader) -> None:
        """Return None for non-existent request_id."""
        content = basic_loader.get_file_content("nonexistent-id")
        assert content is None

    def test_get_file_content_truncation(self, basic_loader: JSDataLoader) -> None:
        """Content is truncated when exceeding max_chars."""
        content = basic_loader.get_file_content("js-001", max_chars=20)
        assert content is not None
        assert len(content) > 20  # includes truncation message
        assert "truncated" in content


# --- List Files Tests ---


class TestListFiles:
    """Tests for list_files method."""

    def test_list_files_returns_all(self, basic_loader: JSDataLoader) -> None:
        """list_files returns info for all files."""
        files = basic_loader.list_files()
        assert len(files) == 5

    def test_list_files_structure(self, basic_loader: JSDataLoader) -> None:
        """list_files returns correct structure."""
        files = basic_loader.list_files()
        file_info = files[0]
        assert "request_id" in file_info
        assert "url" in file_info
        assert "size" in file_info

    def test_list_files_size_calculation(self, basic_loader: JSDataLoader) -> None:
        """list_files calculates correct size."""
        files = basic_loader.list_files()
        for file_info in files:
            entry = basic_loader.get_file(file_info["request_id"])
            expected_size = len(entry.response_body) if entry and entry.response_body else 0
            assert file_info["size"] == expected_size


# --- Search by URL Tests ---


class TestSearchByUrl:
    """Tests for search_by_url method."""

    def test_search_by_url_glob_match(self, urls_loader: JSDataLoader) -> None:
        """Find files matching glob pattern."""
        results = urls_loader.search_by_url("*bundle*")
        assert len(results) == 2  # bundle.js and bundle.min.js

    def test_search_by_url_vendor_match(self, urls_loader: JSDataLoader) -> None:
        """Find files in vendor directory."""
        results = urls_loader.search_by_url("*/vendor/*")
        assert len(results) == 2  # lodash.js and react.min.js

    def test_search_by_url_no_match(self, urls_loader: JSDataLoader) -> None:
        """Return empty list when no matches."""
        results = urls_loader.search_by_url("*nonexistent*")
        assert results == []

    def test_search_by_url_host_match(self, urls_loader: JSDataLoader) -> None:
        """Find files by host pattern."""
        results = urls_loader.search_by_url("https://cdn.example.com/*")
        assert len(results) == 1


# --- Search by Terms Tests ---


class TestSearchByTerms:
    """Tests for search_by_terms method."""

    def test_search_single_term(self, search_loader: JSDataLoader) -> None:
        """Find files with single term."""
        results = search_loader.search_by_terms(["login"])
        assert len(results) > 0
        assert results[0]["id"] == "search-001"  # auth.js contains login

    def test_search_multiple_terms(self, search_loader: JSDataLoader) -> None:
        """Find files with multiple terms."""
        results = search_loader.search_by_terms(["User", "email"])
        assert len(results) > 0

    def test_search_case_insensitive(self, search_loader: JSDataLoader) -> None:
        """Search is case-insensitive."""
        results_lower = search_loader.search_by_terms(["login"])
        results_upper = search_loader.search_by_terms(["LOGIN"])
        assert len(results_lower) == len(results_upper)

    def test_search_no_matches(self, search_loader: JSDataLoader) -> None:
        """Return empty list when no matches."""
        results = search_loader.search_by_terms(["xyznonexistent123"])
        assert results == []

    def test_search_empty_terms(self, search_loader: JSDataLoader) -> None:
        """Return empty list for empty terms."""
        results = search_loader.search_by_terms([])
        assert results == []

    def test_search_result_structure(self, search_loader: JSDataLoader) -> None:
        """Verify search result structure."""
        results = search_loader.search_by_terms(["function"])
        assert len(results) > 0
        result = results[0]
        assert "id" in result
        assert "url" in result
        assert "unique_terms_found" in result
        assert "total_hits" in result
        assert "score" in result

    def test_search_top_n_limit(self, search_loader: JSDataLoader) -> None:
        """Results are limited to top_n."""
        results = search_loader.search_by_terms(["function"], top_n=2)
        assert len(results) <= 2

    def test_search_scoring(self, search_loader: JSDataLoader) -> None:
        """Results are sorted by score descending."""
        results = search_loader.search_by_terms(["function"])
        if len(results) > 1:
            for i in range(len(results) - 1):
                assert results[i]["score"] >= results[i + 1]["score"]


# --- Search by Regex Tests ---


class TestSearchByRegex:
    """Tests for search_by_regex method."""

    def test_search_regex_basic(self, search_loader: JSDataLoader) -> None:
        """Find files with basic regex pattern."""
        result = search_loader.search_by_regex(r"function\s+\w+")
        assert result["error"] is None
        assert len(result["files"]) > 0

    def test_search_regex_token_pattern(self, search_loader: JSDataLoader) -> None:
        """Find files with token pattern."""
        result = search_loader.search_by_regex(r"bearer_\w+")
        assert result["error"] is None
        assert len(result["files"]) >= 2  # In auth.js and payment.js

    def test_search_regex_no_matches(self, search_loader: JSDataLoader) -> None:
        """Return empty list when no matches."""
        result = search_loader.search_by_regex(r"xyznonexistent123")
        assert result["error"] is None
        assert result["files"] == []

    def test_search_regex_invalid_pattern(self, search_loader: JSDataLoader) -> None:
        """Return error for invalid regex."""
        result = search_loader.search_by_regex(r"[invalid")
        assert result["error"] is not None
        assert "Invalid regex" in result["error"]

    def test_search_regex_result_structure(self, search_loader: JSDataLoader) -> None:
        """Verify regex search result structure."""
        result = search_loader.search_by_regex(r"const\s+\w+")
        assert "files" in result
        assert "timed_out" in result
        assert "error" in result
        if result["files"]:
            file_result = result["files"][0]
            assert "request_id" in file_result
            assert "url" in file_result
            assert "match_count" in file_result
            assert "matches" in file_result

    def test_search_regex_match_snippet(self, search_loader: JSDataLoader) -> None:
        """Regex matches include context snippets."""
        result = search_loader.search_by_regex(r"AUTH_TOKEN")
        assert result["files"]
        matches = result["files"][0]["matches"]
        assert len(matches) > 0
        match = matches[0]
        assert "match" in match
        assert "position" in match
        assert "snippet" in match

    def test_search_regex_top_n_limit(self, search_loader: JSDataLoader) -> None:
        """Results are limited to top_n files."""
        result = search_loader.search_by_regex(r"function", top_n=2)
        assert len(result["files"]) <= 2

    def test_search_regex_max_matches_per_file(self, search_loader: JSDataLoader) -> None:
        """Matches per file are limited to max_matches_per_file."""
        result = search_loader.search_by_regex(r"\w+", max_matches_per_file=3)
        for file_result in result["files"]:
            assert len(file_result["matches"]) <= 3


# --- Feature Detection via Stats Tests ---


class TestFeatureDetection:
    """Tests for feature detection via computed stats."""

    def test_multiple_hosts(self, basic_loader: JSDataLoader) -> None:
        """Loader with multiple hosts is correctly detected."""
        stats = basic_loader.stats
        assert len(stats.hosts) > 1

    def test_empty_loader_stats(self, js_events_dir: Path) -> None:
        """Empty loader has zero stats."""
        loader = JSDataLoader(str(js_events_dir / "js_empty.jsonl"))
        stats = loader.stats
        assert stats.total_files == 0
        assert stats.unique_urls == 0
        assert stats.total_bytes == 0
        assert len(stats.hosts) == 0
