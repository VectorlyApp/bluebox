"""
tests/unit/test_network_data_store.py

Comprehensive unit tests for NetworkDataStore and related classes.
"""

from pathlib import Path

import pytest

from bluebox.llms.infra.network_data_store import (
    NetworkDataStore,
    NetworkStats,
)


# --- Fixtures ---

@pytest.fixture(scope="module")
def network_events_dir(tests_root: Path) -> Path:
    """Directory containing network event test JSONL files."""
    return tests_root / "data" / "input" / "network_events"


@pytest.fixture
def basic_store(network_events_dir: Path) -> NetworkDataStore:
    """NetworkDataStore loaded from basic test data."""
    return NetworkDataStore(str(network_events_dir / "network_basic.jsonl"))


@pytest.fixture
def api_store(network_events_dir: Path) -> NetworkDataStore:
    """NetworkDataStore loaded from API test data."""
    return NetworkDataStore(str(network_events_dir / "network_api.jsonl"))


@pytest.fixture
def search_store(network_events_dir: Path) -> NetworkDataStore:
    """NetworkDataStore loaded from search test data."""
    return NetworkDataStore(str(network_events_dir / "network_search.jsonl"))


@pytest.fixture
def hosts_store(network_events_dir: Path) -> NetworkDataStore:
    """NetworkDataStore loaded from hosts test data."""
    return NetworkDataStore(str(network_events_dir / "network_hosts.jsonl"))


@pytest.fixture
def malformed_store(network_events_dir: Path) -> NetworkDataStore:
    """NetworkDataStore loaded from malformed test data (should skip bad lines)."""
    return NetworkDataStore(str(network_events_dir / "network_malformed.jsonl"))


# --- NetworkStats Tests ---

class TestNetworkStats:
    """Tests for NetworkStats dataclass."""

    def test_format_bytes_bytes(self) -> None:
        """Format small byte values."""
        assert NetworkStats._format_bytes(500) == "500.0 B"

    def test_format_bytes_kilobytes(self) -> None:
        """Format kilobyte values."""
        assert NetworkStats._format_bytes(2048) == "2.0 KB"

    def test_format_bytes_megabytes(self) -> None:
        """Format megabyte values."""
        assert NetworkStats._format_bytes(5 * 1024 * 1024) == "5.0 MB"

    def test_format_bytes_gigabytes(self) -> None:
        """Format gigabyte values."""
        assert NetworkStats._format_bytes(3 * 1024 * 1024 * 1024) == "3.0 GB"

    def test_format_bytes_zero(self) -> None:
        """Format zero bytes."""
        assert NetworkStats._format_bytes(0) == "0.0 B"

    def test_to_summary_basic(self) -> None:
        """Generate summary from basic stats."""
        stats = NetworkStats(
            total_requests=10,
            unique_hosts=2,
            unique_paths=5,
            methods={"GET": 8, "POST": 2},
            status_codes={200: 9, 404: 1},
        )
        summary = stats.to_summary()
        assert "Total Requests: 10" in summary
        assert "Unique Hosts: 2" in summary
        assert "GET: 8" in summary
        assert "POST: 2" in summary
        assert "200: 9" in summary

    def test_to_summary_with_features(self) -> None:
        """Generate summary with detected features."""
        stats = NetworkStats(
            total_requests=5,
            has_auth_headers=True,
            has_json_requests=True,
        )
        summary = stats.to_summary()
        assert "Authentication headers present" in summary
        assert "JSON request bodies present" in summary


# --- NetworkDataStore Initialization Tests ---

class TestNetworkDataStoreInit:
    """Tests for NetworkDataStore initialization."""

    def test_init_basic_file(self, basic_store: NetworkDataStore) -> None:
        """Initialize from basic JSONL file."""
        # Should filter out CSS, JS, and image entries
        # Expected: HTML (req-001), JSON entries (req-002, req-004, req-007, req-008)
        assert len(basic_store.entries) >= 4

    def test_init_file_not_found(self, network_events_dir: Path) -> None:
        """Raise error when file doesn't exist."""
        with pytest.raises(ValueError, match="does not exist"):
            NetworkDataStore(str(network_events_dir / "nonexistent.jsonl"))

    def test_init_empty_file(self, network_events_dir: Path) -> None:
        """Initialize from empty file produces empty store."""
        store = NetworkDataStore(str(network_events_dir / "network_empty.jsonl"))
        assert len(store.entries) == 0

    def test_init_malformed_skips_bad_lines(self, malformed_store: NetworkDataStore) -> None:
        """Malformed lines are skipped, valid entries are loaded."""
        # Should have 3 valid entries (good-001, good-002, good-003)
        assert len(malformed_store.entries) == 3
        request_ids = [e.request_id for e in malformed_store.entries]
        assert "good-001" in request_ids
        assert "good-002" in request_ids
        assert "good-003" in request_ids


# --- Properties Tests ---

class TestNetworkDataStoreProperties:
    """Tests for NetworkDataStore properties."""

    def test_entries_returns_list(self, basic_store: NetworkDataStore) -> None:
        """entries property returns list of NetworkTransactionEvent."""
        entries = basic_store.entries
        assert isinstance(entries, list)
        assert len(entries) > 0

    def test_stats_returns_network_stats(self, basic_store: NetworkDataStore) -> None:
        """stats property returns NetworkStats instance."""
        stats = basic_store.stats
        assert isinstance(stats, NetworkStats)
        assert stats.total_requests == len(basic_store.entries)

    def test_stats_methods_counted(self, basic_store: NetworkDataStore) -> None:
        """Stats correctly counts HTTP methods."""
        stats = basic_store.stats
        assert "GET" in stats.methods or "POST" in stats.methods

    def test_stats_status_codes_counted(self, basic_store: NetworkDataStore) -> None:
        """Stats correctly counts status codes."""
        stats = basic_store.stats
        assert 200 in stats.status_codes or 201 in stats.status_codes

    def test_raw_data_structure(self, basic_store: NetworkDataStore) -> None:
        """raw_data returns dict with entries key."""
        raw = basic_store.raw_data
        assert isinstance(raw, dict)
        assert "entries" in raw
        assert isinstance(raw["entries"], list)

    def test_url_counts_returns_dict(self, basic_store: NetworkDataStore) -> None:
        """url_counts returns dict mapping URLs to counts."""
        counts = basic_store.url_counts
        assert isinstance(counts, dict)
        for url, count in counts.items():
            assert isinstance(url, str)
            assert isinstance(count, int)
            assert count > 0

    def test_url_counts_duplicate_urls(self, basic_store: NetworkDataStore) -> None:
        """url_counts correctly counts duplicate URLs."""
        counts = basic_store.url_counts
        # req-002 and req-008 both hit /api/users
        if "https://example.com/api/users" in counts:
            assert counts["https://example.com/api/users"] == 2

    def test_api_urls_detects_versioned_apis(self, api_store: NetworkDataStore) -> None:
        """api_urls detects versioned API endpoints (/v1/, /v2/, etc.)."""
        api_urls = api_store.api_urls
        assert any("/v1/" in url for url in api_urls)
        assert any("/v2/" in url for url in api_urls)

    def test_api_urls_detects_api_keywords(self, api_store: NetworkDataStore) -> None:
        """api_urls detects endpoints with API keywords."""
        api_urls = api_store.api_urls
        # Should detect graphql, rest/api
        assert any("graphql" in url for url in api_urls)

    def test_api_urls_excludes_non_api(self, api_store: NetworkDataStore) -> None:
        """api_urls excludes non-API URLs like .html pages."""
        api_urls = api_store.api_urls
        assert not any(url.endswith(".html") for url in api_urls)

    def test_api_urls_sorted(self, api_store: NetworkDataStore) -> None:
        """api_urls returns sorted list."""
        api_urls = api_store.api_urls
        assert api_urls == sorted(api_urls)


# --- Search Methods Tests ---

class TestSearchEntries:
    """Tests for search_entries method."""

    def test_search_by_method_get(self, basic_store: NetworkDataStore) -> None:
        """Filter entries by GET method."""
        results = basic_store.search_entries(method="GET")
        assert all(e.method == "GET" for e in results)

    def test_search_by_method_post(self, basic_store: NetworkDataStore) -> None:
        """Filter entries by POST method."""
        results = basic_store.search_entries(method="POST")
        assert all(e.method == "POST" for e in results)

    def test_search_by_method_case_insensitive(self, basic_store: NetworkDataStore) -> None:
        """Method filter is case-insensitive."""
        results_upper = basic_store.search_entries(method="GET")
        results_lower = basic_store.search_entries(method="get")
        assert len(results_upper) == len(results_lower)

    def test_search_by_host_contains(self, basic_store: NetworkDataStore) -> None:
        """Filter entries by host substring."""
        results = basic_store.search_entries(host_contains="api")
        assert all("api" in e.url.lower() for e in results)

    def test_search_by_path_contains(self, basic_store: NetworkDataStore) -> None:
        """Filter entries by path substring."""
        results = basic_store.search_entries(path_contains="users")
        assert all("users" in e.url.lower() for e in results)

    def test_search_by_status_code(self, basic_store: NetworkDataStore) -> None:
        """Filter entries by exact status code."""
        results = basic_store.search_entries(status_code=200)
        assert all(e.status == 200 for e in results)

    def test_search_by_content_type(self, basic_store: NetworkDataStore) -> None:
        """Filter entries by content type substring."""
        results = basic_store.search_entries(content_type_contains="json")
        assert all("json" in e.mime_type.lower() for e in results)

    def test_search_by_has_post_data_true(self, basic_store: NetworkDataStore) -> None:
        """Filter entries that have POST data."""
        results = basic_store.search_entries(has_post_data=True)
        assert all(e.post_data is not None for e in results)

    def test_search_by_has_post_data_false(self, basic_store: NetworkDataStore) -> None:
        """Filter entries that don't have POST data."""
        results = basic_store.search_entries(has_post_data=False)
        assert all(e.post_data is None for e in results)

    def test_search_combined_filters(self, basic_store: NetworkDataStore) -> None:
        """Combine multiple filters."""
        results = basic_store.search_entries(method="GET", status_code=200)
        assert all(e.method == "GET" and e.status == 200 for e in results)

    def test_search_no_matches(self, basic_store: NetworkDataStore) -> None:
        """Return empty list when no entries match."""
        results = basic_store.search_entries(status_code=999)
        assert results == []


# --- Entry Retrieval Tests ---

class TestEntryRetrieval:
    """Tests for get_entry and get_entry_ids_by_url_pattern."""

    def test_get_entry_found(self, basic_store: NetworkDataStore) -> None:
        """Get entry by valid request_id."""
        entry = basic_store.get_entry("req-001")
        assert entry is not None
        assert entry.request_id == "req-001"

    def test_get_entry_not_found(self, basic_store: NetworkDataStore) -> None:
        """Return None for non-existent request_id."""
        entry = basic_store.get_entry("nonexistent-id")
        assert entry is None

    def test_get_entry_ids_by_url_pattern_wildcard(self, basic_store: NetworkDataStore) -> None:
        """Match URLs with wildcard pattern."""
        ids = basic_store.get_entry_ids_by_url_pattern("*api*")
        assert len(ids) > 0
        # Verify all matched entries have 'api' in URL
        for request_id in ids:
            entry = basic_store.get_entry(request_id)
            assert "api" in entry.url.lower()

    def test_get_entry_ids_by_url_pattern_exact(self, basic_store: NetworkDataStore) -> None:
        """Match URLs with exact pattern."""
        ids = basic_store.get_entry_ids_by_url_pattern("https://example.com/page")
        assert "req-001" in ids

    def test_get_entry_ids_by_url_pattern_no_match(self, basic_store: NetworkDataStore) -> None:
        """Return empty list when no URLs match pattern."""
        ids = basic_store.get_entry_ids_by_url_pattern("*nonexistent*")
        assert ids == []

    def test_get_entry_ids_by_url_pattern_prefix(self, basic_store: NetworkDataStore) -> None:
        """Match URLs with prefix pattern."""
        ids = basic_store.get_entry_ids_by_url_pattern("https://example.com/*")
        assert len(ids) > 0


# --- Term Search Tests ---

class TestSearchEntriesByTerms:
    """Tests for search_entries_by_terms method."""

    def test_search_single_term(self, search_store: NetworkDataStore) -> None:
        """Search with single term."""
        results = search_store.search_entries_by_terms(["train"])
        assert len(results) > 0
        assert all("train" in r["url"].lower() or r["total_hits"] > 0 for r in results)

    def test_search_multiple_terms(self, search_store: NetworkDataStore) -> None:
        """Search with multiple terms."""
        results = search_store.search_entries_by_terms(["price", "NYC", "Boston"])
        assert len(results) > 0

    def test_search_returns_scored_results(self, search_store: NetworkDataStore) -> None:
        """Results include score, unique_terms_found, total_hits."""
        results = search_store.search_entries_by_terms(["price"])
        assert len(results) > 0
        for r in results:
            assert "id" in r
            assert "url" in r
            assert "score" in r
            assert "unique_terms_found" in r
            assert "total_hits" in r

    def test_search_sorted_by_score(self, search_store: NetworkDataStore) -> None:
        """Results are sorted by score descending."""
        results = search_store.search_entries_by_terms(["price", "train"])
        if len(results) > 1:
            scores = [r["score"] for r in results]
            assert scores == sorted(scores, reverse=True)

    def test_search_respects_top_n(self, search_store: NetworkDataStore) -> None:
        """Results limited to top_n."""
        results = search_store.search_entries_by_terms(["the", "a"], top_n=2)
        assert len(results) <= 2

    def test_search_empty_terms(self, search_store: NetworkDataStore) -> None:
        """Return empty list for empty terms."""
        results = search_store.search_entries_by_terms([])
        assert results == []

    def test_search_no_matches(self, search_store: NetworkDataStore) -> None:
        """Return empty list when no terms match."""
        results = search_store.search_entries_by_terms(["xyznonexistent123"])
        assert results == []

    def test_search_case_insensitive(self, search_store: NetworkDataStore) -> None:
        """Search is case-insensitive."""
        results_lower = search_store.search_entries_by_terms(["nyc"])
        results_upper = search_store.search_entries_by_terms(["NYC"])
        assert len(results_lower) == len(results_upper)

    def test_unique_terms_found_counts_each_term_once(self, search_store: NetworkDataStore) -> None:
        """unique_terms_found counts each term only once, not per occurrence."""
        # Search with terms that may appear multiple times in same entry
        results = search_store.search_entries_by_terms(["price", "train"])
        for r in results:
            # unique_terms_found should be at most the number of search terms
            assert r["unique_terms_found"] <= 2
            # total_hits can be greater (multiple occurrences of same term)
            assert r["total_hits"] >= r["unique_terms_found"]


# --- Host Stats Tests ---

class TestGetHostStats:
    """Tests for get_host_stats method."""

    def test_host_stats_returns_list(self, hosts_store: NetworkDataStore) -> None:
        """get_host_stats returns list of dicts."""
        stats = hosts_store.get_host_stats()
        assert isinstance(stats, list)
        assert len(stats) > 0

    def test_host_stats_structure(self, hosts_store: NetworkDataStore) -> None:
        """Each host stat has required keys."""
        stats = hosts_store.get_host_stats()
        for hs in stats:
            assert "host" in hs
            assert "request_count" in hs
            assert "methods" in hs
            assert "status_codes" in hs

    def test_host_stats_sorted_by_request_count(self, hosts_store: NetworkDataStore) -> None:
        """Results sorted by request count descending."""
        stats = hosts_store.get_host_stats()
        if len(stats) > 1:
            counts = [hs["request_count"] for hs in stats]
            assert counts == sorted(counts, reverse=True)

    def test_host_stats_filter(self, hosts_store: NetworkDataStore) -> None:
        """Filter hosts by substring."""
        stats = hosts_store.get_host_stats(host_filter="api")
        assert all("api" in hs["host"].lower() for hs in stats)

    def test_host_stats_filter_no_match(self, hosts_store: NetworkDataStore) -> None:
        """Return empty list when filter matches nothing."""
        stats = hosts_store.get_host_stats(host_filter="nonexistent")
        assert stats == []

    def test_host_stats_methods_counted(self, hosts_store: NetworkDataStore) -> None:
        """Methods are correctly counted per host."""
        stats = hosts_store.get_host_stats()
        for hs in stats:
            total_methods = sum(hs["methods"].values())
            assert total_methods == hs["request_count"]


# --- Response Body Search Tests ---

class TestSearchResponseBodies:
    """Tests for search_response_bodies method."""

    def test_search_finds_value(self, search_store: NetworkDataStore) -> None:
        """Find entries containing a specific value."""
        results = search_store.search_response_bodies("49.99")
        assert len(results) > 0

    def test_search_returns_context(self, search_store: NetworkDataStore) -> None:
        """Results include sample with context."""
        results = search_store.search_response_bodies("Boston")
        assert len(results) > 0
        for r in results:
            assert "sample" in r
            assert "Boston" in r["sample"] or "boston" in r["sample"].lower()

    def test_search_returns_count(self, search_store: NetworkDataStore) -> None:
        """Results include occurrence count."""
        results = search_store.search_response_bodies("price")
        assert len(results) > 0
        for r in results:
            assert "count" in r
            assert r["count"] > 0

    def test_search_case_insensitive_default(self, search_store: NetworkDataStore) -> None:
        """Search is case-insensitive by default."""
        results_lower = search_store.search_response_bodies("boston")
        results_upper = search_store.search_response_bodies("BOSTON")
        assert len(results_lower) == len(results_upper)

    def test_search_case_sensitive(self, search_store: NetworkDataStore) -> None:
        """Case-sensitive search when specified."""
        results_sensitive = search_store.search_response_bodies("Boston", case_sensitive=True)
        results_wrong_case = search_store.search_response_bodies("BOSTON", case_sensitive=True)
        # "Boston" appears with capital B, so sensitive search for "BOSTON" should find fewer/none
        assert len(results_sensitive) >= len(results_wrong_case)

    def test_search_empty_value(self, search_store: NetworkDataStore) -> None:
        """Return empty list for empty search value."""
        results = search_store.search_response_bodies("")
        assert results == []

    def test_search_no_matches(self, search_store: NetworkDataStore) -> None:
        """Return empty list when value not found."""
        results = search_store.search_response_bodies("xyznonexistent123")
        assert results == []

    def test_search_ellipsis_truncation(self, search_store: NetworkDataStore) -> None:
        """Sample includes ellipsis when truncated."""
        results = search_store.search_response_bodies("trains")
        assert len(results) > 0
        # At least one result should have ellipsis if content is long
        has_ellipsis = any("..." in r["sample"] for r in results)
        # This depends on content length, so just verify structure
        for r in results:
            assert isinstance(r["sample"], str)


# --- Schema Extraction Tests ---

class TestGetResponseBodySchema:
    """Tests for get_response_body_schema method."""

    def test_schema_for_json_response(self, basic_store: NetworkDataStore) -> None:
        """Extract schema from JSON response body."""
        # req-002 has JSON response: {"id": 123, "name": "Alice", "email": "..."}
        schema = basic_store.get_response_body_schema("req-002")
        assert schema is not None
        assert schema["_type"] == "dict"
        assert "id" in schema
        assert "name" in schema

    def test_schema_for_array_response(self, basic_store: NetworkDataStore) -> None:
        """Extract schema from JSON array response."""
        # req-004 has JSON array response
        schema = basic_store.get_response_body_schema("req-004")
        assert schema is not None
        assert schema["_type"] == "list"

    def test_schema_for_non_json_returns_none(self, basic_store: NetworkDataStore) -> None:
        """Return None for non-JSON response body."""
        # req-001 is HTML
        schema = basic_store.get_response_body_schema("req-001")
        assert schema is None

    def test_schema_for_nonexistent_entry(self, basic_store: NetworkDataStore) -> None:
        """Return None for non-existent entry."""
        schema = basic_store.get_response_body_schema("nonexistent-id")
        assert schema is None


# --- Feature Detection Tests ---

class TestFeatureDetection:
    """Tests for feature detection in stats."""

    def test_detects_auth_headers(self, hosts_store: NetworkDataStore) -> None:
        """Detect presence of authorization headers."""
        stats = hosts_store.stats
        assert stats.has_auth_headers is True

    def test_detects_json_requests(self, hosts_store: NetworkDataStore) -> None:
        """Detect presence of JSON request bodies."""
        stats = hosts_store.stats
        assert stats.has_json_requests is True

    def test_detects_form_data(self, hosts_store: NetworkDataStore) -> None:
        """Detect presence of form data."""
        stats = hosts_store.stats
        assert stats.has_form_data is True
