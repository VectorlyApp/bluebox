"""
tests/unit/test_storage_data_store.py

Comprehensive unit tests for StorageDataStore and related classes.
"""

from pathlib import Path

import pytest

from bluebox.llms.infra.storage_data_store import (
    StorageDataStore,
    StorageStats,
)


# --- Fixtures ---


@pytest.fixture(scope="module")
def storage_events_dir(tests_root: Path) -> Path:
    """Directory containing storage event test JSONL files."""
    return tests_root / "data" / "input" / "storage_events"


@pytest.fixture
def basic_store(storage_events_dir: Path) -> StorageDataStore:
    """StorageDataStore loaded from basic test data."""
    return StorageDataStore(str(storage_events_dir / "storage_basic.jsonl"))


@pytest.fixture
def search_store(storage_events_dir: Path) -> StorageDataStore:
    """StorageDataStore loaded from search test data."""
    return StorageDataStore(str(storage_events_dir / "storage_search.jsonl"))


@pytest.fixture
def origins_store(storage_events_dir: Path) -> StorageDataStore:
    """StorageDataStore loaded from origins test data."""
    return StorageDataStore(str(storage_events_dir / "storage_origins.jsonl"))


@pytest.fixture
def malformed_store(storage_events_dir: Path) -> StorageDataStore:
    """StorageDataStore loaded from malformed test data (should skip bad lines)."""
    return StorageDataStore(str(storage_events_dir / "storage_malformed.jsonl"))


# --- StorageStats Tests ---


class TestStorageStats:
    """Tests for StorageStats dataclass."""

    def test_to_summary_basic(self) -> None:
        """Generate summary from basic stats."""
        stats = StorageStats(
            total_events=10,
            cookie_events=3,
            local_storage_events=4,
            session_storage_events=2,
            indexed_db_events=1,
            unique_origins=2,
            unique_keys=5,
        )
        summary = stats.to_summary()
        assert "Total Events: 10" in summary
        assert "Cookie: 3" in summary
        assert "LocalStorage: 4" in summary
        assert "SessionStorage: 2" in summary
        assert "IndexedDB: 1" in summary
        assert "Unique Origins: 2" in summary
        assert "Unique Keys: 5" in summary

    def test_to_summary_zero_values(self) -> None:
        """Generate summary with zero values."""
        stats = StorageStats()
        summary = stats.to_summary()
        assert "Total Events: 0" in summary
        assert "Cookie: 0" in summary


# --- StorageDataStore Initialization Tests ---


class TestStorageDataStoreInit:
    """Tests for StorageDataStore initialization."""

    def test_init_basic_file(self, basic_store: StorageDataStore) -> None:
        """Initialize from basic JSONL file."""
        # Should have 8 events
        assert len(basic_store.entries) == 8

    def test_init_file_not_found(self, storage_events_dir: Path) -> None:
        """Raise FileNotFoundError when file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            StorageDataStore(str(storage_events_dir / "nonexistent.jsonl"))

    def test_init_empty_file(self, storage_events_dir: Path) -> None:
        """Initialize from empty file produces empty store."""
        store = StorageDataStore(str(storage_events_dir / "storage_empty.jsonl"))
        assert len(store.entries) == 0

    def test_init_malformed_skips_bad_lines(self, malformed_store: StorageDataStore) -> None:
        """Malformed lines are skipped, valid entries are loaded."""
        # Should have 3 valid entries (good_key_1, good_key_2, good_key_3)
        assert len(malformed_store.entries) == 3
        keys = [e.key for e in malformed_store.entries]
        assert "good_key_1" in keys
        assert "good_key_2" in keys
        assert "good_key_3" in keys


# --- Properties Tests ---


class TestStorageDataStoreProperties:
    """Tests for StorageDataStore properties."""

    def test_entries_returns_list(self, basic_store: StorageDataStore) -> None:
        """entries property returns list of StorageEvent."""
        entries = basic_store.entries
        assert isinstance(entries, list)
        assert len(entries) > 0

    def test_stats_returns_storage_stats(self, basic_store: StorageDataStore) -> None:
        """stats property returns StorageStats instance."""
        stats = basic_store.stats
        assert isinstance(stats, StorageStats)
        assert stats.total_events == len(basic_store.entries)

    def test_stats_counts_cookie_events(self, basic_store: StorageDataStore) -> None:
        """Stats correctly counts cookie events."""
        stats = basic_store.stats
        # We have initialCookies and cookieChange events
        assert stats.cookie_events == 2

    def test_stats_counts_local_storage_events(self, basic_store: StorageDataStore) -> None:
        """Stats correctly counts localStorage events."""
        stats = basic_store.stats
        # localStorageItemAdded, localStorageItemUpdated, localStorageItemRemoved
        assert stats.local_storage_events == 3

    def test_stats_counts_session_storage_events(self, basic_store: StorageDataStore) -> None:
        """Stats correctly counts sessionStorage events."""
        stats = basic_store.stats
        # Two sessionStorageItemAdded events
        assert stats.session_storage_events == 2

    def test_stats_counts_indexed_db_events(self, basic_store: StorageDataStore) -> None:
        """Stats correctly counts IndexedDB events."""
        stats = basic_store.stats
        assert stats.indexed_db_events == 1

    def test_stats_counts_unique_origins(self, basic_store: StorageDataStore) -> None:
        """Stats correctly counts unique origins."""
        stats = basic_store.stats
        # https://example.com and https://api.example.com
        assert stats.unique_origins == 2

    def test_stats_counts_unique_keys(self, basic_store: StorageDataStore) -> None:
        """Stats correctly counts unique keys."""
        stats = basic_store.stats
        # auth_token, cart_items, request_cache, temp_data (cookies don't have key field)
        assert stats.unique_keys >= 4


# --- Entry Retrieval Tests ---


class TestEntryRetrieval:
    """Tests for get_entry and get_entries_by_* methods."""

    def test_get_entry_found(self, basic_store: StorageDataStore) -> None:
        """Get entry by valid index."""
        entry = basic_store.get_entry(0)
        assert entry is not None
        assert entry.type == "initialCookies"

    def test_get_entry_not_found(self, basic_store: StorageDataStore) -> None:
        """Return None for non-existent index."""
        entry = basic_store.get_entry(999)
        assert entry is None

    def test_get_entries_by_origin_found(self, origins_store: StorageDataStore) -> None:
        """Get entries by exact origin match."""
        results = origins_store.get_entries_by_origin("https://app.example.com")
        assert len(results) == 3
        assert all(e.origin == "https://app.example.com" for e in results)

    def test_get_entries_by_origin_not_found(self, origins_store: StorageDataStore) -> None:
        """Return empty list when origin not found."""
        results = origins_store.get_entries_by_origin("https://nonexistent.com")
        assert results == []

    def test_get_entries_by_key_found(self, origins_store: StorageDataStore) -> None:
        """Get entries by exact key match."""
        results = origins_store.get_entries_by_key("setting_1")
        # Two entries have key "setting_1" (different origins)
        assert len(results) == 2
        assert all(e.key == "setting_1" for e in results)

    def test_get_entries_by_key_not_found(self, origins_store: StorageDataStore) -> None:
        """Return empty list when key not found."""
        results = origins_store.get_entries_by_key("nonexistent_key")
        assert results == []

    def test_get_entries_by_origin_and_key(self, origins_store: StorageDataStore) -> None:
        """Get entries by both origin and key."""
        results = origins_store.get_entries_by_origin_and_key(
            "https://app.example.com", "setting_1"
        )
        assert len(results) == 1
        assert results[0].origin == "https://app.example.com"
        assert results[0].key == "setting_1"

    def test_get_entries_by_origin_and_key_no_match(self, origins_store: StorageDataStore) -> None:
        """Return empty list when origin/key combination not found."""
        results = origins_store.get_entries_by_origin_and_key(
            "https://app.example.com", "cache_key"
        )
        assert results == []


# --- Search Values Tests ---


class TestSearchValues:
    """Tests for search_values method (token tracing)."""

    def test_search_finds_value_in_value_field(self, search_store: StorageDataStore) -> None:
        """Find entries with matching value field."""
        results = search_store.search_values("Alice Johnson")
        assert len(results) > 0
        assert any("value" in r["match_locations"] for r in results)

    def test_search_finds_value_in_new_value_field(self, search_store: StorageDataStore) -> None:
        """Find entries with matching new_value field."""
        results = search_store.search_values("Alice Smith")
        assert len(results) > 0
        assert any("new_value" in r["match_locations"] for r in results)

    def test_search_finds_value_in_old_value_field(self, search_store: StorageDataStore) -> None:
        """Find entries with matching old_value field."""
        # Search for value that appears in old_value
        results = search_store.search_values("Alice Johnson")
        # The update event has old_value with Alice Johnson
        has_old_value_match = any("old_value" in r["match_locations"] for r in results)
        assert has_old_value_match

    def test_search_finds_value_in_added_field(self, search_store: StorageDataStore) -> None:
        """Find entries with matching added cookies/items."""
        results = search_store.search_values("secret_token_12345")
        assert len(results) > 0
        # Should find in cookie added field
        assert any("added" in r["match_locations"] for r in results)

    def test_search_finds_value_in_modified_field(self, search_store: StorageDataStore) -> None:
        """Find entries with matching modified cookies/items."""
        results = search_store.search_values("new_secret_token_67890")
        assert len(results) > 0
        assert any("modified" in r["match_locations"] for r in results)

    def test_search_multiple_locations_same_entry(self, search_store: StorageDataStore) -> None:
        """Value can be found in multiple locations in same entry."""
        # "secret_token_12345" appears in value field and added field in different entries
        # and also appears in checkout_state value
        results = search_store.search_values("secret_token_12345")
        assert len(results) >= 2  # At least in added cookies and in checkout_state value

    def test_search_case_insensitive_default(self, search_store: StorageDataStore) -> None:
        """Search is case-insensitive by default."""
        results_lower = search_store.search_values("alice")
        results_upper = search_store.search_values("ALICE")
        assert len(results_lower) == len(results_upper)

    def test_search_case_sensitive(self, search_store: StorageDataStore) -> None:
        """Case-sensitive search when specified."""
        results_sensitive = search_store.search_values("Alice", case_sensitive=True)
        results_wrong_case = search_store.search_values("ALICE", case_sensitive=True)
        # "Alice" appears with capital A, so case-sensitive search for "ALICE" should find fewer/none
        assert len(results_sensitive) >= len(results_wrong_case)

    def test_search_empty_value(self, search_store: StorageDataStore) -> None:
        """Return empty list for empty search value."""
        results = search_store.search_values("")
        assert results == []

    def test_search_no_matches(self, search_store: StorageDataStore) -> None:
        """Return empty list when value not found."""
        results = search_store.search_values("xyznonexistent123")
        assert results == []

    def test_search_result_structure(self, search_store: StorageDataStore) -> None:
        """Verify search result structure contains all expected fields."""
        results = search_store.search_values("Widget")
        assert len(results) > 0
        result = results[0]
        assert "index" in result
        assert "type" in result
        assert "origin" in result
        assert "key" in result
        assert "match_locations" in result
        assert isinstance(result["match_locations"], list)

    def test_search_returns_correct_origin(self, search_store: StorageDataStore) -> None:
        """Search results include correct origin."""
        results = search_store.search_values("TRACK-ABC-123")
        assert len(results) == 1
        assert results[0]["origin"] == "https://analytics.example.com"

    def test_search_returns_correct_key(self, search_store: StorageDataStore) -> None:
        """Search results include correct key."""
        results = search_store.search_values("TRACK-ABC-123")
        assert len(results) == 1
        assert results[0]["key"] == "tracking_id"


# --- Feature Detection via Stats Tests ---


class TestFeatureDetection:
    """Tests for feature detection via computed stats."""

    def test_mixed_storage_types(self, basic_store: StorageDataStore) -> None:
        """Store with mixed storage types is correctly categorized."""
        stats = basic_store.stats
        assert stats.cookie_events > 0
        assert stats.local_storage_events > 0
        assert stats.session_storage_events > 0
        assert stats.indexed_db_events > 0

    def test_empty_store_stats(self, storage_events_dir: Path) -> None:
        """Empty store has zero stats."""
        store = StorageDataStore(str(storage_events_dir / "storage_empty.jsonl"))
        stats = store.stats
        assert stats.total_events == 0
        assert stats.cookie_events == 0
        assert stats.local_storage_events == 0
        assert stats.session_storage_events == 0
        assert stats.indexed_db_events == 0
        assert stats.unique_origins == 0
        assert stats.unique_keys == 0
