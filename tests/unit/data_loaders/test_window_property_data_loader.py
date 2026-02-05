"""
tests/unit/data_loaders/test_window_property_data_loader.py

Comprehensive unit tests for WindowPropertyDataLoader and related classes.
"""

from pathlib import Path

import pytest

from bluebox.data_models.cdp import WindowPropertyChangeType
from bluebox.llms.data_loaders.window_property_data_loader import (
    WindowPropertyDataLoader,
    WindowPropertyStats,
)


# --- Fixtures ---


@pytest.fixture(scope="module")
def window_property_events_dir(tests_root: Path) -> Path:
    """Directory containing window property event test JSONL files."""
    return tests_root / "data" / "input" / "window_property_events"


@pytest.fixture
def basic_store(window_property_events_dir: Path) -> WindowPropertyDataLoader:
    """WindowPropertyDataLoader loaded from basic test data."""
    return WindowPropertyDataLoader(str(window_property_events_dir / "window_basic.jsonl"))


@pytest.fixture
def search_store(window_property_events_dir: Path) -> WindowPropertyDataLoader:
    """WindowPropertyDataLoader loaded from search test data."""
    return WindowPropertyDataLoader(str(window_property_events_dir / "window_search.jsonl"))


@pytest.fixture
def paths_store(window_property_events_dir: Path) -> WindowPropertyDataLoader:
    """WindowPropertyDataLoader loaded from paths test data."""
    return WindowPropertyDataLoader(str(window_property_events_dir / "window_paths.jsonl"))


@pytest.fixture
def malformed_store(window_property_events_dir: Path) -> WindowPropertyDataLoader:
    """WindowPropertyDataLoader loaded from malformed test data (should skip bad lines)."""
    return WindowPropertyDataLoader(str(window_property_events_dir / "window_malformed.jsonl"))


# --- WindowPropertyStats Tests ---


class TestWindowPropertyStats:
    """Tests for WindowPropertyStats dataclass."""

    def test_to_summary_basic(self) -> None:
        """Generate summary from basic stats."""
        stats = WindowPropertyStats(
            total_events=10,
            total_changes=25,
            changes_added=15,
            changes_changed=8,
            changes_deleted=2,
            unique_urls=3,
            unique_property_paths=12,
        )
        summary = stats.to_summary()
        assert "Total Events: 10" in summary
        assert "Total Changes: 25" in summary
        assert "Added: 15" in summary
        assert "Changed: 8" in summary
        assert "Deleted: 2" in summary
        assert "Unique URLs: 3" in summary
        assert "Unique Property Paths: 12" in summary

    def test_to_summary_zero_values(self) -> None:
        """Generate summary with zero values."""
        stats = WindowPropertyStats()
        summary = stats.to_summary()
        assert "Total Events: 0" in summary
        assert "Total Changes: 0" in summary
        assert "Added: 0" in summary


# --- WindowPropertyDataLoader Initialization Tests ---


class TestWindowPropertyDataLoaderInit:
    """Tests for WindowPropertyDataLoader initialization."""

    def test_init_basic_file(self, basic_store: WindowPropertyDataLoader) -> None:
        """Initialize from basic JSONL file."""
        # Should have 5 events
        assert len(basic_store.entries) == 5

    def test_init_file_not_found(self, window_property_events_dir: Path) -> None:
        """Raise FileNotFoundError when file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            WindowPropertyDataLoader(str(window_property_events_dir / "nonexistent.jsonl"))

    def test_init_empty_file(self, window_property_events_dir: Path) -> None:
        """Initialize from empty file produces empty store."""
        store = WindowPropertyDataLoader(str(window_property_events_dir / "window_empty.jsonl"))
        assert len(store.entries) == 0

    def test_init_malformed_skips_bad_lines(self, malformed_store: WindowPropertyDataLoader) -> None:
        """Malformed lines are skipped, valid entries are loaded."""
        # Should have 3 valid entries
        assert len(malformed_store.entries) == 3
        # Verify all entries have the expected paths
        all_paths = []
        for entry in malformed_store.entries:
            for change in entry.changes:
                all_paths.append(change.path)
        assert "good_prop_1" in all_paths
        assert "good_prop_2" in all_paths
        assert "good_prop_3" in all_paths


# --- Properties Tests ---


class TestWindowPropertyDataLoaderProperties:
    """Tests for WindowPropertyDataLoader properties."""

    def test_entries_returns_list(self, basic_store: WindowPropertyDataLoader) -> None:
        """entries property returns list of WindowPropertyEvent."""
        entries = basic_store.entries
        assert isinstance(entries, list)
        assert len(entries) > 0

    def test_stats_returns_window_property_stats(self, basic_store: WindowPropertyDataLoader) -> None:
        """stats property returns WindowPropertyStats instance."""
        stats = basic_store.stats
        assert isinstance(stats, WindowPropertyStats)
        assert stats.total_events == len(basic_store.entries)

    def test_stats_counts_total_changes(self, basic_store: WindowPropertyDataLoader) -> None:
        """Stats correctly counts total changes across all events."""
        stats = basic_store.stats
        # Count manually: event1=4, event2=2, event3=2, event4=2, event5=2 = 12
        assert stats.total_changes == 12

    def test_stats_counts_added_changes(self, basic_store: WindowPropertyDataLoader) -> None:
        """Stats correctly counts added changes."""
        stats = basic_store.stats
        # Most changes in basic_store are "added"
        assert stats.changes_added >= 8

    def test_stats_counts_changed_changes(self, basic_store: WindowPropertyDataLoader) -> None:
        """Stats correctly counts changed changes."""
        stats = basic_store.stats
        # scrollY changed, innerWidth changed
        assert stats.changes_changed >= 2

    def test_stats_counts_deleted_changes(self, basic_store: WindowPropertyDataLoader) -> None:
        """Stats correctly counts deleted changes."""
        stats = basic_store.stats
        # __userData was deleted
        assert stats.changes_deleted >= 1

    def test_stats_counts_unique_urls(self, basic_store: WindowPropertyDataLoader) -> None:
        """Stats correctly counts unique URLs."""
        stats = basic_store.stats
        # https://example.com/, https://example.com/page, https://api.example.com/dashboard
        assert stats.unique_urls == 3

    def test_stats_counts_unique_property_paths(self, basic_store: WindowPropertyDataLoader) -> None:
        """Stats correctly counts unique property paths."""
        stats = basic_store.stats
        # name, innerWidth, innerHeight, origin, scrollY, scrollX, __userData, __config, devicePixelRatio
        assert stats.unique_property_paths >= 9


# --- Entry Retrieval Tests ---


class TestEntryRetrieval:
    """Tests for get_entry method."""

    def test_get_entry_found(self, basic_store: WindowPropertyDataLoader) -> None:
        """Get entry by valid index."""
        entry = basic_store.get_entry(0)
        assert entry is not None
        assert entry.url == "https://example.com/"

    def test_get_entry_not_found(self, basic_store: WindowPropertyDataLoader) -> None:
        """Return None for non-existent index."""
        entry = basic_store.get_entry(999)
        assert entry is None

    def test_get_entry_has_changes(self, basic_store: WindowPropertyDataLoader) -> None:
        """Retrieved entry has changes list."""
        entry = basic_store.get_entry(0)
        assert entry is not None
        assert len(entry.changes) > 0
        assert entry.changes[0].path == "name"


# --- Get Changes By Path Tests ---


class TestGetChangesByPath:
    """Tests for get_changes_by_path method."""

    def test_exact_path_match(self, paths_store: WindowPropertyDataLoader) -> None:
        """Get changes by exact path match."""
        results = paths_store.get_changes_by_path("__config.apiKey", exact=True)
        assert len(results) == 3  # added, changed, deleted
        assert all(r["path"] == "__config.apiKey" for r in results)

    def test_exact_path_no_match(self, paths_store: WindowPropertyDataLoader) -> None:
        """Return empty list when exact path not found."""
        results = paths_store.get_changes_by_path("nonexistent.path", exact=True)
        assert results == []

    def test_substring_path_match(self, paths_store: WindowPropertyDataLoader) -> None:
        """Get changes by substring path match."""
        results = paths_store.get_changes_by_path("__config", exact=False)
        # Should match __config.apiKey (3), __config.env (1), __config.debug (1) = 5
        assert len(results) >= 4
        assert all("__config" in r["path"].lower() for r in results)

    def test_substring_path_case_insensitive(self, paths_store: WindowPropertyDataLoader) -> None:
        """Substring path match is case-insensitive."""
        results_lower = paths_store.get_changes_by_path("__config", exact=False)
        results_upper = paths_store.get_changes_by_path("__CONFIG", exact=False)
        assert len(results_lower) == len(results_upper)

    def test_result_structure(self, paths_store: WindowPropertyDataLoader) -> None:
        """Verify result structure contains all expected fields."""
        results = paths_store.get_changes_by_path("__user.name", exact=True)
        assert len(results) > 0
        result = results[0]
        assert "index" in result
        assert "timestamp" in result
        assert "url" in result
        assert "path" in result
        assert "change_type" in result
        assert "value" in result

    def test_tracks_change_types(self, paths_store: WindowPropertyDataLoader) -> None:
        """Results correctly include different change types."""
        results = paths_store.get_changes_by_path("__config.apiKey", exact=True)
        change_types = [r["change_type"] for r in results]
        assert WindowPropertyChangeType.ADDED in change_types
        assert WindowPropertyChangeType.CHANGED in change_types
        assert WindowPropertyChangeType.DELETED in change_types

    def test_includes_url(self, paths_store: WindowPropertyDataLoader) -> None:
        """Results include correct URL."""
        results = paths_store.get_changes_by_path("__user.email", exact=True)
        assert len(results) >= 1
        # __user.email added on /profile
        urls = [r["url"] for r in results]
        assert any("profile" in url for url in urls)


# --- Search Values Tests ---


class TestSearchValues:
    """Tests for search_values method (token tracing)."""

    def test_search_finds_string_value(self, search_store: WindowPropertyDataLoader) -> None:
        """Find changes with matching string value."""
        results = search_store.search_values("Alice Johnson")
        assert len(results) > 0

    def test_search_finds_nested_value(self, search_store: WindowPropertyDataLoader) -> None:
        """Find changes with matching value in nested object."""
        results = search_store.search_values("bearer_token_12345")
        assert len(results) >= 2  # In __userProfile and __paymentInfo

    def test_search_finds_numeric_value(self, search_store: WindowPropertyDataLoader) -> None:
        """Find changes with matching numeric value."""
        results = search_store.search_values("29.99")
        assert len(results) > 0

    def test_search_case_insensitive_default(self, search_store: WindowPropertyDataLoader) -> None:
        """Search is case-insensitive by default."""
        results_lower = search_store.search_values("alice")
        results_upper = search_store.search_values("ALICE")
        assert len(results_lower) == len(results_upper)

    def test_search_case_sensitive(self, search_store: WindowPropertyDataLoader) -> None:
        """Case-sensitive search when specified."""
        results_sensitive = search_store.search_values("Alice", case_sensitive=True)
        results_wrong_case = search_store.search_values("ALICE", case_sensitive=True)
        # "Alice" appears with capital A
        assert len(results_sensitive) >= len(results_wrong_case)

    def test_search_empty_value(self, search_store: WindowPropertyDataLoader) -> None:
        """Return empty list for empty search value."""
        results = search_store.search_values("")
        assert results == []

    def test_search_no_matches(self, search_store: WindowPropertyDataLoader) -> None:
        """Return empty list when value not found."""
        results = search_store.search_values("xyznonexistent123")
        assert results == []

    def test_search_skips_null_values(self, search_store: WindowPropertyDataLoader) -> None:
        """Search skips changes with null values (deletions)."""
        # The __cart deletion has value=null, shouldn't match anything
        results = search_store.search_values("null")
        # Should not find the actual null value, only string "null" if present
        for r in results:
            assert r["value"] is not None

    def test_search_result_structure(self, search_store: WindowPropertyDataLoader) -> None:
        """Verify search result structure contains all expected fields."""
        results = search_store.search_values("Widget")
        assert len(results) > 0
        result = results[0]
        assert "index" in result
        assert "timestamp" in result
        assert "url" in result
        assert "path" in result
        assert "change_type" in result
        assert "value" in result

    def test_search_returns_correct_url(self, search_store: WindowPropertyDataLoader) -> None:
        """Search results include correct URL."""
        results = search_store.search_values("TRACK-ABC-123")
        assert len(results) == 1
        assert results[0]["url"] == "https://analytics.example.com/"

    def test_search_returns_correct_path(self, search_store: WindowPropertyDataLoader) -> None:
        """Search results include correct path."""
        results = search_store.search_values("TRACK-ABC-123")
        assert len(results) == 1
        assert results[0]["path"] == "__trackingId"

    def test_search_returns_change_type(self, search_store: WindowPropertyDataLoader) -> None:
        """Search results include change type."""
        results = search_store.search_values("TRACK-ABC-123")
        assert len(results) == 1
        assert results[0]["change_type"] == WindowPropertyChangeType.ADDED


# --- Feature Detection via Stats Tests ---


class TestFeatureDetection:
    """Tests for feature detection via computed stats."""

    def test_mixed_change_types(self, basic_store: WindowPropertyDataLoader) -> None:
        """Store with mixed change types is correctly categorized."""
        stats = basic_store.stats
        assert stats.changes_added > 0
        assert stats.changes_changed > 0
        assert stats.changes_deleted > 0

    def test_empty_store_stats(self, window_property_events_dir: Path) -> None:
        """Empty store has zero stats."""
        store = WindowPropertyDataLoader(str(window_property_events_dir / "window_empty.jsonl"))
        stats = store.stats
        assert stats.total_events == 0
        assert stats.total_changes == 0
        assert stats.changes_added == 0
        assert stats.changes_changed == 0
        assert stats.changes_deleted == 0
        assert stats.unique_urls == 0
        assert stats.unique_property_paths == 0

    def test_changes_sum_equals_total(self, basic_store: WindowPropertyDataLoader) -> None:
        """Sum of change types equals total changes."""
        stats = basic_store.stats
        assert stats.changes_added + stats.changes_changed + stats.changes_deleted == stats.total_changes
