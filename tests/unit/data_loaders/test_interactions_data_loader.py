"""
tests/unit/data_loaders/test_interactions_data_loader.py

Comprehensive unit tests for InteractionsDataLoader and related classes.
"""

from pathlib import Path

import pytest

from bluebox.data_models.cdp import UIInteractionEvent, InteractionType
from bluebox.llms.data_loaders.interactions_data_loader import (
    InteractionsDataLoader,
    InteractionStats,
)


# --- Fixtures ---


@pytest.fixture(scope="module")
def interaction_events_dir(tests_root: Path) -> Path:
    """Directory containing interaction event test JSONL files."""
    return tests_root / "data" / "input" / "interaction_events"


@pytest.fixture
def basic_loader(interaction_events_dir: Path) -> InteractionsDataLoader:
    """InteractionsDataLoader loaded from basic test data."""
    return InteractionsDataLoader.from_jsonl(str(interaction_events_dir / "interactions_basic.jsonl"))


@pytest.fixture
def search_loader(interaction_events_dir: Path) -> InteractionsDataLoader:
    """InteractionsDataLoader loaded from search test data."""
    return InteractionsDataLoader.from_jsonl(str(interaction_events_dir / "interactions_search.jsonl"))


@pytest.fixture
def filter_loader(interaction_events_dir: Path) -> InteractionsDataLoader:
    """InteractionsDataLoader loaded from filter test data."""
    return InteractionsDataLoader.from_jsonl(str(interaction_events_dir / "interactions_filter.jsonl"))


@pytest.fixture
def malformed_loader(interaction_events_dir: Path) -> InteractionsDataLoader:
    """InteractionsDataLoader loaded from malformed test data (should skip bad lines)."""
    return InteractionsDataLoader.from_jsonl(str(interaction_events_dir / "interactions_malformed.jsonl"))


# --- InteractionStats Tests ---


class TestInteractionStats:
    """Tests for InteractionStats dataclass."""

    def test_to_summary_basic(self) -> None:
        """Generate summary from basic stats."""
        stats = InteractionStats(
            total_events=10,
            unique_urls=3,
            events_by_type={"click": 5, "input": 3, "change": 2},
            unique_elements=8,
        )
        summary = stats.to_summary()
        assert "Total Events: 10" in summary
        assert "Unique URLs: 3" in summary
        assert "Unique Elements: 8" in summary
        assert "Events by Type:" in summary
        assert "click: 5" in summary

    def test_to_summary_zero_values(self) -> None:
        """Generate summary with zero values."""
        stats = InteractionStats()
        summary = stats.to_summary()
        assert "Total Events: 0" in summary
        assert "Unique URLs: 0" in summary
        assert "Unique Elements: 0" in summary


# --- InteractionsDataLoader Initialization Tests ---


class TestInteractionsDataLoaderInit:
    """Tests for InteractionsDataLoader initialization."""

    def test_init_from_list(self) -> None:
        """Initialize from list of events."""
        events = [
            UIInteractionEvent(
                type=InteractionType.CLICK,
                url="https://example.com/",
                element={"tag_name": "button", "id": "btn-1"},
            )
        ]
        loader = InteractionsDataLoader(events)
        assert len(loader.entries) == 1

    def test_from_jsonl_basic_file(self, basic_loader: InteractionsDataLoader) -> None:
        """Initialize from basic JSONL file."""
        assert len(basic_loader.entries) == 8

    def test_from_jsonl_file_not_found(self, interaction_events_dir: Path) -> None:
        """Raise ValueError when file doesn't exist."""
        with pytest.raises(ValueError, match="does not exist"):
            InteractionsDataLoader.from_jsonl(str(interaction_events_dir / "nonexistent.jsonl"))

    def test_from_jsonl_empty_file(self, interaction_events_dir: Path) -> None:
        """Initialize from empty file produces empty loader."""
        loader = InteractionsDataLoader.from_jsonl(str(interaction_events_dir / "interactions_empty.jsonl"))
        assert len(loader.entries) == 0

    def test_from_jsonl_malformed_skips_bad_lines(self, malformed_loader: InteractionsDataLoader) -> None:
        """Malformed lines are skipped, valid entries are loaded."""
        # Should have 3 valid entries
        assert len(malformed_loader.entries) == 3
        element_ids = [e.element.id for e in malformed_loader.entries]
        assert "good-btn-1" in element_ids
        assert "good-input-2" in element_ids
        assert "good-btn-3" in element_ids


# --- Properties Tests ---


class TestInteractionsDataLoaderProperties:
    """Tests for InteractionsDataLoader properties."""

    def test_entries_returns_list(self, basic_loader: InteractionsDataLoader) -> None:
        """entries property returns list of UIInteractionEvent."""
        entries = basic_loader.entries
        assert isinstance(entries, list)
        assert len(entries) > 0
        assert all(isinstance(e, UIInteractionEvent) for e in entries)

    def test_stats_returns_interaction_stats(self, basic_loader: InteractionsDataLoader) -> None:
        """stats property returns InteractionStats instance."""
        stats = basic_loader.stats
        assert isinstance(stats, InteractionStats)
        assert stats.total_events == len(basic_loader.entries)

    def test_stats_counts_events_by_type(self, basic_loader: InteractionsDataLoader) -> None:
        """Stats correctly counts events by type."""
        stats = basic_loader.stats
        # We have click, input, change, focus, blur, keydown in basic data
        assert "click" in stats.events_by_type
        assert "input" in stats.events_by_type
        assert stats.events_by_type["click"] == 2
        assert stats.events_by_type["input"] == 2

    def test_stats_counts_unique_urls(self, basic_loader: InteractionsDataLoader) -> None:
        """Stats correctly counts unique URLs."""
        stats = basic_loader.stats
        # https://example.com/, https://example.com/dashboard, https://example.com/settings
        assert stats.unique_urls == 3

    def test_stats_counts_unique_elements(self, basic_loader: InteractionsDataLoader) -> None:
        """Stats correctly counts unique elements."""
        stats = basic_loader.stats
        # 6 unique elements (search-box appears 3x with focus/blur/keydown but same element)
        assert stats.unique_elements == 6


# --- Abstract Method Implementations ---


class TestAbstractMethodImplementations:
    """Tests for AbstractDataLoader method implementations."""

    def test_get_entry_id(self, basic_loader: InteractionsDataLoader) -> None:
        """get_entry_id returns index as string."""
        entry = basic_loader.entries[0]
        assert basic_loader.get_entry_id(entry) == "0"

    def test_get_searchable_content(self, basic_loader: InteractionsDataLoader) -> None:
        """get_searchable_content returns element content."""
        entry = basic_loader.entries[1]  # input with email
        content = basic_loader.get_searchable_content(entry)
        assert content is not None
        assert "alice@example.com" in content

    def test_get_searchable_content_includes_text(self, basic_loader: InteractionsDataLoader) -> None:
        """get_searchable_content includes text."""
        entry = basic_loader.entries[0]  # button with "Submit" text
        content = basic_loader.get_searchable_content(entry)
        assert content is not None
        assert "Submit" in content

    def test_get_searchable_content_includes_placeholder(self, basic_loader: InteractionsDataLoader) -> None:
        """get_searchable_content includes placeholder."""
        entry = basic_loader.entries[1]  # input with placeholder
        content = basic_loader.get_searchable_content(entry)
        assert content is not None
        assert "Enter email" in content

    def test_get_entry_url(self, basic_loader: InteractionsDataLoader) -> None:
        """get_entry_url returns URL."""
        entry = basic_loader.entries[0]
        assert basic_loader.get_entry_url(entry) == "https://example.com/"


# --- Filter by Type Tests ---


class TestFilterByType:
    """Tests for filter_by_type method."""

    def test_filter_single_type(self, filter_loader: InteractionsDataLoader) -> None:
        """Filter by single interaction type."""
        results = filter_loader.filter_by_type(["click"])
        assert len(results) == 2
        assert all(e.type == InteractionType.CLICK for e in results)

    def test_filter_multiple_types(self, filter_loader: InteractionsDataLoader) -> None:
        """Filter by multiple interaction types."""
        results = filter_loader.filter_by_type(["click", "input"])
        assert len(results) == 4
        assert all(e.type in (InteractionType.CLICK, InteractionType.INPUT) for e in results)

    def test_filter_case_insensitive(self, filter_loader: InteractionsDataLoader) -> None:
        """Type filter is case-insensitive."""
        results_lower = filter_loader.filter_by_type(["click"])
        results_upper = filter_loader.filter_by_type(["CLICK"])
        assert len(results_lower) == len(results_upper)

    def test_filter_no_matches(self, filter_loader: InteractionsDataLoader) -> None:
        """Return empty list when no matches."""
        results = filter_loader.filter_by_type(["dblclick"])
        assert results == []


# --- Filter by Element Tests ---


class TestFilterByElement:
    """Tests for filter_by_element method."""

    def test_filter_by_tag_name(self, filter_loader: InteractionsDataLoader) -> None:
        """Filter by HTML tag name."""
        results = filter_loader.filter_by_element(tag_name="input")
        assert len(results) == 4  # input-text (3 events) + input-email (1 event)
        assert all(e.element.tag_name.lower() == "input" for e in results)

    def test_filter_by_element_id(self, filter_loader: InteractionsDataLoader) -> None:
        """Filter by element ID."""
        results = filter_loader.filter_by_element(element_id="btn-1")
        assert len(results) == 1
        assert results[0].element.id == "btn-1"

    def test_filter_by_class_name(self, filter_loader: InteractionsDataLoader) -> None:
        """Filter by CSS class name."""
        results = filter_loader.filter_by_element(class_name="form-control")
        # Only input events have class_names in test data (not keydown/blur)
        assert len(results) == 2

    def test_filter_by_type_attr(self, filter_loader: InteractionsDataLoader) -> None:
        """Filter by input type attribute."""
        results = filter_loader.filter_by_element(type_attr="email")
        assert len(results) == 1
        assert results[0].element.type_attr == "email"

    def test_filter_by_multiple_criteria(self, filter_loader: InteractionsDataLoader) -> None:
        """Filter by multiple element attributes."""
        results = filter_loader.filter_by_element(tag_name="input", type_attr="text")
        assert len(results) == 3  # input-text has 3 events

    def test_filter_no_matches(self, filter_loader: InteractionsDataLoader) -> None:
        """Return empty list when no matches."""
        results = filter_loader.filter_by_element(element_id="nonexistent")
        assert results == []


# --- Get Form Inputs Tests ---


class TestGetFormInputs:
    """Tests for get_form_inputs method."""

    def test_get_form_inputs_returns_input_events(self, filter_loader: InteractionsDataLoader) -> None:
        """get_form_inputs returns input and change events."""
        results = filter_loader.get_form_inputs()
        # 2 input events + 1 change event
        assert len(results) == 3

    def test_get_form_inputs_structure(self, filter_loader: InteractionsDataLoader) -> None:
        """get_form_inputs returns correct structure."""
        results = filter_loader.get_form_inputs()
        result = results[0]
        assert "type" in result
        assert "value" in result
        assert "tag_name" in result
        assert "element_id" in result
        assert "element_name" in result
        assert "type_attr" in result
        assert "placeholder" in result
        assert "css_path" in result
        assert "url" in result

    def test_get_form_inputs_excludes_other_types(self, filter_loader: InteractionsDataLoader) -> None:
        """get_form_inputs excludes click, focus, blur, keydown events."""
        results = filter_loader.get_form_inputs()
        types = [r["type"] for r in results]
        assert all(t in ("input", "change") for t in types)


# --- Get Unique Elements Tests ---


class TestGetUniqueElements:
    """Tests for get_unique_elements method."""

    def test_get_unique_elements_deduplicates(self, filter_loader: InteractionsDataLoader) -> None:
        """get_unique_elements returns deduplicated elements."""
        results = filter_loader.get_unique_elements()
        # Should have fewer unique elements than total events
        assert len(results) < len(filter_loader.entries)

    def test_get_unique_elements_structure(self, filter_loader: InteractionsDataLoader) -> None:
        """get_unique_elements returns correct structure."""
        results = filter_loader.get_unique_elements()
        result = results[0]
        assert "tag_name" in result
        assert "element_id" in result
        assert "element_name" in result
        assert "type_attr" in result
        assert "css_path" in result
        assert "placeholder" in result
        assert "interaction_count" in result
        assert "interaction_types" in result

    def test_get_unique_elements_counts_interactions(self, filter_loader: InteractionsDataLoader) -> None:
        """get_unique_elements correctly counts interactions per element."""
        results = filter_loader.get_unique_elements()
        # input-text has 3 interactions (input, keydown, blur)
        input_text = next((r for r in results if r["element_id"] == "input-text"), None)
        assert input_text is not None
        assert input_text["interaction_count"] == 3

    def test_get_unique_elements_tracks_interaction_types(self, filter_loader: InteractionsDataLoader) -> None:
        """get_unique_elements tracks all interaction types for element."""
        results = filter_loader.get_unique_elements()
        input_text = next((r for r in results if r["element_id"] == "input-text"), None)
        assert input_text is not None
        assert "input" in input_text["interaction_types"]
        assert "keydown" in input_text["interaction_types"]
        assert "blur" in input_text["interaction_types"]

    def test_get_unique_elements_sorted_by_count(self, filter_loader: InteractionsDataLoader) -> None:
        """get_unique_elements returns results sorted by interaction count."""
        results = filter_loader.get_unique_elements()
        for i in range(len(results) - 1):
            assert results[i]["interaction_count"] >= results[i + 1]["interaction_count"]


# --- Get Event Detail Tests ---


class TestGetEventDetail:
    """Tests for get_event_detail method."""

    def test_get_event_detail_found(self, basic_loader: InteractionsDataLoader) -> None:
        """Get event detail by valid index."""
        result = basic_loader.get_event_detail(0)
        assert result is not None
        assert "type" in result
        assert "element" in result
        assert "url" in result

    def test_get_event_detail_not_found(self, basic_loader: InteractionsDataLoader) -> None:
        """Return None for non-existent index."""
        result = basic_loader.get_event_detail(999)
        assert result is None

    def test_get_event_detail_negative_index(self, basic_loader: InteractionsDataLoader) -> None:
        """Return None for negative index."""
        result = basic_loader.get_event_detail(-1)
        assert result is None


# --- Inherited Search Methods Tests ---


class TestInheritedSearchMethods:
    """Tests for inherited search methods from AbstractDataLoader."""

    def test_search_by_terms(self, search_loader: InteractionsDataLoader) -> None:
        """Search by terms finds matching entries."""
        results = search_loader.search_by_terms(["Alice"])
        assert len(results) > 0

    def test_search_by_terms_multiple(self, search_loader: InteractionsDataLoader) -> None:
        """Search by multiple terms."""
        results = search_loader.search_by_terms(["Widget", "Cart"])
        assert len(results) > 0

    def test_search_content(self, search_loader: InteractionsDataLoader) -> None:
        """Search content finds matching entries."""
        results = search_loader.search_content("alice@example.com")
        assert len(results) > 0
        assert results[0]["count"] >= 1

    def test_search_content_case_insensitive(self, search_loader: InteractionsDataLoader) -> None:
        """Search content is case-insensitive by default."""
        results_lower = search_loader.search_content("alice")
        results_upper = search_loader.search_content("ALICE")
        assert len(results_lower) == len(results_upper)

    def test_search_by_regex(self, search_loader: InteractionsDataLoader) -> None:
        """Search by regex finds matching entries."""
        result = search_loader.search_by_regex(r"TRACK-\w+-\d+")
        assert result["error"] is None
        assert len(result["matches"]) > 0


# --- Feature Detection via Stats Tests ---


class TestFeatureDetection:
    """Tests for feature detection via computed stats."""

    def test_mixed_event_types(self, basic_loader: InteractionsDataLoader) -> None:
        """Loader with mixed event types is correctly categorized."""
        stats = basic_loader.stats
        assert len(stats.events_by_type) >= 4  # click, input, change, focus, etc.

    def test_empty_loader_stats(self, interaction_events_dir: Path) -> None:
        """Empty loader has zero stats."""
        loader = InteractionsDataLoader.from_jsonl(str(interaction_events_dir / "interactions_empty.jsonl"))
        stats = loader.stats
        assert stats.total_events == 0
        assert stats.unique_urls == 0
        assert stats.unique_elements == 0
        assert len(stats.events_by_type) == 0

    def test_event_type_sum_equals_total(self, basic_loader: InteractionsDataLoader) -> None:
        """Sum of event types equals total events."""
        stats = basic_loader.stats
        total_from_types = sum(stats.events_by_type.values())
        assert total_from_types == stats.total_events
