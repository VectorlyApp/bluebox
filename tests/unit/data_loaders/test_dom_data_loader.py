"""
tests/unit/data_loaders/test_dom_data_loader.py

Unit tests for DOMDataLoader and related classes.

Test data (dom_basic.jsonl) contains 4 snapshots:
  [0] https://www.example.com/         — home page (form with email input, submit, links)
  [1] https://www.example.com/search   — search page (form with text input, select, button)
  [2] https://www.example.com/search?q=laptops — results page (table, link, button)
  [3] https://auth.example.com/login   — login page (form with username + password inputs, button, link)
"""

from pathlib import Path

import pytest

from bluebox.llms.data_loaders.dom_data_loader import (
    DOMDataLoader,
    DOMStats,
)


# --- Fixtures ---


@pytest.fixture(scope="module")
def dom_events_dir(tests_root: Path) -> Path:
    """Directory containing DOM event test JSONL files."""
    return tests_root / "data" / "input" / "dom_events"


@pytest.fixture
def basic_loader(dom_events_dir: Path) -> DOMDataLoader:
    """DOMDataLoader loaded from basic test data (4 snapshots)."""
    return DOMDataLoader(str(dom_events_dir / "dom_basic.jsonl"))


@pytest.fixture
def malformed_loader(dom_events_dir: Path) -> DOMDataLoader:
    """DOMDataLoader loaded from malformed test data (should skip bad lines)."""
    return DOMDataLoader(str(dom_events_dir / "dom_malformed.jsonl"))


# --- DOMStats Tests ---


class TestDOMStats:
    """Tests for DOMStats dataclass."""

    def test_to_summary_basic(self) -> None:
        """Generate summary from basic stats."""
        stats = DOMStats(
            total_snapshots=3,
            unique_urls=2,
            unique_titles=2,
            urls=["https://example.com/", "https://example.com/search"],
            titles=["Home", "Search"],
            avg_string_count=50.0,
            avg_document_count=1.5,
            total_strings=150,
            hosts={"example.com": 3},
        )
        summary = stats.to_summary()
        assert "Total Snapshots: 3" in summary
        assert "Unique URLs: 2" in summary
        assert "Total Strings Across All Snapshots: 150" in summary
        assert "example.com" in summary

    def test_to_summary_zero_values(self) -> None:
        """Generate summary with zero values."""
        stats = DOMStats()
        summary = stats.to_summary()
        assert "Total Snapshots: 0" in summary
        assert "Unique URLs: 0" in summary


# --- Initialization Tests ---


class TestDOMDataLoaderInit:
    """Tests for DOMDataLoader initialization."""

    def test_init_basic_file(self, basic_loader: DOMDataLoader) -> None:
        """Initialize from basic JSONL file with 4 snapshots."""
        assert len(basic_loader.entries) == 4

    def test_init_file_not_found(self, dom_events_dir: Path) -> None:
        """Raise FileNotFoundError when file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            DOMDataLoader(str(dom_events_dir / "nonexistent.jsonl"))

    def test_init_empty_file(self, dom_events_dir: Path) -> None:
        """Initialize from empty file produces empty loader."""
        loader = DOMDataLoader(str(dom_events_dir / "dom_empty.jsonl"))
        assert len(loader.entries) == 0

    def test_init_malformed_skips_bad_lines(self, malformed_loader: DOMDataLoader) -> None:
        """Malformed lines are skipped, valid entries are loaded."""
        assert len(malformed_loader.entries) == 2
        urls = [e.url for e in malformed_loader.entries]
        assert "https://good.example.com/" in urls
        assert "https://good.example.com/page2" in urls


# --- Stats Tests ---


class TestDOMDataLoaderStats:
    """Tests for DOMDataLoader computed stats."""

    def test_stats_type(self, basic_loader: DOMDataLoader) -> None:
        """stats property returns DOMStats instance."""
        assert isinstance(basic_loader.stats, DOMStats)

    def test_stats_total_snapshots(self, basic_loader: DOMDataLoader) -> None:
        """Stats correctly counts total snapshots."""
        assert basic_loader.stats.total_snapshots == 4

    def test_stats_unique_urls(self, basic_loader: DOMDataLoader) -> None:
        """Stats correctly counts unique URLs."""
        assert basic_loader.stats.unique_urls == 4

    def test_stats_hosts(self, basic_loader: DOMDataLoader) -> None:
        """Stats correctly counts per-host snapshot counts."""
        hosts = basic_loader.stats.hosts
        assert hosts["www.example.com"] == 3
        assert hosts["auth.example.com"] == 1

    def test_stats_total_strings(self, basic_loader: DOMDataLoader) -> None:
        """Stats computes total strings across all snapshots."""
        total = sum(len(e.strings) for e in basic_loader.entries)
        assert basic_loader.stats.total_strings == total

    def test_empty_stats(self, dom_events_dir: Path) -> None:
        """Empty loader has zero stats."""
        loader = DOMDataLoader(str(dom_events_dir / "dom_empty.jsonl"))
        assert loader.stats.total_snapshots == 0


# --- Snapshot Retrieval Tests ---


class TestSnapshotRetrieval:
    """Tests for snapshot retrieval methods."""

    def test_get_snapshot_valid(self, basic_loader: DOMDataLoader) -> None:
        """Get snapshot by valid index."""
        snapshot = basic_loader.get_snapshot(0)
        assert snapshot is not None
        assert snapshot.title == "Example Home"

    def test_get_snapshot_out_of_range(self, basic_loader: DOMDataLoader) -> None:
        """Return None for out-of-range index."""
        assert basic_loader.get_snapshot(99) is None
        assert basic_loader.get_snapshot(-1) is None

    def test_get_snapshots_by_url(self, basic_loader: DOMDataLoader) -> None:
        """Get snapshots by URL substring."""
        results = basic_loader.get_snapshots_by_url("search")
        assert len(results) == 2

    def test_get_snapshots_by_url_pattern(self, basic_loader: DOMDataLoader) -> None:
        """Get snapshots by URL glob pattern."""
        results = basic_loader.get_snapshots_by_url_pattern("*example.com/search*")
        assert len(results) == 2

    def test_get_snapshots_by_host(self, basic_loader: DOMDataLoader) -> None:
        """Get snapshots by host."""
        results = basic_loader.get_snapshots_by_host("auth.example.com")
        assert len(results) == 1
        assert results[0].title == "Login — Example"


# --- Page Summary Tests ---


class TestPageSummaries:
    """Tests for page-level summary methods."""

    def test_list_pages(self, basic_loader: DOMDataLoader) -> None:
        """List all pages with summary info."""
        pages = basic_loader.list_pages()
        assert len(pages) == 4
        assert all(k in pages[0] for k in ["index", "url", "title", "string_count", "document_count", "timestamp"])

    def test_get_page_titles(self, basic_loader: DOMDataLoader) -> None:
        """Get all page titles."""
        titles = basic_loader.get_page_titles()
        assert len(titles) == 4
        assert titles[0]["title"] == "Example Home"


# --- String Table Analysis Tests ---


class TestStringTableAnalysis:
    """Tests for string table analysis methods."""

    def test_get_text_content_valid(self, basic_loader: DOMDataLoader) -> None:
        """Get text content from a valid snapshot."""
        content = basic_loader.get_text_content(0)
        assert content is not None
        assert "Welcome to Example" in content

    def test_get_text_content_invalid_index(self, basic_loader: DOMDataLoader) -> None:
        """Return None for invalid index."""
        assert basic_loader.get_text_content(99) is None

    def test_get_text_content_truncation(self, basic_loader: DOMDataLoader) -> None:
        """Content is truncated at max_chars."""
        content = basic_loader.get_text_content(0, max_chars=20)
        assert content is not None
        assert content.endswith("...")

    def test_search_strings_found(self, basic_loader: DOMDataLoader) -> None:
        """Search strings finds matching content."""
        results = basic_loader.search_strings("example")
        assert len(results) > 0

    def test_search_strings_case_insensitive(self, basic_loader: DOMDataLoader) -> None:
        """String search is case-insensitive by default."""
        results_lower = basic_loader.search_strings("example")
        results_upper = basic_loader.search_strings("EXAMPLE")
        assert len(results_lower) == len(results_upper)

    def test_search_strings_specific_snapshot(self, basic_loader: DOMDataLoader) -> None:
        """Search strings in a specific snapshot only."""
        results = basic_loader.search_strings("laptops", snapshot_index=2)
        assert len(results) == 1
        assert results[0]["snapshot_index"] == 2

    def test_search_strings_no_match(self, basic_loader: DOMDataLoader) -> None:
        """Return empty list when string not found."""
        assert basic_loader.search_strings("xyznonexistent") == []

    def test_search_strings_empty_value(self, basic_loader: DOMDataLoader) -> None:
        """Return empty list for empty search value."""
        assert basic_loader.search_strings("") == []


# --- Input Elements Tests ---


class TestGetInputs:
    """Tests for get_inputs method (INPUT, SELECT, TEXTAREA)."""

    def test_home_page_inputs(self, basic_loader: DOMDataLoader) -> None:
        """Home page has email input and hidden + submit inputs."""
        results = basic_loader.get_inputs(snapshot_index=0)
        assert len(results) == 1
        elements = results[0]["elements"]
        tags = [e["tag"] for e in elements]
        assert "INPUT" in tags

    def test_home_page_email_input_attrs(self, basic_loader: DOMDataLoader) -> None:
        """Home page email input has correct attributes."""
        results = basic_loader.get_inputs(snapshot_index=0)
        elements = results[0]["elements"]
        email_inputs = [e for e in elements if e["attrs"].get("type") == "email"]
        assert len(email_inputs) == 1
        assert email_inputs[0]["attrs"]["name"] == "email_field"
        assert email_inputs[0]["attrs"]["placeholder"] == "Enter your email"

    def test_search_page_has_text_input_and_select(self, basic_loader: DOMDataLoader) -> None:
        """Search page has a text input and a select dropdown."""
        results = basic_loader.get_inputs(snapshot_index=1)
        elements = results[0]["elements"]
        tags = [e["tag"] for e in elements]
        assert "INPUT" in tags
        assert "SELECT" in tags

    def test_login_page_has_username_and_password(self, basic_loader: DOMDataLoader) -> None:
        """Login page has username and password inputs."""
        results = basic_loader.get_inputs(snapshot_index=3)
        elements = results[0]["elements"]
        names = [e["attrs"].get("name") for e in elements]
        assert "username" in names
        assert "password" in names

    def test_input_value_captured(self, basic_loader: DOMDataLoader) -> None:
        """Input elements have inputValue when present."""
        results = basic_loader.get_inputs(snapshot_index=0)
        elements = results[0]["elements"]
        # The home page email input has an inputValue
        inputs_with_value = [e for e in elements if "input_value" in e]
        assert len(inputs_with_value) >= 1

    def test_results_page_no_inputs(self, basic_loader: DOMDataLoader) -> None:
        """Results page has no input elements."""
        results = basic_loader.get_inputs(snapshot_index=2)
        assert results == []

    def test_all_snapshots(self, basic_loader: DOMDataLoader) -> None:
        """Get inputs across all snapshots."""
        results = basic_loader.get_inputs()
        # Home, search, and login pages have inputs; results page does not
        assert len(results) == 3


# --- Button Tests ---


class TestGetButtons:
    """Tests for get_buttons method."""

    def test_search_page_has_button(self, basic_loader: DOMDataLoader) -> None:
        """Search page has a search button."""
        results = basic_loader.get_buttons(snapshot_index=1)
        assert len(results) == 1
        elements = results[0]["elements"]
        assert any(e["tag"] == "BUTTON" for e in elements)

    def test_home_page_submit_input_counted_as_button(self, basic_loader: DOMDataLoader) -> None:
        """Home page has <input type='submit'> which counts as a button."""
        results = basic_loader.get_buttons(snapshot_index=0)
        assert len(results) == 1
        elements = results[0]["elements"]
        submit_inputs = [e for e in elements if e["attrs"].get("type") == "submit"]
        assert len(submit_inputs) == 1

    def test_login_page_has_button(self, basic_loader: DOMDataLoader) -> None:
        """Login page has a Sign In button."""
        results = basic_loader.get_buttons(snapshot_index=3)
        assert len(results) == 1

    def test_results_page_has_next_button(self, basic_loader: DOMDataLoader) -> None:
        """Results page has a Next Page button."""
        results = basic_loader.get_buttons(snapshot_index=2)
        assert len(results) == 1


# --- Link Tests ---


class TestGetLinks:
    """Tests for get_links method."""

    def test_home_page_has_links(self, basic_loader: DOMDataLoader) -> None:
        """Home page has navigation links."""
        results = basic_loader.get_links(snapshot_index=0)
        assert len(results) == 1
        elements = results[0]["elements"]
        hrefs = [e["attrs"].get("href") for e in elements]
        assert "https://www.example.com/about" in hrefs
        assert "/login" in hrefs

    def test_results_page_has_product_link(self, basic_loader: DOMDataLoader) -> None:
        """Results page has product links."""
        results = basic_loader.get_links(snapshot_index=2)
        assert len(results) == 1
        elements = results[0]["elements"]
        hrefs = [e["attrs"].get("href") for e in elements]
        assert any("/product/" in h for h in hrefs if h)

    def test_login_page_has_link(self, basic_loader: DOMDataLoader) -> None:
        """Login page has a link (forgot password)."""
        results = basic_loader.get_links(snapshot_index=3)
        assert len(results) == 1
        elements = results[0]["elements"]
        assert len(elements) >= 1
        assert elements[0]["tag"] == "A"

    def test_all_snapshots(self, basic_loader: DOMDataLoader) -> None:
        """Get links across all snapshots."""
        results = basic_loader.get_links()
        assert len(results) >= 3  # home, results, login all have links


# --- Form Tests ---


class TestGetForms:
    """Tests for get_forms method (forms with child inputs)."""

    def test_home_page_form(self, basic_loader: DOMDataLoader) -> None:
        """Home page has a signup form with email input."""
        results = basic_loader.get_forms(snapshot_index=0)
        assert len(results) == 1
        forms = results[0]["forms"]
        assert len(forms) == 1
        form = forms[0]
        assert form["attrs"].get("id") == "signup-form"

    def test_home_page_form_has_child_inputs(self, basic_loader: DOMDataLoader) -> None:
        """Home page form contains its child input elements."""
        results = basic_loader.get_forms(snapshot_index=0)
        form = results[0]["forms"][0]
        assert len(form["inputs"]) >= 1
        tags = [inp["tag"] for inp in form["inputs"]]
        assert "INPUT" in tags

    def test_search_page_form(self, basic_loader: DOMDataLoader) -> None:
        """Search page has a form with action and method."""
        results = basic_loader.get_forms(snapshot_index=1)
        assert len(results) == 1
        form = results[0]["forms"][0]
        assert form["attrs"].get("action") == "/api/search"
        assert form["attrs"].get("method") == "GET"

    def test_search_page_form_has_input_and_select(self, basic_loader: DOMDataLoader) -> None:
        """Search form contains text input, select, and button."""
        results = basic_loader.get_forms(snapshot_index=1)
        form = results[0]["forms"][0]
        tags = [inp["tag"] for inp in form["inputs"]]
        assert "INPUT" in tags
        assert "SELECT" in tags
        assert "BUTTON" in tags

    def test_login_page_form(self, basic_loader: DOMDataLoader) -> None:
        """Login page has a POST form to /api/auth/login."""
        results = basic_loader.get_forms(snapshot_index=3)
        assert len(results) == 1
        form = results[0]["forms"][0]
        assert form["attrs"]["action"] == "/api/auth/login"
        assert form["attrs"]["method"] == "POST"

    def test_login_form_has_username_and_password(self, basic_loader: DOMDataLoader) -> None:
        """Login form has username and password fields."""
        results = basic_loader.get_forms(snapshot_index=3)
        form = results[0]["forms"][0]
        input_names = [inp["attrs"].get("name") for inp in form["inputs"]]
        assert "username" in input_names
        assert "password" in input_names

    def test_results_page_no_forms(self, basic_loader: DOMDataLoader) -> None:
        """Results page has no forms."""
        results = basic_loader.get_forms(snapshot_index=2)
        assert results == []


# --- Table Tests ---


class TestGetTables:
    """Tests for get_tables method."""

    def test_results_page_has_table(self, basic_loader: DOMDataLoader) -> None:
        """Results page has a results table."""
        results = basic_loader.get_tables(snapshot_index=2)
        assert len(results) == 1
        tables = results[0]["tables"]
        assert len(tables) == 1

    def test_results_table_attrs(self, basic_loader: DOMDataLoader) -> None:
        """Results table has correct attributes."""
        results = basic_loader.get_tables(snapshot_index=2)
        table = results[0]["tables"][0]
        assert table["attrs"].get("class") == "results-table"

    def test_home_page_no_tables(self, basic_loader: DOMDataLoader) -> None:
        """Home page has no tables."""
        results = basic_loader.get_tables(snapshot_index=0)
        assert results == []

    def test_all_snapshots(self, basic_loader: DOMDataLoader) -> None:
        """Only results page has tables."""
        results = basic_loader.get_tables()
        assert len(results) == 1
        assert results[0]["snapshot_index"] == 2


# --- Heading Tests ---


class TestGetHeadings:
    """Tests for get_headings method."""

    def test_home_page_heading(self, basic_loader: DOMDataLoader) -> None:
        """Home page has H1 heading."""
        results = basic_loader.get_headings(snapshot_index=0)
        assert len(results) == 1
        headings = results[0]["headings"]
        assert any(h["tag"] == "H1" for h in headings)

    def test_heading_text_content(self, basic_loader: DOMDataLoader) -> None:
        """Heading has text content extracted from child text nodes."""
        results = basic_loader.get_headings(snapshot_index=0)
        headings = results[0]["headings"]
        h1 = [h for h in headings if h["tag"] == "H1"][0]
        assert h1["text"] is not None
        assert "Welcome to Example" in h1["text"]

    def test_all_pages_have_headings(self, basic_loader: DOMDataLoader) -> None:
        """All 4 pages have at least one heading."""
        results = basic_loader.get_headings()
        assert len(results) == 4


# --- Clickable Elements Tests ---


class TestGetClickableElements:
    """Tests for get_clickable_elements method."""

    def test_home_page_clickable(self, basic_loader: DOMDataLoader) -> None:
        """Home page has clickable elements (input, button, links)."""
        results = basic_loader.get_clickable_elements(snapshot_index=0)
        assert len(results) == 1
        elements = results[0]["elements"]
        assert len(elements) > 0

    def test_login_page_clickable(self, basic_loader: DOMDataLoader) -> None:
        """Login page has clickable elements."""
        results = basic_loader.get_clickable_elements(snapshot_index=3)
        assert len(results) == 1
        elements = results[0]["elements"]
        tags = [e["tag"] for e in elements]
        # Inputs, button, and link should be clickable
        assert "INPUT" in tags or "BUTTON" in tags

    def test_clickable_elements_have_attrs(self, basic_loader: DOMDataLoader) -> None:
        """Clickable elements include their attributes."""
        results = basic_loader.get_clickable_elements(snapshot_index=0)
        for element in results[0]["elements"]:
            assert "tag" in element
            assert "node_index" in element
            assert "attrs" in element

    def test_all_snapshots(self, basic_loader: DOMDataLoader) -> None:
        """All pages have some clickable elements."""
        results = basic_loader.get_clickable_elements()
        assert len(results) == 4


# --- Snapshot Diff Tests ---


class TestSnapshotDiff:
    """Tests for get_snapshot_diff method."""

    def test_diff_valid_snapshots(self, basic_loader: DOMDataLoader) -> None:
        """Diff between two valid snapshots."""
        diff = basic_loader.get_snapshot_diff(0, 1)
        assert diff is not None
        assert diff["added_count"] > 0
        assert diff["removed_count"] > 0
        assert diff["shared_count"] > 0

    def test_diff_snapshot_urls(self, basic_loader: DOMDataLoader) -> None:
        """Diff includes snapshot URLs."""
        diff = basic_loader.get_snapshot_diff(0, 2)
        assert diff is not None
        assert diff["snapshot_a"]["url"] == "https://www.example.com/"
        assert diff["snapshot_b"]["url"] == "https://www.example.com/search?q=laptops"

    def test_diff_invalid_index(self, basic_loader: DOMDataLoader) -> None:
        """Return None when either index is invalid."""
        assert basic_loader.get_snapshot_diff(0, 99) is None
        assert basic_loader.get_snapshot_diff(99, 0) is None


# --- Navigation Sequence Tests ---


class TestNavigationSequence:
    """Tests for get_navigation_sequence method."""

    def test_navigation_sequence_length(self, basic_loader: DOMDataLoader) -> None:
        """Navigation sequence has one entry per snapshot."""
        sequence = basic_loader.get_navigation_sequence()
        assert len(sequence) == 4

    def test_navigation_sequence_ordered_by_timestamp(self, basic_loader: DOMDataLoader) -> None:
        """Navigation sequence is ordered by timestamp."""
        sequence = basic_loader.get_navigation_sequence()
        timestamps = [s["timestamp"] for s in sequence]
        assert timestamps == sorted(timestamps)

    def test_navigation_sequence_hosts(self, basic_loader: DOMDataLoader) -> None:
        """Navigation sequence correctly extracts hosts."""
        sequence = basic_loader.get_navigation_sequence()
        hosts = [s["host"] for s in sequence]
        assert "www.example.com" in hosts
        assert "auth.example.com" in hosts


# --- Abstract Interface Tests ---


class TestAbstractInterface:
    """Tests for inherited AbstractDataLoader methods."""

    def test_search_by_terms(self, basic_loader: DOMDataLoader) -> None:
        """Inherited search_by_terms works on DOM string tables."""
        results = basic_loader.search_by_terms(["laptop"])
        assert len(results) > 0

    def test_search_content(self, basic_loader: DOMDataLoader) -> None:
        """Inherited search_content works on DOM string tables."""
        results = basic_loader.search_content("password")
        assert len(results) > 0

    def test_get_entry_id(self, basic_loader: DOMDataLoader) -> None:
        """get_entry_id returns string index."""
        entry = basic_loader.entries[0]
        assert basic_loader.get_entry_id(entry) == "0"

    def test_get_searchable_content(self, basic_loader: DOMDataLoader) -> None:
        """get_searchable_content returns joined string table."""
        entry = basic_loader.entries[0]
        content = basic_loader.get_searchable_content(entry)
        assert content is not None
        assert "Welcome to Example" in content


# --- Meta Tags Tests ---


class TestGetMetaTags:
    """Tests for get_meta_tags method."""

    def test_home_page_has_meta_tags(self, basic_loader: DOMDataLoader) -> None:
        """Home page has meta tags (csrf-token and viewport)."""
        results = basic_loader.get_meta_tags(snapshot_index=0)
        assert len(results) == 1
        meta_tags = results[0]["meta_tags"]
        assert len(meta_tags) == 2

    def test_csrf_meta_tag_attrs(self, basic_loader: DOMDataLoader) -> None:
        """CSRF meta tag has name and content attributes."""
        results = basic_loader.get_meta_tags(snapshot_index=0)
        meta_tags = results[0]["meta_tags"]
        csrf_meta = [m for m in meta_tags if m["attrs"].get("name") == "csrf-token"]
        assert len(csrf_meta) == 1
        assert csrf_meta[0]["attrs"]["content"] == "abc123def456"

    def test_viewport_meta_tag(self, basic_loader: DOMDataLoader) -> None:
        """Viewport meta tag is present."""
        results = basic_loader.get_meta_tags(snapshot_index=0)
        meta_tags = results[0]["meta_tags"]
        viewport = [m for m in meta_tags if m["attrs"].get("name") == "viewport"]
        assert len(viewport) == 1
        assert viewport[0]["attrs"]["content"] == "width=device-width"

    def test_login_page_has_csrf_meta(self, basic_loader: DOMDataLoader) -> None:
        """Login page has a CSRF meta tag."""
        results = basic_loader.get_meta_tags(snapshot_index=3)
        assert len(results) == 1
        meta_tags = results[0]["meta_tags"]
        csrf_meta = [m for m in meta_tags if m["attrs"].get("name") == "csrf-token"]
        assert len(csrf_meta) == 1
        assert csrf_meta[0]["attrs"]["content"] == "login_csrf_abc789"

    def test_meta_tags_no_clickable(self, basic_loader: DOMDataLoader) -> None:
        """Meta tag results don't include is_clickable (simplified output)."""
        results = basic_loader.get_meta_tags(snapshot_index=0)
        for meta in results[0]["meta_tags"]:
            assert "is_clickable" not in meta

    def test_pages_without_meta(self, basic_loader: DOMDataLoader) -> None:
        """Pages without meta tags return empty results for that snapshot."""
        results = basic_loader.get_meta_tags(snapshot_index=1)
        assert len(results) == 0

    def test_all_snapshots(self, basic_loader: DOMDataLoader) -> None:
        """Searching all snapshots returns results from pages with meta tags."""
        results = basic_loader.get_meta_tags()
        # Home page and login page have meta tags
        assert len(results) == 2


# --- Script Tags Tests ---


class TestGetScripts:
    """Tests for get_scripts method."""

    def test_home_page_has_scripts(self, basic_loader: DOMDataLoader) -> None:
        """Home page has script tags."""
        results = basic_loader.get_scripts(snapshot_index=0)
        assert len(results) == 1
        scripts = results[0]["scripts"]
        assert len(scripts) == 2  # __NEXT_DATA__ + external

    def test_next_data_script(self, basic_loader: DOMDataLoader) -> None:
        """__NEXT_DATA__ script has id, type, and inline content."""
        results = basic_loader.get_scripts(snapshot_index=0)
        scripts = results[0]["scripts"]
        next_data = [s for s in scripts if s["attrs"].get("id") == "__NEXT_DATA__"]
        assert len(next_data) == 1
        assert next_data[0]["attrs"]["type"] == "application/json"
        assert "inline_content" in next_data[0]
        assert '"page":"/"' in next_data[0]["inline_content"]

    def test_external_script(self, basic_loader: DOMDataLoader) -> None:
        """External script has src attribute and no inline content."""
        results = basic_loader.get_scripts(snapshot_index=0)
        scripts = results[0]["scripts"]
        external = [s for s in scripts if "src" in s["attrs"]]
        assert len(external) == 1
        assert external[0]["attrs"]["src"] == "/js/app.js"
        assert "inline_content" not in external[0]

    def test_inline_content_truncation(self, basic_loader: DOMDataLoader) -> None:
        """Inline content is truncated when exceeding max_inline_chars."""
        results = basic_loader.get_scripts(snapshot_index=0, max_inline_chars=10)
        scripts = results[0]["scripts"]
        next_data = [s for s in scripts if s["attrs"].get("id") == "__NEXT_DATA__"]
        assert len(next_data) == 1
        assert next_data[0].get("inline_content_truncated") is True
        assert next_data[0]["inline_content"].endswith("...")

    def test_pages_without_scripts(self, basic_loader: DOMDataLoader) -> None:
        """Pages without scripts return empty results."""
        results = basic_loader.get_scripts(snapshot_index=1)
        assert len(results) == 0

    def test_all_snapshots(self, basic_loader: DOMDataLoader) -> None:
        """Only snapshots with scripts are returned."""
        results = basic_loader.get_scripts()
        assert len(results) == 1  # Only home page has scripts


# --- Hidden Inputs Tests ---


class TestGetHiddenInputs:
    """Tests for get_hidden_inputs method."""

    def test_home_page_has_hidden_input(self, basic_loader: DOMDataLoader) -> None:
        """Home page has a hidden CSRF input."""
        results = basic_loader.get_hidden_inputs(snapshot_index=0)
        assert len(results) == 1
        elements = results[0]["elements"]
        assert len(elements) == 1

    def test_hidden_input_attrs(self, basic_loader: DOMDataLoader) -> None:
        """Hidden input has correct name and value attributes."""
        results = basic_loader.get_hidden_inputs(snapshot_index=0)
        el = results[0]["elements"][0]
        assert el["attrs"]["type"] == "hidden"
        assert el["attrs"]["name"] == "csrf_token"
        assert el["attrs"]["value"] == "hidden_csrf_value"

    def test_login_page_hidden_input(self, basic_loader: DOMDataLoader) -> None:
        """Login page has a hidden form token."""
        results = basic_loader.get_hidden_inputs(snapshot_index=3)
        assert len(results) == 1
        el = results[0]["elements"][0]
        assert el["attrs"]["name"] == "_token"
        assert el["attrs"]["value"] == "form_token_xyz"

    def test_pages_without_hidden_inputs(self, basic_loader: DOMDataLoader) -> None:
        """Pages without hidden inputs return empty."""
        results = basic_loader.get_hidden_inputs(snapshot_index=1)
        assert len(results) == 0

    def test_all_snapshots(self, basic_loader: DOMDataLoader) -> None:
        """Searching all snapshots finds hidden inputs across pages."""
        results = basic_loader.get_hidden_inputs()
        # Home page and login page have hidden inputs
        assert len(results) == 2

    def test_excludes_non_hidden_inputs(self, basic_loader: DOMDataLoader) -> None:
        """Regular inputs (text, email, etc.) are excluded."""
        results = basic_loader.get_hidden_inputs(snapshot_index=0)
        for el in results[0]["elements"]:
            assert el["attrs"].get("type") == "hidden"
