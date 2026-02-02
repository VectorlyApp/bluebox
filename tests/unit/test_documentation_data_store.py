"""
tests/unit/test_documentation_data_store.py

Comprehensive unit tests for DocumentationDataStore and related classes.
"""

from pathlib import Path

import pytest

from bluebox.llms.infra.documentation_data_store import (
    DocumentationDataStore,
    DocumentationStats,
    FileEntry,
    FileType,
)


# --- Fixtures ---

@pytest.fixture(scope="module")
def documentation_files_dir(tests_root: Path) -> Path:
    """Directory containing documentation test files."""
    return tests_root / "data" / "input" / "documentation_files"


@pytest.fixture
def basic_store(documentation_files_dir: Path) -> DocumentationDataStore:
    """DocumentationDataStore loaded with all test files."""
    return DocumentationDataStore(
        documentation_paths=[str(documentation_files_dir / "*.md")],
        code_paths=[str(documentation_files_dir / "*.py")],
    )


@pytest.fixture
def docs_only_store(documentation_files_dir: Path) -> DocumentationDataStore:
    """DocumentationDataStore loaded with only documentation files."""
    return DocumentationDataStore(
        documentation_paths=[str(documentation_files_dir / "*.md")],
    )


@pytest.fixture
def code_only_store(documentation_files_dir: Path) -> DocumentationDataStore:
    """DocumentationDataStore loaded with only code files."""
    return DocumentationDataStore(
        code_paths=[str(documentation_files_dir / "*.py")],
    )


@pytest.fixture
def empty_store() -> DocumentationDataStore:
    """Empty DocumentationDataStore with no files."""
    return DocumentationDataStore()


# --- DocumentationStats Tests ---

class TestDocumentationStats:
    """Tests for DocumentationStats dataclass."""

    def test_to_summary_basic(self) -> None:
        """Generate summary from basic stats."""
        stats = DocumentationStats(
            total_files=10,
            total_docs=6,
            total_code=4,
            total_bytes=5000,
            extensions={".md": 6, ".py": 4},
        )
        summary = stats.to_summary()
        assert "Total Files: 10" in summary
        assert "Documentation: 6" in summary
        assert "Code: 4" in summary
        assert ".md: 6" in summary
        assert ".py: 4" in summary

    def test_to_summary_with_quality_metrics(self) -> None:
        """Generate summary with documentation quality metrics."""
        stats = DocumentationStats(
            total_files=5,
            total_docs=3,
            total_code=2,
            docs_with_summary=2,
            docs_with_title=3,
            code_with_docstring=1,
        )
        summary = stats.to_summary()
        assert "With Summary: 2/3" in summary
        assert "With Title: 3/3" in summary
        assert "With Docstring: 1/2" in summary


# --- FileEntry Tests ---

class TestFileEntry:
    """Tests for FileEntry dataclass."""

    def test_to_dict(self) -> None:
        """Convert FileEntry to dictionary."""
        entry = FileEntry(
            path=Path("/test/file.md"),
            file_type=FileType.DOCUMENTATION,
            content="# Test",
            summary="Test summary",
            title="Test Title",
            extension=".md",
            size_bytes=100,
        )
        d = entry.to_dict()
        assert d["path"] == "/test/file.md"
        assert d["file_type"] == FileType.DOCUMENTATION
        assert d["summary"] == "Test summary"
        assert d["title"] == "Test Title"
        assert d["extension"] == ".md"
        assert d["size_bytes"] == 100

    def test_to_dict_without_optional_fields(self) -> None:
        """Convert FileEntry to dict when optional fields are None."""
        entry = FileEntry(
            path=Path("/test/file.md"),
            file_type=FileType.DOCUMENTATION,
            content="# Test",
        )
        d = entry.to_dict()
        assert d["summary"] is None
        assert d["title"] is None


# --- DocumentationDataStore Initialization Tests ---

class TestDocumentationDataStoreInit:
    """Tests for DocumentationDataStore initialization."""

    def test_init_with_docs_and_code(self, basic_store: DocumentationDataStore) -> None:
        """Initialize with both documentation and code paths."""
        assert len(basic_store.entries) > 0
        assert basic_store.stats.total_docs > 0
        assert basic_store.stats.total_code > 0

    def test_init_docs_only(self, docs_only_store: DocumentationDataStore) -> None:
        """Initialize with only documentation paths."""
        assert len(docs_only_store.entries) > 0
        assert docs_only_store.stats.total_docs > 0
        assert docs_only_store.stats.total_code == 0

    def test_init_code_only(self, code_only_store: DocumentationDataStore) -> None:
        """Initialize with only code paths."""
        assert len(code_only_store.entries) > 0
        assert code_only_store.stats.total_code > 0
        assert code_only_store.stats.total_docs == 0

    def test_init_empty(self, empty_store: DocumentationDataStore) -> None:
        """Initialize with no paths produces empty store."""
        assert len(empty_store.entries) == 0
        assert empty_store.stats.total_files == 0

    def test_init_nonexistent_path(self) -> None:
        """Non-existent paths are handled gracefully (no error)."""
        store = DocumentationDataStore(
            documentation_paths=["/nonexistent/path/*.md"],
        )
        assert len(store.entries) == 0


# --- Properties Tests ---

class TestDocumentationDataStoreProperties:
    """Tests for DocumentationDataStore properties."""

    def test_entries_returns_list(self, basic_store: DocumentationDataStore) -> None:
        """entries property returns list of FileEntry."""
        entries = basic_store.entries
        assert isinstance(entries, list)
        assert all(isinstance(e, FileEntry) for e in entries)

    def test_stats_returns_documentation_stats(self, basic_store: DocumentationDataStore) -> None:
        """stats property returns DocumentationStats instance."""
        stats = basic_store.stats
        assert isinstance(stats, DocumentationStats)
        assert stats.total_files == len(basic_store.entries)

    def test_documentation_files_property(self, basic_store: DocumentationDataStore) -> None:
        """documentation_files returns only doc entries."""
        docs = basic_store.documentation_files
        assert all(e.file_type == FileType.DOCUMENTATION for e in docs)

    def test_code_files_property(self, basic_store: DocumentationDataStore) -> None:
        """code_files returns only code entries."""
        code = basic_store.code_files
        assert all(e.file_type == FileType.CODE for e in code)


# --- Title and Summary Parsing Tests ---

class TestTitleSummaryParsing:
    """Tests for title and summary parsing."""

    def test_doc_with_title_and_summary(self, basic_store: DocumentationDataStore) -> None:
        """Parse title and summary from markdown with both."""
        entry = basic_store.get_file_by_path("doc_with_title_and_summary.md")
        assert entry is not None
        assert entry.title == "Getting Started Guide"
        assert entry.summary == "A comprehensive guide for getting started with the framework."

    def test_doc_with_title_only(self, basic_store: DocumentationDataStore) -> None:
        """Parse title from markdown without summary."""
        entry = basic_store.get_file_by_path("doc_with_title_only.md")
        assert entry is not None
        assert entry.title == "API Reference"
        assert entry.summary is None

    def test_doc_without_title(self, basic_store: DocumentationDataStore) -> None:
        """Handle markdown without title."""
        entry = basic_store.get_file_by_path("doc_no_title.md")
        assert entry is not None
        assert entry.title is None

    def test_code_with_docstring(self, basic_store: DocumentationDataStore) -> None:
        """Extract docstring from Python file."""
        entry = basic_store.get_file_by_path("code_with_docstring.py")
        assert entry is not None
        assert entry.summary is not None
        assert "code_with_docstring.py" in entry.summary
        assert "module-level docstring" in entry.summary

    def test_code_without_docstring(self, basic_store: DocumentationDataStore) -> None:
        """Handle Python file without docstring."""
        entry = basic_store.get_file_by_path("code_no_docstring.py")
        assert entry is not None
        assert entry.summary is None


# --- File Retrieval Tests ---

class TestFileRetrieval:
    """Tests for file retrieval methods."""

    def test_get_file_by_path_exact(self, basic_store: DocumentationDataStore) -> None:
        """Get file by exact path match."""
        # Get any entry and use its full path
        entry = basic_store.entries[0]
        result = basic_store.get_file_by_path(str(entry.path))
        assert result is not None
        assert result.path == entry.path

    def test_get_file_by_path_partial(self, basic_store: DocumentationDataStore) -> None:
        """Get file by partial path (filename)."""
        entry = basic_store.get_file_by_path("doc_with_title_and_summary.md")
        assert entry is not None
        assert "doc_with_title_and_summary.md" in str(entry.path)

    def test_get_file_by_path_not_found(self, basic_store: DocumentationDataStore) -> None:
        """Return None for non-existent file."""
        entry = basic_store.get_file_by_path("nonexistent_file.md")
        assert entry is None

    def test_get_file_content(self, basic_store: DocumentationDataStore) -> None:
        """Get file content by path."""
        content = basic_store.get_file_content("doc_with_title_and_summary.md")
        assert content is not None
        assert "# Getting Started Guide" in content

    def test_get_file_content_not_found(self, basic_store: DocumentationDataStore) -> None:
        """Return None for non-existent file content."""
        content = basic_store.get_file_content("nonexistent_file.md")
        assert content is None


# --- Search Content Tests ---

class TestSearchContent:
    """Tests for search_content method."""

    def test_search_finds_matches(self, basic_store: DocumentationDataStore) -> None:
        """Search finds entries containing query."""
        results = basic_store.search_content("placeholder")
        assert len(results) > 0

    def test_search_returns_count(self, basic_store: DocumentationDataStore) -> None:
        """Results include occurrence count."""
        results = basic_store.search_content("placeholder")
        assert len(results) > 0
        for r in results:
            assert "count" in r
            assert r["count"] > 0

    def test_search_returns_sample(self, basic_store: DocumentationDataStore) -> None:
        """Results include sample context."""
        results = basic_store.search_content("placeholder")
        assert len(results) > 0
        for r in results:
            assert "sample" in r
            assert isinstance(r["sample"], str)

    def test_search_case_insensitive_default(self, basic_store: DocumentationDataStore) -> None:
        """Search is case-insensitive by default."""
        results_lower = basic_store.search_content("placeholder")
        results_upper = basic_store.search_content("PLACEHOLDER")
        assert len(results_lower) == len(results_upper)

    def test_search_case_sensitive(self, basic_store: DocumentationDataStore) -> None:
        """Case-sensitive search when specified."""
        results_sensitive = basic_store.search_content("UPPERCASE", case_sensitive=True)
        results_wrong_case = basic_store.search_content("uppercase", case_sensitive=True)
        # Results should differ based on case
        assert len(results_sensitive) != len(results_wrong_case) or len(results_sensitive) == 0

    def test_search_filter_by_file_type(self, basic_store: DocumentationDataStore) -> None:
        """Filter search results by file type."""
        results = basic_store.search_content("def", file_type=FileType.CODE)
        assert len(results) > 0
        for r in results:
            assert r["file_type"] == FileType.CODE

    def test_search_no_matches(self, basic_store: DocumentationDataStore) -> None:
        """Return empty list when no matches."""
        results = basic_store.search_content("xyznonexistent123")
        assert results == []

    def test_search_empty_query(self, basic_store: DocumentationDataStore) -> None:
        """Return empty list for empty query."""
        results = basic_store.search_content("")
        assert results == []

    def test_search_sorted_by_count(self, basic_store: DocumentationDataStore) -> None:
        """Results sorted by count descending."""
        results = basic_store.search_content("the")
        if len(results) > 1:
            counts = [r["count"] for r in results]
            assert counts == sorted(counts, reverse=True)


# --- Search Content With Lines Tests ---

class TestSearchContentWithLines:
    """Tests for search_content_with_lines method."""

    def test_search_returns_line_numbers(self, basic_store: DocumentationDataStore) -> None:
        """Search returns line numbers for matches."""
        results = basic_store.search_content_with_lines("placeholder")
        assert len(results) > 0
        for r in results:
            assert "matches" in r
            for match in r["matches"]:
                assert "line_number" in match
                assert "line_content" in match
                assert isinstance(match["line_number"], int)
                assert match["line_number"] > 0

    def test_search_line_numbers_are_1_indexed(self, basic_store: DocumentationDataStore) -> None:
        """Line numbers are 1-indexed (not 0-indexed)."""
        results = basic_store.search_content_with_lines("# Getting Started")
        assert len(results) > 0
        # First line of doc should be line 1
        for r in results:
            if "doc_with_title_and_summary" in r["path"]:
                assert r["matches"][0]["line_number"] == 1

    def test_search_limits_matches_per_file(self, basic_store: DocumentationDataStore) -> None:
        """Respects max_matches_per_file limit."""
        results = basic_store.search_content_with_lines("the", max_matches_per_file=2)
        for r in results:
            assert len(r["matches"]) <= 2

    def test_search_filter_by_file_type(self, basic_store: DocumentationDataStore) -> None:
        """Filter by file type."""
        results = basic_store.search_content_with_lines("def", file_type=FileType.CODE)
        for r in results:
            assert r["file_type"] == FileType.CODE

    def test_search_case_sensitive(self, basic_store: DocumentationDataStore) -> None:
        """Case-sensitive search."""
        results = basic_store.search_content_with_lines("UPPERCASE", case_sensitive=True)
        # Should find matches for exact case
        assert len(results) > 0

    def test_search_sorted_by_matches(self, basic_store: DocumentationDataStore) -> None:
        """Results sorted by total matches descending."""
        results = basic_store.search_content_with_lines("the")
        if len(results) > 1:
            match_counts = [r["total_matches"] for r in results]
            assert match_counts == sorted(match_counts, reverse=True)


# --- Get File Lines Tests ---

class TestGetFileLines:
    """Tests for get_file_lines method."""

    def test_get_all_lines(self, basic_store: DocumentationDataStore) -> None:
        """Get all lines when no range specified."""
        result = basic_store.get_file_lines("doc_with_title_and_summary.md")
        assert result is not None
        content, total_lines = result
        assert total_lines > 0
        assert "# Getting Started Guide" in content

    def test_get_specific_range(self, basic_store: DocumentationDataStore) -> None:
        """Get specific line range."""
        result = basic_store.get_file_lines(
            "doc_with_title_and_summary.md",
            start_line=1,
            end_line=3,
        )
        assert result is not None
        content, total_lines = result
        # Should have first 3 lines
        lines = content.split("\n")
        assert len(lines) == 3
        assert "# Getting Started Guide" in lines[0]

    def test_get_lines_from_start(self, basic_store: DocumentationDataStore) -> None:
        """Get lines from start when only end_line specified."""
        result = basic_store.get_file_lines(
            "doc_with_title_and_summary.md",
            end_line=5,
        )
        assert result is not None
        content, _ = result
        lines = content.split("\n")
        assert len(lines) == 5

    def test_get_lines_to_end(self, basic_store: DocumentationDataStore) -> None:
        """Get lines to end when only start_line specified."""
        result_all = basic_store.get_file_lines("doc_with_title_and_summary.md")
        result_partial = basic_store.get_file_lines(
            "doc_with_title_and_summary.md",
            start_line=5,
        )
        assert result_all is not None
        assert result_partial is not None
        _, total_all = result_all
        content_partial, total_partial = result_partial
        # Should get lines from 5 to end
        assert total_partial == total_all
        lines = content_partial.split("\n")
        assert len(lines) == total_all - 4  # Lines 5 to end

    def test_get_lines_file_not_found(self, basic_store: DocumentationDataStore) -> None:
        """Return None for non-existent file."""
        result = basic_store.get_file_lines("nonexistent.md")
        assert result is None

    def test_get_lines_clamps_range(self, basic_store: DocumentationDataStore) -> None:
        """Line range is clamped to valid range."""
        result = basic_store.get_file_lines(
            "doc_with_title_and_summary.md",
            start_line=1,
            end_line=10000,  # Beyond file length
        )
        assert result is not None
        content, total_lines = result
        # Should return all lines, not error
        lines = content.split("\n")
        assert len(lines) == total_lines


# --- Index Methods Tests ---

class TestIndexMethods:
    """Tests for get_documentation_index and get_code_index."""

    def test_documentation_index_structure(self, basic_store: DocumentationDataStore) -> None:
        """Documentation index has correct structure."""
        index = basic_store.get_documentation_index()
        assert isinstance(index, list)
        for item in index:
            assert "path" in item
            assert "filename" in item
            assert "title" in item
            assert "summary" in item

    def test_documentation_index_sorted_by_path(self, basic_store: DocumentationDataStore) -> None:
        """Documentation index is sorted by path."""
        index = basic_store.get_documentation_index()
        paths = [item["path"] for item in index]
        assert paths == sorted(paths)

    def test_code_index_structure(self, basic_store: DocumentationDataStore) -> None:
        """Code index has correct structure."""
        index = basic_store.get_code_index()
        assert isinstance(index, list)
        for item in index:
            assert "path" in item
            assert "filename" in item
            assert "docstring" in item
            assert "extension" in item

    def test_code_index_sorted_by_path(self, basic_store: DocumentationDataStore) -> None:
        """Code index is sorted by path."""
        index = basic_store.get_code_index()
        paths = [item["path"] for item in index]
        assert paths == sorted(paths)


# --- Stats Computation Tests ---

class TestStatsComputation:
    """Tests for stats computation."""

    def test_stats_total_files(self, basic_store: DocumentationDataStore) -> None:
        """Total files equals entries length."""
        assert basic_store.stats.total_files == len(basic_store.entries)

    def test_stats_docs_and_code_sum(self, basic_store: DocumentationDataStore) -> None:
        """Docs + code equals total files."""
        stats = basic_store.stats
        assert stats.total_docs + stats.total_code == stats.total_files

    def test_stats_extensions_counted(self, basic_store: DocumentationDataStore) -> None:
        """Extensions are correctly counted."""
        stats = basic_store.stats
        assert ".md" in stats.extensions
        assert ".py" in stats.extensions
        # Sum of extension counts should equal total files
        assert sum(stats.extensions.values()) == stats.total_files

    def test_stats_quality_metrics(self, basic_store: DocumentationDataStore) -> None:
        """Quality metrics are computed."""
        stats = basic_store.stats
        # Should have some docs with titles (at least 2 test files have titles)
        assert stats.docs_with_title >= 2
        # Should have some docs with summaries
        assert stats.docs_with_summary >= 1
        # Should have at least 1 code file with docstring
        assert stats.code_with_docstring >= 1

    def test_stats_bytes_accumulated(self, basic_store: DocumentationDataStore) -> None:
        """Total bytes is accumulated from all files."""
        stats = basic_store.stats
        assert stats.total_bytes > 0
        # Should equal sum of entry sizes
        total = sum(e.size_bytes for e in basic_store.entries)
        assert stats.total_bytes == total
