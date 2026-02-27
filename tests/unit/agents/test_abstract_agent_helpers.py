"""
tests/unit/agents/test_abstract_agent_helpers.py

Unit tests for module-level helper functions in abstract_agent.py:
  - _normalize_file_scope
  - _parse_search_terms
"""

import pytest

from bluebox.agents.abstract_agent import (
    _normalize_file_scope,
    _parse_search_terms,
)


# =============================================================================
# _normalize_file_scope
# =============================================================================


class TestNormalizeFileScope:
    """Tests for _normalize_file_scope."""

    def test_workspace_lowercase(self) -> None:
        assert _normalize_file_scope("workspace") == "workspace"

    def test_docs_lowercase(self) -> None:
        assert _normalize_file_scope("docs") == "docs"

    def test_uppercase_normalized(self) -> None:
        assert _normalize_file_scope("WORKSPACE") == "workspace"
        assert _normalize_file_scope("DOCS") == "docs"

    def test_mixed_case_normalized(self) -> None:
        assert _normalize_file_scope("Workspace") == "workspace"
        assert _normalize_file_scope("Docs") == "docs"

    def test_leading_trailing_whitespace_stripped(self) -> None:
        assert _normalize_file_scope("  workspace  ") == "workspace"
        assert _normalize_file_scope("\tdocs\n") == "docs"

    def test_invalid_scope_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="scope must be 'workspace' or 'docs'"):
            _normalize_file_scope("files")

    def test_empty_string_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="scope must be 'workspace' or 'docs'"):
            _normalize_file_scope("")

    def test_whitespace_only_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="scope must be 'workspace' or 'docs'"):
            _normalize_file_scope("   ")

    def test_partial_match_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            _normalize_file_scope("work")

    def test_extra_word_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            _normalize_file_scope("workspace docs")


# =============================================================================
# _parse_search_terms
# =============================================================================


class TestParseSearchTerms:
    """Tests for _parse_search_terms."""

    def test_single_term(self) -> None:
        assert _parse_search_terms("hello") == ["hello"]

    def test_multiple_space_separated_terms(self) -> None:
        assert _parse_search_terms("foo bar baz") == ["foo", "bar", "baz"]

    def test_comma_separated_terms(self) -> None:
        assert _parse_search_terms("foo,bar,baz") == ["foo", "bar", "baz"]

    def test_mixed_separators(self) -> None:
        assert _parse_search_terms("foo, bar  baz,,qux") == ["foo", "bar", "baz", "qux"]

    def test_duplicates_removed_preserving_order(self) -> None:
        assert _parse_search_terms("foo bar foo baz bar") == ["foo", "bar", "baz"]

    def test_empty_string_returns_empty_list(self) -> None:
        assert _parse_search_terms("") == []

    def test_whitespace_only_returns_empty_list(self) -> None:
        assert _parse_search_terms("   ") == []

    def test_commas_only_returns_empty_list(self) -> None:
        assert _parse_search_terms(",,,") == []

    def test_leading_trailing_whitespace_in_terms(self) -> None:
        # The regex split + strip should handle this
        assert _parse_search_terms("  foo  ,  bar  ") == ["foo", "bar"]

    def test_tab_separated(self) -> None:
        assert _parse_search_terms("foo\tbar\tbaz") == ["foo", "bar", "baz"]

    def test_newline_separated(self) -> None:
        assert _parse_search_terms("foo\nbar\nbaz") == ["foo", "bar", "baz"]

    def test_single_character_terms(self) -> None:
        assert _parse_search_terms("a b c") == ["a", "b", "c"]

    def test_preserves_case(self) -> None:
        assert _parse_search_terms("Foo BAR baz") == ["Foo", "BAR", "baz"]

    def test_special_characters_preserved(self) -> None:
        assert _parse_search_terms("foo-bar baz_qux") == ["foo-bar", "baz_qux"]
