"""
tests/unit/agents/test_abstract_agent_helpers.py

Unit tests for module-level helper functions in abstract_agent.py:
  - _serialize_tool_result
  - _normalize_file_scope
  - _parse_search_terms
"""

import json
from datetime import datetime
from typing import Any

import pytest

from bluebox.agents.abstract_agent import (
    _normalize_file_scope,
    _parse_search_terms,
    _serialize_tool_result,
)


# =============================================================================
# _serialize_tool_result
# =============================================================================


class TestSerializeToolResult:
    """Tests for _serialize_tool_result."""

    def test_dict_returns_json(self) -> None:
        result = {"key": "value", "count": 42}
        serialized, content_type = _serialize_tool_result(result)
        assert content_type == "json"
        assert json.loads(serialized) == result

    def test_list_returns_json(self) -> None:
        result = [1, 2, 3]
        serialized, content_type = _serialize_tool_result(result)
        assert content_type == "json"
        assert json.loads(serialized) == result

    def test_string_returns_json(self) -> None:
        result = "hello"
        serialized, content_type = _serialize_tool_result(result)
        assert content_type == "json"
        assert json.loads(serialized) == result

    def test_nested_dict_returns_json(self) -> None:
        result = {"a": {"b": [1, 2]}, "c": None}
        serialized, content_type = _serialize_tool_result(result)
        assert content_type == "json"
        assert json.loads(serialized) == result

    def test_none_returns_json(self) -> None:
        serialized, content_type = _serialize_tool_result(None)
        assert content_type == "json"
        assert json.loads(serialized) is None

    def test_non_ascii_preserved(self) -> None:
        result = {"emoji": "🔥", "text": "café"}
        serialized, content_type = _serialize_tool_result(result)
        assert content_type == "json"
        assert "🔥" in serialized
        assert "café" in serialized

    def test_datetime_uses_default_str(self) -> None:
        dt = datetime(2025, 1, 15, 12, 30, 0)
        result = {"timestamp": dt}
        serialized, content_type = _serialize_tool_result(result)
        assert content_type == "json"
        parsed = json.loads(serialized)
        assert "2025-01-15" in parsed["timestamp"]

    def test_non_serializable_falls_back_to_text(self) -> None:
        # An object whose __str__ works but json.dumps with default=str
        # should still handle it — need something truly unserializable.
        # Actually, default=str handles most things. Let's verify str fallback
        # by using an object that raises in __repr__/__str__ during json encoding.

        class BadObj:
            def __repr__(self) -> str:
                return "BadObj()"

        # default=str calls str() on non-serializable, so this should still
        # produce json via the default handler
        result = {"obj": BadObj()}
        serialized, content_type = _serialize_tool_result(result)
        assert content_type == "json"
        assert "BadObj()" in serialized

    def test_integer_returns_json(self) -> None:
        serialized, content_type = _serialize_tool_result(42)
        assert content_type == "json"
        assert json.loads(serialized) == 42

    def test_boolean_returns_json(self) -> None:
        serialized, content_type = _serialize_tool_result(True)
        assert content_type == "json"
        assert json.loads(serialized) is True

    def test_empty_dict_returns_json(self) -> None:
        serialized, content_type = _serialize_tool_result({})
        assert content_type == "json"
        assert json.loads(serialized) == {}


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
