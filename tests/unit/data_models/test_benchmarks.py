"""
tests/unit/test_benchmarks.py

Unit tests for benchmarks data models and expression evaluation.
"""

import pytest
import re

from unittest.mock import MagicMock, patch

from web_hacker.data_models.benchmarks import (
    ExpressionOperator,
    OPERATOR_SYMBOLS,
    _format_value,
    _get_value_at_path,
    _resolve_value,
    PathReference,
    SimpleExpression,
    CompositeExpression,
    DeterministicTest,
    LLMTest,
    LLMTestResult,
    evaluate_expression,
    stringify_expression,
)


def P(path: str) -> PathReference:
    """Helper to create a PathReference for tests."""
    return PathReference(path=path)


class TestFormatValue:
    """Test cases for _format_value helper function."""

    def test_format_none(self):
        assert _format_value(None) == "null"

    def test_format_string(self):
        assert _format_value("hello") == '"hello"'
        assert _format_value("") == '""'

    def test_format_bool_true(self):
        assert _format_value(True) == "true"

    def test_format_bool_false(self):
        assert _format_value(False) == "false"

    def test_format_integer(self):
        assert _format_value(42) == "42"
        assert _format_value(0) == "0"
        assert _format_value(-5) == "-5"

    def test_format_float(self):
        assert _format_value(3.14) == "3.14"

    def test_format_list(self):
        assert _format_value([1, 2, 3]) == "[1, 2, 3]"
        assert _format_value([]) == "[]"

    def test_format_dict(self):
        result = _format_value({"key": "value"})
        assert result == '{"key": "value"}'


class TestGetValueAtPath:
    """Test cases for _get_value_at_path helper function using JMESPath."""

    def test_empty_path_returns_data(self):
        data = {"key": "value"}
        exists, value = _get_value_at_path(data, "")
        assert exists is True
        assert value == data

    def test_simple_key_access(self):
        data = {"name": "John"}
        exists, value = _get_value_at_path(data, "name")
        assert exists is True
        assert value == "John"

    def test_nested_key_access(self):
        data = {"user": {"profile": {"name": "John"}}}
        exists, value = _get_value_at_path(data, "user.profile.name")
        assert exists is True
        assert value == "John"

    def test_array_index_access(self):
        data = {"items": ["a", "b", "c"]}
        exists, value = _get_value_at_path(data, "items[0]")
        assert exists is True
        assert value == "a"

        exists, value = _get_value_at_path(data, "items[2]")
        assert exists is True
        assert value == "c"

    def test_nested_array_access(self):
        data = {"users": [{"name": "Alice"}, {"name": "Bob"}]}
        exists, value = _get_value_at_path(data, "users[1].name")
        assert exists is True
        assert value == "Bob"

    def test_missing_key_returns_not_exists(self):
        data = {"name": "John"}
        exists, value = _get_value_at_path(data, "age")
        assert exists is False
        assert value is None

    def test_missing_nested_key(self):
        data = {"user": {"name": "John"}}
        exists, value = _get_value_at_path(data, "user.age")
        assert exists is False

    def test_array_index_out_of_bounds(self):
        data = {"items": ["a", "b"]}
        exists, value = _get_value_at_path(data, "items[5]")
        assert exists is False

    def test_negative_index_access(self):
        """Negative indices are supported for lists."""
        data = {"items": ["a", "b", "c"]}
        exists, value = _get_value_at_path(data, "items[-1]")
        assert exists is True
        assert value == "c"

        exists, value = _get_value_at_path(data, "items[-2]")
        assert exists is True
        assert value == "b"

        exists, value = _get_value_at_path(data, "items[-3]")
        assert exists is True
        assert value == "a"

    def test_negative_index_out_of_bounds(self):
        """Negative index out of bounds returns not exists."""
        data = {"items": ["a", "b", "c"]}
        exists, value = _get_value_at_path(data, "items[-4]")
        assert exists is False

    def test_negative_index_nested(self):
        """Negative indices work in nested paths."""
        data = {"users": [{"name": "Alice"}, {"name": "Bob"}, {"name": "Charlie"}]}
        exists, value = _get_value_at_path(data, "users[-1].name")
        assert exists is True
        assert value == "Charlie"

    def test_path_through_none_value(self):
        data = {"user": None}
        exists, value = _get_value_at_path(data, "user.name")
        assert exists is False

    def test_none_data_with_path(self):
        exists, value = _get_value_at_path(None, "key")
        assert exists is False

    def test_none_data_with_empty_path(self):
        exists, value = _get_value_at_path(None, "")
        assert exists is True
        assert value is None

    # JMESPath-specific features
    def test_wildcard_access(self):
        """Test wildcard [*] to get all elements."""
        data = {"users": [{"name": "Alice"}, {"name": "Bob"}, {"name": "Charlie"}]}
        exists, value = _get_value_at_path(data, "users[*].name")
        assert exists is True
        assert value == ["Alice", "Bob", "Charlie"]

    def test_filter_expression(self):
        """Test filter expressions with [?...]."""
        data = {
            "steps": [
                {"type": "navigate", "url": "https://google.com"},
                {"type": "click", "selector": "#btn"},
                {"type": "navigate", "url": "https://example.com"},
            ]
        }
        exists, value = _get_value_at_path(data, "steps[?type == 'navigate']")
        assert exists is True
        assert len(value) == 2
        assert value[0]["url"] == "https://google.com"
        assert value[1]["url"] == "https://example.com"

    def test_filter_with_pipe_and_index(self):
        """Test filter with pipe to get specific element."""
        data = {
            "steps": [
                {"type": "navigate", "url": "https://google.com"},
                {"type": "click", "selector": "#btn"},
                {"type": "navigate", "url": "https://example.com"},
            ]
        }
        # Get the last navigate step
        exists, value = _get_value_at_path(data, "steps[?type == 'navigate'] | [-1]")
        assert exists is True
        assert value["url"] == "https://example.com"

    def test_filter_get_field(self):
        """Test filter and get specific field."""
        data = {
            "steps": [
                {"type": "navigate", "url": "https://google.com"},
                {"type": "click", "selector": "#btn"},
                {"type": "navigate", "url": "https://example.com"},
            ]
        }
        # Get the URL of the last navigate step
        exists, value = _get_value_at_path(data, "steps[?type == 'navigate'] | [-1].url")
        assert exists is True
        assert value == "https://example.com"

    def test_length_function(self):
        """Test JMESPath length() function."""
        data = {"items": [1, 2, 3, 4, 5]}
        exists, value = _get_value_at_path(data, "length(items)")
        assert exists is True
        assert value == 5

    def test_contains_function(self):
        """Test JMESPath contains() function."""
        data = {"text": "hello world"}
        exists, value = _get_value_at_path(data, "contains(text, 'world')")
        assert exists is True
        assert value is True

    def test_keys_function(self):
        """Test JMESPath keys() function."""
        data = {"user": {"name": "John", "age": 30}}
        exists, value = _get_value_at_path(data, "keys(user)")
        assert exists is True
        assert set(value) == {"name", "age"}


class TestSimpleExpressionEvaluate:
    """Test cases for SimpleExpression.evaluate method."""

    # Existence operators
    def test_exists_when_path_exists(self):
        expr = SimpleExpression(value_1=P("name"), operator=ExpressionOperator.EXISTS)
        assert expr.evaluate({"name": "John"}) is True

    def test_exists_when_path_missing(self):
        expr = SimpleExpression(value_1=P("age"), operator=ExpressionOperator.EXISTS)
        assert expr.evaluate({"name": "John"}) is False

    def test_exists_with_none_value(self):
        """Path exists even if value is None."""
        expr = SimpleExpression(value_1=P("name"), operator=ExpressionOperator.EXISTS)
        assert expr.evaluate({"name": None}) is True

    def test_not_exists_when_path_missing(self):
        expr = SimpleExpression(value_1=P("age"), operator=ExpressionOperator.NOT_EXISTS)
        assert expr.evaluate({"name": "John"}) is True

    def test_not_exists_when_path_exists(self):
        expr = SimpleExpression(value_1=P("name"), operator=ExpressionOperator.NOT_EXISTS)
        assert expr.evaluate({"name": "John"}) is False

    # Null checks
    def test_is_null_with_none_value(self):
        expr = SimpleExpression(value_1=P("value"), operator=ExpressionOperator.IS_NULL)
        assert expr.evaluate({"value": None}) is True

    def test_is_null_with_non_none_value(self):
        expr = SimpleExpression(value_1=P("value"), operator=ExpressionOperator.IS_NULL)
        assert expr.evaluate({"value": "something"}) is False

    def test_is_null_with_missing_path(self):
        expr = SimpleExpression(value_1=P("missing"), operator=ExpressionOperator.IS_NULL)
        assert expr.evaluate({"value": "something"}) is False

    def test_is_not_null_with_value(self):
        expr = SimpleExpression(value_1=P("value"), operator=ExpressionOperator.IS_NOT_NULL)
        assert expr.evaluate({"value": "something"}) is True

    def test_is_not_null_with_none(self):
        expr = SimpleExpression(value_1=P("value"), operator=ExpressionOperator.IS_NOT_NULL)
        assert expr.evaluate({"value": None}) is False

    # Type checks
    def test_is_type_string(self):
        expr = SimpleExpression(value_1=P("val"), operator=ExpressionOperator.IS_TYPE, value_2="string")
        assert expr.evaluate({"val": "hello"}) is True
        assert expr.evaluate({"val": 123}) is False

    def test_is_type_integer(self):
        expr = SimpleExpression(value_1=P("val"), operator=ExpressionOperator.IS_TYPE, value_2="int")
        assert expr.evaluate({"val": 42}) is True
        assert expr.evaluate({"val": 3.14}) is False

    def test_is_type_number(self):
        """'number' type should match both int and float."""
        expr = SimpleExpression(value_1=P("val"), operator=ExpressionOperator.IS_TYPE, value_2="number")
        assert expr.evaluate({"val": 42}) is True
        assert expr.evaluate({"val": 3.14}) is True
        assert expr.evaluate({"val": "42"}) is False

    def test_is_type_bool(self):
        expr = SimpleExpression(value_1=P("val"), operator=ExpressionOperator.IS_TYPE, value_2="bool")
        assert expr.evaluate({"val": True}) is True
        assert expr.evaluate({"val": False}) is True
        assert expr.evaluate({"val": 1}) is False

    def test_is_type_list(self):
        expr = SimpleExpression(value_1=P("val"), operator=ExpressionOperator.IS_TYPE, value_2="list")
        assert expr.evaluate({"val": [1, 2, 3]}) is True
        assert expr.evaluate({"val": (1, 2, 3)}) is False

    def test_is_type_dict(self):
        expr = SimpleExpression(value_1=P("val"), operator=ExpressionOperator.IS_TYPE, value_2="dict")
        assert expr.evaluate({"val": {"key": "value"}}) is True
        assert expr.evaluate({"val": [1, 2]}) is False

    def test_is_type_null(self):
        expr = SimpleExpression(value_1=P("val"), operator=ExpressionOperator.IS_TYPE, value_2="null")
        assert expr.evaluate({"val": None}) is True
        assert expr.evaluate({"val": ""}) is False

    def test_is_type_unknown_type(self):
        expr = SimpleExpression(value_1=P("val"), operator=ExpressionOperator.IS_TYPE, value_2="unknown_type")
        assert expr.evaluate({"val": "hello"}) is False

    # Empty checks
    def test_is_empty_with_empty_string(self):
        expr = SimpleExpression(value_1=P("val"), operator=ExpressionOperator.IS_EMPTY)
        assert expr.evaluate({"val": ""}) is True

    def test_is_empty_with_non_empty_string(self):
        expr = SimpleExpression(value_1=P("val"), operator=ExpressionOperator.IS_EMPTY)
        assert expr.evaluate({"val": "hello"}) is False

    def test_is_empty_with_empty_list(self):
        expr = SimpleExpression(value_1=P("val"), operator=ExpressionOperator.IS_EMPTY)
        assert expr.evaluate({"val": []}) is True

    def test_is_empty_with_non_empty_list(self):
        expr = SimpleExpression(value_1=P("val"), operator=ExpressionOperator.IS_EMPTY)
        assert expr.evaluate({"val": [1, 2, 3]}) is False

    def test_is_empty_with_empty_dict(self):
        expr = SimpleExpression(value_1=P("val"), operator=ExpressionOperator.IS_EMPTY)
        assert expr.evaluate({"val": {}}) is True

    def test_is_empty_with_none(self):
        expr = SimpleExpression(value_1=P("val"), operator=ExpressionOperator.IS_EMPTY)
        assert expr.evaluate({"val": None}) is True

    def test_is_empty_with_number(self):
        """Numbers don't have length, so is_empty returns False."""
        expr = SimpleExpression(value_1=P("val"), operator=ExpressionOperator.IS_EMPTY)
        assert expr.evaluate({"val": 0}) is False
        assert expr.evaluate({"val": 42}) is False

    def test_is_not_empty_with_value(self):
        expr = SimpleExpression(value_1=P("val"), operator=ExpressionOperator.IS_NOT_EMPTY)
        assert expr.evaluate({"val": "hello"}) is True
        assert expr.evaluate({"val": [1]}) is True

    def test_is_not_empty_with_empty(self):
        expr = SimpleExpression(value_1=P("val"), operator=ExpressionOperator.IS_NOT_EMPTY)
        assert expr.evaluate({"val": ""}) is False
        assert expr.evaluate({"val": []}) is False

    def test_is_not_empty_with_none(self):
        expr = SimpleExpression(value_1=P("val"), operator=ExpressionOperator.IS_NOT_EMPTY)
        assert expr.evaluate({"val": None}) is False

    def test_is_not_empty_with_number(self):
        """Numbers without __len__ return True for is_not_empty."""
        expr = SimpleExpression(value_1=P("val"), operator=ExpressionOperator.IS_NOT_EMPTY)
        assert expr.evaluate({"val": 42}) is True

    # Equality operators
    def test_equals_string(self):
        expr = SimpleExpression(value_1=P("name"), operator=ExpressionOperator.EQUALS, value_2="John")
        assert expr.evaluate({"name": "John"}) is True
        assert expr.evaluate({"name": "Jane"}) is False

    def test_equals_number(self):
        expr = SimpleExpression(value_1=P("age"), operator=ExpressionOperator.EQUALS, value_2=30)
        assert expr.evaluate({"age": 30}) is True
        assert expr.evaluate({"age": 25}) is False

    def test_equals_bool(self):
        expr = SimpleExpression(value_1=P("active"), operator=ExpressionOperator.EQUALS, value_2=True)
        assert expr.evaluate({"active": True}) is True
        assert expr.evaluate({"active": False}) is False

    def test_equals_none(self):
        expr = SimpleExpression(value_1=P("val"), operator=ExpressionOperator.EQUALS, value_2=None)
        assert expr.evaluate({"val": None}) is True
        assert expr.evaluate({"val": ""}) is False

    def test_equals_list(self):
        expr = SimpleExpression(value_1=P("items"), operator=ExpressionOperator.EQUALS, value_2=[1, 2, 3])
        assert expr.evaluate({"items": [1, 2, 3]}) is True
        assert expr.evaluate({"items": [1, 2]}) is False

    def test_not_equals(self):
        expr = SimpleExpression(value_1=P("name"), operator=ExpressionOperator.NOT_EQUALS, value_2="John")
        assert expr.evaluate({"name": "Jane"}) is True
        assert expr.evaluate({"name": "John"}) is False

    def test_equals_missing_path(self):
        expr = SimpleExpression(value_1=P("missing"), operator=ExpressionOperator.EQUALS, value_2="value")
        assert expr.evaluate({"name": "John"}) is False

    # Containment operators
    def test_contains_string_in_string(self):
        expr = SimpleExpression(value_1=P("text"), operator=ExpressionOperator.CONTAINS, value_2="world")
        assert expr.evaluate({"text": "hello world"}) is True
        assert expr.evaluate({"text": "hello"}) is False

    def test_contains_item_in_list(self):
        expr = SimpleExpression(value_1=P("items"), operator=ExpressionOperator.CONTAINS, value_2=2)
        assert expr.evaluate({"items": [1, 2, 3]}) is True
        assert expr.evaluate({"items": [1, 3, 5]}) is False

    def test_contains_key_in_dict(self):
        expr = SimpleExpression(value_1=P("obj"), operator=ExpressionOperator.CONTAINS, value_2="key")
        assert expr.evaluate({"obj": {"key": "value"}}) is True
        assert expr.evaluate({"obj": {"other": "value"}}) is False

    def test_contains_item_in_tuple(self):
        expr = SimpleExpression(value_1=P("coords"), operator=ExpressionOperator.CONTAINS, value_2=10)
        assert expr.evaluate({"coords": (10, 20, 30)}) is True

    def test_contains_incompatible_types(self):
        """When actual is not a string/list/tuple/dict, returns False."""
        expr = SimpleExpression(value_1=P("val"), operator=ExpressionOperator.CONTAINS, value_2="x")
        assert expr.evaluate({"val": 12345}) is False

    def test_not_contains_string(self):
        expr = SimpleExpression(value_1=P("text"), operator=ExpressionOperator.NOT_CONTAINS, value_2="foo")
        assert expr.evaluate({"text": "hello world"}) is True
        assert expr.evaluate({"text": "foo bar"}) is False

    def test_not_contains_list(self):
        expr = SimpleExpression(value_1=P("items"), operator=ExpressionOperator.NOT_CONTAINS, value_2=5)
        assert expr.evaluate({"items": [1, 2, 3]}) is True
        assert expr.evaluate({"items": [1, 5, 3]}) is False

    def test_not_contains_incompatible_types(self):
        """When actual is not a string/list/tuple/dict, returns True."""
        expr = SimpleExpression(value_1=P("val"), operator=ExpressionOperator.NOT_CONTAINS, value_2="x")
        assert expr.evaluate({"val": 12345}) is True

    # Comparison operators
    def test_greater_than(self):
        expr = SimpleExpression(value_1=P("age"), operator=ExpressionOperator.GREATER_THAN, value_2=18)
        assert expr.evaluate({"age": 25}) is True
        assert expr.evaluate({"age": 18}) is False
        assert expr.evaluate({"age": 10}) is False

    def test_greater_than_or_equal(self):
        expr = SimpleExpression(value_1=P("age"), operator=ExpressionOperator.GREATER_THAN_OR_EQUAL, value_2=18)
        assert expr.evaluate({"age": 25}) is True
        assert expr.evaluate({"age": 18}) is True
        assert expr.evaluate({"age": 10}) is False

    def test_less_than(self):
        expr = SimpleExpression(value_1=P("count"), operator=ExpressionOperator.LESS_THAN, value_2=10)
        assert expr.evaluate({"count": 5}) is True
        assert expr.evaluate({"count": 10}) is False
        assert expr.evaluate({"count": 15}) is False

    def test_less_than_or_equal(self):
        expr = SimpleExpression(value_1=P("count"), operator=ExpressionOperator.LESS_THAN_OR_EQUAL, value_2=10)
        assert expr.evaluate({"count": 5}) is True
        assert expr.evaluate({"count": 10}) is True
        assert expr.evaluate({"count": 15}) is False

    def test_comparison_with_floats(self):
        expr = SimpleExpression(value_1=P("price"), operator=ExpressionOperator.GREATER_THAN, value_2=9.99)
        assert expr.evaluate({"price": 10.50}) is True
        assert expr.evaluate({"price": 9.99}) is False

    def test_comparison_with_non_numeric(self):
        """Comparison with non-numeric values returns False."""
        expr = SimpleExpression(value_1=P("val"), operator=ExpressionOperator.GREATER_THAN, value_2=10)
        assert expr.evaluate({"val": "not a number"}) is False

    def test_comparison_with_numeric_string(self):
        """Numeric strings are converted to float for comparison."""
        expr = SimpleExpression(value_1=P("val"), operator=ExpressionOperator.GREATER_THAN, value_2=10)
        assert expr.evaluate({"val": "15"}) is True

    # String operators
    def test_starts_with(self):
        expr = SimpleExpression(value_1=P("text"), operator=ExpressionOperator.STARTS_WITH, value_2="Hello")
        assert expr.evaluate({"text": "Hello World"}) is True
        assert expr.evaluate({"text": "World Hello"}) is False

    def test_starts_with_non_string(self):
        expr = SimpleExpression(value_1=P("val"), operator=ExpressionOperator.STARTS_WITH, value_2="1")
        assert expr.evaluate({"val": 123}) is False

    def test_ends_with(self):
        expr = SimpleExpression(value_1=P("text"), operator=ExpressionOperator.ENDS_WITH, value_2="World")
        assert expr.evaluate({"text": "Hello World"}) is True
        assert expr.evaluate({"text": "World Hello"}) is False

    def test_ends_with_non_string(self):
        expr = SimpleExpression(value_1=P("val"), operator=ExpressionOperator.ENDS_WITH, value_2="3")
        assert expr.evaluate({"val": 123}) is False

    def test_matches_regex(self):
        expr = SimpleExpression(value_1=P("email"), operator=ExpressionOperator.MATCHES_REGEX, value_2=r"^\w+@\w+\.\w+$")
        assert expr.evaluate({"email": "test@example.com"}) is True
        assert expr.evaluate({"email": "invalid-email"}) is False

    def test_matches_regex_partial_match(self):
        """Regex uses search, so partial matches work."""
        expr = SimpleExpression(value_1=P("text"), operator=ExpressionOperator.MATCHES_REGEX, value_2=r"\d+")
        assert expr.evaluate({"text": "abc123def"}) is True

    def test_matches_regex_non_string(self):
        expr = SimpleExpression(value_1=P("val"), operator=ExpressionOperator.MATCHES_REGEX, value_2=r"\d+")
        assert expr.evaluate({"val": 123}) is False

    def test_matches_regex_invalid_pattern(self):
        """Invalid regex pattern raises an exception."""
        expr = SimpleExpression(value_1=P("text"), operator=ExpressionOperator.MATCHES_REGEX, value_2="[invalid")
        with pytest.raises(re.error):
            expr.evaluate({"text": "test"})

    # Length operators
    def test_length_equals_string(self):
        expr = SimpleExpression(value_1=P("text"), operator=ExpressionOperator.LENGTH_EQUALS, value_2=5)
        assert expr.evaluate({"text": "hello"}) is True
        assert expr.evaluate({"text": "hi"}) is False

    def test_length_equals_list(self):
        expr = SimpleExpression(value_1=P("items"), operator=ExpressionOperator.LENGTH_EQUALS, value_2=3)
        assert expr.evaluate({"items": [1, 2, 3]}) is True
        assert expr.evaluate({"items": [1, 2]}) is False

    def test_length_equals_dict(self):
        expr = SimpleExpression(value_1=P("obj"), operator=ExpressionOperator.LENGTH_EQUALS, value_2=2)
        assert expr.evaluate({"obj": {"a": 1, "b": 2}}) is True

    def test_length_greater_than(self):
        expr = SimpleExpression(value_1=P("items"), operator=ExpressionOperator.LENGTH_GREATER_THAN, value_2=2)
        assert expr.evaluate({"items": [1, 2, 3]}) is True
        assert expr.evaluate({"items": [1, 2]}) is False

    def test_length_less_than(self):
        expr = SimpleExpression(value_1=P("items"), operator=ExpressionOperator.LENGTH_LESS_THAN, value_2=3)
        assert expr.evaluate({"items": [1, 2]}) is True
        assert expr.evaluate({"items": [1, 2, 3]}) is False

    def test_length_on_non_collection(self):
        """Length operators return False for values without __len__."""
        expr = SimpleExpression(value_1=P("val"), operator=ExpressionOperator.LENGTH_EQUALS, value_2=3)
        assert expr.evaluate({"val": 123}) is False


class TestSimpleExpressionStringify:
    """Test cases for SimpleExpression.stringify method."""

    def test_stringify_equals(self):
        expr = SimpleExpression(value_1=P("user.name"), operator=ExpressionOperator.EQUALS, value_2="John")
        assert expr.stringify() == '${user.name} == "John"'

    def test_stringify_greater_than(self):
        expr = SimpleExpression(value_1=P("age"), operator=ExpressionOperator.GREATER_THAN, value_2=18)
        assert expr.stringify() == "${age} > 18"

    def test_stringify_is_null(self):
        expr = SimpleExpression(value_1=P("value"), operator=ExpressionOperator.IS_NULL)
        assert expr.stringify() == "${value} is null"

    def test_stringify_is_not_null(self):
        expr = SimpleExpression(value_1=P("value"), operator=ExpressionOperator.IS_NOT_NULL)
        assert expr.stringify() == "${value} is not null"

    def test_stringify_is_empty(self):
        expr = SimpleExpression(value_1=P("items"), operator=ExpressionOperator.IS_EMPTY)
        assert expr.stringify() == "${items} is empty"

    def test_stringify_exists(self):
        expr = SimpleExpression(value_1=P("optional_field"), operator=ExpressionOperator.EXISTS)
        assert expr.stringify() == "${optional_field} exists"

    def test_stringify_contains(self):
        expr = SimpleExpression(value_1=P("text"), operator=ExpressionOperator.CONTAINS, value_2="hello")
        assert expr.stringify() == '${text} contains "hello"'

    def test_stringify_with_bool_value(self):
        expr = SimpleExpression(value_1=P("active"), operator=ExpressionOperator.EQUALS, value_2=True)
        assert expr.stringify() == "${active} == true"

    def test_stringify_with_list_value(self):
        expr = SimpleExpression(value_1=P("items"), operator=ExpressionOperator.EQUALS, value_2=[1, 2])
        assert expr.stringify() == "${items} == [1, 2]"


class TestCompositeExpression:
    """Test cases for CompositeExpression."""

    def test_and_all_true(self):
        expr = CompositeExpression(
            logic="and",
            expressions=[
                SimpleExpression(value_1=P("age"), operator=ExpressionOperator.GREATER_THAN, value_2=18),
                SimpleExpression(value_1=P("verified"), operator=ExpressionOperator.EQUALS, value_2=True),
            ]
        )
        assert expr.evaluate({"age": 25, "verified": True}) is True

    def test_and_one_false(self):
        expr = CompositeExpression(
            logic="and",
            expressions=[
                SimpleExpression(value_1=P("age"), operator=ExpressionOperator.GREATER_THAN, value_2=18),
                SimpleExpression(value_1=P("verified"), operator=ExpressionOperator.EQUALS, value_2=True),
            ]
        )
        assert expr.evaluate({"age": 25, "verified": False}) is False

    def test_and_all_false(self):
        expr = CompositeExpression(
            logic="and",
            expressions=[
                SimpleExpression(value_1=P("age"), operator=ExpressionOperator.GREATER_THAN, value_2=18),
                SimpleExpression(value_1=P("verified"), operator=ExpressionOperator.EQUALS, value_2=True),
            ]
        )
        assert expr.evaluate({"age": 10, "verified": False}) is False

    def test_or_one_true(self):
        expr = CompositeExpression(
            logic="or",
            expressions=[
                SimpleExpression(value_1=P("admin"), operator=ExpressionOperator.EQUALS, value_2=True),
                SimpleExpression(value_1=P("moderator"), operator=ExpressionOperator.EQUALS, value_2=True),
            ]
        )
        assert expr.evaluate({"admin": True, "moderator": False}) is True
        assert expr.evaluate({"admin": False, "moderator": True}) is True

    def test_or_all_false(self):
        expr = CompositeExpression(
            logic="or",
            expressions=[
                SimpleExpression(value_1=P("admin"), operator=ExpressionOperator.EQUALS, value_2=True),
                SimpleExpression(value_1=P("moderator"), operator=ExpressionOperator.EQUALS, value_2=True),
            ]
        )
        assert expr.evaluate({"admin": False, "moderator": False}) is False

    def test_or_all_true(self):
        expr = CompositeExpression(
            logic="or",
            expressions=[
                SimpleExpression(value_1=P("admin"), operator=ExpressionOperator.EQUALS, value_2=True),
                SimpleExpression(value_1=P("moderator"), operator=ExpressionOperator.EQUALS, value_2=True),
            ]
        )
        assert expr.evaluate({"admin": True, "moderator": True}) is True

    def test_or_with_missing_path(self):
        """Test OR when one path exists and another is missing entirely."""
        expr = CompositeExpression(
            logic="or",
            expressions=[
                SimpleExpression(value_1=P("admin"), operator=ExpressionOperator.EQUALS, value_2=True),
                SimpleExpression(value_1=P("moderator"), operator=ExpressionOperator.EQUALS, value_2=True),
            ]
        )
        # Only "admin" exists and is True - should pass
        assert expr.evaluate({"admin": True}) is True
        # Only "admin" exists and is False, "moderator" missing - should fail
        assert expr.evaluate({"admin": False}) is False
        # Only "moderator" exists and is True - should pass
        assert expr.evaluate({"moderator": True}) is True
        # Neither path exists - should fail
        assert expr.evaluate({"other": "value"}) is False

    def test_and_with_missing_path(self):
        """Test AND when one path exists and another is missing entirely."""
        expr = CompositeExpression(
            logic="and",
            expressions=[
                SimpleExpression(value_1=P("admin"), operator=ExpressionOperator.EQUALS, value_2=True),
                SimpleExpression(value_1=P("verified"), operator=ExpressionOperator.EQUALS, value_2=True),
            ]
        )
        # Only "admin" exists - "verified" missing means it fails equality check
        assert expr.evaluate({"admin": True}) is False
        # Both exist and are True - should pass
        assert expr.evaluate({"admin": True, "verified": True}) is True

    def test_nested_composite(self):
        """Test nested composite expressions."""
        expr = CompositeExpression(
            logic="and",
            expressions=[
                SimpleExpression(value_1=P("active"), operator=ExpressionOperator.EQUALS, value_2=True),
                CompositeExpression(
                    logic="or",
                    expressions=[
                        SimpleExpression(value_1=P("role"), operator=ExpressionOperator.EQUALS, value_2="admin"),
                        SimpleExpression(value_1=P("role"), operator=ExpressionOperator.EQUALS, value_2="moderator"),
                    ]
                )
            ]
        )
        assert expr.evaluate({"active": True, "role": "admin"}) is True
        assert expr.evaluate({"active": True, "role": "moderator"}) is True
        assert expr.evaluate({"active": True, "role": "user"}) is False
        assert expr.evaluate({"active": False, "role": "admin"}) is False

    def test_empty_and_expression(self):
        """Empty AND expression should return True (vacuous truth)."""
        expr = CompositeExpression(logic="and", expressions=[])
        assert expr.evaluate({"any": "data"}) is True

    def test_empty_or_expression(self):
        """Empty OR expression should return False."""
        expr = CompositeExpression(logic="or", expressions=[])
        assert expr.evaluate({"any": "data"}) is False

    def test_stringify_and(self):
        expr = CompositeExpression(
            logic="and",
            expressions=[
                SimpleExpression(value_1=P("age"), operator=ExpressionOperator.GREATER_THAN, value_2=18),
                SimpleExpression(value_1=P("verified"), operator=ExpressionOperator.EQUALS, value_2=True),
            ]
        )
        assert expr.stringify() == "(${age} > 18 AND ${verified} == true)"

    def test_stringify_or(self):
        expr = CompositeExpression(
            logic="or",
            expressions=[
                SimpleExpression(value_1=P("admin"), operator=ExpressionOperator.EQUALS, value_2=True),
                SimpleExpression(value_1=P("moderator"), operator=ExpressionOperator.EQUALS, value_2=True),
            ]
        )
        assert expr.stringify() == "(${admin} == true OR ${moderator} == true)"

    def test_stringify_nested(self):
        expr = CompositeExpression(
            logic="and",
            expressions=[
                SimpleExpression(value_1=P("active"), operator=ExpressionOperator.EQUALS, value_2=True),
                CompositeExpression(
                    logic="or",
                    expressions=[
                        SimpleExpression(value_1=P("a"), operator=ExpressionOperator.EQUALS, value_2=1),
                        SimpleExpression(value_1=P("b"), operator=ExpressionOperator.EQUALS, value_2=2),
                    ]
                )
            ]
        )
        assert expr.stringify() == "(${active} == true AND (${a} == 1 OR ${b} == 2))"


class TestDeterministicTest:
    """Test cases for DeterministicTest model."""

    def test_from_dict_simple_expression(self):
        data = {
            "name": "check_name",
            "description": "Check that name equals John",
            "expression": {
                "type": "simple",
                "value_1": {"type": "path", "path": "name"},
                "operator": "equals",
                "value_2": "John"
            }
        }
        test = DeterministicTest.model_validate(data)
        assert test.name == "check_name"
        assert test.description == "Check that name equals John"
        assert isinstance(test.expression, SimpleExpression)
        assert test.expression.evaluate({"name": "John"}) is True

    def test_from_dict_composite_expression(self):
        data = {
            "name": "check_user",
            "expression": {
                "type": "composite",
                "logic": "and",
                "expressions": [
                    {"type": "simple", "value_1": {"type": "path", "path": "age"}, "operator": "greater_than", "value_2": 18},
                    {"type": "simple", "value_1": {"type": "path", "path": "verified"}, "operator": "equals", "value_2": True}
                ]
            }
        }
        test = DeterministicTest.model_validate(data)
        assert test.name == "check_user"
        assert isinstance(test.expression, CompositeExpression)

    def test_from_json(self):
        json_str = '{"name": "test", "expression": {"type": "simple", "value_1": {"type": "path", "path": "val"}, "operator": "equals", "value_2": 42}}'
        test = DeterministicTest.model_validate_json(json_str)
        assert test.name == "test"
        assert test.expression.evaluate({"val": 42}) is True

    def test_default_description(self):
        data = {
            "name": "test",
            "expression": {"type": "simple", "value_1": {"type": "path", "path": "x"}, "operator": "exists"}
        }
        test = DeterministicTest.model_validate(data)
        assert test.description == ""


class TestEvaluateExpression:
    """Test cases for evaluate_expression helper function."""

    def test_evaluate_simple(self):
        expr = SimpleExpression(value_1=P("x"), operator=ExpressionOperator.EQUALS, value_2=10)
        assert evaluate_expression(expr, {"x": 10}) is True

    def test_evaluate_composite(self):
        expr = CompositeExpression(
            logic="and",
            expressions=[
                SimpleExpression(value_1=P("a"), operator=ExpressionOperator.EQUALS, value_2=1),
                SimpleExpression(value_1=P("b"), operator=ExpressionOperator.EQUALS, value_2=2),
            ]
        )
        assert evaluate_expression(expr, {"a": 1, "b": 2}) is True


class TestStringifyExpression:
    """Test cases for stringify_expression helper function."""

    def test_stringify_simple(self):
        expr = SimpleExpression(value_1=P("x"), operator=ExpressionOperator.EQUALS, value_2=10)
        assert stringify_expression(expr) == "${x} == 10"

    def test_stringify_composite(self):
        expr = CompositeExpression(
            logic="or",
            expressions=[
                SimpleExpression(value_1=P("a"), operator=ExpressionOperator.EQUALS, value_2=1),
                SimpleExpression(value_1=P("b"), operator=ExpressionOperator.EQUALS, value_2=2),
            ]
        )
        assert stringify_expression(expr) == "(${a} == 1 OR ${b} == 2)"


class TestOperatorSymbols:
    """Test that all operators have symbols defined."""

    def test_all_operators_have_symbols(self):
        for op in ExpressionOperator:
            assert op.value in OPERATOR_SYMBOLS, f"Missing symbol for operator: {op.value}"


class TestEdgeCases:
    """Test edge cases and potential bugs."""

    def test_deep_nested_path(self):
        data = {"a": {"b": {"c": {"d": {"e": "deep"}}}}}
        expr = SimpleExpression(value_1=P("a.b.c.d.e"), operator=ExpressionOperator.EQUALS, value_2="deep")
        assert expr.evaluate(data) is True

    def test_dict_with_string_numeric_keys(self):
        """JMESPath can access dict keys that look like numbers."""
        data = {"items": {"0": "string_key", "1": "another"}}
        # JMESPath treats this as a dict key access
        exists, value = _get_value_at_path(data, 'items."0"')
        assert exists is True
        assert value == "string_key"

    def test_expression_with_empty_string_value(self):
        expr = SimpleExpression(value_1=P("name"), operator=ExpressionOperator.EQUALS, value_2="")
        assert expr.evaluate({"name": ""}) is True
        assert expr.evaluate({"name": "John"}) is False

    def test_expression_with_zero_value(self):
        expr = SimpleExpression(value_1=P("count"), operator=ExpressionOperator.EQUALS, value_2=0)
        assert expr.evaluate({"count": 0}) is True
        assert expr.evaluate({"count": 1}) is False

    def test_expression_with_false_value(self):
        expr = SimpleExpression(value_1=P("active"), operator=ExpressionOperator.EQUALS, value_2=False)
        assert expr.evaluate({"active": False}) is True
        assert expr.evaluate({"active": True}) is False

    def test_contains_empty_string(self):
        """Empty string is contained in any string."""
        expr = SimpleExpression(value_1=P("text"), operator=ExpressionOperator.CONTAINS, value_2="")
        assert expr.evaluate({"text": "hello"}) is True
        assert expr.evaluate({"text": ""}) is True

    def test_length_equals_zero(self):
        expr = SimpleExpression(value_1=P("items"), operator=ExpressionOperator.LENGTH_EQUALS, value_2=0)
        assert expr.evaluate({"items": []}) is True
        assert expr.evaluate({"items": [1]}) is False

    def test_regex_special_characters(self):
        expr = SimpleExpression(value_1=P("text"), operator=ExpressionOperator.MATCHES_REGEX, value_2=r"\$\d+\.\d{2}")
        assert expr.evaluate({"text": "Price: $19.99"}) is True
        assert expr.evaluate({"text": "Price: 19.99"}) is False

    def test_comparison_with_none_value(self):
        """Comparison operators should handle None gracefully."""
        expr = SimpleExpression(value_1=P("val"), operator=ExpressionOperator.GREATER_THAN, value_2=10)
        assert expr.evaluate({"val": None}) is False

    def test_array_of_objects(self):
        data = {
            "users": [
                {"name": "Alice", "age": 30},
                {"name": "Bob", "age": 25},
            ]
        }
        expr = SimpleExpression(value_1=P("users[0].name"), operator=ExpressionOperator.EQUALS, value_2="Alice")
        assert expr.evaluate(data) is True

        expr = SimpleExpression(value_1=P("users[1].age"), operator=ExpressionOperator.LESS_THAN, value_2=30)
        assert expr.evaluate(data) is True

    def test_type_check_case_insensitive(self):
        """Type check should be case-insensitive."""
        expr = SimpleExpression(value_1=P("val"), operator=ExpressionOperator.IS_TYPE, value_2="STRING")
        assert expr.evaluate({"val": "hello"}) is True

        expr = SimpleExpression(value_1=P("val"), operator=ExpressionOperator.IS_TYPE, value_2="String")
        assert expr.evaluate({"val": "hello"}) is True


class TestSerializationRoundtrip:
    """Test model_dump/model_dump_json and back."""

    def test_simple_expression_model_dump_roundtrip(self):
        original = SimpleExpression(
            value_1=P("user.name"),
            operator=ExpressionOperator.EQUALS,
            value_2="John"
        )
        dumped = original.model_dump()
        restored = SimpleExpression.model_validate(dumped)

        # Roundtrip should yield identical objects
        assert restored == original

    def test_simple_expression_model_dump_json_roundtrip(self):
        original = SimpleExpression(
            value_1=P("count"),
            operator=ExpressionOperator.GREATER_THAN,
            value_2=10
        )
        json_str = original.model_dump_json()
        restored = SimpleExpression.model_validate_json(json_str)

        # Roundtrip should yield identical objects
        assert restored == original

    def test_simple_expression_with_none_value_roundtrip(self):
        original = SimpleExpression(
            value_1=P("optional"),
            operator=ExpressionOperator.IS_NULL
        )
        dumped = original.model_dump()
        restored = SimpleExpression.model_validate(dumped)

        assert restored.value_2 is None
        assert restored.operator == ExpressionOperator.IS_NULL

    def test_simple_expression_with_complex_value_roundtrip(self):
        original = SimpleExpression(
            value_1=P("data"),
            operator=ExpressionOperator.EQUALS,
            value_2={"nested": [1, 2, {"key": "value"}]}
        )
        json_str = original.model_dump_json()
        restored = SimpleExpression.model_validate_json(json_str)

        assert restored.value_2 == {"nested": [1, 2, {"key": "value"}]}

    def test_composite_expression_model_dump_roundtrip(self):
        original = CompositeExpression(
            logic="and",
            expressions=[
                SimpleExpression(value_1=P("age"), operator=ExpressionOperator.GREATER_THAN, value_2=18),
                SimpleExpression(value_1=P("verified"), operator=ExpressionOperator.EQUALS, value_2=True),
            ]
        )
        dumped = original.model_dump()
        restored = CompositeExpression.model_validate(dumped)

        assert restored.logic == original.logic
        assert restored.type == "composite"
        assert len(restored.expressions) == 2
        assert restored.expressions[0].value_1.path == "age"
        assert restored.expressions[1].value_1.path == "verified"

    def test_composite_expression_model_dump_json_roundtrip(self):
        original = CompositeExpression(
            logic="or",
            expressions=[
                SimpleExpression(value_1=P("admin"), operator=ExpressionOperator.EQUALS, value_2=True),
                SimpleExpression(value_1=P("moderator"), operator=ExpressionOperator.EQUALS, value_2=True),
            ]
        )
        json_str = original.model_dump_json()
        restored = CompositeExpression.model_validate_json(json_str)

        assert restored.logic == "or"
        assert len(restored.expressions) == 2

    def test_nested_composite_expression_roundtrip(self):
        original = CompositeExpression(
            logic="and",
            expressions=[
                SimpleExpression(value_1=P("active"), operator=ExpressionOperator.EQUALS, value_2=True),
                CompositeExpression(
                    logic="or",
                    expressions=[
                        SimpleExpression(value_1=P("role"), operator=ExpressionOperator.EQUALS, value_2="admin"),
                        SimpleExpression(value_1=P("role"), operator=ExpressionOperator.EQUALS, value_2="moderator"),
                    ]
                )
            ]
        )
        json_str = original.model_dump_json()
        restored = CompositeExpression.model_validate_json(json_str)

        assert restored.logic == "and"
        assert len(restored.expressions) == 2
        assert isinstance(restored.expressions[0], SimpleExpression)
        assert isinstance(restored.expressions[1], CompositeExpression)
        assert restored.expressions[1].logic == "or"
        assert len(restored.expressions[1].expressions) == 2

    def test_deterministic_test_model_dump_roundtrip(self):
        original = DeterministicTest(
            name="check_user",
            description="Verify user is valid",
            expression=SimpleExpression(
                value_1=P("user.active"),
                operator=ExpressionOperator.EQUALS,
                value_2=True
            )
        )
        dumped = original.model_dump()
        restored = DeterministicTest.model_validate(dumped)

        assert restored.name == original.name
        assert restored.description == original.description
        assert isinstance(restored.expression, SimpleExpression)
        assert restored.expression.value_1.path == "user.active"

    def test_deterministic_test_model_dump_json_roundtrip(self):
        original = DeterministicTest(
            name="complex_test",
            description="Test with composite expression",
            expression=CompositeExpression(
                logic="and",
                expressions=[
                    SimpleExpression(value_1=P("age"), operator=ExpressionOperator.GREATER_THAN_OR_EQUAL, value_2=18),
                    SimpleExpression(value_1=P("country"), operator=ExpressionOperator.EQUALS, value_2="US"),
                ]
            )
        )
        json_str = original.model_dump_json()
        restored = DeterministicTest.model_validate_json(json_str)

        assert restored.name == "complex_test"
        assert isinstance(restored.expression, CompositeExpression)
        assert len(restored.expression.expressions) == 2

    def test_roundtrip_preserves_evaluation_behavior(self):
        """Ensure serialized and restored expressions evaluate identically."""
        original = CompositeExpression(
            logic="and",
            expressions=[
                SimpleExpression(value_1=P("user.age"), operator=ExpressionOperator.GREATER_THAN, value_2=18),
                SimpleExpression(value_1=P("user.verified"), operator=ExpressionOperator.EQUALS, value_2=True),
                CompositeExpression(
                    logic="or",
                    expressions=[
                        SimpleExpression(value_1=P("user.role"), operator=ExpressionOperator.EQUALS, value_2="admin"),
                        SimpleExpression(value_1=P("user.premium"), operator=ExpressionOperator.EQUALS, value_2=True),
                    ]
                )
            ]
        )

        json_str = original.model_dump_json()
        restored = CompositeExpression.model_validate_json(json_str)

        test_data = [
            {"user": {"age": 25, "verified": True, "role": "admin", "premium": False}},
            {"user": {"age": 25, "verified": True, "role": "user", "premium": True}},
            {"user": {"age": 25, "verified": True, "role": "user", "premium": False}},
            {"user": {"age": 15, "verified": True, "role": "admin", "premium": False}},
            {"user": {"age": 25, "verified": False, "role": "admin", "premium": False}},
        ]

        for data in test_data:
            assert original.evaluate(data) == restored.evaluate(data), f"Mismatch for data: {data}"

    def test_all_operators_serialize_correctly(self):
        """Ensure all operators can be serialized and deserialized."""
        for op in ExpressionOperator:
            original = SimpleExpression(value_1=P("test"), operator=op, value_2="test_value")
            json_str = original.model_dump_json()
            restored = SimpleExpression.model_validate_json(json_str)
            assert restored.operator == op, f"Failed for operator: {op}"

    def test_model_dump_structure(self):
        """Verify the structure of model_dump output."""
        expr = SimpleExpression(
            value_1=P("user.name"),
            operator=ExpressionOperator.EQUALS,
            value_2="John"
        )
        dumped = expr.model_dump()

        assert "type" in dumped
        assert dumped["type"] == "simple"
        assert "value_1" in dumped
        assert dumped["value_1"]["path"] == "user.name"
        assert "operator" in dumped
        assert dumped["operator"] == "equals"
        assert "value_2" in dumped
        assert dumped["value_2"] == "John"

    def test_composite_model_dump_structure(self):
        """Verify the structure of composite model_dump output."""
        expr = CompositeExpression(
            logic="and",
            expressions=[
                SimpleExpression(value_1=P("a"), operator=ExpressionOperator.EQUALS, value_2=1),
            ]
        )
        dumped = expr.model_dump()

        assert dumped["type"] == "composite"
        assert dumped["logic"] == "and"
        assert "expressions" in dumped
        assert len(dumped["expressions"]) == 1
        assert dumped["expressions"][0]["type"] == "simple"

    def test_deterministic_test_with_deeply_nested_expression(self):
        """Test serialization of deeply nested expressions."""
        original = DeterministicTest(
            name="deep_test",
            expression=CompositeExpression(
                logic="and",
                expressions=[
                    CompositeExpression(
                        logic="or",
                        expressions=[
                            CompositeExpression(
                                logic="and",
                                expressions=[
                                    SimpleExpression(value_1=P("a"), operator=ExpressionOperator.EQUALS, value_2=1),
                                    SimpleExpression(value_1=P("b"), operator=ExpressionOperator.EQUALS, value_2=2),
                                ]
                            ),
                            SimpleExpression(value_1=P("c"), operator=ExpressionOperator.EQUALS, value_2=3),
                        ]
                    ),
                    SimpleExpression(value_1=P("d"), operator=ExpressionOperator.EXISTS),
                ]
            )
        )

        json_str = original.model_dump_json()
        restored = DeterministicTest.model_validate_json(json_str)

        # Verify structure is preserved
        assert restored.name == "deep_test"
        assert isinstance(restored.expression, CompositeExpression)
        assert restored.expression.logic == "and"
        inner = restored.expression.expressions[0]
        assert isinstance(inner, CompositeExpression)
        assert inner.logic == "or"

        # Verify evaluation works
        data = {"a": 1, "b": 2, "c": 3, "d": "exists"}
        assert original.expression.evaluate(data) == restored.expression.evaluate(data)


class TestPathReference:
    """Test cases for PathReference and path-to-path comparisons."""

    def test_path_to_path_equals(self):
        """Test comparing values at two different paths."""
        data = {
            "source": {"value": 42},
            "target": {"value": 42}
        }
        expr = SimpleExpression(
            value_1=P("source.value"),
            operator=ExpressionOperator.EQUALS,
            value_2=P("target.value")
        )
        assert expr.evaluate(data) is True

        data["target"]["value"] = 100
        assert expr.evaluate(data) is False

    def test_path_to_path_not_equals(self):
        """Test comparing values at two different paths are not equal."""
        data = {
            "a": "hello",
            "b": "world"
        }
        expr = SimpleExpression(
            value_1=P("a"),
            operator=ExpressionOperator.NOT_EQUALS,
            value_2=P("b")
        )
        assert expr.evaluate(data) is True

    def test_path_to_path_with_jmespath_filter(self):
        """Test path-to-path comparison using JMESPath filters."""
        data = {
            "steps": [
                {"type": "fetch", "session": "abc123"},
                {"type": "click", "session": "abc123"},
                {"type": "return", "session": "abc123"},
            ]
        }
        # Compare session of last fetch with session of last return
        expr = SimpleExpression(
            value_1=P("steps[?type == 'fetch'] | [-1].session"),
            operator=ExpressionOperator.EQUALS,
            value_2=P("steps[?type == 'return'] | [-1].session")
        )
        assert expr.evaluate(data) is True

        # Change the return session
        data["steps"][2]["session"] = "xyz789"
        assert expr.evaluate(data) is False

    def test_path_to_path_greater_than(self):
        """Test numeric comparison between two paths."""
        data = {
            "before": {"count": 10},
            "after": {"count": 20}
        }
        expr = SimpleExpression(
            value_1=P("after.count"),
            operator=ExpressionOperator.GREATER_THAN,
            value_2=P("before.count")
        )
        assert expr.evaluate(data) is True

    def test_path_to_path_missing_first_path(self):
        """Test when first path doesn't exist."""
        data = {"b": "value"}
        expr = SimpleExpression(
            value_1=P("a"),
            operator=ExpressionOperator.EQUALS,
            value_2=P("b")
        )
        assert expr.evaluate(data) is False

    def test_path_to_path_missing_second_path(self):
        """Test when second path doesn't exist."""
        data = {"a": "value"}
        expr = SimpleExpression(
            value_1=P("a"),
            operator=ExpressionOperator.EQUALS,
            value_2=P("b")
        )
        assert expr.evaluate(data) is False

    def test_path_to_path_both_missing(self):
        """Test when both paths don't exist."""
        data = {"c": "value"}
        expr = SimpleExpression(
            value_1=P("a"),
            operator=ExpressionOperator.EQUALS,
            value_2=P("b")
        )
        assert expr.evaluate(data) is False

    def test_path_reference_dict_format(self):
        """Test PathReference as dict (from JSON deserialization)."""
        data = {"x": 5, "y": 5}
        expr = SimpleExpression(
            value_1={"type": "path", "path": "x"},
            operator=ExpressionOperator.EQUALS,
            value_2={"type": "path", "path": "y"}
        )
        assert expr.evaluate(data) is True

    def test_literal_to_literal(self):
        """Test comparing two literal values (edge case)."""
        data = {"anything": "data"}
        expr = SimpleExpression(
            value_1=42,
            operator=ExpressionOperator.EQUALS,
            value_2=42
        )
        assert expr.evaluate(data) is True

        expr = SimpleExpression(
            value_1="hello",
            operator=ExpressionOperator.NOT_EQUALS,
            value_2="world"
        )
        assert expr.evaluate(data) is True

    def test_stringify_path_to_path(self):
        """Test stringify shows both paths."""
        expr = SimpleExpression(
            value_1=P("source.value"),
            operator=ExpressionOperator.EQUALS,
            value_2=P("target.value")
        )
        result = expr.stringify()
        assert "${source.value}" in result
        assert "${target.value}" in result
        assert "==" in result

    def test_stringify_path_to_literal(self):
        """Test stringify shows path and literal value."""
        expr = SimpleExpression(
            value_1=P("user.name"),
            operator=ExpressionOperator.EQUALS,
            value_2="John"
        )
        assert expr.stringify() == '${user.name} == "John"'

    def test_resolve_value_with_path_reference(self):
        """Test _resolve_value helper function."""
        data = {"nested": {"key": "value"}}

        # PathReference object
        exists, val = _resolve_value(PathReference(path="nested.key"), data)
        assert exists is True
        assert val == "value"

        # Dict format
        exists, val = _resolve_value({"type": "path", "path": "nested.key"}, data)
        assert exists is True
        assert val == "value"

        # Literal value
        exists, val = _resolve_value("literal", data)
        assert exists is True
        assert val == "literal"

        # Missing path
        exists, val = _resolve_value(PathReference(path="missing.path"), data)
        assert exists is False


class TestLLMTestResult:
    """Test cases for LLMTestResult model."""

    def test_passed_with_threshold_above(self):
        result = LLMTestResult(score=0.8, rationale="Good quality")
        assert result.passed(0.7) is True

    def test_passed_with_threshold_equal(self):
        result = LLMTestResult(score=0.7, rationale="Meets threshold")
        assert result.passed(0.7) is True

    def test_passed_with_threshold_below(self):
        result = LLMTestResult(score=0.5, rationale="Below threshold")
        assert result.passed(0.7) is False

    def test_passed_with_none_threshold(self):
        result = LLMTestResult(score=0.8, rationale="No threshold")
        assert result.passed(None) is None

    def test_default_values(self):
        result = LLMTestResult(score=0.5)
        assert result.rationale is None
        assert result.confidence is None

    def test_all_fields(self):
        result = LLMTestResult(
            score=0.85,
            rationale="The response was comprehensive",
            confidence=0.9
        )
        assert result.score == 0.85
        assert result.rationale == "The response was comprehensive"
        assert result.confidence == 0.9

    def test_serialization_roundtrip(self):
        original = LLMTestResult(score=0.75, rationale="Test", confidence=0.8)
        json_str = original.model_dump_json()
        restored = LLMTestResult.model_validate_json(json_str)
        assert restored.score == original.score
        assert restored.rationale == original.rationale
        assert restored.confidence == original.confidence


class TestLLMTest:
    """Test cases for LLMTest model."""

    def test_default_values(self):
        test = LLMTest(
            name="test",
            prompt="Evaluate this",
            model="gpt-4.1"
        )
        assert test.description == ""
        assert test.n_trials == 3
        assert test.score_range == (0.0, 1.0)
        assert test.passing_threshold is None
        assert test.aggregation == "mean"

    def test_all_fields(self):
        test = LLMTest(
            name="quality_check",
            description="Check response quality",
            prompt="Rate the quality of this response",
            model="gpt-4.1",
            n_trials=5,
            score_range=(1.0, 10.0),
            passing_threshold=7.0,
            aggregation="median"
        )
        assert test.name == "quality_check"
        assert test.description == "Check response quality"
        assert test.n_trials == 5
        assert test.score_range == (1.0, 10.0)
        assert test.passing_threshold == 7.0
        assert test.aggregation == "median"

    def test_serialization_roundtrip(self):
        original = LLMTest(
            name="test",
            prompt="Evaluate",
            model="gpt-4.1",
            n_trials=2,
            aggregation="max"
        )
        json_str = original.model_dump_json()
        restored = LLMTest.model_validate_json(json_str)
        assert restored.name == original.name
        assert restored.n_trials == original.n_trials
        assert restored.aggregation == original.aggregation

    def test_run_with_mean_aggregation(self):
        """Test run method with mean aggregation using mocked OpenAI client."""
        test = LLMTest(
            name="test",
            prompt="Evaluate the data",
            model="gpt-4.1",
            n_trials=3,
            aggregation="mean"
        )

        # Create mock client and responses
        mock_client = MagicMock()
        mock_responses = [
            MagicMock(output_parsed=LLMTestResult(score=0.7, rationale="Good", confidence=0.8)),
            MagicMock(output_parsed=LLMTestResult(score=0.8, rationale="Very good", confidence=0.9)),
            MagicMock(output_parsed=LLMTestResult(score=0.9, rationale="Excellent", confidence=0.85)),
        ]
        mock_client.responses.parse.side_effect = mock_responses

        result = test.run(data={"key": "value"}, client=mock_client)

        assert mock_client.responses.parse.call_count == 3
        assert result.score == pytest.approx(0.8, rel=1e-6)  # mean of 0.7, 0.8, 0.9
        assert result.rationale == "Very good"  # closest to mean
        assert result.confidence == pytest.approx(0.85, rel=1e-6)  # mean of confidences

    def test_run_with_median_aggregation(self):
        """Test run method with median aggregation."""
        test = LLMTest(
            name="test",
            prompt="Evaluate",
            model="gpt-4.1",
            n_trials=3,
            aggregation="median"
        )

        mock_client = MagicMock()
        mock_responses = [
            MagicMock(output_parsed=LLMTestResult(score=0.5, rationale="Low")),
            MagicMock(output_parsed=LLMTestResult(score=0.9, rationale="High")),
            MagicMock(output_parsed=LLMTestResult(score=0.7, rationale="Medium")),
        ]
        mock_client.responses.parse.side_effect = mock_responses

        result = test.run(data="test data", client=mock_client)

        assert result.score == 0.7  # median of 0.5, 0.7, 0.9
        assert result.rationale == "Medium"  # exactly at median

    def test_run_with_min_aggregation(self):
        """Test run method with min aggregation."""
        test = LLMTest(
            name="test",
            prompt="Evaluate",
            model="gpt-4.1",
            n_trials=3,
            aggregation="min"
        )

        mock_client = MagicMock()
        mock_responses = [
            MagicMock(output_parsed=LLMTestResult(score=0.6, rationale="Lowest")),
            MagicMock(output_parsed=LLMTestResult(score=0.8, rationale="Middle")),
            MagicMock(output_parsed=LLMTestResult(score=0.9, rationale="Highest")),
        ]
        mock_client.responses.parse.side_effect = mock_responses

        result = test.run(data="test", client=mock_client)

        assert result.score == 0.6
        assert result.rationale == "Lowest"

    def test_run_with_max_aggregation(self):
        """Test run method with max aggregation."""
        test = LLMTest(
            name="test",
            prompt="Evaluate",
            model="gpt-4.1",
            n_trials=3,
            aggregation="max"
        )

        mock_client = MagicMock()
        mock_responses = [
            MagicMock(output_parsed=LLMTestResult(score=0.6, rationale="Lowest")),
            MagicMock(output_parsed=LLMTestResult(score=0.8, rationale="Middle")),
            MagicMock(output_parsed=LLMTestResult(score=0.95, rationale="Highest")),
        ]
        mock_client.responses.parse.side_effect = mock_responses

        result = test.run(data="test", client=mock_client)

        assert result.score == 0.95
        assert result.rationale == "Highest"

    def test_run_single_trial(self):
        """Test run method with single trial."""
        test = LLMTest(
            name="test",
            prompt="Evaluate",
            model="gpt-4.1",
            n_trials=1
        )

        mock_client = MagicMock()
        mock_client.responses.parse.return_value = MagicMock(
            output_parsed=LLMTestResult(score=0.75, rationale="Single trial", confidence=0.9)
        )

        result = test.run(data="test", client=mock_client)

        assert mock_client.responses.parse.call_count == 1
        assert result.score == 0.75
        assert result.rationale == "Single trial"
        assert result.confidence == 0.9

    def test_run_without_confidence(self):
        """Test run method when no confidence is provided."""
        test = LLMTest(
            name="test",
            prompt="Evaluate",
            model="gpt-4.1",
            n_trials=2
        )

        mock_client = MagicMock()
        mock_responses = [
            MagicMock(output_parsed=LLMTestResult(score=0.6, rationale="No confidence")),
            MagicMock(output_parsed=LLMTestResult(score=0.8, rationale="Also no confidence")),
        ]
        mock_client.responses.parse.side_effect = mock_responses

        result = test.run(data="test", client=mock_client)

        assert result.confidence is None

    def test_run_prompt_includes_data_and_score_range(self):
        """Test that the prompt sent to the LLM includes data and score range."""
        test = LLMTest(
            name="test",
            prompt="Rate the quality",
            model="gpt-4.1",
            n_trials=1,
            score_range=(1.0, 5.0)
        )

        mock_client = MagicMock()
        mock_client.responses.parse.return_value = MagicMock(
            output_parsed=LLMTestResult(score=3.5, rationale="Average")
        )

        test.run(data="my test data", client=mock_client)

        # Verify the call was made with correct arguments
        call_args = mock_client.responses.parse.call_args
        assert call_args.kwargs["model"] == "gpt-4.1"

        input_messages = call_args.kwargs["input"]
        assert len(input_messages) == 1
        assert input_messages[0]["role"] == "user"

        content = input_messages[0]["content"]
        assert "Rate the quality" in content
        assert "my test data" in content
        assert "1.0" in content
        assert "5.0" in content

    def test_run_uses_correct_text_format(self):
        """Test that run uses LLMTestResult as text_format."""
        test = LLMTest(
            name="test",
            prompt="Evaluate",
            model="gpt-4.1",
            n_trials=1
        )

        mock_client = MagicMock()
        mock_client.responses.parse.return_value = MagicMock(
            output_parsed=LLMTestResult(score=0.5)
        )

        test.run(data="test", client=mock_client)

        call_args = mock_client.responses.parse.call_args
        assert call_args.kwargs["text_format"] == LLMTestResult
