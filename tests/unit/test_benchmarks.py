"""
tests/unit/test_benchmarks.py

Unit tests for benchmarks data models and expression evaluation.
"""

import pytest
import re

from web_hacker.data_models.benchmarks import (
    ExpressionOperator,
    OPERATOR_SYMBOLS,
    _format_value,
    _get_value_at_path,
    SimpleExpression,
    CompositeExpression,
    DeterministicTest,
    evaluate_expression,
    stringify_expression,
)


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
    """Test cases for _get_value_at_path helper function."""

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
        exists, value = _get_value_at_path(data, "items.0")
        assert exists is True
        assert value == "a"

        exists, value = _get_value_at_path(data, "items.2")
        assert exists is True
        assert value == "c"

    def test_nested_array_access(self):
        data = {"users": [{"name": "Alice"}, {"name": "Bob"}]}
        exists, value = _get_value_at_path(data, "users.1.name")
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
        exists, value = _get_value_at_path(data, "items.5")
        assert exists is False

    def test_negative_index_not_supported(self):
        """Negative indices are not supported - they're treated as string keys."""
        data = {"items": ["a", "b", "c"]}
        exists, value = _get_value_at_path(data, "items.-1")
        assert exists is False

    def test_path_through_none_value(self):
        data = {"user": None}
        exists, value = _get_value_at_path(data, "user.name")
        assert exists is False

    def test_tuple_index_access(self):
        data = {"coords": (10, 20, 30)}
        exists, value = _get_value_at_path(data, "coords.1")
        assert exists is True
        assert value == 20

    def test_object_attribute_access(self):
        class Obj:
            name = "TestObject"

        exists, value = _get_value_at_path(Obj(), "name")
        assert exists is True
        assert value == "TestObject"

    def test_none_data_with_path(self):
        exists, value = _get_value_at_path(None, "key")
        assert exists is False

    def test_none_data_with_empty_path(self):
        exists, value = _get_value_at_path(None, "")
        assert exists is True
        assert value is None


class TestSimpleExpressionEvaluate:
    """Test cases for SimpleExpression.evaluate method."""

    # Existence operators
    def test_exists_when_path_exists(self):
        expr = SimpleExpression(path="name", operator=ExpressionOperator.EXISTS)
        assert expr.evaluate({"name": "John"}) is True

    def test_exists_when_path_missing(self):
        expr = SimpleExpression(path="age", operator=ExpressionOperator.EXISTS)
        assert expr.evaluate({"name": "John"}) is False

    def test_exists_with_none_value(self):
        """Path exists even if value is None."""
        expr = SimpleExpression(path="name", operator=ExpressionOperator.EXISTS)
        assert expr.evaluate({"name": None}) is True

    def test_not_exists_when_path_missing(self):
        expr = SimpleExpression(path="age", operator=ExpressionOperator.NOT_EXISTS)
        assert expr.evaluate({"name": "John"}) is True

    def test_not_exists_when_path_exists(self):
        expr = SimpleExpression(path="name", operator=ExpressionOperator.NOT_EXISTS)
        assert expr.evaluate({"name": "John"}) is False

    # Null checks
    def test_is_null_with_none_value(self):
        expr = SimpleExpression(path="value", operator=ExpressionOperator.IS_NULL)
        assert expr.evaluate({"value": None}) is True

    def test_is_null_with_non_none_value(self):
        expr = SimpleExpression(path="value", operator=ExpressionOperator.IS_NULL)
        assert expr.evaluate({"value": "something"}) is False

    def test_is_null_with_missing_path(self):
        expr = SimpleExpression(path="missing", operator=ExpressionOperator.IS_NULL)
        assert expr.evaluate({"value": "something"}) is False

    def test_is_not_null_with_value(self):
        expr = SimpleExpression(path="value", operator=ExpressionOperator.IS_NOT_NULL)
        assert expr.evaluate({"value": "something"}) is True

    def test_is_not_null_with_none(self):
        expr = SimpleExpression(path="value", operator=ExpressionOperator.IS_NOT_NULL)
        assert expr.evaluate({"value": None}) is False

    # Type checks
    def test_is_type_string(self):
        expr = SimpleExpression(path="val", operator=ExpressionOperator.IS_TYPE, value="string")
        assert expr.evaluate({"val": "hello"}) is True
        assert expr.evaluate({"val": 123}) is False

    def test_is_type_integer(self):
        expr = SimpleExpression(path="val", operator=ExpressionOperator.IS_TYPE, value="int")
        assert expr.evaluate({"val": 42}) is True
        assert expr.evaluate({"val": 3.14}) is False

    def test_is_type_number(self):
        """'number' type should match both int and float."""
        expr = SimpleExpression(path="val", operator=ExpressionOperator.IS_TYPE, value="number")
        assert expr.evaluate({"val": 42}) is True
        assert expr.evaluate({"val": 3.14}) is True
        assert expr.evaluate({"val": "42"}) is False

    def test_is_type_bool(self):
        expr = SimpleExpression(path="val", operator=ExpressionOperator.IS_TYPE, value="bool")
        assert expr.evaluate({"val": True}) is True
        assert expr.evaluate({"val": False}) is True
        assert expr.evaluate({"val": 1}) is False

    def test_is_type_list(self):
        expr = SimpleExpression(path="val", operator=ExpressionOperator.IS_TYPE, value="list")
        assert expr.evaluate({"val": [1, 2, 3]}) is True
        assert expr.evaluate({"val": (1, 2, 3)}) is False

    def test_is_type_dict(self):
        expr = SimpleExpression(path="val", operator=ExpressionOperator.IS_TYPE, value="dict")
        assert expr.evaluate({"val": {"key": "value"}}) is True
        assert expr.evaluate({"val": [1, 2]}) is False

    def test_is_type_null(self):
        expr = SimpleExpression(path="val", operator=ExpressionOperator.IS_TYPE, value="null")
        assert expr.evaluate({"val": None}) is True
        assert expr.evaluate({"val": ""}) is False

    def test_is_type_unknown_type(self):
        expr = SimpleExpression(path="val", operator=ExpressionOperator.IS_TYPE, value="unknown_type")
        assert expr.evaluate({"val": "hello"}) is False

    # Empty checks
    def test_is_empty_with_empty_string(self):
        expr = SimpleExpression(path="val", operator=ExpressionOperator.IS_EMPTY)
        assert expr.evaluate({"val": ""}) is True

    def test_is_empty_with_non_empty_string(self):
        expr = SimpleExpression(path="val", operator=ExpressionOperator.IS_EMPTY)
        assert expr.evaluate({"val": "hello"}) is False

    def test_is_empty_with_empty_list(self):
        expr = SimpleExpression(path="val", operator=ExpressionOperator.IS_EMPTY)
        assert expr.evaluate({"val": []}) is True

    def test_is_empty_with_non_empty_list(self):
        expr = SimpleExpression(path="val", operator=ExpressionOperator.IS_EMPTY)
        assert expr.evaluate({"val": [1, 2, 3]}) is False

    def test_is_empty_with_empty_dict(self):
        expr = SimpleExpression(path="val", operator=ExpressionOperator.IS_EMPTY)
        assert expr.evaluate({"val": {}}) is True

    def test_is_empty_with_none(self):
        expr = SimpleExpression(path="val", operator=ExpressionOperator.IS_EMPTY)
        assert expr.evaluate({"val": None}) is True

    def test_is_empty_with_number(self):
        """Numbers don't have length, so is_empty returns False."""
        expr = SimpleExpression(path="val", operator=ExpressionOperator.IS_EMPTY)
        assert expr.evaluate({"val": 0}) is False
        assert expr.evaluate({"val": 42}) is False

    def test_is_not_empty_with_value(self):
        expr = SimpleExpression(path="val", operator=ExpressionOperator.IS_NOT_EMPTY)
        assert expr.evaluate({"val": "hello"}) is True
        assert expr.evaluate({"val": [1]}) is True

    def test_is_not_empty_with_empty(self):
        expr = SimpleExpression(path="val", operator=ExpressionOperator.IS_NOT_EMPTY)
        assert expr.evaluate({"val": ""}) is False
        assert expr.evaluate({"val": []}) is False

    def test_is_not_empty_with_none(self):
        expr = SimpleExpression(path="val", operator=ExpressionOperator.IS_NOT_EMPTY)
        assert expr.evaluate({"val": None}) is False

    def test_is_not_empty_with_number(self):
        """Numbers without __len__ return True for is_not_empty."""
        expr = SimpleExpression(path="val", operator=ExpressionOperator.IS_NOT_EMPTY)
        assert expr.evaluate({"val": 42}) is True

    # Equality operators
    def test_equals_string(self):
        expr = SimpleExpression(path="name", operator=ExpressionOperator.EQUALS, value="John")
        assert expr.evaluate({"name": "John"}) is True
        assert expr.evaluate({"name": "Jane"}) is False

    def test_equals_number(self):
        expr = SimpleExpression(path="age", operator=ExpressionOperator.EQUALS, value=30)
        assert expr.evaluate({"age": 30}) is True
        assert expr.evaluate({"age": 25}) is False

    def test_equals_bool(self):
        expr = SimpleExpression(path="active", operator=ExpressionOperator.EQUALS, value=True)
        assert expr.evaluate({"active": True}) is True
        assert expr.evaluate({"active": False}) is False

    def test_equals_none(self):
        expr = SimpleExpression(path="val", operator=ExpressionOperator.EQUALS, value=None)
        assert expr.evaluate({"val": None}) is True
        assert expr.evaluate({"val": ""}) is False

    def test_equals_list(self):
        expr = SimpleExpression(path="items", operator=ExpressionOperator.EQUALS, value=[1, 2, 3])
        assert expr.evaluate({"items": [1, 2, 3]}) is True
        assert expr.evaluate({"items": [1, 2]}) is False

    def test_not_equals(self):
        expr = SimpleExpression(path="name", operator=ExpressionOperator.NOT_EQUALS, value="John")
        assert expr.evaluate({"name": "Jane"}) is True
        assert expr.evaluate({"name": "John"}) is False

    def test_equals_missing_path(self):
        expr = SimpleExpression(path="missing", operator=ExpressionOperator.EQUALS, value="value")
        assert expr.evaluate({"name": "John"}) is False

    # Containment operators
    def test_contains_string_in_string(self):
        expr = SimpleExpression(path="text", operator=ExpressionOperator.CONTAINS, value="world")
        assert expr.evaluate({"text": "hello world"}) is True
        assert expr.evaluate({"text": "hello"}) is False

    def test_contains_item_in_list(self):
        expr = SimpleExpression(path="items", operator=ExpressionOperator.CONTAINS, value=2)
        assert expr.evaluate({"items": [1, 2, 3]}) is True
        assert expr.evaluate({"items": [1, 3, 5]}) is False

    def test_contains_key_in_dict(self):
        expr = SimpleExpression(path="obj", operator=ExpressionOperator.CONTAINS, value="key")
        assert expr.evaluate({"obj": {"key": "value"}}) is True
        assert expr.evaluate({"obj": {"other": "value"}}) is False

    def test_contains_item_in_tuple(self):
        expr = SimpleExpression(path="coords", operator=ExpressionOperator.CONTAINS, value=10)
        assert expr.evaluate({"coords": (10, 20, 30)}) is True

    def test_contains_incompatible_types(self):
        """When actual is not a string/list/tuple/dict, returns False."""
        expr = SimpleExpression(path="val", operator=ExpressionOperator.CONTAINS, value="x")
        assert expr.evaluate({"val": 12345}) is False

    def test_not_contains_string(self):
        expr = SimpleExpression(path="text", operator=ExpressionOperator.NOT_CONTAINS, value="foo")
        assert expr.evaluate({"text": "hello world"}) is True
        assert expr.evaluate({"text": "foo bar"}) is False

    def test_not_contains_list(self):
        expr = SimpleExpression(path="items", operator=ExpressionOperator.NOT_CONTAINS, value=5)
        assert expr.evaluate({"items": [1, 2, 3]}) is True
        assert expr.evaluate({"items": [1, 5, 3]}) is False

    def test_not_contains_incompatible_types(self):
        """When actual is not a string/list/tuple/dict, returns True."""
        expr = SimpleExpression(path="val", operator=ExpressionOperator.NOT_CONTAINS, value="x")
        assert expr.evaluate({"val": 12345}) is True

    # Comparison operators
    def test_greater_than(self):
        expr = SimpleExpression(path="age", operator=ExpressionOperator.GREATER_THAN, value=18)
        assert expr.evaluate({"age": 25}) is True
        assert expr.evaluate({"age": 18}) is False
        assert expr.evaluate({"age": 10}) is False

    def test_greater_than_or_equal(self):
        expr = SimpleExpression(path="age", operator=ExpressionOperator.GREATER_THAN_OR_EQUAL, value=18)
        assert expr.evaluate({"age": 25}) is True
        assert expr.evaluate({"age": 18}) is True
        assert expr.evaluate({"age": 10}) is False

    def test_less_than(self):
        expr = SimpleExpression(path="count", operator=ExpressionOperator.LESS_THAN, value=10)
        assert expr.evaluate({"count": 5}) is True
        assert expr.evaluate({"count": 10}) is False
        assert expr.evaluate({"count": 15}) is False

    def test_less_than_or_equal(self):
        expr = SimpleExpression(path="count", operator=ExpressionOperator.LESS_THAN_OR_EQUAL, value=10)
        assert expr.evaluate({"count": 5}) is True
        assert expr.evaluate({"count": 10}) is True
        assert expr.evaluate({"count": 15}) is False

    def test_comparison_with_floats(self):
        expr = SimpleExpression(path="price", operator=ExpressionOperator.GREATER_THAN, value=9.99)
        assert expr.evaluate({"price": 10.50}) is True
        assert expr.evaluate({"price": 9.99}) is False

    def test_comparison_with_non_numeric(self):
        """Comparison with non-numeric values returns False."""
        expr = SimpleExpression(path="val", operator=ExpressionOperator.GREATER_THAN, value=10)
        assert expr.evaluate({"val": "not a number"}) is False

    def test_comparison_with_numeric_string(self):
        """Numeric strings are converted to float for comparison."""
        expr = SimpleExpression(path="val", operator=ExpressionOperator.GREATER_THAN, value=10)
        assert expr.evaluate({"val": "15"}) is True

    # String operators
    def test_starts_with(self):
        expr = SimpleExpression(path="text", operator=ExpressionOperator.STARTS_WITH, value="Hello")
        assert expr.evaluate({"text": "Hello World"}) is True
        assert expr.evaluate({"text": "World Hello"}) is False

    def test_starts_with_non_string(self):
        expr = SimpleExpression(path="val", operator=ExpressionOperator.STARTS_WITH, value="1")
        assert expr.evaluate({"val": 123}) is False

    def test_ends_with(self):
        expr = SimpleExpression(path="text", operator=ExpressionOperator.ENDS_WITH, value="World")
        assert expr.evaluate({"text": "Hello World"}) is True
        assert expr.evaluate({"text": "World Hello"}) is False

    def test_ends_with_non_string(self):
        expr = SimpleExpression(path="val", operator=ExpressionOperator.ENDS_WITH, value="3")
        assert expr.evaluate({"val": 123}) is False

    def test_matches_regex(self):
        expr = SimpleExpression(path="email", operator=ExpressionOperator.MATCHES_REGEX, value=r"^\w+@\w+\.\w+$")
        assert expr.evaluate({"email": "test@example.com"}) is True
        assert expr.evaluate({"email": "invalid-email"}) is False

    def test_matches_regex_partial_match(self):
        """Regex uses search, so partial matches work."""
        expr = SimpleExpression(path="text", operator=ExpressionOperator.MATCHES_REGEX, value=r"\d+")
        assert expr.evaluate({"text": "abc123def"}) is True

    def test_matches_regex_non_string(self):
        expr = SimpleExpression(path="val", operator=ExpressionOperator.MATCHES_REGEX, value=r"\d+")
        assert expr.evaluate({"val": 123}) is False

    def test_matches_regex_invalid_pattern(self):
        """Invalid regex pattern raises an exception."""
        expr = SimpleExpression(path="text", operator=ExpressionOperator.MATCHES_REGEX, value="[invalid")
        with pytest.raises(re.error):
            expr.evaluate({"text": "test"})

    # Length operators
    def test_length_equals_string(self):
        expr = SimpleExpression(path="text", operator=ExpressionOperator.LENGTH_EQUALS, value=5)
        assert expr.evaluate({"text": "hello"}) is True
        assert expr.evaluate({"text": "hi"}) is False

    def test_length_equals_list(self):
        expr = SimpleExpression(path="items", operator=ExpressionOperator.LENGTH_EQUALS, value=3)
        assert expr.evaluate({"items": [1, 2, 3]}) is True
        assert expr.evaluate({"items": [1, 2]}) is False

    def test_length_equals_dict(self):
        expr = SimpleExpression(path="obj", operator=ExpressionOperator.LENGTH_EQUALS, value=2)
        assert expr.evaluate({"obj": {"a": 1, "b": 2}}) is True

    def test_length_greater_than(self):
        expr = SimpleExpression(path="items", operator=ExpressionOperator.LENGTH_GREATER_THAN, value=2)
        assert expr.evaluate({"items": [1, 2, 3]}) is True
        assert expr.evaluate({"items": [1, 2]}) is False

    def test_length_less_than(self):
        expr = SimpleExpression(path="items", operator=ExpressionOperator.LENGTH_LESS_THAN, value=3)
        assert expr.evaluate({"items": [1, 2]}) is True
        assert expr.evaluate({"items": [1, 2, 3]}) is False

    def test_length_on_non_collection(self):
        """Length operators return False for values without __len__."""
        expr = SimpleExpression(path="val", operator=ExpressionOperator.LENGTH_EQUALS, value=3)
        assert expr.evaluate({"val": 123}) is False


class TestSimpleExpressionStringify:
    """Test cases for SimpleExpression.stringify method."""

    def test_stringify_equals(self):
        expr = SimpleExpression(path="user.name", operator=ExpressionOperator.EQUALS, value="John")
        assert expr.stringify() == 'user.name == "John"'

    def test_stringify_greater_than(self):
        expr = SimpleExpression(path="age", operator=ExpressionOperator.GREATER_THAN, value=18)
        assert expr.stringify() == "age > 18"

    def test_stringify_is_null(self):
        expr = SimpleExpression(path="value", operator=ExpressionOperator.IS_NULL)
        assert expr.stringify() == "value is null"

    def test_stringify_is_not_null(self):
        expr = SimpleExpression(path="value", operator=ExpressionOperator.IS_NOT_NULL)
        assert expr.stringify() == "value is not null"

    def test_stringify_is_empty(self):
        expr = SimpleExpression(path="items", operator=ExpressionOperator.IS_EMPTY)
        assert expr.stringify() == "items is empty"

    def test_stringify_exists(self):
        expr = SimpleExpression(path="optional_field", operator=ExpressionOperator.EXISTS)
        assert expr.stringify() == "optional_field exists"

    def test_stringify_contains(self):
        expr = SimpleExpression(path="text", operator=ExpressionOperator.CONTAINS, value="hello")
        assert expr.stringify() == 'text contains "hello"'

    def test_stringify_with_bool_value(self):
        expr = SimpleExpression(path="active", operator=ExpressionOperator.EQUALS, value=True)
        assert expr.stringify() == "active == true"

    def test_stringify_with_list_value(self):
        expr = SimpleExpression(path="items", operator=ExpressionOperator.EQUALS, value=[1, 2])
        assert expr.stringify() == "items == [1, 2]"


class TestCompositeExpression:
    """Test cases for CompositeExpression."""

    def test_and_all_true(self):
        expr = CompositeExpression(
            logic="and",
            expressions=[
                SimpleExpression(path="age", operator=ExpressionOperator.GREATER_THAN, value=18),
                SimpleExpression(path="verified", operator=ExpressionOperator.EQUALS, value=True),
            ]
        )
        assert expr.evaluate({"age": 25, "verified": True}) is True

    def test_and_one_false(self):
        expr = CompositeExpression(
            logic="and",
            expressions=[
                SimpleExpression(path="age", operator=ExpressionOperator.GREATER_THAN, value=18),
                SimpleExpression(path="verified", operator=ExpressionOperator.EQUALS, value=True),
            ]
        )
        assert expr.evaluate({"age": 25, "verified": False}) is False

    def test_and_all_false(self):
        expr = CompositeExpression(
            logic="and",
            expressions=[
                SimpleExpression(path="age", operator=ExpressionOperator.GREATER_THAN, value=18),
                SimpleExpression(path="verified", operator=ExpressionOperator.EQUALS, value=True),
            ]
        )
        assert expr.evaluate({"age": 10, "verified": False}) is False

    def test_or_one_true(self):
        expr = CompositeExpression(
            logic="or",
            expressions=[
                SimpleExpression(path="admin", operator=ExpressionOperator.EQUALS, value=True),
                SimpleExpression(path="moderator", operator=ExpressionOperator.EQUALS, value=True),
            ]
        )
        assert expr.evaluate({"admin": True, "moderator": False}) is True
        assert expr.evaluate({"admin": False, "moderator": True}) is True

    def test_or_all_false(self):
        expr = CompositeExpression(
            logic="or",
            expressions=[
                SimpleExpression(path="admin", operator=ExpressionOperator.EQUALS, value=True),
                SimpleExpression(path="moderator", operator=ExpressionOperator.EQUALS, value=True),
            ]
        )
        assert expr.evaluate({"admin": False, "moderator": False}) is False

    def test_or_all_true(self):
        expr = CompositeExpression(
            logic="or",
            expressions=[
                SimpleExpression(path="admin", operator=ExpressionOperator.EQUALS, value=True),
                SimpleExpression(path="moderator", operator=ExpressionOperator.EQUALS, value=True),
            ]
        )
        assert expr.evaluate({"admin": True, "moderator": True}) is True

    def test_nested_composite(self):
        """Test nested composite expressions."""
        expr = CompositeExpression(
            logic="and",
            expressions=[
                SimpleExpression(path="active", operator=ExpressionOperator.EQUALS, value=True),
                CompositeExpression(
                    logic="or",
                    expressions=[
                        SimpleExpression(path="role", operator=ExpressionOperator.EQUALS, value="admin"),
                        SimpleExpression(path="role", operator=ExpressionOperator.EQUALS, value="moderator"),
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
                SimpleExpression(path="age", operator=ExpressionOperator.GREATER_THAN, value=18),
                SimpleExpression(path="verified", operator=ExpressionOperator.EQUALS, value=True),
            ]
        )
        assert expr.stringify() == "(age > 18 AND verified == true)"

    def test_stringify_or(self):
        expr = CompositeExpression(
            logic="or",
            expressions=[
                SimpleExpression(path="admin", operator=ExpressionOperator.EQUALS, value=True),
                SimpleExpression(path="moderator", operator=ExpressionOperator.EQUALS, value=True),
            ]
        )
        assert expr.stringify() == "(admin == true OR moderator == true)"

    def test_stringify_nested(self):
        expr = CompositeExpression(
            logic="and",
            expressions=[
                SimpleExpression(path="active", operator=ExpressionOperator.EQUALS, value=True),
                CompositeExpression(
                    logic="or",
                    expressions=[
                        SimpleExpression(path="a", operator=ExpressionOperator.EQUALS, value=1),
                        SimpleExpression(path="b", operator=ExpressionOperator.EQUALS, value=2),
                    ]
                )
            ]
        )
        assert expr.stringify() == "(active == true AND (a == 1 OR b == 2))"


class TestDeterministicTest:
    """Test cases for DeterministicTest model."""

    def test_from_dict_simple_expression(self):
        data = {
            "name": "check_name",
            "description": "Check that name equals John",
            "expression": {
                "type": "simple",
                "path": "name",
                "operator": "equals",
                "value": "John"
            }
        }
        test = DeterministicTest.from_dict(data)
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
                    {"type": "simple", "path": "age", "operator": "greater_than", "value": 18},
                    {"type": "simple", "path": "verified", "operator": "equals", "value": True}
                ]
            }
        }
        test = DeterministicTest.from_dict(data)
        assert test.name == "check_user"
        assert isinstance(test.expression, CompositeExpression)

    def test_from_json(self):
        json_str = '{"name": "test", "expression": {"type": "simple", "path": "val", "operator": "equals", "value": 42}}'
        test = DeterministicTest.from_json(json_str)
        assert test.name == "test"
        assert test.expression.evaluate({"val": 42}) is True

    def test_default_description(self):
        data = {
            "name": "test",
            "expression": {"type": "simple", "path": "x", "operator": "exists"}
        }
        test = DeterministicTest.from_dict(data)
        assert test.description == ""


class TestEvaluateExpression:
    """Test cases for evaluate_expression helper function."""

    def test_evaluate_simple(self):
        expr = SimpleExpression(path="x", operator=ExpressionOperator.EQUALS, value=10)
        assert evaluate_expression(expr, {"x": 10}) is True

    def test_evaluate_composite(self):
        expr = CompositeExpression(
            logic="and",
            expressions=[
                SimpleExpression(path="a", operator=ExpressionOperator.EQUALS, value=1),
                SimpleExpression(path="b", operator=ExpressionOperator.EQUALS, value=2),
            ]
        )
        assert evaluate_expression(expr, {"a": 1, "b": 2}) is True


class TestStringifyExpression:
    """Test cases for stringify_expression helper function."""

    def test_stringify_simple(self):
        expr = SimpleExpression(path="x", operator=ExpressionOperator.EQUALS, value=10)
        assert stringify_expression(expr) == "x == 10"

    def test_stringify_composite(self):
        expr = CompositeExpression(
            logic="or",
            expressions=[
                SimpleExpression(path="a", operator=ExpressionOperator.EQUALS, value=1),
                SimpleExpression(path="b", operator=ExpressionOperator.EQUALS, value=2),
            ]
        )
        assert stringify_expression(expr) == "(a == 1 OR b == 2)"


class TestOperatorSymbols:
    """Test that all operators have symbols defined."""

    def test_all_operators_have_symbols(self):
        for op in ExpressionOperator:
            assert op.value in OPERATOR_SYMBOLS, f"Missing symbol for operator: {op.value}"


class TestEdgeCases:
    """Test edge cases and potential bugs."""

    def test_deep_nested_path(self):
        data = {"a": {"b": {"c": {"d": {"e": "deep"}}}}}
        expr = SimpleExpression(path="a.b.c.d.e", operator=ExpressionOperator.EQUALS, value="deep")
        assert expr.evaluate(data) is True

    def test_path_with_numeric_keys_in_dict(self):
        """Numeric path parts are treated as indices, not dict keys."""
        data = {"items": {"0": "string_key", "1": "another"}}
        # "0" is treated as index, not key, so this won't find the value
        exists, value = _get_value_at_path(data, "items.0")
        assert exists is False

    def test_expression_with_empty_string_value(self):
        expr = SimpleExpression(path="name", operator=ExpressionOperator.EQUALS, value="")
        assert expr.evaluate({"name": ""}) is True
        assert expr.evaluate({"name": "John"}) is False

    def test_expression_with_zero_value(self):
        expr = SimpleExpression(path="count", operator=ExpressionOperator.EQUALS, value=0)
        assert expr.evaluate({"count": 0}) is True
        assert expr.evaluate({"count": 1}) is False

    def test_expression_with_false_value(self):
        expr = SimpleExpression(path="active", operator=ExpressionOperator.EQUALS, value=False)
        assert expr.evaluate({"active": False}) is True
        assert expr.evaluate({"active": True}) is False

    def test_contains_empty_string(self):
        """Empty string is contained in any string."""
        expr = SimpleExpression(path="text", operator=ExpressionOperator.CONTAINS, value="")
        assert expr.evaluate({"text": "hello"}) is True
        assert expr.evaluate({"text": ""}) is True

    def test_length_equals_zero(self):
        expr = SimpleExpression(path="items", operator=ExpressionOperator.LENGTH_EQUALS, value=0)
        assert expr.evaluate({"items": []}) is True
        assert expr.evaluate({"items": [1]}) is False

    def test_regex_special_characters(self):
        expr = SimpleExpression(path="text", operator=ExpressionOperator.MATCHES_REGEX, value=r"\$\d+\.\d{2}")
        assert expr.evaluate({"text": "Price: $19.99"}) is True
        assert expr.evaluate({"text": "Price: 19.99"}) is False

    def test_comparison_with_none_value(self):
        """Comparison operators should handle None gracefully."""
        expr = SimpleExpression(path="val", operator=ExpressionOperator.GREATER_THAN, value=10)
        assert expr.evaluate({"val": None}) is False

    def test_array_of_objects(self):
        data = {
            "users": [
                {"name": "Alice", "age": 30},
                {"name": "Bob", "age": 25},
            ]
        }
        expr = SimpleExpression(path="users.0.name", operator=ExpressionOperator.EQUALS, value="Alice")
        assert expr.evaluate(data) is True

        expr = SimpleExpression(path="users.1.age", operator=ExpressionOperator.LESS_THAN, value=30)
        assert expr.evaluate(data) is True

    def test_type_check_case_insensitive(self):
        """Type check should be case-insensitive."""
        expr = SimpleExpression(path="val", operator=ExpressionOperator.IS_TYPE, value="STRING")
        assert expr.evaluate({"val": "hello"}) is True

        expr = SimpleExpression(path="val", operator=ExpressionOperator.IS_TYPE, value="String")
        assert expr.evaluate({"val": "hello"}) is True


class TestSerializationRoundtrip:
    """Test model_dump/model_dump_json and back."""

    def test_simple_expression_model_dump_roundtrip(self):
        original = SimpleExpression(
            path="user.name",
            operator=ExpressionOperator.EQUALS,
            value="John"
        )
        dumped = original.model_dump()
        restored = SimpleExpression.model_validate(dumped)

        assert restored.path == original.path
        assert restored.operator == original.operator
        assert restored.value == original.value
        assert restored.type == "simple"

    def test_simple_expression_model_dump_json_roundtrip(self):
        original = SimpleExpression(
            path="count",
            operator=ExpressionOperator.GREATER_THAN,
            value=10
        )
        json_str = original.model_dump_json()
        restored = SimpleExpression.model_validate_json(json_str)

        assert restored.path == original.path
        assert restored.operator == original.operator
        assert restored.value == original.value

    def test_simple_expression_with_none_value_roundtrip(self):
        original = SimpleExpression(
            path="optional",
            operator=ExpressionOperator.IS_NULL
        )
        dumped = original.model_dump()
        restored = SimpleExpression.model_validate(dumped)

        assert restored.value is None
        assert restored.operator == ExpressionOperator.IS_NULL

    def test_simple_expression_with_complex_value_roundtrip(self):
        original = SimpleExpression(
            path="data",
            operator=ExpressionOperator.EQUALS,
            value={"nested": [1, 2, {"key": "value"}]}
        )
        json_str = original.model_dump_json()
        restored = SimpleExpression.model_validate_json(json_str)

        assert restored.value == {"nested": [1, 2, {"key": "value"}]}

    def test_composite_expression_model_dump_roundtrip(self):
        original = CompositeExpression(
            logic="and",
            expressions=[
                SimpleExpression(path="age", operator=ExpressionOperator.GREATER_THAN, value=18),
                SimpleExpression(path="verified", operator=ExpressionOperator.EQUALS, value=True),
            ]
        )
        dumped = original.model_dump()
        restored = CompositeExpression.model_validate(dumped)

        assert restored.logic == original.logic
        assert restored.type == "composite"
        assert len(restored.expressions) == 2
        assert restored.expressions[0].path == "age"
        assert restored.expressions[1].path == "verified"

    def test_composite_expression_model_dump_json_roundtrip(self):
        original = CompositeExpression(
            logic="or",
            expressions=[
                SimpleExpression(path="admin", operator=ExpressionOperator.EQUALS, value=True),
                SimpleExpression(path="moderator", operator=ExpressionOperator.EQUALS, value=True),
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
                SimpleExpression(path="active", operator=ExpressionOperator.EQUALS, value=True),
                CompositeExpression(
                    logic="or",
                    expressions=[
                        SimpleExpression(path="role", operator=ExpressionOperator.EQUALS, value="admin"),
                        SimpleExpression(path="role", operator=ExpressionOperator.EQUALS, value="moderator"),
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
                path="user.active",
                operator=ExpressionOperator.EQUALS,
                value=True
            )
        )
        dumped = original.model_dump()
        restored = DeterministicTest.model_validate(dumped)

        assert restored.name == original.name
        assert restored.description == original.description
        assert isinstance(restored.expression, SimpleExpression)
        assert restored.expression.path == "user.active"

    def test_deterministic_test_model_dump_json_roundtrip(self):
        original = DeterministicTest(
            name="complex_test",
            description="Test with composite expression",
            expression=CompositeExpression(
                logic="and",
                expressions=[
                    SimpleExpression(path="age", operator=ExpressionOperator.GREATER_THAN_OR_EQUAL, value=18),
                    SimpleExpression(path="country", operator=ExpressionOperator.EQUALS, value="US"),
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
                SimpleExpression(path="user.age", operator=ExpressionOperator.GREATER_THAN, value=18),
                SimpleExpression(path="user.verified", operator=ExpressionOperator.EQUALS, value=True),
                CompositeExpression(
                    logic="or",
                    expressions=[
                        SimpleExpression(path="user.role", operator=ExpressionOperator.EQUALS, value="admin"),
                        SimpleExpression(path="user.premium", operator=ExpressionOperator.EQUALS, value=True),
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
            original = SimpleExpression(path="test", operator=op, value="test_value")
            json_str = original.model_dump_json()
            restored = SimpleExpression.model_validate_json(json_str)
            assert restored.operator == op, f"Failed for operator: {op}"

    def test_model_dump_structure(self):
        """Verify the structure of model_dump output."""
        expr = SimpleExpression(
            path="user.name",
            operator=ExpressionOperator.EQUALS,
            value="John"
        )
        dumped = expr.model_dump()

        assert "type" in dumped
        assert dumped["type"] == "simple"
        assert "path" in dumped
        assert dumped["path"] == "user.name"
        assert "operator" in dumped
        assert dumped["operator"] == "equals"
        assert "value" in dumped
        assert dumped["value"] == "John"

    def test_composite_model_dump_structure(self):
        """Verify the structure of composite model_dump output."""
        expr = CompositeExpression(
            logic="and",
            expressions=[
                SimpleExpression(path="a", operator=ExpressionOperator.EQUALS, value=1),
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
                                    SimpleExpression(path="a", operator=ExpressionOperator.EQUALS, value=1),
                                    SimpleExpression(path="b", operator=ExpressionOperator.EQUALS, value=2),
                                ]
                            ),
                            SimpleExpression(path="c", operator=ExpressionOperator.EQUALS, value=3),
                        ]
                    ),
                    SimpleExpression(path="d", operator=ExpressionOperator.EXISTS),
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
