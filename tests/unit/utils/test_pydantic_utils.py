"""
tests/unit/utils/test_pydantic_utils.py

Unit tests for Pydantic markdown schema utilities.
"""

from enum import StrEnum
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field

from bluebox.utils.pydantic_utils import (
    format_default,
    format_field_type,
    format_model_fields,
)


# ---------------------------------------------------------------------------
# Test fixtures — tiny models / enums used across tests
# ---------------------------------------------------------------------------

class Color(StrEnum):
    RED = "red"
    GREEN = "green"
    BLUE = "blue"


class SingleValueEnum(StrEnum):
    ONLY = "only"


class HTTPMethod(StrEnum):
    GET = "GET"
    POST = "POST"
    DELETE = "DELETE"


class NestedModel(BaseModel):
    value: int


class SimpleModel(BaseModel):
    name: str
    age: int
    active: bool = True


class AllOptionalModel(BaseModel):
    label: str | None = None
    count: int = 0
    flag: bool = False
    score: float = 3.14


class MixedModel(BaseModel):
    """Model with required, optional, enum, nested, and list fields."""
    id: str
    method: HTTPMethod
    color: Color = Color.RED
    tags: list[str] = Field(default_factory=list)
    nested: NestedModel | None = None
    metadata: dict[str, Any] | None = None


class DefaultFactoryModel(BaseModel):
    items: list[int] = Field(default_factory=list)
    mapping: dict[str, str] = Field(default_factory=dict)


class LiteralModel(BaseModel):
    direction: Literal["up", "down"]
    mode: Literal["fast", "slow"] = "fast"


# ---------------------------------------------------------------------------
# format_field_type
# ---------------------------------------------------------------------------

class TestFormatFieldType:
    """Tests for format_field_type()."""

    # -- Basic types --

    def test_str(self) -> None:
        assert format_field_type(str) == "str"

    def test_int(self) -> None:
        assert format_field_type(int) == "int"

    def test_float(self) -> None:
        assert format_field_type(float) == "float"

    def test_bool(self) -> None:
        assert format_field_type(bool) == "bool"

    def test_bytes(self) -> None:
        assert format_field_type(bytes) == "bytes"

    # -- None annotation --

    def test_none_annotation(self) -> None:
        assert format_field_type(None) == "any"

    # -- Optional / Union with None (pipe syntax) --

    def test_optional_str_pipe(self) -> None:
        result = format_field_type(str | None)
        assert result == "str?"

    def test_optional_int_pipe(self) -> None:
        result = format_field_type(int | None)
        assert result == "int?"

    def test_optional_float_pipe(self) -> None:
        result = format_field_type(float | None)
        assert result == "float?"

    # -- Optional (typing.Optional syntax) --

    def test_optional_str_typing(self) -> None:
        result = format_field_type(Optional[str])
        assert result == "str?"

    def test_optional_int_typing(self) -> None:
        result = format_field_type(Optional[int])
        assert result == "int?"

    # -- Multi-type Union --

    def test_union_str_int(self) -> None:
        result = format_field_type(str | int)
        assert "str" in result
        assert "int" in result
        assert "?" not in result  # no None in the union

    def test_union_str_int_none(self) -> None:
        result = format_field_type(str | int | None)
        assert "str" in result
        assert "int" in result
        assert result.endswith("?")

    def test_union_typing_syntax(self) -> None:
        result = format_field_type(Union[str, int, float])
        assert "str" in result
        assert "int" in result
        assert "float" in result

    # -- Literal --

    def test_literal_strings(self) -> None:
        result = format_field_type(Literal["a", "b", "c"])
        assert result == '"a" | "b" | "c"'

    def test_literal_single_string(self) -> None:
        result = format_field_type(Literal["only"])
        assert result == '"only"'

    def test_literal_ints(self) -> None:
        result = format_field_type(Literal[1, 2, 3])
        assert result == "1 | 2 | 3"

    def test_literal_mixed(self) -> None:
        result = format_field_type(Literal["auto", 0])
        assert '"auto"' in result
        assert "0" in result

    # -- StrEnum --

    def test_strenum(self) -> None:
        result = format_field_type(Color)
        assert result == '"red" | "green" | "blue"'

    def test_strenum_single_value(self) -> None:
        result = format_field_type(SingleValueEnum)
        assert result == '"only"'

    def test_strenum_http_method(self) -> None:
        result = format_field_type(HTTPMethod)
        assert '"GET"' in result
        assert '"POST"' in result
        assert '"DELETE"' in result

    # -- dict --

    def test_plain_dict(self) -> None:
        result = format_field_type(dict)
        assert result == "dict"

    def test_typed_dict(self) -> None:
        result = format_field_type(dict[str, Any])
        assert result == "dict"

    def test_typed_dict_specific(self) -> None:
        result = format_field_type(dict[str, int])
        assert result == "dict"

    # -- list --

    def test_plain_list(self) -> None:
        result = format_field_type(list)
        assert result == "list"

    def test_typed_list_str(self) -> None:
        result = format_field_type(list[str])
        assert result == "list[str]"

    def test_typed_list_int(self) -> None:
        result = format_field_type(list[int])
        assert result == "list[int]"

    def test_list_of_optional(self) -> None:
        # list[str | None] — the inner type should show "str?"
        result = format_field_type(list[str | None])
        assert result == "list[str?]"

    # -- BaseModel subclass --

    def test_basemodel_subclass(self) -> None:
        result = format_field_type(NestedModel)
        assert result == "NestedModel"

    def test_basemodel_base(self) -> None:
        result = format_field_type(BaseModel)
        assert result == "BaseModel"

    # -- Optional BaseModel --

    def test_optional_model(self) -> None:
        result = format_field_type(NestedModel | None)
        assert result == "NestedModel?"

    # -- Optional dict --

    def test_optional_dict(self) -> None:
        result = format_field_type(dict[str, Any] | None)
        assert result == "dict?"

    # -- Literal with enum values --

    def test_literal_with_enum_value(self) -> None:
        """Literal containing an enum member should use .value."""
        result = format_field_type(Literal[Color.RED])
        assert result == '"red"'


# ---------------------------------------------------------------------------
# format_default
# ---------------------------------------------------------------------------

class TestFormatDefault:
    """Tests for format_default()."""

    # -- None --

    def test_none(self) -> None:
        assert format_default(None) == "null"

    # -- bool (must be checked before int since bool is subclass of int) --

    def test_true(self) -> None:
        assert format_default(True) == "true"

    def test_false(self) -> None:
        assert format_default(False) == "false"

    # -- int --

    def test_int_zero(self) -> None:
        assert format_default(0) == "0"

    def test_int_positive(self) -> None:
        assert format_default(42) == "42"

    def test_int_negative(self) -> None:
        assert format_default(-1) == "-1"

    # -- float --

    def test_float(self) -> None:
        assert format_default(3.14) == "3.14"

    def test_float_whole(self) -> None:
        assert format_default(5.0) == "5.0"

    # -- enum member --

    def test_enum_member(self) -> None:
        assert format_default(Color.RED) == '"red"'

    def test_enum_member_http(self) -> None:
        assert format_default(HTTPMethod.GET) == '"GET"'

    # -- str --

    def test_str(self) -> None:
        assert format_default("hello") == '"hello"'

    def test_str_empty(self) -> None:
        assert format_default("") == '""'

    def test_str_with_special_chars(self) -> None:
        assert format_default("same-origin") == '"same-origin"'

    def test_str_wildcard(self) -> None:
        assert format_default("*") == '"*"'

    # -- list --

    def test_empty_list(self) -> None:
        assert format_default([]) == "[]"

    def test_nonempty_list(self) -> None:
        # Non-empty lists also show "[]" — format_default just checks isinstance
        assert format_default([1, 2, 3]) == "[]"

    # -- fallback --

    def test_dict_fallback(self) -> None:
        result = format_default({"key": "val"})
        assert result == repr({"key": "val"})

    def test_tuple_fallback(self) -> None:
        result = format_default((1, 2))
        assert result == repr((1, 2))

    # -- bool vs int ordering --

    def test_bool_not_treated_as_int(self) -> None:
        """bool is a subclass of int — ensure True != '1'."""
        assert format_default(True) == "true"
        assert format_default(True) != "1"


# ---------------------------------------------------------------------------
# format_model_fields
# ---------------------------------------------------------------------------

class TestFormatModelFields:
    """Tests for format_model_fields()."""

    # -- Basic required / optional --

    def test_simple_model_all_fields(self) -> None:
        lines = format_model_fields(SimpleModel)
        assert "- name: str (required)" in lines
        assert "- age: int (required)" in lines
        assert "- active: bool = true" in lines

    def test_required_fields_say_required(self) -> None:
        lines = format_model_fields(SimpleModel)
        required_lines = [l for l in lines if "(required)" in l]
        assert len(required_lines) == 2  # name and age

    def test_optional_fields_show_default(self) -> None:
        lines = format_model_fields(AllOptionalModel)
        assert "- label: str? = null" in lines
        assert "- count: int = 0" in lines
        assert "- flag: bool = false" in lines
        assert "- score: float = 3.14" in lines

    def test_no_required_in_all_optional(self) -> None:
        lines = format_model_fields(AllOptionalModel)
        required_lines = [l for l in lines if "(required)" in l]
        assert len(required_lines) == 0

    # -- skip_fields --

    def test_skip_fields(self) -> None:
        lines = format_model_fields(SimpleModel, skip_fields={"age"})
        field_names = [l.split(":")[0].strip("- ") for l in lines]
        assert "age" not in field_names
        assert "name" in field_names
        assert "active" in field_names

    def test_skip_multiple_fields(self) -> None:
        lines = format_model_fields(SimpleModel, skip_fields={"name", "active"})
        assert len(lines) == 1
        assert "age" in lines[0]

    def test_skip_all_fields(self) -> None:
        lines = format_model_fields(SimpleModel, skip_fields={"name", "age", "active"})
        assert lines == []

    def test_skip_nonexistent_field(self) -> None:
        """Skipping a field that doesn't exist should be a no-op."""
        lines_with_skip = format_model_fields(SimpleModel, skip_fields={"nonexistent"})
        lines_without_skip = format_model_fields(SimpleModel)
        assert lines_with_skip == lines_without_skip

    def test_skip_fields_none(self) -> None:
        """Passing None for skip_fields should be same as no skipping."""
        lines = format_model_fields(SimpleModel, skip_fields=None)
        assert len(lines) == 3

    # -- Enum fields --

    def test_enum_field_with_default(self) -> None:
        lines = format_model_fields(MixedModel)
        color_line = [l for l in lines if "color:" in l][0]
        assert '"red" | "green" | "blue"' in color_line
        assert '= "red"' in color_line

    def test_enum_field_required(self) -> None:
        lines = format_model_fields(MixedModel)
        method_line = [l for l in lines if "method:" in l][0]
        assert '"GET" | "POST" | "DELETE"' in method_line
        assert "(required)" in method_line

    # -- default_factory --

    def test_default_factory_list(self) -> None:
        lines = format_model_fields(DefaultFactoryModel)
        items_line = [l for l in lines if "items:" in l][0]
        assert "= []" in items_line

    def test_default_factory_dict(self) -> None:
        lines = format_model_fields(DefaultFactoryModel)
        mapping_line = [l for l in lines if "mapping:" in l][0]
        # dict default_factory returns {}, format_default falls through to repr
        assert "= " in mapping_line

    def test_field_with_default_factory_not_required(self) -> None:
        lines = format_model_fields(DefaultFactoryModel)
        required_lines = [l for l in lines if "(required)" in l]
        assert len(required_lines) == 0

    # -- Nested model / optional dict --

    def test_nested_model_optional(self) -> None:
        lines = format_model_fields(MixedModel)
        nested_line = [l for l in lines if "nested:" in l][0]
        assert "NestedModel?" in nested_line
        assert "= null" in nested_line

    def test_optional_dict_field(self) -> None:
        lines = format_model_fields(MixedModel)
        meta_line = [l for l in lines if "metadata:" in l][0]
        assert "dict?" in meta_line
        assert "= null" in meta_line

    # -- Literal fields --

    def test_literal_required(self) -> None:
        lines = format_model_fields(LiteralModel)
        dir_line = [l for l in lines if "direction:" in l][0]
        assert '"up" | "down"' in dir_line
        assert "(required)" in dir_line

    def test_literal_with_default(self) -> None:
        lines = format_model_fields(LiteralModel)
        mode_line = [l for l in lines if "mode:" in l][0]
        assert '"fast" | "slow"' in mode_line
        assert '= "fast"' in mode_line

    # -- Field ordering --

    def test_field_order_preserved(self) -> None:
        """Fields should appear in model definition order."""
        lines = format_model_fields(MixedModel)
        field_names = [l.split(":")[0].strip("- ") for l in lines]
        assert field_names == ["id", "method", "color", "tags", "nested", "metadata"]

    # -- All lines have markdown bullet prefix --

    def test_all_lines_start_with_dash(self) -> None:
        lines = format_model_fields(MixedModel)
        for line in lines:
            assert line.startswith("- "), f"Line missing '- ' prefix: {line}"

    # -- Empty model --

    def test_empty_model(self) -> None:
        class EmptyModel(BaseModel):
            pass

        lines = format_model_fields(EmptyModel)
        assert lines == []
