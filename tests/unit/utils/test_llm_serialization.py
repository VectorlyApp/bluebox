"""
tests/unit/utils/test_llm_serialization.py

Tests for bluebox.utils.llm_serialization:
  - LLMExclude marker
  - _excluded_fields (cached introspection)
  - strip_llm_excluded (recursive stripping)
  - SerializedContentType / SerializedToolResult
  - serialize_tool_result
"""

import json
from datetime import datetime
from typing import Annotated, Optional

from pydantic import BaseModel, Field, computed_field

from bluebox.utils.llm_serialization import (
    LLMExclude,
    SerializedContentType,
    SerializedToolResult,
    _excluded_fields,
    serialize_tool_result,
    strip_llm_excluded,
)


# ---------------------------------------------------------------------------
# Test models — defined at module level so _excluded_fields cache works
# ---------------------------------------------------------------------------


class SimpleModel(BaseModel):
    visible: str
    hidden: Annotated[str, LLMExclude()]


class AllExcluded(BaseModel):
    a: Annotated[str, LLMExclude()]
    b: Annotated[int, LLMExclude()]


class NoExclusions(BaseModel):
    x: str
    y: int


class NestedOuter(BaseModel):
    name: str
    inner: SimpleModel
    secret: Annotated[str, LLMExclude()]


class WithOptional(BaseModel):
    required: str
    maybe: Annotated[Optional[str], LLMExclude()] = None


class WithDefault(BaseModel):
    keep: str
    drop: Annotated[str, LLMExclude()] = "default_val"


class WithFieldAndExclude(BaseModel):
    """LLMExclude combined with Pydantic Field metadata."""
    name: str
    internal_id: Annotated[int, Field(description="DB primary key"), LLMExclude()]


class Parent(BaseModel):
    children: list[SimpleModel]
    tag: Annotated[str, LLMExclude()]


class TupleContainer(BaseModel):
    items: tuple[SimpleModel, ...]
    removed: Annotated[str, LLMExclude()]


class WithComputedField(BaseModel):
    first: str
    last: str
    secret: Annotated[str, LLMExclude()]

    @computed_field
    @property
    def full_name(self) -> str:
        return f"{self.first} {self.last}"


class ComputedOnly(BaseModel):
    """Model where only computed fields exist alongside excluded regular fields."""
    raw: Annotated[str, LLMExclude()]

    @computed_field
    @property
    def derived(self) -> str:
        return self.raw.upper()


class NestedWithComputed(BaseModel):
    label: str
    item: WithComputedField


# =============================================================================
# LLMExclude marker
# =============================================================================


class TestLLMExclude:
    """Basic tests for the LLMExclude marker class."""

    def test_instantiable(self) -> None:
        marker = LLMExclude()
        assert isinstance(marker, LLMExclude)

    def test_stored_in_pydantic_metadata(self) -> None:
        info = SimpleModel.model_fields["hidden"]
        assert any(isinstance(m, LLMExclude) for m in info.metadata)

    def test_not_on_regular_field(self) -> None:
        info = SimpleModel.model_fields["visible"]
        assert not any(isinstance(m, LLMExclude) for m in info.metadata)


# =============================================================================
# _excluded_fields
# =============================================================================


class TestExcludedFields:
    """Tests for the cached _excluded_fields helper."""

    def test_returns_marked_fields(self) -> None:
        assert _excluded_fields(SimpleModel) == frozenset({"hidden"})

    def test_all_excluded(self) -> None:
        assert _excluded_fields(AllExcluded) == frozenset({"a", "b"})

    def test_none_excluded(self) -> None:
        assert _excluded_fields(NoExclusions) == frozenset()

    def test_nested_model_own_exclusions(self) -> None:
        # NestedOuter's own exclusions, not inner model's
        assert _excluded_fields(NestedOuter) == frozenset({"secret"})

    def test_with_optional_field(self) -> None:
        assert _excluded_fields(WithOptional) == frozenset({"maybe"})

    def test_with_default_field(self) -> None:
        assert _excluded_fields(WithDefault) == frozenset({"drop"})

    def test_combined_with_pydantic_field(self) -> None:
        assert _excluded_fields(WithFieldAndExclude) == frozenset({"internal_id"})

    def test_result_is_cached(self) -> None:
        result1 = _excluded_fields(SimpleModel)
        result2 = _excluded_fields(SimpleModel)
        assert result1 is result2  # same object from cache


# =============================================================================
# strip_llm_excluded — BaseModel inputs
# =============================================================================


class TestStripLLMExcludedModels:
    """Tests for strip_llm_excluded with BaseModel instances."""

    def test_simple_model(self) -> None:
        obj = SimpleModel(visible="yes", hidden="no")
        result = strip_llm_excluded(obj)
        assert result == {"visible": "yes"}
        assert "hidden" not in result

    def test_all_fields_excluded_returns_empty_dict(self) -> None:
        obj = AllExcluded(a="x", b=42)
        assert strip_llm_excluded(obj) == {}

    def test_no_exclusions_returns_all_fields(self) -> None:
        obj = NoExclusions(x="hello", y=99)
        result = strip_llm_excluded(obj)
        assert result == {"x": "hello", "y": 99}

    def test_nested_model_both_levels_stripped(self) -> None:
        inner = SimpleModel(visible="kept", hidden="dropped")
        outer = NestedOuter(name="test", inner=inner, secret="shh")
        result = strip_llm_excluded(outer)
        assert result == {
            "name": "test",
            "inner": {"visible": "kept"},
        }
        assert "secret" not in result
        assert "hidden" not in result["inner"]

    def test_optional_none_excluded(self) -> None:
        obj = WithOptional(required="yes")
        result = strip_llm_excluded(obj)
        assert result == {"required": "yes"}

    def test_optional_with_value_excluded(self) -> None:
        obj = WithOptional(required="yes", maybe="should vanish")
        result = strip_llm_excluded(obj)
        assert result == {"required": "yes"}

    def test_default_value_excluded(self) -> None:
        obj = WithDefault(keep="visible")
        result = strip_llm_excluded(obj)
        assert result == {"keep": "visible"}

    def test_field_with_pydantic_field_and_exclude(self) -> None:
        obj = WithFieldAndExclude(name="widget", internal_id=12345)
        result = strip_llm_excluded(obj)
        assert result == {"name": "widget"}

    def test_list_of_models_in_parent(self) -> None:
        children = [
            SimpleModel(visible="a", hidden="x"),
            SimpleModel(visible="b", hidden="y"),
        ]
        obj = Parent(children=children, tag="remove_me")
        result = strip_llm_excluded(obj)
        assert result == {
            "children": [{"visible": "a"}, {"visible": "b"}],
        }

    def test_tuple_of_models_in_parent(self) -> None:
        items = (
            SimpleModel(visible="a", hidden="x"),
            SimpleModel(visible="b", hidden="y"),
        )
        obj = TupleContainer(items=items, removed="gone")
        result = strip_llm_excluded(obj)
        assert result == {
            "items": ({"visible": "a"}, {"visible": "b"}),
        }

    def test_returns_dict_not_model(self) -> None:
        """strip_llm_excluded always converts BaseModel to dict."""
        obj = NoExclusions(x="a", y=1)
        result = strip_llm_excluded(obj)
        assert isinstance(result, dict)
        assert not isinstance(result, BaseModel)

    def test_computed_field_preserved(self) -> None:
        """@computed_field values are included in stripped output."""
        obj = WithComputedField(first="Jane", last="Doe", secret="shh")
        result = strip_llm_excluded(obj)
        assert result == {"first": "Jane", "last": "Doe", "full_name": "Jane Doe"}
        assert "secret" not in result

    def test_computed_field_with_all_regular_excluded(self) -> None:
        """Computed fields survive even when all regular fields are excluded."""
        obj = ComputedOnly(raw="hello")
        result = strip_llm_excluded(obj)
        assert result == {"derived": "HELLO"}
        assert "raw" not in result

    def test_nested_model_with_computed_field(self) -> None:
        """Nested model's computed fields are preserved recursively."""
        inner = WithComputedField(first="Jane", last="Doe", secret="x")
        obj = NestedWithComputed(label="test", item=inner)
        result = strip_llm_excluded(obj)
        assert result == {
            "label": "test",
            "item": {"first": "Jane", "last": "Doe", "full_name": "Jane Doe"},
        }


# =============================================================================
# strip_llm_excluded — dict / list / tuple / primitive inputs
# =============================================================================


class TestStripLLMExcludedContainers:
    """Tests for strip_llm_excluded with non-model containers."""

    def test_plain_dict_passthrough(self) -> None:
        d = {"a": 1, "b": "two"}
        assert strip_llm_excluded(d) == d

    def test_dict_with_model_value(self) -> None:
        obj = SimpleModel(visible="yes", hidden="no")
        result = strip_llm_excluded({"data": obj, "count": 1})
        assert result == {"data": {"visible": "yes"}, "count": 1}

    def test_dict_nested_dicts_with_models(self) -> None:
        obj = SimpleModel(visible="v", hidden="h")
        result = strip_llm_excluded({"level1": {"level2": obj}})
        assert result == {"level1": {"level2": {"visible": "v"}}}

    def test_list_of_models(self) -> None:
        models = [
            SimpleModel(visible="a", hidden="1"),
            SimpleModel(visible="b", hidden="2"),
        ]
        result = strip_llm_excluded(models)
        assert result == [{"visible": "a"}, {"visible": "b"}]

    def test_list_of_primitives_passthrough(self) -> None:
        lst = [1, "two", 3.0, None, True]
        assert strip_llm_excluded(lst) == lst

    def test_tuple_of_models(self) -> None:
        models = (
            SimpleModel(visible="x", hidden="y"),
        )
        result = strip_llm_excluded(models)
        assert result == ({"visible": "x"},)
        assert isinstance(result, tuple)

    def test_tuple_of_primitives_passthrough(self) -> None:
        t = (1, 2, 3)
        result = strip_llm_excluded(t)
        assert result == (1, 2, 3)
        assert isinstance(result, tuple)

    def test_empty_dict(self) -> None:
        assert strip_llm_excluded({}) == {}

    def test_empty_list(self) -> None:
        assert strip_llm_excluded([]) == []

    def test_empty_tuple(self) -> None:
        assert strip_llm_excluded(()) == ()

    def test_mixed_list(self) -> None:
        """List with models, dicts, and primitives."""
        obj = SimpleModel(visible="v", hidden="h")
        result = strip_llm_excluded([obj, {"plain": "dict"}, 42, None])
        assert result == [{"visible": "v"}, {"plain": "dict"}, 42, None]


# =============================================================================
# strip_llm_excluded — primitives
# =============================================================================


class TestStripLLMExcludedPrimitives:
    """Primitives pass through unchanged."""

    def test_string(self) -> None:
        assert strip_llm_excluded("hello") == "hello"

    def test_int(self) -> None:
        assert strip_llm_excluded(42) == 42

    def test_float(self) -> None:
        assert strip_llm_excluded(3.14) == 3.14

    def test_bool(self) -> None:
        assert strip_llm_excluded(True) is True

    def test_none(self) -> None:
        assert strip_llm_excluded(None) is None


# =============================================================================
# SerializedContentType
# =============================================================================


class TestSerializedContentType:
    """Tests for the SerializedContentType enum."""

    def test_json_value(self) -> None:
        assert SerializedContentType.JSON == "json"

    def test_text_value(self) -> None:
        assert SerializedContentType.TEXT == "text"

    def test_is_str_enum(self) -> None:
        assert isinstance(SerializedContentType.JSON, str)


# =============================================================================
# SerializedToolResult
# =============================================================================


class TestSerializedToolResult:
    """Tests for the SerializedToolResult named tuple."""

    def test_tuple_unpacking(self) -> None:
        r = SerializedToolResult(serialized="{}", content_type=SerializedContentType.JSON)
        s, ct = r
        assert s == "{}"
        assert ct == SerializedContentType.JSON

    def test_named_access(self) -> None:
        r = SerializedToolResult(serialized="hi", content_type=SerializedContentType.TEXT)
        assert r.serialized == "hi"
        assert r.content_type == SerializedContentType.TEXT

    def test_equality_with_plain_tuple(self) -> None:
        r = SerializedToolResult(serialized="x", content_type=SerializedContentType.JSON)
        assert r == ("x", "json")  # NamedTuple is tuple-compatible


# =============================================================================
# serialize_tool_result
# =============================================================================


class TestSerializeToolResult:
    """Tests for serialize_tool_result."""

    def test_dict_returns_json(self) -> None:
        result = serialize_tool_result({"key": "value", "count": 42})
        assert result.content_type == SerializedContentType.JSON
        assert json.loads(result.serialized) == {"key": "value", "count": 42}

    def test_list_returns_json(self) -> None:
        result = serialize_tool_result([1, 2, 3])
        assert result.content_type == SerializedContentType.JSON
        assert json.loads(result.serialized) == [1, 2, 3]

    def test_string_returns_json(self) -> None:
        result = serialize_tool_result("hello")
        assert result.content_type == SerializedContentType.JSON
        assert json.loads(result.serialized) == "hello"

    def test_nested_dict_returns_json(self) -> None:
        data = {"a": {"b": [1, 2]}, "c": None}
        result = serialize_tool_result(data)
        assert result.content_type == SerializedContentType.JSON
        assert json.loads(result.serialized) == data

    def test_none_returns_json(self) -> None:
        result = serialize_tool_result(None)
        assert result.content_type == SerializedContentType.JSON
        assert json.loads(result.serialized) is None

    def test_non_ascii_preserved(self) -> None:
        data = {"emoji": "\U0001f525", "text": "caf\u00e9"}
        result = serialize_tool_result(data)
        assert result.content_type == SerializedContentType.JSON
        assert "\U0001f525" in result.serialized
        assert "caf\u00e9" in result.serialized

    def test_datetime_uses_default_str(self) -> None:
        dt = datetime(2025, 1, 15, 12, 30, 0)
        result = serialize_tool_result({"timestamp": dt})
        assert result.content_type == SerializedContentType.JSON
        parsed = json.loads(result.serialized)
        assert "2025-01-15" in parsed["timestamp"]

    def test_custom_object_handled_by_default_str(self) -> None:
        class Widget:
            def __str__(self) -> str:
                return "Widget()"

        result = serialize_tool_result({"obj": Widget()})
        assert result.content_type == SerializedContentType.JSON
        assert "Widget()" in result.serialized

    def test_integer_returns_json(self) -> None:
        result = serialize_tool_result(42)
        assert result.content_type == SerializedContentType.JSON
        assert json.loads(result.serialized) == 42

    def test_boolean_returns_json(self) -> None:
        result = serialize_tool_result(True)
        assert result.content_type == SerializedContentType.JSON
        assert json.loads(result.serialized) is True

    def test_empty_dict_returns_json(self) -> None:
        result = serialize_tool_result({})
        assert result.content_type == SerializedContentType.JSON
        assert json.loads(result.serialized) == {}

    def test_result_is_named_tuple(self) -> None:
        result = serialize_tool_result({"a": 1})
        assert isinstance(result, SerializedToolResult)
        assert isinstance(result, tuple)

    def test_tuple_unpacking_still_works(self) -> None:
        """Backwards compat: callers that do `s, ct = serialize_tool_result(...)` still work."""
        serialized, content_type = serialize_tool_result({"x": 1})
        assert isinstance(serialized, str)
        assert content_type == "json"

    def test_output_is_indented(self) -> None:
        result = serialize_tool_result({"a": 1})
        assert "\n" in result.serialized  # indent=2 produces newlines

    def test_large_payload(self) -> None:
        data = {f"key_{i}": f"value_{i}" for i in range(500)}
        result = serialize_tool_result(data)
        assert result.content_type == SerializedContentType.JSON
        assert json.loads(result.serialized) == data
