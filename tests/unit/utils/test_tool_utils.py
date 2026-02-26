"""
tests/unit/test_tool_utils.py

Unit tests for LLM tool utilities.
"""

from pydantic import BaseModel

from bluebox.llms.tools.tool_utils import (
    extract_description_from_docstring,
    generate_parameters_schema,
)


class TestExtractDescriptionFromDocstring:
    """Tests for docstring description extraction."""

    def test_single_line_docstring(self) -> None:
        docstring = "This is a simple description."
        result = extract_description_from_docstring(docstring)
        assert result == "This is a simple description."

    def test_multiline_first_paragraph(self) -> None:
        docstring = """This is a description
        that spans multiple lines."""
        result = extract_description_from_docstring(docstring)
        assert result == "This is a description that spans multiple lines."

    def test_extracts_everything_before_args(self) -> None:
        docstring = """First paragraph here.

        Args:
            foo: Some argument.
        """
        result = extract_description_from_docstring(docstring)
        assert result == "First paragraph here."

    def test_extracts_multiple_paragraphs_before_args(self) -> None:
        docstring = """First paragraph here.

        Second paragraph with more details.

        Args:
            foo: Some argument.
        """
        result = extract_description_from_docstring(docstring)
        assert result == "First paragraph here. Second paragraph with more details."

    def test_none_docstring(self) -> None:
        result = extract_description_from_docstring(None)
        assert result == ""

    def test_empty_docstring(self) -> None:
        result = extract_description_from_docstring("")
        assert result == ""

    def test_strips_leading_whitespace(self) -> None:
        docstring = """
        Description with leading whitespace.
        """
        result = extract_description_from_docstring(docstring)
        assert result == "Description with leading whitespace."


class TestGenerateParametersSchema:
    """Tests for function parameter schema generation."""

    def test_simple_string_params(self) -> None:
        def example_func(name: str, value: str) -> None:
            pass

        schema = generate_parameters_schema(example_func)
        assert schema["type"] == "object"
        assert schema["required"] == ["name", "value"]
        assert schema["properties"]["name"]["type"] == "string"
        assert schema["properties"]["value"]["type"] == "string"

    def test_optional_params_not_required(self) -> None:
        def example_func(required_param: str, optional_param: str | None = None) -> None:
            pass

        schema = generate_parameters_schema(example_func)
        assert schema["required"] == ["required_param"]
        assert "optional_param" in schema["properties"]
        assert "optional_param" not in schema["required"]

    def test_list_type(self) -> None:
        def example_func(items: list[str]) -> None:
            pass

        schema = generate_parameters_schema(example_func)
        assert schema["properties"]["items"]["type"] == "array"
        assert schema["properties"]["items"]["items"]["type"] == "string"

    def test_dict_type(self) -> None:
        def example_func(data: dict[str, int]) -> None:
            pass

        schema = generate_parameters_schema(example_func)
        props = schema["properties"]["data"]
        assert props["type"] == "object"
        assert props["additionalProperties"]["type"] == "integer"

    def test_skips_self_parameter(self) -> None:
        class Example:
            def method(self, name: str) -> None:
                pass

        schema = generate_parameters_schema(Example.method)
        assert "self" not in schema["properties"]
        assert schema["required"] == ["name"]

    def test_nullable_type_uses_anyof(self) -> None:
        def example_func(value: str | None) -> None:
            pass

        schema = generate_parameters_schema(example_func)
        # pydantic represents str | None as anyOf
        assert "anyOf" in schema["properties"]["value"]

    def test_pydantic_model_defs_hoisted_to_root(self) -> None:
        """$defs from Pydantic model params should be at the schema root,
        not nested inside individual properties."""
        class Item(BaseModel):
            name: str
            value: int

        def example_func(items: list[Item]) -> None:
            pass

        schema = generate_parameters_schema(example_func)
        # $defs should be at root level
        assert "$defs" in schema
        assert "Item" in schema["$defs"]
        # $defs should NOT be inside the property
        assert "$defs" not in schema["properties"]["items"]
        # property should reference via $ref
        assert schema["properties"]["items"]["items"]["$ref"] == "#/$defs/Item"

    def test_nullable_pydantic_model_list_defs_hoisted(self) -> None:
        """list[PydanticModel] | None should have $defs at root, not in property."""
        class Config(BaseModel):
            key: str
            value: str

        def example_func(configs: list[Config] | None = None) -> None:
            pass

        schema = generate_parameters_schema(example_func)
        # $defs at root
        assert "$defs" in schema
        assert "Config" in schema["$defs"]
        # not inside the property
        assert "$defs" not in schema["properties"]["configs"]

    def test_multiple_pydantic_params_merge_defs(self) -> None:
        """$defs from multiple params should all be merged at the root."""
        class Alpha(BaseModel):
            x: int

        class Beta(BaseModel):
            y: str

        def example_func(a: list[Alpha], b: list[Beta]) -> None:
            pass

        schema = generate_parameters_schema(example_func)
        assert "$defs" in schema
        assert "Alpha" in schema["$defs"]
        assert "Beta" in schema["$defs"]

    def test_no_defs_when_no_pydantic_models(self) -> None:
        """Simple types should not produce $defs at the root."""
        def example_func(name: str, count: int) -> None:
            pass

        schema = generate_parameters_schema(example_func)
        assert "$defs" not in schema
