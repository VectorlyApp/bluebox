"""
tests/unit/test_tool_utils.py

Unit tests for LLM tool utilities.
"""

from web_hacker.llms.tools.tool_utils import (
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
