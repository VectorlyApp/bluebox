"""
tests/unit/utils/test_llm_utils.py

Unit tests for LLM utility functions.
"""

from toon import encode

from bluebox.utils.llm_utils import token_optimized


class TestTokenOptimized:
    """Test cases for the token_optimized decorator."""

    def test_returns_string_not_dict(self) -> None:
        """Decorator returns encoded string, not original dict."""
        @token_optimized
        def get_data() -> dict:
            return {"name": "test", "value": 123}

        result = get_data()

        assert isinstance(result, str)
        assert not isinstance(result, dict)

    def test_output_matches_toon_encode(self) -> None:
        """Decorator output matches direct toon.encode() call."""
        data = {"name": "test", "value": 123}

        @token_optimized
        def get_data() -> dict:
            return data

        result = get_data()
        expected = encode(data)

        assert result == expected

    def test_encoded_format_simple(self) -> None:
        """Show actual toon encoded format for simple dict."""
        @token_optimized
        def get_simple() -> dict:
            return {"status": "ok", "count": 5}

        result = get_simple()

        # toon uses compact notation - verify it's shorter than JSON
        import json
        json_str = json.dumps({"status": "ok", "count": 5})
        assert len(result) <= len(json_str)
        # Verify the encoded string contains the actual values
        assert "ok" in result
        assert "5" in result

    def test_encoded_format_nested(self) -> None:
        """Show actual toon encoded format for nested dict."""
        @token_optimized
        def get_nested() -> dict:
            return {"user": {"id": 1, "name": "Alice"}, "active": True}

        result = get_nested()

        # Verify key content is present in encoded form
        assert "Alice" in result
        assert "1" in result

    def test_preserves_function_metadata(self) -> None:
        """Decorator preserves __name__ and __doc__ via @wraps."""
        @token_optimized
        def my_tool() -> dict:
            """Tool docstring."""
            return {"x": 1}

        assert my_tool.__name__ == "my_tool"
        assert my_tool.__doc__ == "Tool docstring."

    def test_passes_arguments_through(self) -> None:
        """Decorator passes args and kwargs to wrapped function."""
        @token_optimized
        def compute(a: int, b: int, multiplier: int = 1) -> dict:
            return {"result": (a + b) * multiplier}

        result = compute(3, 4, multiplier=2)
        expected = encode({"result": 14})

        assert result == expected

    def test_works_on_instance_method(self) -> None:
        """Decorator works on class methods (common use case)."""
        class ToolHandler:
            def __init__(self, prefix: str) -> None:
                self.prefix = prefix

            @token_optimized
            def process(self, value: str) -> dict:
                return {"output": f"{self.prefix}_{value}"}

        handler = ToolHandler(prefix="test")
        result = handler.process("data")
        expected = encode({"output": "test_data"})

        assert result == expected

    def test_empty_dict(self) -> None:
        """Decorator handles empty dict."""
        @token_optimized
        def get_empty() -> dict:
            return {}

        result = get_empty()
        expected = encode({})

        assert result == expected

    def test_list_values(self) -> None:
        """Decorator handles dicts with list values."""
        @token_optimized
        def get_list() -> dict:
            return {"items": [1, 2, 3]}

        result = get_list()
        expected = encode({"items": [1, 2, 3]})

        assert result == expected

    def test_none_values(self) -> None:
        """Decorator handles None values in dict."""
        @token_optimized
        def get_nullable() -> dict:
            return {"data": None}

        result = get_nullable()
        expected = encode({"data": None})

        assert result == expected


class TestToonEncodedFormat:
    """Tests showing exact toon encoded output format."""

    def test_simple_dict_encoded_format(self) -> None:
        """Simple dict encodes to YAML-like key: value format."""
        @token_optimized
        def get_status() -> dict:
            return {"status": "ok", "count": 5}

        result = get_status()

        # toon produces YAML-like output with newline-separated key: value pairs
        assert result == "status: ok\ncount: 5"

    def test_nested_dict_encoded_format(self) -> None:
        """Nested dict encodes with indentation."""
        @token_optimized
        def get_user() -> dict:
            return {"user": {"id": 1, "name": "Bob"}, "active": True}

        result = get_user()

        # Nested structures use indentation, booleans become lowercase
        expected = "user:\n  id: 1\n  name: Bob\nactive: true"
        assert result == expected

    def test_list_in_dict_encoded_format(self) -> None:
        """Lists encode inline as [item,item,item]."""
        @token_optimized
        def get_items() -> dict:
            return {"items": [1, 2, 3], "total": 3}

        result = get_items()

        # Lists are inline comma-separated, no spaces
        assert result == "items: [1,2,3]\ntotal: 3"
