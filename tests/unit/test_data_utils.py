"""
tests/unit/test_data_utils.py

Unit tests for data utility functions.
"""

import datetime
from decimal import Decimal
from pathlib import Path
import logging
import json
import pytest

from bluebox.utils.data_utils import (
    assert_balanced_js_delimiters,
    convert_decimals_to_floats,
    convert_floats_to_decimals,
    load_data,
    serialize_datetime,
    resolve_dotted_path,
    get_text_from_html,
    apply_params_to_str,
    apply_params_to_json,
)
from bluebox.utils.exceptions import UnsupportedFileFormat


class TestGetTextFromHtml:
    """Test cases for get_text_from_html function."""

    def test_basic_html_with_text(self):
        """Test basic HTML extraction with visible text."""
        html = "<html><body><p>Hello World</p></body></html>"
        result = get_text_from_html(html)
        assert result == "Hello World"

    def test_multiple_paragraphs(self):
        """Test HTML with multiple paragraphs."""
        html = "<html><body><p>First paragraph</p><p>Second paragraph</p></body></html>"
        result = get_text_from_html(html)
        assert result == "First paragraph\nSecond paragraph"

    def test_removes_script_tags(self):
        """Test that script tags are removed."""
        html = """
        <html>
            <body>
                <p>Visible text</p>
                <script>console.log('hidden');</script>
                <p>More visible text</p>
            </body>
        </html>
        """
        result = get_text_from_html(html)
        assert result == "Visible text\nMore visible text"

    def test_removes_style_tags(self):
        """Test that style tags are removed."""
        html = """
        <html>
            <head>
                <style>
                    body { color: red; }
                </style>
            </head>
            <body>
                <p>Visible text</p>
            </body>
        </html>
        """
        result = get_text_from_html(html)
        assert result == "Visible text"

    def test_removes_noscript_tags(self):
        """Test that noscript tags are removed."""
        html = """
        <html>
            <body>
                <p>Visible text</p>
                <noscript>Please enable JavaScript</noscript>
                <p>More text</p>
            </body>
        </html>
        """
        result = get_text_from_html(html)
        assert result == "Visible text\nMore text"

    def test_removes_all_non_visible_tags(self):
        """Test that script, style, and noscript are all removed."""
        html = """
        <html>
            <head>
                <style>body { margin: 0; }</style>
                <script>var x = 1;</script>
            </head>
            <body>
                <p>Content</p>
                <noscript>No JS</noscript>
            </body>
        </html>
        """
        result = get_text_from_html(html)
        assert result == "Content"

    def test_normalizes_whitespace(self):
        """Test that extra whitespace is normalized."""
        html = "<html><body><p>Text    with    multiple    spaces</p></body></html>"
        result = get_text_from_html(html)
        # BeautifulSoup converts multiple spaces to newlines, which are then normalized
        assert result == "Text\nwith\nmultiple\nspaces"

    def test_normalizes_consecutive_newlines(self):
        """Test that consecutive newlines are normalized to single newline."""
        html = """
        <html>
            <body>
                <p>First</p>
                
                
                <p>Second</p>
            </body>
        </html>
        """
        result = get_text_from_html(html)
        # Should have single newline between First and Second
        assert result == "First\nSecond"

    def test_handles_carriage_return_line_endings(self):
        """Test that \\r\\n line endings are handled correctly."""
        html = "<html><body><p>Line 1</p>\r\n\r\n<p>Line 2</p></body></html>"
        result = get_text_from_html(html)
        assert result == "Line 1\nLine 2"

    def test_removes_leading_trailing_whitespace(self):
        """Test that leading and trailing whitespace is removed."""
        html = """
        
        
        <html>
            <body>
                <p>Content</p>
            </body>
        </html>
        
        
        """
        result = get_text_from_html(html)
        assert result == "Content"

    def test_empty_html(self):
        """Test empty HTML string."""
        html = ""
        result = get_text_from_html(html)
        assert result == ""

    def test_html_with_only_non_visible_elements(self):
        """Test HTML with only script/style/noscript tags."""
        html = """
        <html>
            <head>
                <style>body { color: red; }</style>
                <script>console.log('test');</script>
            </head>
            <body>
                <noscript>No JS</noscript>
            </body>
        </html>
        """
        result = get_text_from_html(html)
        assert result == ""  # Should be empty after removing all non-visible elements

    def test_nested_elements(self):
        """Test HTML with nested elements."""
        html = """
        <html>
            <body>
                <div>
                    <h1>Title</h1>
                    <p>Paragraph with <strong>bold</strong> text</p>
                </div>
            </body>
        </html>
        """
        result = get_text_from_html(html)
        # BeautifulSoup separates inline elements with newlines
        assert result == "Title\nParagraph with\nbold\ntext"

    def test_complex_html_structure(self):
        """Test complex HTML with various elements."""
        html = """
        <html>
            <head>
                <style>.hidden { display: none; }</style>
                <script>function test() { return true; }</script>
            </head>
            <body>
                <header>
                    <h1>Main Title</h1>
                </header>
                <main>
                    <article>
                        <h2>Article Title</h2>
                        <p>Article content goes here.</p>
                        <ul>
                            <li>Item 1</li>
                            <li>Item 2</li>
                        </ul>
                    </article>
                </main>
                <noscript>JavaScript disabled</noscript>
            </body>
        </html>
        """
        result = get_text_from_html(html)
        assert result == "Main Title\nArticle Title\nArticle content goes here.\nItem 1\nItem 2"

    def test_html_with_attributes(self):
        """Test HTML with various attributes."""
        html = """
        <html>
            <body>
                <p id="test" class="content" data-value="123">Text content</p>
                <a href="https://example.com">Link text</a>
            </body>
        </html>
        """
        result = get_text_from_html(html)
        assert result == "Text content\nLink text"

    def test_html_with_comments(self):
        """Test HTML with comments (should be removed by BeautifulSoup)."""
        html = """
        <html>
            <body>
                <!-- This is a comment -->
                <p>Visible text</p>
                <!-- Another comment -->
            </body>
        </html>
        """
        result = get_text_from_html(html)
        assert result == "Visible text"

    def test_multiple_spaces_in_text(self):
        """Test that multiple spaces within text are normalized."""
        html = "<html><body><p>Word1    Word2     Word3</p></body></html>"
        result = get_text_from_html(html)
        # BeautifulSoup converts multiple spaces to newlines
        assert result == "Word1\nWord2\nWord3"

    def test_html_with_forms(self):
        """Test HTML with form elements."""
        html = """
        <html>
            <body>
                <form>
                    <label>Name:</label>
                    <input type="text" name="name" />
                    <button type="submit">Submit</button>
                </form>
            </body>
        </html>
        """
        result = get_text_from_html(html)
        assert result == "Name:\nSubmit"

    def test_html_with_tables(self):
        """Test HTML with table elements."""
        html = """
        <html>
            <body>
                <table>
                    <tr>
                        <th>Header 1</th>
                        <th>Header 2</th>
                    </tr>
                    <tr>
                        <td>Data 1</td>
                        <td>Data 2</td>
                    </tr>
                </table>
            </body>
        </html>
        """
        result = get_text_from_html(html)
        assert result == "Header 1\nHeader 2\nData 1\nData 2"

    def test_html_with_mixed_line_endings(self):
        """Test HTML with mixed \\n and \\r\\n line endings."""
        html = "<html><body><p>Line 1</p>\n\r\n<p>Line 2</p>\r\n<p>Line 3</p></body></html>"
        result = get_text_from_html(html)
        assert result == "Line 1\nLine 2\nLine 3"

    def test_html_with_only_whitespace(self):
        """Test HTML with only whitespace characters."""
        html = "   \n\n   \r\n   "
        result = get_text_from_html(html)
        assert result == ""

    def test_preserves_text_structure(self):
        """Test that text structure is preserved (newlines between elements)."""
        html = """
        <html>
            <body>
                <h1>Title</h1>
                <p>Paragraph 1</p>
                <p>Paragraph 2</p>
            </body>
        </html>
        """
        result = get_text_from_html(html)
        # Should have newlines between elements
        assert result == "Title\nParagraph 1\nParagraph 2"


class TestLoadData:
    """Test cases for the load_data function."""

    def test_load_dict_data(self, input_data_dir: Path) -> None:
        """Test loading a JSON file containing a dictionary."""
        file_path = input_data_dir / "sample_dict.json"
        result = load_data(file_path)

        assert isinstance(result, dict)
        assert result["name"] == "John Doe"
        assert result["age"] == 30
        assert result["city"] == "New York"
        assert result["is_active"] is True
        assert result["scores"] == [85.5, 92.0, 78.5]
        assert result["metadata"]["created_at"] == "2023-01-15T10:30:00"
        assert result["metadata"]["version"] == 1.2
        assert result["metadata"]["tags"] == ["test", "sample"]

    def test_load_list_data(self, input_data_dir: Path) -> None:
        """Test loading a JSON file containing a list."""
        file_path = input_data_dir / "sample_list.json"
        result = load_data(file_path)

        assert isinstance(result, list)
        assert len(result) == 3
        assert result[0]["id"] == 1
        assert result[0]["name"] == "Item 1"
        assert result[0]["price"] == 19.99
        assert result[1]["id"] == 2
        assert result[2]["id"] == 3

    def test_load_empty_dict(self, input_data_dir: Path) -> None:
        """Test loading an empty JSON file."""
        file_path = input_data_dir / "empty.json"
        result = load_data(file_path)

        assert isinstance(result, dict)
        assert result == {}

    def test_load_unsupported_file_format(self, input_data_dir: Path) -> None:
        """Test that UnsupportedFileFormat is raised for unsupported file types."""
        file_path = input_data_dir / "unsupported.txt"

        with pytest.raises(UnsupportedFileFormat) as exc_info:
            load_data(file_path)

        assert "No support for provided file type" in str(exc_info.value)
        assert "unsupported.txt" in str(exc_info.value)

    def test_load_nonexistent_file(self, input_data_dir: Path) -> None:
        """Test that FileNotFoundError is raised for nonexistent files."""
        file_path = input_data_dir / "nonexistent.json"

        with pytest.raises(FileNotFoundError):
            load_data(file_path)

    def test_load_with_path_object(self, input_data_dir: Path) -> None:
        """Test that function works with Path objects."""
        file_path = input_data_dir / "sample_dict.json"
        result = load_data(file_path)

        assert isinstance(result, dict)
        assert result["name"] == "John Doe"


class TestConvertFloatsToDecimals:
    """Test cases for the convert_floats_to_decimals function."""

    def test_convert_single_float(self) -> None:
        """Test converting a single float to Decimal."""
        result = convert_floats_to_decimals(3.14)
        assert isinstance(result, Decimal)
        assert result == Decimal("3.14")

    def test_convert_dict_with_floats(self) -> None:
        """Test converting floats in a dictionary."""
        data = {"price": 19.99, "tax": 0.08, "name": "item"}
        result = convert_floats_to_decimals(data)

        assert isinstance(result["price"], Decimal)
        assert result["price"] == Decimal("19.99")
        assert isinstance(result["tax"], Decimal)
        assert result["tax"] == Decimal("0.08")
        assert result["name"] == "item"  # string unchanged

    def test_convert_list_with_floats(self) -> None:
        """Test converting floats in a list."""
        data = [1.5, 2.7, "string", 3.14]
        result = convert_floats_to_decimals(data)

        assert isinstance(result[0], Decimal)
        assert result[0] == Decimal("1.5")
        assert isinstance(result[1], Decimal)
        assert result[1] == Decimal("2.7")
        assert result[2] == "string"  # string unchanged
        assert isinstance(result[3], Decimal)
        assert result[3] == Decimal("3.14")

    def test_convert_nested_structure(self) -> None:
        """Test converting floats in nested structures."""
        data = {
            "items": [
                {"price": 10.5, "quantity": 2},
                {"price": 20.0, "quantity": 1}
            ],
            "total": 41.0
        }
        result = convert_floats_to_decimals(data)

        assert isinstance(result["total"], Decimal)
        assert result["total"] == Decimal("41.0")
        assert isinstance(result["items"][0]["price"], Decimal)
        assert result["items"][0]["price"] == Decimal("10.5")
        assert isinstance(result["items"][1]["price"], Decimal)
        assert result["items"][1]["price"] == Decimal("20.0")
        assert result["items"][0]["quantity"] == 2  # int unchanged

    def test_convert_no_floats(self) -> None:
        """Test that non-float values are unchanged."""
        data = {"name": "test", "count": 5, "active": True}
        result = convert_floats_to_decimals(data)

        assert result == data  # no changes

    def test_convert_empty_structures(self) -> None:
        """Test converting empty structures."""
        assert convert_floats_to_decimals({}) == {}
        assert convert_floats_to_decimals([]) == []
        assert convert_floats_to_decimals("string") == "string"
        assert convert_floats_to_decimals(42) == 42


class TestConvertDecimalsToFloats:
    """Test cases for the convert_decimals_to_floats function."""

    def test_convert_single_decimal(self) -> None:
        """Test converting a single Decimal to float."""
        result = convert_decimals_to_floats(Decimal("3.14"))
        assert isinstance(result, float)
        assert result == 3.14

    def test_convert_dict_with_decimals(self) -> None:
        """Test converting Decimals in a dictionary."""
        data = {"price": Decimal("19.99"), "tax": Decimal("0.08"), "name": "item"}
        result = convert_decimals_to_floats(data)
        assert isinstance(result["price"], float)
        assert result["price"] == 19.99
        assert isinstance(result["tax"], float)
        assert result["tax"] == 0.08
        assert result["name"] == "item"  # string unchanged

    def test_convert_list_with_decimals(self) -> None:
        """Test converting Decimals in a list."""
        data = [Decimal("1.5"), Decimal("2.7"), "string", Decimal("3.14")]
        result = convert_decimals_to_floats(data)

        assert isinstance(result[0], float)
        assert result[0] == 1.5
        assert isinstance(result[1], float)
        assert result[1] == 2.7
        assert result[2] == "string"  # string unchanged
        assert isinstance(result[3], float)
        assert result[3] == 3.14

    def test_convert_nested_structure(self) -> None:
        """Test converting Decimals in nested structures."""
        data = {
            "items": [
                {"price": Decimal("10.5"), "quantity": 2},
                {"price": Decimal("20.0"), "quantity": 1}
            ],
            "total": Decimal("41.0")
        }
        result = convert_decimals_to_floats(data)
        assert isinstance(result["total"], float)
        assert result["total"] == 41.0
        assert isinstance(result["items"][0]["price"], float)
        assert result["items"][0]["price"] == 10.5
        assert isinstance(result["items"][1]["price"], float)
        assert result["items"][1]["price"] == 20.0
        assert result["items"][0]["quantity"] == 2  # int unchanged

    def test_convert_no_decimals(self) -> None:
        """Test that non-Decimal values are unchanged."""
        data = {"name": "test", "count": 5, "active": True}
        result = convert_decimals_to_floats(data)
        assert result == data  # no changes

    def test_convert_empty_structures(self) -> None:
        """Test converting empty structures."""
        assert convert_decimals_to_floats({}) == {}
        assert convert_decimals_to_floats([]) == []
        assert convert_decimals_to_floats("string") == "string"
        assert convert_decimals_to_floats(42) == 42


class TestSerializeDatetime:
    """Test cases for the serialize_datetime function."""

    def test_convert_single_datetime(self) -> None:
        """Test converting a single datetime to ISO string."""
        dt = datetime.datetime(2023, 1, 15, 10, 30, 45)
        result = serialize_datetime(dt)
        assert isinstance(result, str)
        assert result == "2023-01-15T10:30:45"

    def test_convert_dict_with_datetime(self) -> None:
        """Test converting datetime in a dictionary."""
        dt = datetime.datetime(2023, 1, 15, 10, 30, 45)
        data = {"created_at": dt, "name": "test", "count": 5}
        result = serialize_datetime(data)
        assert result["created_at"] == "2023-01-15T10:30:45"
        assert result["name"] == "test"  # unchanged
        assert result["count"] == 5  # unchanged

    def test_convert_list_with_datetime(self) -> None:
        """Test converting datetime in a list."""
        dt1 = datetime.datetime(2023, 1, 15, 10, 30, 45)
        dt2 = datetime.datetime(2023, 2, 20, 14, 15, 30)
        data = [dt1, "string", dt2, 42]
        result = serialize_datetime(data)
        assert result[0] == "2023-01-15T10:30:45"
        assert result[1] == "string"  # unchanged
        assert result[2] == "2023-02-20T14:15:30"
        assert result[3] == 42  # unchanged

    def test_convert_nested_structure(self) -> None:
        """Test converting datetime in nested structures."""
        dt1 = datetime.datetime(2023, 1, 15, 10, 30, 45)
        dt2 = datetime.datetime(2023, 2, 20, 14, 15, 30)
        data = {
            "events": [
                {"timestamp": dt1, "action": "create"},
                {"timestamp": dt2, "action": "update"}
            ],
            "last_modified": dt2
        }
        result = serialize_datetime(data)
        assert result["last_modified"] == "2023-02-20T14:15:30"
        assert result["events"][0]["timestamp"] == "2023-01-15T10:30:45"
        assert result["events"][0]["action"] == "create"  # unchanged
        assert result["events"][1]["timestamp"] == "2023-02-20T14:15:30"
        assert result["events"][1]["action"] == "update"  # unchanged

    def test_convert_no_datetime(self) -> None:
        """Test that non-datetime values are unchanged."""
        data = {"name": "test", "count": 5, "active": True}
        result = serialize_datetime(data)
        assert result == data  # no changes

    def test_convert_empty_structures(self) -> None:
        """Test converting empty structures."""
        assert serialize_datetime({}) == {}
        assert serialize_datetime([]) == []
        assert serialize_datetime("string") == "string"
        assert serialize_datetime(42) == 42

    def test_convert_datetime_with_microseconds(self) -> None:
        """Test converting datetime with microseconds."""
        dt = datetime.datetime(2023, 1, 15, 10, 30, 45, 123456)
        result = serialize_datetime(dt)
        assert result == "2023-01-15T10:30:45.123456"


class TestResolveDottedPath:
    """Tests for resolve_dotted_path in data_utils."""

    def test_resolve_simple_dict_path(self) -> None:
        logger = logging.getLogger("test_resolve_simple_dict_path")
        data = {"a": {"b": {"c": 123}}}
        result = resolve_dotted_path(logger, data, "a.b.c")
        assert result == "123"  # coerced to string

    def test_resolve_list_index_in_path(self) -> None:
        logger = logging.getLogger("test_resolve_list_index_in_path")
        data = {"items": [{"name": "alpha"}, {"name": "beta"}]}
        result = resolve_dotted_path(logger, data, "items.1.name")
        assert result == "beta"

    def test_resolve_with_json_string_root(self) -> None:
        logger = logging.getLogger("test_resolve_with_json_string_root")
        obj = json.dumps({"x": {"y": [10, 20, {"z": "ok"}]}})
        result = resolve_dotted_path(logger, obj, "x.y.2.z")
        assert result == "ok"

    def test_empty_path_returns_stringified_object(self) -> None:
        logger = logging.getLogger("test_empty_path_returns_stringified_object")
        data = {"k": 1}
        result = resolve_dotted_path(logger, data, [])
        assert result == str(data)

    def test_missing_key_returns_none(self) -> None:
        logger = logging.getLogger("test_missing_key_returns_none")
        data = {"a": {"b": 1}}
        result = resolve_dotted_path(logger, data, "a.c")
        assert result is None

    def test_none_final_value_returns_none(self) -> None:
        logger = logging.getLogger("test_none_final_value_returns_none")
        data = {"a": {"b": None}}
        result = resolve_dotted_path(logger, data, "a.b")
        assert result is None


class TestAssertBalancedJsDelimiters:
    """Test cases for assert_balanced_js_delimiters function."""

    # ============================================================================
    # Valid Balanced Brackets Tests
    # ============================================================================

    def test_valid_simple_parentheses(self) -> None:
        """Test valid balanced parentheses."""
        assert_balanced_js_delimiters("(function() { return 42; })()")

    def test_valid_nested_parentheses(self) -> None:
        """Test valid nested parentheses."""
        assert_balanced_js_delimiters("((()))")

    def test_valid_simple_braces(self) -> None:
        """Test valid balanced braces."""
        assert_balanced_js_delimiters("{ key: 'value' }")

    def test_valid_nested_braces(self) -> None:
        """Test valid nested braces."""
        assert_balanced_js_delimiters("{{{{}}}}")

    def test_valid_square_brackets(self) -> None:
        """Test valid balanced square brackets."""
        assert_balanced_js_delimiters("[1, 2, [3, 4]]")

    def test_valid_mixed_brackets(self) -> None:
        """Test valid mixed bracket types."""
        assert_balanced_js_delimiters("({[()]})")

    def test_valid_complex_iife(self) -> None:
        """Test valid complex IIFE with multiple bracket types."""
        assert_balanced_js_delimiters("(function() { return { items: [1, 2, 3] }; })()")

    # ============================================================================
    # Unbalanced Brackets Tests
    # ============================================================================

    def test_unbalanced_missing_closing_parenthesis(self) -> None:
        """Test unbalanced missing closing parenthesis."""
        with pytest.raises(ValueError, match="Unbalanced brackets"):
            assert_balanced_js_delimiters("(function() { return 42; }")

    def test_unbalanced_missing_opening_parenthesis(self) -> None:
        """Test unbalanced missing opening parenthesis."""
        with pytest.raises(ValueError, match="Unbalanced brackets"):
            assert_balanced_js_delimiters("function() { return 42; })")

    def test_unbalanced_mismatched_brackets(self) -> None:
        """Test unbalanced mismatched bracket types."""
        with pytest.raises(ValueError, match="Unbalanced brackets"):
            assert_balanced_js_delimiters("([)]")

    def test_unbalanced_nested_mismatch(self) -> None:
        """Test unbalanced nested brackets with wrong closing type."""
        with pytest.raises(ValueError, match="Unbalanced brackets"):
            assert_balanced_js_delimiters("({)}")

    def test_unbalanced_extra_closing_brace(self) -> None:
        """Test unbalanced extra closing brace."""
        with pytest.raises(ValueError, match="Unbalanced brackets"):
            assert_balanced_js_delimiters("{ key: 'value' }}")

    # ============================================================================
    # String Literal Tests - Valid
    # ============================================================================

    def test_valid_double_quoted_string(self) -> None:
        """Test valid double-quoted string."""
        assert_balanced_js_delimiters('const str = "hello world";')

    def test_valid_single_quoted_string(self) -> None:
        """Test valid single-quoted string."""
        assert_balanced_js_delimiters("const str = 'hello world';")

    def test_valid_template_literal(self) -> None:
        """Test valid template literal."""
        assert_balanced_js_delimiters("const str = `hello world`;")

    def test_valid_multiple_strings(self) -> None:
        """Test valid multiple string literals."""
        assert_balanced_js_delimiters('const a = "hello"; const b = \'world\'; const c = `test`;')

    def test_valid_string_with_brackets_inside(self) -> None:
        """Test valid string containing brackets (should be ignored)."""
        assert_balanced_js_delimiters('const str = "value: {key: 123}";')

    def test_valid_nested_strings_different_types(self) -> None:
        """Test valid nested strings of different quote types."""
        assert_balanced_js_delimiters('const str = "outer \'inner\' string";')

    # ============================================================================
    # String Literal Tests - Unterminated
    # ============================================================================

    def test_unterminated_double_quoted_string(self) -> None:
        """Test unterminated double-quoted string."""
        with pytest.raises(ValueError, match="Unterminated string literal"):
            assert_balanced_js_delimiters('const str = "hello world;')

    def test_unterminated_single_quoted_string(self) -> None:
        """Test unterminated single-quoted string."""
        with pytest.raises(ValueError, match="Unterminated string literal"):
            assert_balanced_js_delimiters("const str = 'hello world;")

    def test_unterminated_template_literal(self) -> None:
        """Test unterminated template literal."""
        with pytest.raises(ValueError, match="Unterminated string literal"):
            assert_balanced_js_delimiters("const str = `hello world;")

    # ============================================================================
    # Escaped Quotes Tests
    # ============================================================================

    def test_valid_escaped_double_quote_in_double_string(self) -> None:
        """Test escaped double quote within double-quoted string."""
        assert_balanced_js_delimiters('const str = "He said \\"hello\\"";')

    def test_valid_escaped_single_quote_in_single_string(self) -> None:
        """Test escaped single quote within single-quoted string."""
        assert_balanced_js_delimiters("const str = 'It\\'s working';")

    def test_valid_escaped_backtick_in_template_literal(self) -> None:
        """Test escaped backtick within template literal."""
        assert_balanced_js_delimiters("const str = `Template with \\`backtick\\``;")

    def test_valid_escaped_backslash(self) -> None:
        """Test escaped backslash in string."""
        assert_balanced_js_delimiters('const str = "path\\\\to\\\\file";')

    def test_valid_double_escaped_backslash(self) -> None:
        """Test double escaped backslash (two backslashes = one escaped backslash in JS)."""
        # "\\\\" in Python string becomes "\\" in JS (one escaped backslash)
        assert_balanced_js_delimiters('const str = "\\\\";')

    def test_valid_escaped_newline(self) -> None:
        """Test escaped newline in string."""
        assert_balanced_js_delimiters('const str = "line1\\nline2";')

    def test_valid_escaped_tab(self) -> None:
        """Test escaped tab in string."""
        assert_balanced_js_delimiters('const str = "col1\\tcol2";')

    # ============================================================================
    # Template Literal Edge Cases
    # ============================================================================

    def test_valid_template_literal_with_expression(self) -> None:
        """Test template literal with ${} expression."""
        assert_balanced_js_delimiters("const str = `Value: ${value}`;")

    def test_valid_template_literal_with_nested_expression(self) -> None:
        """Test template literal with nested ${} expression containing brackets."""
        assert_balanced_js_delimiters("const str = `Value: ${obj[key]}`;")

    def test_valid_template_literal_with_complex_expression(self) -> None:
        """Test template literal with complex ${} expression."""
        assert_balanced_js_delimiters("const str = `Value: ${func({key: value})}`;")

    def test_valid_template_literal_nested(self) -> None:
        """Test nested template literals (quotes of different types)."""
        assert_balanced_js_delimiters('const str = `outer "inner" string`;')

    def test_valid_template_literal_escaped_dollar(self) -> None:
        """Test escaped $ in template literal (not an expression)."""
        assert_balanced_js_delimiters("const str = `Price: \\$100`;")

    def test_valid_template_literal_escaped_brace(self) -> None:
        """Test escaped braces in template literal."""
        assert_balanced_js_delimiters("const str = `Text \\${not expression}`;")

    # ============================================================================
    # Complex Real-World Scenarios
    # ============================================================================

    def test_valid_iife_with_template_literal(self) -> None:
        """Test valid IIFE with template literal."""
        assert_balanced_js_delimiters("(function() { return `Result: ${value}`; })()")

    def test_valid_object_literal_with_strings(self) -> None:
        """Test valid object literal with various string types."""
        assert_balanced_js_delimiters(
            '{key1: "value1", key2: \'value2\', key3: `value3`}'
        )

    def test_valid_array_with_strings_and_brackets(self) -> None:
        """Test valid array containing strings with brackets."""
        assert_balanced_js_delimiters(
            '["item1", "item2: {value}", \'item3\']'
        )

    def test_valid_function_call_with_strings(self) -> None:
        """Test valid function call with string arguments."""
        assert_balanced_js_delimiters(
            'func("arg1", \'arg2\', `arg3`)'
        )

    def test_valid_complex_nested_structure(self) -> None:
        """Test valid complex nested structure with strings and brackets."""
        assert_balanced_js_delimiters(
            '(function() { return {items: ["a", "b"], nested: {key: "value"}}; })()'
        )

    # ============================================================================
    # Edge Cases
    # ============================================================================

    def test_valid_empty_string(self) -> None:
        """Test empty string."""
        assert_balanced_js_delimiters('""')

    def test_valid_empty_template_literal(self) -> None:
        """Test empty template literal."""
        assert_balanced_js_delimiters('``')

    def test_valid_string_with_only_escapes(self) -> None:
        """Test string containing only escape sequences."""
        assert_balanced_js_delimiters('"\\\\\\"\\n\\t"')

    def test_valid_template_literal_with_curly_braces_in_text(self) -> None:
        """Test template literal with curly braces that are not expressions."""
        assert_balanced_js_delimiters("const str = `{not an expression}`;")

    def test_valid_template_literal_expression_not_closed(self) -> None:
        """Test template literal with ${ expression that has brackets."""
        assert_balanced_js_delimiters("const str = `${obj[key]}`;")

    def test_valid_unicode_in_string(self) -> None:
        """Test string with unicode characters."""
        assert_balanced_js_delimiters('const str = "Hello 世界";')

    def test_valid_newline_in_template_literal(self) -> None:
        """Test template literal with actual newline."""
        assert_balanced_js_delimiters("const str = `line1\nline2`;")

    # ============================================================================
    # Invalid Edge Cases
    # ============================================================================

    def test_invalid_string_starts_with_escape_sequence_only(self) -> None:
        """Test string that ends with just a backslash."""
        with pytest.raises(ValueError, match="Unterminated string literal"):
            assert_balanced_js_delimiters('const str = "test\\";')

    def test_invalid_unescaped_quote_in_string(self) -> None:
        """Test unescaped quote of wrong type in string (should terminate)."""
        # This is actually valid - different quote types can be inside
        assert_balanced_js_delimiters('const str = "It\'s valid";')

    def test_invalid_backslash_at_end_of_string(self) -> None:
        """Test backslash at end of string (unterminated)."""
        with pytest.raises(ValueError, match="Unterminated string literal"):
            assert_balanced_js_delimiters('const str = "test\\')

    def test_invalid_bracket_after_unterminated_string(self) -> None:
        """Test bracket after unterminated string."""
        with pytest.raises(ValueError, match="Unterminated string literal"):
            assert_balanced_js_delimiters('"unterminated {')

    def test_invalid_complex_unbalanced_with_strings(self) -> None:
        """Test complex structure with strings that has unbalanced brackets."""
        with pytest.raises(ValueError, match="Unbalanced brackets"):
            assert_balanced_js_delimiters(
                '(function() { return {items: ["a", "b"]; })'
            )


class TestApplyParamsToDictStandalone:
    """Test apply_params_to_json with standalone placeholder replacement."""

    def test_string_param(self) -> None:
        d = {"name": "{{user_name}}"}
        params = {"user_name": "John Doe"}
        type_map = {"user_name": "string"}
        result = apply_params_to_json(d, params, type_map)
        assert result == {"name": "John Doe"}

    def test_integer_param(self) -> None:
        d = {"age": "{{user_age}}"}
        params = {"user_age": 25}
        type_map = {"user_age": "integer"}
        result = apply_params_to_json(d, params, type_map)
        assert result == {"age": 25}

    def test_number_param(self) -> None:
        d = {"price": "{{item_price}}"}
        params = {"item_price": 19.99}
        type_map = {"item_price": "number"}
        result = apply_params_to_json(d, params, type_map)
        assert result == {"price": 19.99}

    def test_boolean_true(self) -> None:
        d = {"active": "{{is_active}}"}
        params = {"is_active": True}
        type_map = {"is_active": "boolean"}
        result = apply_params_to_json(d, params, type_map)
        assert result == {"active": True}

    def test_boolean_false(self) -> None:
        d = {"enabled": "{{is_enabled}}"}
        params = {"is_enabled": False}
        type_map = {"is_enabled": "boolean"}
        result = apply_params_to_json(d, params, type_map)
        assert result == {"enabled": False}

    def test_string_typed_integer_value(self) -> None:
        """Zip code: value is 2101 but type is string, should produce '2101'."""
        d = {"zip": "{{zip_code}}"}
        params = {"zip_code": 2101}
        type_map = {"zip_code": "string"}
        result = apply_params_to_json(d, params, type_map)
        assert result == {"zip": "2101"}

    def test_integer_from_string_value(self) -> None:
        """Value is '25' string but type is integer, should produce int 25."""
        d = {"age": "{{age}}"}
        params = {"age": "25"}
        type_map = {"age": "integer"}
        result = apply_params_to_json(d, params, type_map)
        assert result == {"age": 25}

    def test_zero_value(self) -> None:
        d = {"count": "{{item_count}}"}
        params = {"item_count": 0}
        type_map = {"item_count": "integer"}
        result = apply_params_to_json(d, params, type_map)
        assert result == {"count": 0}

    def test_empty_string(self) -> None:
        d = {"name": "{{user_name}}"}
        params = {"user_name": ""}
        type_map = {"user_name": "string"}
        result = apply_params_to_json(d, params, type_map)
        assert result == {"name": ""}


class TestApplyParamsToDictSubstring:
    """Test apply_params_to_json with substring interpolation."""

    def test_substring_interpolation(self) -> None:
        d = {"label": "Price: {{direction}}"}
        params = {"direction": "Low to High"}
        type_map = {"direction": "string"}
        result = apply_params_to_json(d, params, type_map)
        assert result == {"label": "Price: Low to High"}

    def test_substring_with_number(self) -> None:
        d = {"msg": "Page {{num}} of results"}
        params = {"num": 5}
        type_map = {"num": "integer"}
        result = apply_params_to_json(d, params, type_map)
        assert result == {"msg": "Page 5 of results"}

    def test_date_suffix(self) -> None:
        d = {"datetime": "{{date}}T00:00:00"}
        params = {"date": "2026-01-30"}
        type_map = {"date": "string"}
        result = apply_params_to_json(d, params, type_map)
        assert result == {"datetime": "2026-01-30T00:00:00"}

    def test_multiple_substrings(self) -> None:
        d = {"greeting": "Hello {{first}} {{last}}!"}
        params = {"first": "John", "last": "Doe"}
        type_map = {"first": "string", "last": "string"}
        result = apply_params_to_json(d, params, type_map)
        assert result == {"greeting": "Hello John Doe!"}


class TestApplyParamsToDictNonMatching:
    """Test that non-parameter placeholders are left untouched."""

    def test_leave_sessionStorage_untouched(self) -> None:
        d = {"token": "{{sessionStorage:auth.token}}"}
        params = {"user_id": "123"}
        type_map = {"user_id": "string"}
        result = apply_params_to_json(d, params, type_map)
        assert result == {"token": "{{sessionStorage:auth.token}}"}

    def test_leave_cookie_untouched(self) -> None:
        d = {"session": "{{cookie:session_id}}"}
        params = {"user_id": "123"}
        type_map = {"user_id": "string"}
        result = apply_params_to_json(d, params, type_map)
        assert result == {"session": "{{cookie:session_id}}"}

    def test_leave_missing_param_untouched(self) -> None:
        d = {"name": "{{user_name}}", "age": "{{user_age}}"}
        params = {"user_name": "George"}
        type_map = {"user_name": "string"}
        result = apply_params_to_json(d, params, type_map)
        assert result == {"name": "George", "age": "{{user_age}}"}


class TestApplyParamsToDictNested:
    """Test apply_params_to_json with nested structures."""

    def test_nested_dict(self) -> None:
        d = {"user": {"name": "{{name}}", "profile": {"age": "{{age}}"}}}
        params = {"name": "Laura", "age": 28}
        type_map = {"name": "string", "age": "integer"}
        result = apply_params_to_json(d, params, type_map)
        assert result == {"user": {"name": "Laura", "profile": {"age": 28}}}

    def test_list_values(self) -> None:
        d = {"items": ["{{item1}}", "{{item2}}"]}
        params = {"item1": "first", "item2": "second"}
        type_map = {"item1": "string", "item2": "string"}
        result = apply_params_to_json(d, params, type_map)
        assert result == {"items": ["first", "second"]}

    def test_mixed_types(self) -> None:
        d = {"name": "{{name}}", "age": "{{age}}", "active": "{{active}}"}
        params = {"name": "Frank", "age": 35, "active": True}
        type_map = {"name": "string", "age": "integer", "active": "boolean"}
        result = apply_params_to_json(d, params, type_map)
        assert result == {"name": "Frank", "age": 35, "active": True}

    def test_preserves_non_string_values(self) -> None:
        """Non-string values in the dict (ints, bools) should pass through unchanged."""
        d = {"name": "{{name}}", "hardcoded_int": 42, "hardcoded_bool": True}
        params = {"name": "test"}
        type_map = {"name": "string"}
        result = apply_params_to_json(d, params, type_map)
        assert result == {"name": "test", "hardcoded_int": 42, "hardcoded_bool": True}

    def test_empty_params(self) -> None:
        d = {"name": "{{user}}"}
        result = apply_params_to_json(d, {}, {})
        assert result == {"name": "{{user}}"}

    def test_same_param_multiple_times(self) -> None:
        d = {"first": "{{value}}", "second": "{{value}}"}
        params = {"value": "repeated"}
        type_map = {"value": "string"}
        result = apply_params_to_json(d, params, type_map)
        assert result == {"first": "repeated", "second": "repeated"}


class TestApplyParamsToDictRealWorld:
    """Test real-world scenarios with apply_params_to_json."""

    def test_http_headers_with_bearer_token(self) -> None:
        d = {
            "Authorization": "Bearer {{token}}",
            "X-Request-ID": "{{uuid}}",
            "Content-Type": "application/json"
        }
        params = {"token": "abc123xyz"}
        type_map = {"token": "string"}
        result = apply_params_to_json(d, params, type_map)
        assert result == {
            "Authorization": "Bearer abc123xyz",
            "X-Request-ID": "{{uuid}}",  # builtin, left untouched
            "Content-Type": "application/json"
        }

    def test_request_body_mixed(self) -> None:
        d = {
            "username": "{{username}}",
            "limit": "{{limit}}",
            "active": "{{active}}",
            "type": "OW"  # hardcoded
        }
        params = {"username": "testuser", "limit": 50, "active": True}
        type_map = {"username": "string", "limit": "integer", "active": "boolean"}
        result = apply_params_to_json(d, params, type_map)
        assert result == {"username": "testuser", "limit": 50, "active": True, "type": "OW"}


class TestApplyParamsToStr:
    """Test apply_params_to_str for plain string interpolation."""

    def test_url_with_params(self) -> None:
        text = "https://api.example.com/users/{{user_id}}/posts/{{post_id}}"
        params = {"user_id": "12345", "post_id": "67890"}
        result = apply_params_to_str(text, params)
        assert result == "https://api.example.com/users/12345/posts/67890"

    def test_url_with_numeric_param(self) -> None:
        text = "https://example.com/api?limit={{limit}}&offset={{offset}}"
        params = {"limit": 20, "offset": 0}
        result = apply_params_to_str(text, params)
        assert result == "https://example.com/api?limit=20&offset=0"

    def test_selector_with_param(self) -> None:
        text = "input[name='{{field_name}}']"
        params = {"field_name": "email"}
        result = apply_params_to_str(text, params)
        assert result == "input[name='email']"

    def test_leave_storage_untouched(self) -> None:
        text = "Bearer {{sessionStorage:auth.token}}"
        params = {"user_id": "123"}
        result = apply_params_to_str(text, params)
        assert result == "Bearer {{sessionStorage:auth.token}}"

    def test_empty_text(self) -> None:
        assert apply_params_to_str("", {"x": "y"}) == ""

    def test_none_text(self) -> None:
        assert apply_params_to_str(None, {"x": "y"}) is None

    def test_none_params(self) -> None:
        assert apply_params_to_str("{{x}}", None) == "{{x}}"

    def test_empty_params(self) -> None:
        assert apply_params_to_str("{{x}}", {}) == "{{x}}"

    def test_no_placeholders(self) -> None:
        assert apply_params_to_str("static text", {"x": "y"}) == "static text"

    def test_whitespace_in_placeholder(self) -> None:
        text = "hello {{ name }}"
        params = {"name": "world"}
        result = apply_params_to_str(text, params)
        assert result == "hello world"

    def test_special_chars_in_value(self) -> None:
        text = "https://example.com/api?filter={{filter}}"
        params = {"filter": "name=John&age>25"}
        result = apply_params_to_str(text, params)
        assert result == "https://example.com/api?filter=name=John&age>25"
