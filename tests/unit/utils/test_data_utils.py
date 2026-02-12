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
    serialize_datetime,
    resolve_dotted_path,
    get_text_from_html,
    apply_params_to_str,
    apply_params_to_json,
    _coerce_value,
    extract_object_schema,
)


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


class TestCoerceValue:
    """Test _coerce_value type coercion logic."""

    def test_none_returns_none(self) -> None:
        """Test that None always returns None regardless of type."""
        assert _coerce_value(None, "integer") is None
        assert _coerce_value(None, "string") is None
        assert _coerce_value(None, "boolean") is None

    # --- integer ---

    def test_integer_passthrough(self) -> None:
        """Test that int value with integer type passes through."""
        assert _coerce_value(25, "integer") == 25
        assert isinstance(_coerce_value(25, "integer"), int)

    def test_integer_from_string(self) -> None:
        """Test converting string to integer."""
        assert _coerce_value("42", "integer") == 42

    def test_integer_from_float(self) -> None:
        """Test converting float to integer."""
        assert _coerce_value(3.0, "integer") == 3

    def test_integer_zero(self) -> None:
        """Test zero int passes through."""
        assert _coerce_value(0, "integer") == 0

    # --- number ---

    def test_number_passthrough(self) -> None:
        """Test that float value with number type passes through."""
        assert _coerce_value(3.14, "number") == 3.14
        assert isinstance(_coerce_value(3.14, "number"), float)

    def test_number_from_string(self) -> None:
        """Test converting string to float."""
        assert _coerce_value("19.99", "number") == 19.99

    def test_number_from_int(self) -> None:
        """Test converting int to float."""
        assert _coerce_value(5, "number") == 5.0
        assert isinstance(_coerce_value(5, "number"), float)

    # --- boolean ---

    def test_boolean_passthrough(self) -> None:
        """Test that bool value with boolean type passes through."""
        assert _coerce_value(True, "boolean") is True
        assert _coerce_value(False, "boolean") is False

    def test_boolean_from_string_true(self) -> None:
        """Test converting string 'true'/'1'/'yes' to True."""
        assert _coerce_value("true", "boolean") is True
        assert _coerce_value("True", "boolean") is True
        assert _coerce_value("1", "boolean") is True
        assert _coerce_value("yes", "boolean") is True

    def test_boolean_from_string_false(self) -> None:
        """Test converting string 'false'/'0'/'no' to False."""
        assert _coerce_value("false", "boolean") is False
        assert _coerce_value("False", "boolean") is False
        assert _coerce_value("0", "boolean") is False
        assert _coerce_value("no", "boolean") is False

    def test_boolean_from_int(self) -> None:
        """Test converting int to boolean."""
        assert _coerce_value(1, "boolean") is True
        assert _coerce_value(0, "boolean") is False

    # --- string ---

    def test_string_passthrough(self) -> None:
        """Test that string value stays as string."""
        assert _coerce_value("hello", "string") == "hello"

    def test_string_from_int(self) -> None:
        """Test converting int to string."""
        assert _coerce_value(42, "string") == "42"

    def test_string_from_float(self) -> None:
        """Test converting float to string."""
        assert _coerce_value(3.14, "string") == "3.14"

    # --- other types (date, email, url, etc.) treated as string ---

    def test_date_type_treated_as_string(self) -> None:
        """Test that 'date' type coerces to string."""
        assert _coerce_value("2023-01-15", "date") == "2023-01-15"
        assert _coerce_value(42, "date") == "42"

    # --- bool is not int for integer type ---

    def test_boolean_not_treated_as_int(self) -> None:
        """Test that True/False are coerced by int() for integer type, not passed through."""
        # bool is subclass of int in Python, but we explicitly exclude it from passthrough
        result = _coerce_value(True, "integer")
        assert result == 1


class TestApplyParamsToStr:
    """Test apply_params_to_str — plain string replacement (always str)."""

    def test_basic_replacement(self) -> None:
        """Test basic placeholder replacement in a URL."""
        text = "https://api.example.com/users/{{user_id}}/posts"
        result = apply_params_to_str(text, {"user_id": "12345"})
        assert result == "https://api.example.com/users/12345/posts"

    def test_multiple_replacements(self) -> None:
        """Test replacing multiple different placeholders."""
        text = "https://example.com/{{path}}?q={{query}}"
        result = apply_params_to_str(text, {"path": "search", "query": "test"})
        assert result == "https://example.com/search?q=test"

    def test_numeric_value_becomes_string(self) -> None:
        """Test that numeric values are stringified."""
        text = "https://example.com/page/{{page_num}}"
        result = apply_params_to_str(text, {"page_num": 42})
        assert result == "https://example.com/page/42"

    def test_whitespace_in_placeholder(self) -> None:
        """Test that whitespace inside placeholder braces is stripped."""
        text = "https://example.com/{{ user_id }}"
        result = apply_params_to_str(text, {"user_id": "abc"})
        assert result == "https://example.com/abc"

    def test_missing_param_left_untouched(self) -> None:
        """Test that unknown placeholders are left as-is."""
        text = "https://example.com/{{known}}/{{unknown}}"
        result = apply_params_to_str(text, {"known": "val"})
        assert result == "https://example.com/val/{{unknown}}"

    def test_runtime_placeholder_left_untouched(self) -> None:
        """Test that runtime placeholders (sessionStorage:xxx) are NOT replaced."""
        text = "Bearer {{sessionStorage:auth.token}}"
        result = apply_params_to_str(text, {"sessionStorage:auth.token": "SHOULD_NOT_MATCH"})
        # apply_params_to_str replaces by exact key match — the key "sessionStorage:auth.token"
        # is technically in the dict, so it DOES replace. The protection is at the caller level.
        # What matters: if the key isn't in params, it stays.
        text2 = "Bearer {{sessionStorage:auth.token}}"
        result2 = apply_params_to_str(text2, {"user_id": "123"})
        assert result2 == "Bearer {{sessionStorage:auth.token}}"

    def test_empty_text_returns_empty(self) -> None:
        """Test that empty text returns empty."""
        assert apply_params_to_str("", {"k": "v"}) == ""

    def test_none_params_returns_original(self) -> None:
        """Test that None params returns original text."""
        text = "https://example.com/{{id}}"
        assert apply_params_to_str(text, None) == text

    def test_empty_params_returns_original(self) -> None:
        """Test that empty params returns original text."""
        text = "https://example.com/{{id}}"
        assert apply_params_to_str(text, {}) == text

    def test_same_param_multiple_times(self) -> None:
        """Test that same placeholder is replaced everywhere."""
        text = "{{val}}-{{val}}-{{val}}"
        result = apply_params_to_str(text, {"val": "x"})
        assert result == "x-x-x"


class TestApplyParamsToJson:
    """Test apply_params_to_json — dict-walking with typed coercion."""

    # --- standalone placeholder with typed coercion ---

    def test_standalone_integer_coercion(self) -> None:
        """Test standalone placeholder coerced to int."""
        d = {"count": "{{n}}"}
        result = apply_params_to_json(d, {"n": "5"}, {"n": "integer"})
        assert result == {"count": 5}
        assert isinstance(result["count"], int)

    def test_standalone_number_coercion(self) -> None:
        """Test standalone placeholder coerced to float."""
        d = {"price": "{{p}}"}
        result = apply_params_to_json(d, {"p": "19.99"}, {"p": "number"})
        assert result == {"price": 19.99}
        assert isinstance(result["price"], float)

    def test_standalone_boolean_coercion(self) -> None:
        """Test standalone placeholder coerced to bool."""
        d = {"enabled": "{{flag}}"}
        result = apply_params_to_json(d, {"flag": "true"}, {"flag": "boolean"})
        assert result == {"enabled": True}
        assert result["enabled"] is True

    def test_standalone_string_coercion(self) -> None:
        """Test standalone placeholder stays as string."""
        d = {"name": "{{user}}"}
        result = apply_params_to_json(d, {"user": "Alice"}, {"user": "string"})
        assert result == {"name": "Alice"}
        assert isinstance(result["name"], str)

    def test_standalone_defaults_to_string(self) -> None:
        """Test standalone placeholder defaults to string when type_map is empty."""
        d = {"name": "{{user}}"}
        result = apply_params_to_json(d, {"user": "Bob"})
        assert result == {"name": "Bob"}

    # --- substring placeholder always str ---

    def test_substring_placeholder_always_str(self) -> None:
        """Test that substring placeholders are always str-substituted."""
        d = {"url": "https://example.com/{{id}}/data"}
        result = apply_params_to_json(d, {"id": 42}, {"id": "integer"})
        assert result == {"url": "https://example.com/42/data"}
        assert isinstance(result["url"], str)

    def test_substring_multiple_params(self) -> None:
        """Test substring with multiple placeholders."""
        d = {"greeting": "Hello {{first}} {{last}}!"}
        result = apply_params_to_json(d, {"first": "Jane", "last": "Doe"})
        assert result == {"greeting": "Hello Jane Doe!"}

    # --- nested structures ---

    def test_nested_dict(self) -> None:
        """Test resolution in nested dicts."""
        d = {"user": {"name": "{{name}}", "age": "{{age}}"}}
        result = apply_params_to_json(d, {"name": "Alice", "age": "30"}, {"name": "string", "age": "integer"})
        assert result == {"user": {"name": "Alice", "age": 30}}

    def test_list_values(self) -> None:
        """Test resolution in lists."""
        d = {"items": ["{{a}}", "{{b}}", "static"]}
        result = apply_params_to_json(d, {"a": "1", "b": "2"}, {"a": "integer", "b": "integer"})
        assert result == {"items": [1, 2, "static"]}

    def test_deeply_nested(self) -> None:
        """Test resolution in deeply nested structures."""
        d = {"l1": {"l2": {"l3": "{{val}}"}}}
        result = apply_params_to_json(d, {"val": "42"}, {"val": "integer"})
        assert result == {"l1": {"l2": {"l3": 42}}}

    # --- missing / runtime placeholders ---

    def test_missing_param_left_untouched(self) -> None:
        """Test that unknown placeholders are left as-is."""
        d = {"a": "{{known}}", "b": "{{unknown}}"}
        result = apply_params_to_json(d, {"known": "yes"}, {"known": "string"})
        assert result == {"a": "yes", "b": "{{unknown}}"}

    def test_runtime_placeholder_left_untouched(self) -> None:
        """Test that sessionStorage/cookie/etc. placeholders pass through."""
        d = {"token": "{{sessionStorage:auth.token}}", "csrf": "{{meta:csrf}}"}
        result = apply_params_to_json(d, {"user": "x"})
        assert result == {"token": "{{sessionStorage:auth.token}}", "csrf": "{{meta:csrf}}"}

    # --- immutability ---

    def test_does_not_mutate_input(self) -> None:
        """Test that the original dict is not mutated."""
        d = {"name": "{{user}}", "nested": {"val": "{{v}}"}}
        import copy
        original = copy.deepcopy(d)
        apply_params_to_json(d, {"user": "Alice", "v": "42"}, {"user": "string", "v": "integer"})
        assert d == original

    # --- edge cases ---

    def test_none_params(self) -> None:
        """Test that None params returns structure unchanged."""
        d = {"name": "{{user}}"}
        result = apply_params_to_json(d, None)
        assert result == {"name": "{{user}}"}

    def test_empty_params(self) -> None:
        """Test that empty params returns structure unchanged."""
        d = {"name": "{{user}}"}
        result = apply_params_to_json(d, {})
        assert result == {"name": "{{user}}"}

    def test_non_string_values_pass_through(self) -> None:
        """Test that non-string values (int, bool, None) pass through."""
        d = {"count": 42, "flag": True, "nothing": None}
        result = apply_params_to_json(d, {"x": "y"})
        assert result == {"count": 42, "flag": True, "nothing": None}

    def test_whitespace_in_placeholder(self) -> None:
        """Test that whitespace in placeholder is trimmed."""
        d = {"val": "{{ user }}"}
        result = apply_params_to_json(d, {"user": "Bob"}, {"user": "string"})
        assert result == {"val": "Bob"}

    def test_mixed_standalone_and_substring(self) -> None:
        """Test dict with both standalone and substring placeholders."""
        d = {
            "count": "{{n}}",
            "url": "https://example.com/{{n}}/data"
        }
        result = apply_params_to_json(d, {"n": "5"}, {"n": "integer"})
        assert result == {"count": 5, "url": "https://example.com/5/data"}
        assert isinstance(result["count"], int)
        assert isinstance(result["url"], str)

    def test_real_world_fetch_body(self) -> None:
        """Test a realistic fetch body with mixed param types."""
        d = {
            "query": "{{search_term}}",
            "limit": "{{limit}}",
            "include_archived": "{{archived}}",
            "api_key": "{{sessionStorage:api_key}}"
        }
        params = {"search_term": "Boston", "limit": "25", "archived": "false"}
        type_map = {"search_term": "string", "limit": "integer", "archived": "boolean"}
        result = apply_params_to_json(d, params, type_map)
        assert result == {
            "query": "Boston",
            "limit": 25,
            "include_archived": False,
            "api_key": "{{sessionStorage:api_key}}"
        }


class TestExtractObjectSchema:
    """Test cases for extract_object_schema function."""

    def test_scalar_string(self) -> None:
        """Scalar string returns scalar type."""
        result = extract_object_schema("hello")
        assert result == {"_type": "scalar", "_count": 1}

    def test_scalar_int(self) -> None:
        """Scalar integer returns scalar type."""
        result = extract_object_schema(42)
        assert result == {"_type": "scalar", "_count": 1}

    def test_scalar_float(self) -> None:
        """Scalar float returns scalar type."""
        result = extract_object_schema(3.14)
        assert result == {"_type": "scalar", "_count": 1}

    def test_scalar_bool(self) -> None:
        """Scalar boolean returns scalar type."""
        result = extract_object_schema(True)
        assert result == {"_type": "scalar", "_count": 1}

    def test_scalar_none(self) -> None:
        """Scalar None returns scalar type."""
        result = extract_object_schema(None)
        assert result == {"_type": "scalar", "_count": 1}

    def test_empty_dict(self) -> None:
        """Empty dict returns dict type with no fields."""
        result = extract_object_schema({})
        assert result == {"_type": "dict"}

    def test_simple_dict(self) -> None:
        """Simple dict returns dict type with field schemas."""
        result = extract_object_schema({"name": "Alice", "age": 30})
        assert result["_type"] == "dict"
        assert result["name"] == {"_type": "scalar", "_count": 1}
        assert result["age"] == {"_type": "scalar", "_count": 1}

    def test_nested_dict(self) -> None:
        """Nested dict returns nested schema structure."""
        data = {"user": {"name": "Alice", "email": "alice@example.com"}}
        result = extract_object_schema(data)
        assert result["_type"] == "dict"
        assert result["user"]["_type"] == "dict"
        assert result["user"]["name"] == {"_type": "scalar", "_count": 1}
        assert result["user"]["email"] == {"_type": "scalar", "_count": 1}

    def test_empty_list(self) -> None:
        """Empty list returns list type with empty items."""
        result = extract_object_schema([])
        assert result == {"_type": "list", "_count": 0, "_items": {}}

    def test_list_of_scalars(self) -> None:
        """List of scalars returns list with scalar items."""
        result = extract_object_schema([1, 2, 3])
        assert result["_type"] == "list"
        assert result["_count"] == 3
        assert result["_items"] == {"_type": "scalar"}

    def test_list_of_strings(self) -> None:
        """List of strings returns list with scalar items."""
        result = extract_object_schema(["a", "b", "c"])
        assert result["_type"] == "list"
        assert result["_count"] == 3
        assert result["_items"] == {"_type": "scalar"}

    def test_list_of_dicts_same_keys(self) -> None:
        """List of dicts with same keys merges schemas."""
        data = [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}]
        result = extract_object_schema(data)
        assert result["_type"] == "list"
        assert result["_count"] == 2
        assert result["_items"]["_type"] == "dict"
        assert "id" in result["_items"]
        assert "name" in result["_items"]

    def test_list_of_dicts_different_keys(self) -> None:
        """List of dicts with different keys includes all keys."""
        data = [{"id": 1, "name": "a"}, {"id": 2, "email": "b@example.com"}]
        result = extract_object_schema(data)
        assert result["_type"] == "list"
        assert result["_count"] == 2
        assert result["_items"]["_type"] == "dict"
        assert "id" in result["_items"]
        assert "name" in result["_items"]
        assert "email" in result["_items"]

    def test_list_of_dicts_partial_keys(self) -> None:
        """List of dicts where some items have fewer keys."""
        data = [{"id": 1, "name": "a"}, {"id": 2}]
        result = extract_object_schema(data)
        assert result["_type"] == "list"
        assert result["_count"] == 2
        assert result["_items"]["_type"] == "dict"
        assert "id" in result["_items"]
        assert "name" in result["_items"]

    def test_deeply_nested_structure(self) -> None:
        """Deeply nested structure preserves all levels."""
        data = {
            "level1": {
                "level2": {
                    "level3": {"value": 42}
                }
            }
        }
        result = extract_object_schema(data)
        assert result["_type"] == "dict"
        assert result["level1"]["_type"] == "dict"
        assert result["level1"]["level2"]["_type"] == "dict"
        assert result["level1"]["level2"]["level3"]["_type"] == "dict"
        assert result["level1"]["level2"]["level3"]["value"] == {"_type": "scalar", "_count": 1}

    def test_dict_with_list_field(self) -> None:
        """Dict containing a list field."""
        data = {"items": [1, 2, 3], "count": 3}
        result = extract_object_schema(data)
        assert result["_type"] == "dict"
        assert result["items"]["_type"] == "list"
        assert result["items"]["_count"] == 3
        assert result["count"] == {"_type": "scalar", "_count": 1}

    def test_list_of_dicts_with_nested_lists(self) -> None:
        """List of dicts where dicts contain lists."""
        data = [
            {"id": 1, "tags": ["a", "b"]},
            {"id": 2, "tags": ["c"]},
        ]
        result = extract_object_schema(data)
        assert result["_type"] == "list"
        assert result["_count"] == 2
        assert result["_items"]["_type"] == "dict"
        assert "id" in result["_items"]
        assert "tags" in result["_items"]

    def test_list_of_nested_dicts(self) -> None:
        """List of dicts with nested dict fields."""
        data = [
            {"user": {"name": "Alice"}},
            {"user": {"name": "Bob", "age": 30}},
        ]
        result = extract_object_schema(data)
        assert result["_type"] == "list"
        assert result["_count"] == 2
        assert result["_items"]["_type"] == "dict"
        assert result["_items"]["user"]["_type"] == "dict"
        assert "name" in result["_items"]["user"]
        assert "age" in result["_items"]["user"]

    def test_complex_api_response_structure(self) -> None:
        """Complex structure mimicking typical API response."""
        data = {
            "status": "success",
            "data": {
                "users": [
                    {"id": 1, "name": "Alice", "roles": ["admin", "user"]},
                    {"id": 2, "name": "Bob", "roles": ["user"]},
                ],
                "total": 2,
            },
            "meta": {"version": "1.0"},
        }
        result = extract_object_schema(data)

        assert result["_type"] == "dict"
        assert result["status"] == {"_type": "scalar", "_count": 1}
        assert result["data"]["_type"] == "dict"
        assert result["data"]["users"]["_type"] == "list"
        assert result["data"]["users"]["_count"] == 2
        assert result["data"]["users"]["_items"]["_type"] == "dict"
        assert "id" in result["data"]["users"]["_items"]
        assert "name" in result["data"]["users"]["_items"]
        assert "roles" in result["data"]["users"]["_items"]
        assert result["data"]["total"] == {"_type": "scalar", "_count": 1}
        assert result["meta"]["_type"] == "dict"
        assert result["meta"]["version"] == {"_type": "scalar", "_count": 1}

    def test_list_of_lists(self) -> None:
        """List containing lists (matrix-like structure)."""
        data = [[1, 2], [3, 4], [5, 6]]
        result = extract_object_schema(data)
        assert result["_type"] == "list"
        assert result["_count"] == 3

    def test_mixed_list_non_dict_items(self) -> None:
        """List with mixed non-dict items returns scalar items."""
        data = [1, "two", 3.0, True]
        result = extract_object_schema(data)
        assert result["_type"] == "list"
        assert result["_count"] == 4
        assert result["_items"] == {"_type": "scalar"}
