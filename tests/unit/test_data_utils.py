"""
tests/unit/test_data_utils.py

Unit tests for data utility functions.
"""

import pytest

from web_hacker.utils.data_utils import get_text_from_html


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

