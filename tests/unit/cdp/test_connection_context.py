"""
tests/unit/cdp/test_connection_context.py

Tests for contextvars-based sticky session support in cdp/connection.py.
"""

import contextvars
from unittest.mock import MagicMock, patch

import pytest
import requests

from bluebox.cdp.connection import (
    cdp_http_session,
    _get_http,
    _ws_cookie_header,
    get_browser_websocket_url,
    get_existing_tabs,
    cdp_new_tab,
    cdp_attach_to_existing_tab,
    dispose_context,
)


# -- Helpers / fixtures -------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_context_var():
    """Ensure cdp_http_session is None before and after each test."""
    token = cdp_http_session.set(None)
    yield
    cdp_http_session.reset(token)


def _make_session_with_cookies(**cookies: str) -> requests.Session:
    """Create a requests.Session pre-loaded with cookies."""
    session = requests.Session()
    for name, value in cookies.items():
        session.cookies.set(name, value)
    return session


# -- _get_http -----------------------------------------------------------------

class TestGetHttp:

    def test_uses_bare_requests_when_no_session(self):
        """Without a context session, _get_http falls back to requests.get."""
        with patch("bluebox.cdp.connection.requests.get") as mock_get:
            mock_get.return_value = MagicMock(status_code=200)
            resp = _get_http("http://chrome:9222", "/json/version")
            mock_get.assert_called_once_with("http://chrome:9222/json/version", timeout=5)

    def test_uses_session_when_set(self):
        """With a context session, _get_http uses session.get."""
        session = requests.Session()
        session.get = MagicMock(return_value=MagicMock(status_code=200))
        cdp_http_session.set(session)

        _get_http("http://chrome:9222", "/json/list")
        session.get.assert_called_once_with("http://chrome:9222/json/list", timeout=5)


# -- _ws_cookie_header ---------------------------------------------------------

class TestWsCookieHeader:

    def test_empty_when_no_session(self):
        assert _ws_cookie_header() == {}

    def test_empty_when_session_has_no_cookies(self):
        cdp_http_session.set(requests.Session())
        assert _ws_cookie_header() == {}

    def test_returns_cookie_header(self):
        session = _make_session_with_cookies(sticky="abc123")
        cdp_http_session.set(session)
        header = _ws_cookie_header()
        assert header == {"Cookie": "sticky=abc123"}

    def test_multiple_cookies(self):
        session = _make_session_with_cookies(sticky="abc", sticky_cors="xyz")
        cdp_http_session.set(session)
        header = _ws_cookie_header()
        # Both cookies present (order may vary)
        assert "sticky=abc" in header["Cookie"]
        assert "sticky_cors=xyz" in header["Cookie"]


# -- get_browser_websocket_url with context session ----------------------------

class TestGetBrowserWebsocketUrlContext:

    @patch("bluebox.cdp.connection._get_http")
    def test_uses_context_session_via_get_http(self, mock_get_http):
        """get_browser_websocket_url delegates to _get_http (which reads context)."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "webSocketDebuggerUrl": "ws://127.0.0.1:9222/devtools/browser/abc-123"
        }
        mock_get_http.return_value = mock_resp

        url = get_browser_websocket_url("http://127.0.0.1:9222")
        mock_get_http.assert_called_once_with("http://127.0.0.1:9222", "/json/version")
        assert "abc-123" in url


# -- get_existing_tabs with context session ------------------------------------

class TestGetExistingTabsContext:

    @patch("bluebox.cdp.connection._get_http")
    def test_uses_context_session_via_get_http(self, mock_get_http):
        mock_resp = MagicMock()
        mock_resp.json.return_value = [{"id": "tab1", "type": "page"}]
        mock_get_http.return_value = mock_resp

        tabs = get_existing_tabs("http://chrome:9222")
        mock_get_http.assert_called_once_with("http://chrome:9222", "/json/list")
        assert tabs == [{"id": "tab1", "type": "page"}]


# -- cdp_new_tab forwards cookies to WS ---------------------------------------

class TestCdpNewTabCookies:

    @patch("bluebox.cdp.connection._ws_cookie_header", return_value={"Cookie": "sticky=yes"})
    @patch("bluebox.cdp.connection.get_browser_websocket_url", return_value="ws://chrome:9222/devtools/browser/abc")
    @patch("bluebox.cdp.connection.websocket.create_connection")
    def test_passes_cookie_header_to_ws(self, mock_ws_create, mock_get_ws_url, mock_cookie):
        """cdp_new_tab should forward sticky cookies to the WebSocket upgrade."""
        mock_ws = MagicMock()
        mock_ws_create.return_value = mock_ws
        # Mock CDP responses for createBrowserContext + createTarget
        mock_ws.recv.side_effect = [
            '{"id": 1, "result": {"browserContextId": "ctx-1"}}',
            '{"id": 2, "result": {"targetId": "target-1"}}',
        ]

        target_id, ctx_id, ws = cdp_new_tab("http://chrome:9222", incognito=True)

        mock_ws_create.assert_called_once_with(
            url="ws://chrome:9222/devtools/browser/abc",
            timeout=10,
            header={"Cookie": "sticky=yes"},
        )
        assert target_id == "target-1"
        assert ctx_id == "ctx-1"


# -- cdp_attach_to_existing_tab forwards cookies to WS ------------------------

class TestCdpAttachCookies:

    @patch("bluebox.cdp.connection._ws_cookie_header", return_value={"Cookie": "sticky=yes"})
    @patch("bluebox.cdp.connection.get_browser_websocket_url", return_value="ws://chrome:9222/devtools/browser/abc")
    @patch("bluebox.cdp.connection.get_existing_tabs", return_value=[{"id": "page-1", "type": "page", "url": "about:blank"}])
    @patch("bluebox.cdp.connection.websocket.create_connection")
    def test_passes_cookie_header_to_ws(self, mock_ws_create, mock_tabs, mock_get_ws_url, mock_cookie):
        mock_ws = MagicMock()
        mock_ws_create.return_value = mock_ws

        target_id, ctx_id, ws = cdp_attach_to_existing_tab("http://chrome:9222")

        mock_ws_create.assert_called_once_with(
            url="ws://chrome:9222/devtools/browser/abc",
            timeout=10,
            header={"Cookie": "sticky=yes"},
        )
        assert target_id == "page-1"
        assert ctx_id is None


# -- dispose_context forwards cookies when opening its own WS ------------------

class TestDisposeContextCookies:

    @patch("bluebox.cdp.connection._ws_cookie_header", return_value={"Cookie": "sticky=yes"})
    @patch("bluebox.cdp.connection.get_browser_websocket_url", return_value="ws://chrome:9222/devtools/browser/abc")
    @patch("bluebox.cdp.connection.websocket.create_connection")
    def test_passes_cookie_header_when_creating_ws(self, mock_ws_create, mock_get_ws_url, mock_cookie):
        mock_ws = MagicMock()
        mock_ws.recv.return_value = '{"id": 1, "result": {}}'
        mock_ws_create.return_value = mock_ws

        dispose_context("ctx-1", remote_debugging_address="http://chrome:9222")

        mock_ws_create.assert_called_once_with(
            "ws://chrome:9222/devtools/browser/abc",
            timeout=10,
            header={"Cookie": "sticky=yes"},
        )
        mock_ws.close.assert_called_once()


# -- Context isolation (contextvars should not leak between tasks) -------------

class TestContextIsolation:

    def test_session_does_not_leak_across_contexts(self):
        """Setting cdp_http_session in one context should not affect another."""
        session = _make_session_with_cookies(sticky="leaked")

        results = {}

        def in_context_a():
            cdp_http_session.set(session)
            results["a"] = _ws_cookie_header()

        def in_context_b():
            # Should see the default (None) — not context A's session
            results["b"] = _ws_cookie_header()

        ctx_a = contextvars.copy_context()
        ctx_b = contextvars.copy_context()

        ctx_a.run(in_context_a)
        ctx_b.run(in_context_b)

        assert results["a"] == {"Cookie": "sticky=leaked"}
        assert results["b"] == {}  # no leakage
