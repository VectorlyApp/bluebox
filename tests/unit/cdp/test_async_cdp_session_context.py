"""
tests/unit/cdp/test_async_cdp_session_context.py

Tests for contextvars-based extra headers in AsyncCDPSession.
"""

import contextvars

import pytest

from bluebox.cdp.async_cdp_session import cdp_ws_extra_headers


@pytest.fixture(autouse=True)
def _reset_context_var():
    """Ensure cdp_ws_extra_headers is empty before and after each test."""
    token = cdp_ws_extra_headers.set({})
    yield
    cdp_ws_extra_headers.reset(token)


class TestCdpWsExtraHeaders:

    def test_default_is_empty(self):
        assert cdp_ws_extra_headers.get() == {}

    def test_set_and_get(self):
        cdp_ws_extra_headers.set({"Cookie": "sticky=abc"})
        assert cdp_ws_extra_headers.get() == {"Cookie": "sticky=abc"}

    def test_isolation_across_contexts(self):
        results = {}

        def ctx_a():
            cdp_ws_extra_headers.set({"Cookie": "sticky=a"})
            results["a"] = cdp_ws_extra_headers.get()

        def ctx_b():
            results["b"] = cdp_ws_extra_headers.get()

        contextvars.copy_context().run(ctx_a)
        contextvars.copy_context().run(ctx_b)

        assert results["a"] == {"Cookie": "sticky=a"}
        assert results["b"] == {}  # no leakage
