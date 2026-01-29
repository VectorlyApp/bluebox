"""
tests/unit/test_cdp_monitors.py

Tests for CDP monitors: AbstractAsyncMonitor, AsyncNetworkMonitor,
AsyncStorageMonitor, AsyncWindowPropertyMonitor, AsyncInteractionMonitor.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock

from bluebox.cdp.monitors.abstract_async_monitor import AbstractAsyncMonitor
from bluebox.cdp.monitors.async_network_monitor import AsyncNetworkMonitor
from bluebox.cdp.monitors.async_storage_monitor import AsyncStorageMonitor
from bluebox.cdp.monitors.async_window_property_monitor import AsyncWindowPropertyMonitor
from bluebox.cdp.monitors.async_interaction_monitor import AsyncInteractionMonitor


class TestAbstractAsyncMonitor:
    """
    Tests for AbstractAsyncMonitor base class.
    """

    def test_subclass_registration(self) -> None:
        """Verify all 4 monitors are registered in _subclasses."""
        subclasses = AbstractAsyncMonitor._subclasses
        subclass_names = [cls.__name__ for cls in subclasses]

        assert "AsyncNetworkMonitor" in subclass_names
        assert "AsyncStorageMonitor" in subclass_names
        assert "AsyncWindowPropertyMonitor" in subclass_names
        assert "AsyncInteractionMonitor" in subclass_names

    def test_get_all_subclasses_returns_copy(self) -> None:
        """Ensure get_all_subclasses returns a copy, not the original."""
        subclasses = AbstractAsyncMonitor.get_all_subclasses()
        original = AbstractAsyncMonitor._subclasses

        # should be equal in content
        assert subclasses == original
        # but not the same object
        assert subclasses is not original

    def test_get_monitor_category(self) -> None:
        """Each subclass returns its class name."""
        assert AsyncNetworkMonitor.get_monitor_category() == "AsyncNetworkMonitor"
        assert AsyncStorageMonitor.get_monitor_category() == "AsyncStorageMonitor"
        assert AsyncWindowPropertyMonitor.get_monitor_category() == "AsyncWindowPropertyMonitor"
        assert AsyncInteractionMonitor.get_monitor_category() == "AsyncInteractionMonitor"


class TestAsyncNetworkMonitorStaticMethods:
    """
    Tests for AsyncNetworkMonitor static methods.
    """

    def test_is_internal_url_chrome(self) -> None:
        """chrome:// URLs should return True."""
        assert AsyncNetworkMonitor._is_internal_url("chrome://settings") is True
        assert AsyncNetworkMonitor._is_internal_url("chrome://newtab") is True

    def test_is_internal_url_https(self) -> None:
        """https:// URLs should return False."""
        assert AsyncNetworkMonitor._is_internal_url("https://example.com") is False

    def test_is_internal_url_none(self) -> None:
        """None should return False."""
        assert AsyncNetworkMonitor._is_internal_url(None) is False

    def test_is_static_asset_css(self) -> None:
        """.css files should return True."""
        assert AsyncNetworkMonitor._is_static_asset("https://example.com/styles.css") is True

    def test_is_static_asset_png(self) -> None:
        """.png files should return True."""
        assert AsyncNetworkMonitor._is_static_asset("https://example.com/image.png") is True

    def test_is_static_asset_api(self) -> None:
        """/api/ URLs should return False."""
        assert AsyncNetworkMonitor._is_static_asset("https://example.com/api/data") is False

    def test_is_static_asset_none(self) -> None:
        """None should return False."""
        assert AsyncNetworkMonitor._is_static_asset(None) is False

    def test_should_block_url_googletagmanager(self) -> None:
        """googletagmanager.com should be blocked."""
        assert AsyncNetworkMonitor._should_block_url(
            "https://www.googletagmanager.com/gtag/js"
        ) is True

    def test_should_block_url_normal(self) -> None:
        """Normal URLs should not be blocked."""
        assert AsyncNetworkMonitor._should_block_url("https://example.com/api/data") is False

    def test_should_block_url_internal(self) -> None:
        """Internal chrome:// URLs should be blocked."""
        assert AsyncNetworkMonitor._should_block_url("chrome://settings") is True

    def test_get_set_cookie_values_simple(self) -> None:
        """Extract Set-Cookie from simple header."""
        headers = {"set-cookie": "session=abc123; Path=/"}
        values = AsyncNetworkMonitor._get_set_cookie_values(headers)
        assert values == ["session=abc123; Path=/"]

    def test_get_set_cookie_values_multiline(self) -> None:
        """Extract Set-Cookie from newline-separated header."""
        headers = {"set-cookie": "session=abc\ntoken=xyz"}
        values = AsyncNetworkMonitor._get_set_cookie_values(headers)
        assert len(values) == 2
        assert "session=abc" in values
        assert "token=xyz" in values

    def test_get_set_cookie_values_list(self) -> None:
        """Extract Set-Cookie from list value."""
        headers = {"set-cookie": ["session=abc", "token=xyz"]}
        values = AsyncNetworkMonitor._get_set_cookie_values(headers)
        assert values == ["session=abc", "token=xyz"]

    def test_get_set_cookie_values_empty(self) -> None:
        """Empty headers should return empty list."""
        assert AsyncNetworkMonitor._get_set_cookie_values({}) == []
        assert AsyncNetworkMonitor._get_set_cookie_values(None) == []

    def test_parse_json_if_applicable_json(self) -> None:
        """JSON content-type should parse JSON."""
        data = '{"key": "value"}'
        result = AsyncNetworkMonitor._parse_json_if_applicable(data, "application/json")
        assert result == {"key": "value"}

    def test_parse_json_if_applicable_plain_text(self) -> None:
        """Non-JSON content-type should return string."""
        data = '{"key": "value"}'
        result = AsyncNetworkMonitor._parse_json_if_applicable(data, "text/plain")
        assert result == '{"key": "value"}'

    def test_parse_json_if_applicable_none(self) -> None:
        """None data should return None."""
        assert AsyncNetworkMonitor._parse_json_if_applicable(None, "application/json") is None

    def test_parse_json_if_applicable_invalid_json(self) -> None:
        """Invalid JSON should return original string."""
        data = "not valid json"
        result = AsyncNetworkMonitor._parse_json_if_applicable(data, "application/json")
        assert result == "not valid json"

    def test_is_html_content_type(self) -> None:
        """HTML content-type should return True."""
        assert AsyncNetworkMonitor._is_html("<html>", "text/html") is True

    def test_is_html_body_pattern(self) -> None:
        """HTML body patterns should return True."""
        assert AsyncNetworkMonitor._is_html("<!DOCTYPE html><html><head>", None) is True

    def test_is_html_false_for_json(self) -> None:
        """JSON response should return False."""
        assert AsyncNetworkMonitor._is_html('{"key": "value"}', "application/json") is False

    def test_is_html_empty(self) -> None:
        """Empty body should return False."""
        assert AsyncNetworkMonitor._is_html(None, None) is False
        assert AsyncNetworkMonitor._is_html("", None) is False

    def test_clean_response_body_truncation(self) -> None:
        """Long response should be truncated."""
        long_body = "x" * 500_000
        result = AsyncNetworkMonitor._clean_response_body(long_body)
        assert len(result) <= AsyncNetworkMonitor.RESPONSE_BODY_MAX_CHARS

    def test_clean_response_body_json_handling(self) -> None:
        """JSON dict should be serialized."""
        data = {"key": "value", "nested": {"a": 1}}
        result = AsyncNetworkMonitor._clean_response_body(data)
        assert "key" in result
        assert "value" in result

    def test_clean_response_body_empty(self) -> None:
        """Empty body should return empty string."""
        assert AsyncNetworkMonitor._clean_response_body(None) == ""
        assert AsyncNetworkMonitor._clean_response_body("") == ""


class TestAsyncNetworkMonitorWsEventSummary:
    """
    Tests for AsyncNetworkMonitor.get_ws_event_summary.
    """

    def test_get_ws_event_summary_basic(self) -> None:
        """Returns correct summary structure with truncated URL."""
        detail = {
            "method": "GET",
            "url": "https://example.com/api/data",
            "status": 200,
            "type": "Fetch",
            "failed": False,
        }
        summary = AsyncNetworkMonitor.get_ws_event_summary(detail)

        assert summary["type"] == "AsyncNetworkMonitor"
        assert summary["method"] == "GET"
        assert summary["url"] == "https://example.com/api/data"
        assert summary["status"] == 200
        assert summary["resource_type"] == "Fetch"
        assert summary["failed"] is False

    def test_get_ws_event_summary_truncates_url(self) -> None:
        """Long URLs should be truncated."""
        long_url = "https://example.com/" + "a" * 300
        detail = {"url": long_url, "method": "GET"}
        summary = AsyncNetworkMonitor.get_ws_event_summary(detail)

        assert len(summary["url"]) == AsyncNetworkMonitor.URL_MAX_CHARS


class TestAsyncStorageMonitorWsEventSummary:
    """
    Tests for AsyncStorageMonitor.get_ws_event_summary.
    """

    def test_get_ws_event_summary_cookie_change(self) -> None:
        """Cookie change events should have correct summary."""
        detail = {
            "type": "cookieChange",
            "added": [{"name": "a"}],
            "modified": [{"name": "b"}],
            "removed": [],
            "total_count": 10,
        }
        summary = AsyncStorageMonitor.get_ws_event_summary(detail)

        assert summary["type"] == "AsyncStorageMonitor"
        assert summary["event_type"] == "cookieChange"
        assert summary["added_count"] == 1
        assert summary["modified_count"] == 1
        assert summary["removed_count"] == 0

    def test_get_ws_event_summary_storage_event(self) -> None:
        """Storage events should have correct summary."""
        detail = {
            "type": "localStorageItemAdded",
            "origin": "https://example.com",
            "key": "user_id",
        }
        summary = AsyncStorageMonitor.get_ws_event_summary(detail)

        assert summary["type"] == "AsyncStorageMonitor"
        assert summary["event_type"] == "localStorageItemAdded"
        assert summary["origin"] == "https://example.com"
        assert summary["key"] == "user_id"


class TestAsyncStorageMonitorStateManagement:
    """
    Tests for AsyncStorageMonitor state management methods.
    """

    @pytest.mark.asyncio
    async def test_handle_dom_storage_added(self, mock_event_callback: AsyncMock) -> None:
        """domStorageItemAdded updates local_storage_state."""
        monitor = AsyncStorageMonitor(event_callback_fn=mock_event_callback)
        msg = {
            "method": "DOMStorage.domStorageItemAdded",
            "params": {
                "storageId": {"securityOrigin": "https://example.com", "isLocalStorage": True},
                "key": "user_id",
                "newValue": "123",
            },
        }

        await monitor._handle_dom_storage_added(msg)

        assert "https://example.com" in monitor.local_storage_state
        assert monitor.local_storage_state["https://example.com"]["user_id"] == "123"
        mock_event_callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_dom_storage_updated(self, mock_event_callback: AsyncMock) -> None:
        """domStorageItemUpdated updates state and emits event."""
        monitor = AsyncStorageMonitor(event_callback_fn=mock_event_callback)
        monitor.local_storage_state["https://example.com"] = {"user_id": "old"}

        msg = {
            "method": "DOMStorage.domStorageItemUpdated",
            "params": {
                "storageId": {"securityOrigin": "https://example.com", "isLocalStorage": True},
                "key": "user_id",
                "oldValue": "old",
                "newValue": "new",
            },
        }

        await monitor._handle_dom_storage_updated(msg)

        assert monitor.local_storage_state["https://example.com"]["user_id"] == "new"
        mock_event_callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_dom_storage_removed(self, mock_event_callback: AsyncMock) -> None:
        """domStorageItemRemoved removes from state."""
        monitor = AsyncStorageMonitor(event_callback_fn=mock_event_callback)
        monitor.local_storage_state["https://example.com"] = {"user_id": "123", "other": "val"}

        msg = {
            "method": "DOMStorage.domStorageItemRemoved",
            "params": {
                "storageId": {"securityOrigin": "https://example.com", "isLocalStorage": True},
                "key": "user_id",
            },
        }

        await monitor._handle_dom_storage_removed(msg)

        assert "user_id" not in monitor.local_storage_state["https://example.com"]
        assert "other" in monitor.local_storage_state["https://example.com"]

    @pytest.mark.asyncio
    async def test_handle_dom_storage_cleared(self, mock_event_callback: AsyncMock) -> None:
        """domStorageItemsCleared clears origin state."""
        monitor = AsyncStorageMonitor(event_callback_fn=mock_event_callback)
        monitor.local_storage_state["https://example.com"] = {"a": "1", "b": "2"}

        msg = {
            "method": "DOMStorage.domStorageItemsCleared",
            "params": {
                "storageId": {"securityOrigin": "https://example.com", "isLocalStorage": True},
            },
        }

        await monitor._handle_dom_storage_cleared(msg)

        assert "https://example.com" not in monitor.local_storage_state

    @pytest.mark.asyncio
    async def test_handle_get_cookies_reply_initial(self, mock_event_callback: AsyncMock) -> None:
        """getAllCookies reply sets initial cookie state silently (no event emitted)."""
        monitor = AsyncStorageMonitor(event_callback_fn=mock_event_callback)
        msg = {
            "result": {
                "cookies": [
                    {"domain": ".example.com", "name": "session", "value": "abc"},
                    {"domain": ".example.com", "name": "token", "value": "xyz"},
                ]
            }
        }
        command_info = {"type": "getAllCookies", "initial": True}

        await monitor._handle_get_cookies_reply(msg, command_info)

        assert len(monitor.cookies_state) == 2
        assert ".example.com:session" in monitor.cookies_state
        mock_event_callback.assert_not_called()

    @pytest.mark.asyncio
    async def test_handle_get_cookies_reply_changes(self, mock_event_callback: AsyncMock) -> None:
        """getAllCookies reply detects added/modified/removed cookies."""
        monitor = AsyncStorageMonitor(event_callback_fn=mock_event_callback)
        # initial state
        monitor.cookies_state = {
            ".example.com:session": {"domain": ".example.com", "name": "session", "value": "old"},
            ".example.com:removed": {"domain": ".example.com", "name": "removed", "value": "gone"},
        }

        msg = {
            "result": {
                "cookies": [
                    {"domain": ".example.com", "name": "session", "value": "new"},  # modified
                    {"domain": ".example.com", "name": "added", "value": "brand_new"},  # added
                    # "removed" cookie is missing -> removed
                ]
            }
        }
        command_info = {"type": "getAllCookies", "initial": False, "triggered_by": "test"}

        await monitor._handle_get_cookies_reply(msg, command_info)

        mock_event_callback.assert_called_once()
        call_args = mock_event_callback.call_args
        event = call_args[0][1]
        assert event.type == "cookieChange"
        assert len(event.added) == 1
        assert len(event.modified) == 1
        assert len(event.removed) == 1


class TestAsyncWindowPropertyMonitorStaticMethods:
    """
    Tests for AsyncWindowPropertyMonitor static methods.
    """

    def test_is_application_object_native_classname(self) -> None:
        """HTMLElement, SVGElement should return False."""
        assert AsyncWindowPropertyMonitor._is_application_object("HTMLElement", "div") is False
        assert AsyncWindowPropertyMonitor._is_application_object("SVGElement", "svg") is False

    def test_is_application_object_native_name(self) -> None:
        """document, window should return False."""
        assert AsyncWindowPropertyMonitor._is_application_object(None, "document") is False
        assert AsyncWindowPropertyMonitor._is_application_object(None, "window") is False
        assert AsyncWindowPropertyMonitor._is_application_object(None, "navigator") is False

    def test_is_application_object_app_object(self) -> None:
        """Custom names with Object className return True."""
        assert AsyncWindowPropertyMonitor._is_application_object("Object", "myAppState") is True
        assert AsyncWindowPropertyMonitor._is_application_object("", "customVar") is True
        assert AsyncWindowPropertyMonitor._is_application_object(None, "appConfig") is True


class TestAsyncWindowPropertyMonitorWsEventSummary:
    """
    Tests for AsyncWindowPropertyMonitor.get_ws_event_summary.
    """

    def test_get_ws_event_summary(self) -> None:
        """Returns correct structure with change_count and changed_paths."""
        detail = {
            "url": "https://example.com",
            "changes": [
                {"path": "appState.user", "value": "john", "change_type": "changed"},
                {"path": "appState.items", "value": [], "change_type": "added"},
            ],
            "total_keys": 10,
        }
        summary = AsyncWindowPropertyMonitor.get_ws_event_summary(detail)

        assert summary["type"] == "AsyncWindowPropertyMonitor"
        assert summary["url"] == "https://example.com"
        assert summary["change_count"] == 2
        assert summary["total_keys"] == 10
        assert "appState.user" in summary["changed_paths"]
        assert "changed" in summary["change_types"]
        assert "added" in summary["change_types"]


class TestAsyncInteractionMonitorWsEventSummary:
    """
    Tests for AsyncInteractionMonitor.get_ws_event_summary.
    """

    def test_get_ws_event_summary(self) -> None:
        """Returns interaction_type and element_tag."""
        detail = {
            "type": "click",
            "url": "https://example.com",
            "element": {"tag_name": "button"},
        }
        summary = AsyncInteractionMonitor.get_ws_event_summary(detail)

        assert summary["type"] == "AsyncInteractionMonitor"
        assert summary["interaction_type"] == "click"
        assert summary["element_tag"] == "button"
        assert summary["url"] == "https://example.com"

    def test_get_ws_event_summary_no_element(self) -> None:
        """Missing element should have None for element_tag."""
        detail = {"type": "keydown", "url": "https://example.com"}
        summary = AsyncInteractionMonitor.get_ws_event_summary(detail)

        assert summary["element_tag"] is None


class TestAsyncInteractionMonitorParseEvent:
    """
    Tests for AsyncInteractionMonitor._parse_interaction_event.
    """

    def test_parse_interaction_event(self, mock_event_callback: AsyncMock) -> None:
        """Parses raw JS data into UIInteractionEvent."""
        monitor = AsyncInteractionMonitor(event_callback_fn=mock_event_callback)
        raw_data = {
            "type": "click",
            "timestamp": 1234567890,
            "url": "https://example.com",
            "event": {
                "mouse_button": 0,
                "mouse_x_viewport": 100,
                "mouse_y_viewport": 200,
            },
            "element": {
                "tag_name": "button",
                "id": "submit-btn",
                "text": "Submit",
                "bounding_box": {"x": 10, "y": 20, "width": 100, "height": 40},
            },
        }

        result = monitor._parse_interaction_event(raw_data)

        assert result is not None
        assert result.type.value == "click"
        assert result.element.tag_name == "button"
        assert result.element.id == "submit-btn"
        assert result.interaction.mouse_button == 0
        assert result.element.bounding_box is not None
        assert result.element.bounding_box.width == 100

    def test_parse_interaction_event_missing_element(
        self, mock_event_callback: AsyncMock
    ) -> None:
        """Returns None for missing element."""
        monitor = AsyncInteractionMonitor(event_callback_fn=mock_event_callback)
        raw_data = {"type": "click", "timestamp": 123}

        result = monitor._parse_interaction_event(raw_data)
        assert result is None


class TestAsyncInteractionMonitorFileOperations:
    """
    Tests for AsyncInteractionMonitor file operations.
    """

    def test_consolidate_interactions(self, tmp_path: Path) -> None:
        """Reads JSONL and returns consolidated dict."""
        events_file = tmp_path / "interactions.jsonl"
        events_file.write_text(
            '{"type": "click", "url": "https://a.com"}\n'
            '{"type": "keydown", "url": "https://a.com"}\n'
            '{"type": "click", "url": "https://b.com"}\n'
        )

        result = AsyncInteractionMonitor.consolidate_interactions(str(events_file))

        assert len(result["interactions"]) == 3
        assert result["summary"]["total"] == 3
        assert result["summary"]["by_type"]["click"] == 2
        assert result["summary"]["by_type"]["keydown"] == 1
        assert result["summary"]["by_url"]["https://a.com"] == 2

    def test_consolidate_interactions_missing_file(self, tmp_path: Path) -> None:
        """Missing file returns empty structure."""
        missing_file = tmp_path / "nonexistent.jsonl"
        result = AsyncInteractionMonitor.consolidate_interactions(str(missing_file))

        assert result["interactions"] == []
        assert result["summary"]["total"] == 0


# =============================================================================
# AsyncNetworkMonitor CDP Message Handling Tests
# =============================================================================


class TestAsyncNetworkMonitorRequestHandling:
    """Tests for network request stage handling."""

    @pytest.mark.asyncio
    async def test_on_request_will_be_sent_stores_metadata(
        self, mock_event_callback: AsyncMock
    ) -> None:
        """requestWillBeSent stores request metadata."""
        monitor = AsyncNetworkMonitor(event_callback_fn=mock_event_callback)
        msg = {
            "method": "Network.requestWillBeSent",
            "params": {
                "requestId": "req-123",
                "request": {
                    "url": "https://api.example.com/data",
                    "method": "POST",
                    "headers": {"content-type": "application/json"},  # lowercase to match code
                    "postData": '{"key": "value"}',
                },
                "type": "Fetch",
            },
        }

        result = await monitor._on_request_will_be_sent(msg)

        assert result is True
        assert "req-123" in monitor.req_meta
        meta = monitor.req_meta["req-123"]
        assert meta["url"] == "https://api.example.com/data"
        assert meta["method"] == "POST"
        assert meta["type"] == "Fetch"
        # JSON postData should be parsed
        assert meta["postData"] == {"key": "value"}

    @pytest.mark.asyncio
    async def test_on_request_will_be_sent_skips_blocked_url(
        self, mock_event_callback: AsyncMock
    ) -> None:
        """requestWillBeSent skips blocked URLs (analytics)."""
        monitor = AsyncNetworkMonitor(event_callback_fn=mock_event_callback)
        msg = {
            "method": "Network.requestWillBeSent",
            "params": {
                "requestId": "req-blocked",
                "request": {
                    "url": "https://www.googletagmanager.com/gtag/js",
                    "method": "GET",
                    "headers": {},
                },
                "type": "Script",
            },
        }

        result = await monitor._on_request_will_be_sent(msg)

        assert result is True
        assert "req-blocked" not in monitor.req_meta

    @pytest.mark.asyncio
    async def test_on_request_will_be_sent_skips_static_asset(
        self, mock_event_callback: AsyncMock
    ) -> None:
        """requestWillBeSent skips static assets."""
        monitor = AsyncNetworkMonitor(event_callback_fn=mock_event_callback)
        msg = {
            "method": "Network.requestWillBeSent",
            "params": {
                "requestId": "req-static",
                "request": {
                    "url": "https://example.com/styles.css",
                    "method": "GET",
                    "headers": {},
                },
                "type": "Stylesheet",
            },
        }

        result = await monitor._on_request_will_be_sent(msg)

        assert result is True
        assert "req-static" not in monitor.req_meta

    @pytest.mark.asyncio
    async def test_on_request_will_be_sent_parses_json_post_data(
        self, mock_event_callback: AsyncMock
    ) -> None:
        """requestWillBeSent parses JSON postData based on content-type."""
        monitor = AsyncNetworkMonitor(event_callback_fn=mock_event_callback)
        msg = {
            "method": "Network.requestWillBeSent",
            "params": {
                "requestId": "req-json",
                "request": {
                    "url": "https://api.example.com/submit",
                    "method": "POST",
                    "headers": {"content-type": "application/json"},
                    "postData": '{"name": "test", "count": 42}',
                },
                "type": "Fetch",
            },
        }

        await monitor._on_request_will_be_sent(msg)

        meta = monitor.req_meta["req-json"]
        assert isinstance(meta["postData"], dict)
        assert meta["postData"]["name"] == "test"
        assert meta["postData"]["count"] == 42


class TestAsyncNetworkMonitorResponseHandling:
    """Tests for network response stage handling."""

    @pytest.mark.asyncio
    async def test_on_response_received_updates_metadata(
        self, mock_event_callback: AsyncMock
    ) -> None:
        """responseReceived updates existing request metadata."""
        monitor = AsyncNetworkMonitor(event_callback_fn=mock_event_callback)
        # Pre-populate request metadata
        monitor.req_meta["req-123"] = {
            "requestId": "req-123",
            "url": "https://api.example.com/data",
            "method": "GET",
            "type": "Fetch",
        }

        msg = {
            "method": "Network.responseReceived",
            "params": {
                "requestId": "req-123",
                "response": {
                    "url": "https://api.example.com/data",
                    "status": 200,
                    "statusText": "OK",
                    "headers": {"Content-Type": "application/json"},
                    "mimeType": "application/json",
                },
            },
        }

        result = await monitor._on_response_received(msg)

        assert result is True
        meta = monitor.req_meta["req-123"]
        assert meta["status"] == 200
        assert meta["statusText"] == "OK"
        assert meta["mimeType"] == "application/json"

    @pytest.mark.asyncio
    async def test_on_response_received_skips_blocked_url(
        self, mock_event_callback: AsyncMock
    ) -> None:
        """responseReceived skips blocked URLs."""
        monitor = AsyncNetworkMonitor(event_callback_fn=mock_event_callback)
        msg = {
            "method": "Network.responseReceived",
            "params": {
                "requestId": "req-blocked",
                "response": {
                    "url": "https://www.google-analytics.com/collect",
                    "status": 200,
                    "headers": {},
                },
            },
        }

        result = await monitor._on_response_received(msg)

        assert result is True  # handled but skipped
        assert "req-blocked" not in monitor.req_meta

    @pytest.mark.asyncio
    async def test_on_response_received_extra_info_extracts_cookies(
        self, mock_event_callback: AsyncMock
    ) -> None:
        """responseReceivedExtraInfo extracts Set-Cookie headers."""
        monitor = AsyncNetworkMonitor(event_callback_fn=mock_event_callback)
        # Pre-populate request metadata
        monitor.req_meta["req-123"] = {"requestId": "req-123"}

        msg = {
            "method": "Network.responseReceivedExtraInfo",
            "params": {
                "requestId": "req-123",
                "headers": {"set-cookie": "session=abc123; Path=/"},
            },
        }

        result = await monitor._on_response_received_extra_info(msg)

        assert result is True
        meta = monitor.req_meta["req-123"]
        assert meta["setCookies"] == ["session=abc123; Path=/"]
        assert meta["cookiesLogged"] is True


class TestAsyncNetworkMonitorLoadingEvents:
    """
    
    Tests for loading finished/failed events."""

    @pytest.mark.asyncio
    async def test_on_loading_finished_emits_transaction(
        self, mock_event_callback: AsyncMock
    ) -> None:
        """loadingFinished emits transaction event via callback."""
        monitor = AsyncNetworkMonitor(event_callback_fn=mock_event_callback)
        # Pre-populate complete request metadata
        monitor.req_meta["req-123"] = {
            "requestId": "req-123",
            "url": "https://api.example.com/data",
            "method": "GET",
            "type": "Fetch",
            "status": 200,
            "statusText": "OK",
            "requestHeaders": {},
            "responseHeaders": {"Content-Type": "application/json"},
            "responseBody": '{"result": "ok"}',
            "mimeType": "application/json",
        }

        msg = {
            "method": "Network.loadingFinished",
            "params": {"requestId": "req-123"},
        }

        result = await monitor._on_loading_finished(msg)

        assert result is True
        mock_event_callback.assert_called_once()
        call_args = mock_event_callback.call_args
        category, event = call_args[0]
        assert category == "AsyncNetworkMonitor"
        assert event.url == "https://api.example.com/data"
        assert event.method == "GET"
        assert event.status == 200

    @pytest.mark.asyncio
    async def test_on_loading_finished_cleans_up_metadata(
        self, mock_event_callback: AsyncMock
    ) -> None:
        """loadingFinished removes request metadata after emission."""
        monitor = AsyncNetworkMonitor(event_callback_fn=mock_event_callback)
        monitor.req_meta["req-123"] = {
            "requestId": "req-123",
            "url": "https://api.example.com/data",
            "method": "GET",
            "type": "Fetch",
            "status": 200,
            "statusText": "OK",
            "requestHeaders": {},
            "responseHeaders": {},
            "mimeType": "application/json",
        }

        msg = {
            "method": "Network.loadingFinished",
            "params": {"requestId": "req-123"},
        }

        await monitor._on_loading_finished(msg)

        assert "req-123" not in monitor.req_meta

    @pytest.mark.asyncio
    async def test_on_loading_finished_skips_blocked_url(
        self, mock_event_callback: AsyncMock
    ) -> None:
        """loadingFinished skips emission for blocked URLs."""
        monitor = AsyncNetworkMonitor(event_callback_fn=mock_event_callback)
        monitor.req_meta["req-blocked"] = {
            "requestId": "req-blocked",
            "url": "https://www.googletagmanager.com/gtag/js",
            "method": "GET",
        }

        msg = {
            "method": "Network.loadingFinished",
            "params": {"requestId": "req-blocked"},
        }

        await monitor._on_loading_finished(msg)

        mock_event_callback.assert_not_called()
        # metadata should be cleaned up even for blocked URLs
        assert "req-blocked" not in monitor.req_meta

    @pytest.mark.asyncio
    async def test_on_loading_failed_emits_failed_transaction(
        self, mock_event_callback: AsyncMock
    ) -> None:
        """loadingFailed emits failed transaction event."""
        monitor = AsyncNetworkMonitor(event_callback_fn=mock_event_callback)
        monitor.req_meta["req-fail"] = {
            "requestId": "req-fail",
            "url": "https://api.example.com/data",
            "method": "GET",
            "type": "Fetch",
            "requestHeaders": {},
        }

        msg = {
            "method": "Network.loadingFailed",
            "params": {
                "requestId": "req-fail",
                "errorText": "net::ERR_CONNECTION_REFUSED",
            },
        }

        result = await monitor._on_loading_failed(msg)

        assert result is True
        mock_event_callback.assert_called_once()
        call_args = mock_event_callback.call_args
        category, event = call_args[0]
        assert category == "AsyncNetworkMonitor"
        assert event.failed is True
        assert event.errorText == "net::ERR_CONNECTION_REFUSED"

    @pytest.mark.asyncio
    async def test_on_loading_failed_includes_error_text(
        self, mock_event_callback: AsyncMock
    ) -> None:
        """loadingFailed includes errorText in emitted event."""
        monitor = AsyncNetworkMonitor(event_callback_fn=mock_event_callback)
        monitor.req_meta["req-fail"] = {
            "requestId": "req-fail",
            "url": "https://api.example.com/data",
            "method": "POST",
            "type": "XHR",
            "requestHeaders": {},
        }

        msg = {
            "method": "Network.loadingFailed",
            "params": {
                "requestId": "req-fail",
                "errorText": "net::ERR_SSL_PROTOCOL_ERROR",
            },
        }

        await monitor._on_loading_failed(msg)

        call_args = mock_event_callback.call_args
        _, event = call_args[0]
        assert event.errorText == "net::ERR_SSL_PROTOCOL_ERROR"
        assert event.url == "https://api.example.com/data"


class TestAsyncNetworkMonitorFetchInterception:
    """Tests for Fetch.requestPaused handling."""

    @pytest.mark.asyncio
    async def test_fetch_request_paused_request_stage(
        self, mock_event_callback: AsyncMock, mock_cdp_session: AsyncMock
    ) -> None:
        """Fetch.requestPaused in request stage stores metadata."""
        monitor = AsyncNetworkMonitor(event_callback_fn=mock_event_callback)
        msg = {
            "method": "Fetch.requestPaused",
            "params": {
                "requestId": "fetch-123",
                "request": {
                    "url": "https://api.example.com/data",
                    "method": "POST",
                    "headers": {"content-type": "application/json"},  # lowercase to match code
                    "postData": '{"key": "value"}',
                },
                "resourceType": "Fetch",
                # No responseStatusCode = request stage
            },
        }

        result = await monitor._on_fetch_request_paused(msg, mock_cdp_session)

        assert result is True
        assert "fetch-123" in monitor.req_meta
        meta = monitor.req_meta["fetch-123"]
        assert meta["method"] == "POST"
        assert meta["postData"] == {"key": "value"}
        # Should continue the request
        mock_cdp_session.send.assert_called_with(
            "Fetch.continueRequest", {"requestId": "fetch-123"}
        )

    @pytest.mark.asyncio
    async def test_fetch_request_paused_response_stage(
        self, mock_event_callback: AsyncMock, mock_cdp_session: AsyncMock
    ) -> None:
        """Fetch.requestPaused in response stage updates metadata."""
        monitor = AsyncNetworkMonitor(event_callback_fn=mock_event_callback)
        # Pre-populate from request stage
        monitor.req_meta["fetch-123"] = {
            "requestId": "fetch-123",
            "url": "https://api.example.com/data",
            "method": "GET",
            "type": "Fetch",
        }

        msg = {
            "method": "Fetch.requestPaused",
            "params": {
                "requestId": "fetch-123",
                "responseStatusCode": 200,
                "responseStatusText": "OK",
                "responseHeaders": [
                    {"name": "Content-Type", "value": "application/json"},
                ],
                "request": {"url": "https://api.example.com/data", "method": "GET"},
                "resourceType": "Fetch",
            },
        }

        result = await monitor._on_fetch_request_paused(msg, mock_cdp_session)

        assert result is True
        meta = monitor.req_meta["fetch-123"]
        assert meta["status"] == 200
        assert meta["responseHeaders"]["Content-Type"] == "application/json"

    @pytest.mark.asyncio
    async def test_fetch_request_paused_requests_body_for_captured_types(
        self, mock_event_callback: AsyncMock, mock_cdp_session: AsyncMock
    ) -> None:
        """Fetch.requestPaused requests body for CAPTURE_RESOURCES types."""
        monitor = AsyncNetworkMonitor(event_callback_fn=mock_event_callback)
        mock_cdp_session.send.return_value = 42  # command ID

        msg = {
            "method": "Fetch.requestPaused",
            "params": {
                "requestId": "fetch-123",
                "responseStatusCode": 200,
                "responseStatusText": "OK",
                "responseHeaders": [],
                "request": {"url": "https://api.example.com/data", "method": "GET"},
                "resourceType": "Fetch",  # in CAPTURE_RESOURCES
            },
        }

        await monitor._on_fetch_request_paused(msg, mock_cdp_session)

        # should call Fetch.getResponseBody
        calls = [
            c for c in mock_cdp_session.send.call_args_list
            if c[0][0] == "Fetch.getResponseBody"
        ]
        assert len(calls) == 1
        # should track the pending body request
        assert 42 in monitor.fetch_get_body_wait

    @pytest.mark.asyncio
    async def test_fetch_request_paused_skips_blocked_urls(
        self, mock_event_callback: AsyncMock, mock_cdp_session: AsyncMock
    ) -> None:
        """Fetch.requestPaused skips blocked URLs and continues."""
        monitor = AsyncNetworkMonitor(event_callback_fn=mock_event_callback)
        msg = {
            "method": "Fetch.requestPaused",
            "params": {
                "requestId": "fetch-blocked",
                "request": {
                    "url": "https://www.googletagmanager.com/gtag/js",
                    "method": "GET",
                },
                "resourceType": "Script",
            },
        }

        result = await monitor._on_fetch_request_paused(msg, mock_cdp_session)

        assert result is True
        assert "fetch-blocked" not in monitor.req_meta
        mock_cdp_session.send.assert_called_with(
            "Fetch.continueRequest", {"requestId": "fetch-blocked"}
        )

    @pytest.mark.asyncio
    async def test_on_fetch_get_body_reply_stores_and_emits(
        self, mock_event_callback: AsyncMock, mock_cdp_session: AsyncMock
    ) -> None:
        """getResponseBody reply stores body and emits transaction."""
        monitor = AsyncNetworkMonitor(event_callback_fn=mock_event_callback)
        cmd_id = 42
        # Setup tracking context
        monitor.fetch_get_body_wait[cmd_id] = {
            "rid": "fetch-123",
            "fetch_id": "fetch-123",
        }
        # Setup request metadata
        monitor.req_meta["fetch-123"] = {
            "requestId": "fetch-123",
            "url": "https://api.example.com/data",
            "method": "GET",
            "type": "Fetch",
            "status": 200,
            "statusText": "OK",
            "mimeType": "application/json",
            "requestHeaders": {},
            "responseHeaders": {},
        }

        msg = {
            "id": cmd_id,
            "result": {
                "body": '{"result": "success"}',
                "base64Encoded": False,
            },
        }

        result = await monitor._on_fetch_get_body_reply(cmd_id, msg, mock_cdp_session)

        assert result is True
        # Should emit transaction
        mock_event_callback.assert_called_once()
        call_args = mock_event_callback.call_args
        _, event = call_args[0]
        assert '{"result": "success"}' in event.response_body
        # Tracking context should be cleaned up
        assert cmd_id not in monitor.fetch_get_body_wait

    @pytest.mark.asyncio
    async def test_on_fetch_get_body_reply_decodes_base64(
        self, mock_event_callback: AsyncMock, mock_cdp_session: AsyncMock
    ) -> None:
        """getResponseBody reply decodes base64-encoded body."""
        import base64

        monitor = AsyncNetworkMonitor(event_callback_fn=mock_event_callback)
        cmd_id = 42
        monitor.fetch_get_body_wait[cmd_id] = {
            "rid": "fetch-123",
            "fetch_id": "fetch-123",
        }
        monitor.req_meta["fetch-123"] = {
            "requestId": "fetch-123",
            "url": "https://api.example.com/data",
            "method": "GET",
            "type": "Fetch",
            "status": 200,
            "statusText": "OK",
            "mimeType": "text/plain",
            "requestHeaders": {},
            "responseHeaders": {},
        }

        original_body = "Hello, World!"
        encoded_body = base64.b64encode(original_body.encode()).decode()

        msg = {
            "id": cmd_id,
            "result": {
                "body": encoded_body,
                "base64Encoded": True,
            },
        }

        await monitor._on_fetch_get_body_reply(cmd_id, msg, mock_cdp_session)

        call_args = mock_event_callback.call_args
        _, event = call_args[0]
        assert "Hello, World!" in event.response_body


class TestAsyncNetworkMonitorDispatch:
    """Tests for handle_network_message dispatch."""

    @pytest.mark.asyncio
    async def test_handle_network_message_routes_fetch_paused(
        self, mock_event_callback: AsyncMock, mock_cdp_session: AsyncMock
    ) -> None:
        """handle_network_message routes Fetch.requestPaused correctly."""
        monitor = AsyncNetworkMonitor(event_callback_fn=mock_event_callback)
        msg = {
            "method": "Fetch.requestPaused",
            "params": {
                "requestId": "fetch-123",
                "request": {
                    "url": "https://api.example.com/data",
                    "method": "GET",
                    "headers": {},
                },
                "resourceType": "Fetch",
            },
        }

        result = await monitor.handle_network_message(msg, mock_cdp_session)

        # Returns False to not swallow event (allows storage monitor to handle)
        assert result is False
        assert "fetch-123" in monitor.req_meta

    @pytest.mark.asyncio
    async def test_handle_network_message_routes_request_will_be_sent(
        self, mock_event_callback: AsyncMock, mock_cdp_session: AsyncMock
    ) -> None:
        """handle_network_message routes Network.requestWillBeSent correctly."""
        monitor = AsyncNetworkMonitor(event_callback_fn=mock_event_callback)
        msg = {
            "method": "Network.requestWillBeSent",
            "params": {
                "requestId": "req-123",
                "request": {
                    "url": "https://api.example.com/data",
                    "method": "GET",
                    "headers": {},
                },
                "type": "Fetch",
            },
        }

        result = await monitor.handle_network_message(msg, mock_cdp_session)

        assert result is True
        assert "req-123" in monitor.req_meta

    @pytest.mark.asyncio
    async def test_handle_network_message_routes_response_received(
        self, mock_event_callback: AsyncMock, mock_cdp_session: AsyncMock
    ) -> None:
        """handle_network_message routes Network.responseReceived correctly."""
        monitor = AsyncNetworkMonitor(event_callback_fn=mock_event_callback)
        monitor.req_meta["req-123"] = {"requestId": "req-123", "url": "https://api.example.com/data"}

        msg = {
            "method": "Network.responseReceived",
            "params": {
                "requestId": "req-123",
                "response": {
                    "url": "https://api.example.com/data",
                    "status": 200,
                    "headers": {},
                },
            },
        }

        result = await monitor.handle_network_message(msg, mock_cdp_session)

        # Returns False to not swallow event (allows storage monitor to handle)
        assert result is False
        assert monitor.req_meta["req-123"]["status"] == 200

    @pytest.mark.asyncio
    async def test_handle_network_message_routes_loading_finished(
        self, mock_event_callback: AsyncMock, mock_cdp_session: AsyncMock
    ) -> None:
        """handle_network_message routes Network.loadingFinished correctly."""
        monitor = AsyncNetworkMonitor(event_callback_fn=mock_event_callback)
        monitor.req_meta["req-123"] = {
            "requestId": "req-123",
            "url": "https://api.example.com/data",
            "method": "GET",
            "status": 200,
            "statusText": "OK",
            "requestHeaders": {},
            "responseHeaders": {},
            "mimeType": "application/json",
        }

        msg = {
            "method": "Network.loadingFinished",
            "params": {"requestId": "req-123"},
        }

        result = await monitor.handle_network_message(msg, mock_cdp_session)

        assert result is True
        mock_event_callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_network_message_returns_false_for_unknown(
        self, mock_event_callback: AsyncMock, mock_cdp_session: AsyncMock
    ) -> None:
        """handle_network_message returns False for unknown methods."""
        monitor = AsyncNetworkMonitor(event_callback_fn=mock_event_callback)
        msg = {"method": "SomeUnknown.method", "params": {}}

        result = await monitor.handle_network_message(msg, mock_cdp_session)

        assert result is False

    @pytest.mark.asyncio
    async def test_handle_network_command_reply_routes_body_reply(
        self, mock_event_callback: AsyncMock, mock_cdp_session: AsyncMock
    ) -> None:
        """handle_network_command_reply routes getResponseBody replies."""
        monitor = AsyncNetworkMonitor(event_callback_fn=mock_event_callback)
        cmd_id = 42
        monitor.fetch_get_body_wait[cmd_id] = {
            "rid": "fetch-123",
            "fetch_id": "fetch-123",
        }
        monitor.req_meta["fetch-123"] = {
            "requestId": "fetch-123",
            "url": "https://api.example.com/data",
            "method": "GET",
            "type": "Fetch",
            "status": 200,
            "statusText": "OK",
            "mimeType": "text/plain",
            "requestHeaders": {},
            "responseHeaders": {},
        }

        msg = {
            "id": cmd_id,
            "result": {"body": "response data", "base64Encoded": False},
        }

        result = await monitor.handle_network_command_reply(msg, mock_cdp_session)

        assert result is True
        mock_event_callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_network_command_reply_returns_false_for_unknown(
        self, mock_event_callback: AsyncMock, mock_cdp_session: AsyncMock
    ) -> None:
        """handle_network_command_reply returns False for unknown command IDs."""
        monitor = AsyncNetworkMonitor(event_callback_fn=mock_event_callback)
        msg = {"id": 999, "result": {}}

        result = await monitor.handle_network_command_reply(msg, mock_cdp_session)

        assert result is False


class TestAsyncNetworkMonitorIntegration:
    """End-to-end flow tests."""

    @pytest.mark.asyncio
    async def test_full_request_response_flow(
        self, mock_event_callback: AsyncMock
    ) -> None:
        """Simulates: requestWillBeSent → responseReceived → loadingFinished."""
        monitor = AsyncNetworkMonitor(event_callback_fn=mock_event_callback)

        # Step 1: Request will be sent
        request_msg = {
            "method": "Network.requestWillBeSent",
            "params": {
                "requestId": "flow-123",
                "request": {
                    "url": "https://api.example.com/users",
                    "method": "GET",
                    "headers": {"Accept": "application/json"},
                },
                "type": "XHR",
                "timestamp": 1000.0,
            },
        }
        await monitor._on_request_will_be_sent(request_msg)
        assert "flow-123" in monitor.req_meta

        # Step 2: Response received
        response_msg = {
            "method": "Network.responseReceived",
            "params": {
                "requestId": "flow-123",
                "response": {
                    "url": "https://api.example.com/users",
                    "status": 200,
                    "statusText": "OK",
                    "headers": {"Content-Type": "application/json"},
                    "mimeType": "application/json",
                },
            },
        }
        await monitor._on_response_received(response_msg)
        meta = monitor.req_meta["flow-123"]
        assert meta["status"] == 200

        # Step 3: Loading finished
        finished_msg = {
            "method": "Network.loadingFinished",
            "params": {"requestId": "flow-123"},
        }
        await monitor._on_loading_finished(finished_msg)

        # Verify event was emitted
        mock_event_callback.assert_called_once()
        _, event = mock_event_callback.call_args[0]
        assert event.url == "https://api.example.com/users"
        assert event.method == "GET"
        assert event.status == 200
        # Metadata should be cleaned up
        assert "flow-123" not in monitor.req_meta

    @pytest.mark.asyncio
    async def test_fetch_interception_flow(
        self, mock_event_callback: AsyncMock, mock_cdp_session: AsyncMock
    ) -> None:
        """Simulates: Fetch.requestPaused (request) → Fetch.requestPaused (response) → getResponseBody reply."""
        monitor = AsyncNetworkMonitor(event_callback_fn=mock_event_callback)
        mock_cdp_session.send.return_value = 42  # command ID for getResponseBody

        # Step 1: Request stage (using form data to keep postData as string)
        request_paused_msg = {
            "method": "Fetch.requestPaused",
            "params": {
                "requestId": "fetch-flow",
                "request": {
                    "url": "https://api.example.com/data",
                    "method": "POST",
                    "headers": {"content-type": "application/x-www-form-urlencoded"},
                    "postData": "action=create&name=test",
                },
                "resourceType": "Fetch",
            },
        }
        await monitor._on_fetch_request_paused(request_paused_msg, mock_cdp_session)
        assert "fetch-flow" in monitor.req_meta
        assert monitor.req_meta["fetch-flow"]["postData"] == "action=create&name=test"

        # Step 2: Response stage
        response_paused_msg = {
            "method": "Fetch.requestPaused",
            "params": {
                "requestId": "fetch-flow",
                "responseStatusCode": 201,
                "responseStatusText": "Created",
                "responseHeaders": [
                    {"name": "Content-Type", "value": "application/json"},
                ],
                "request": {"url": "https://api.example.com/data", "method": "POST"},
                "resourceType": "Fetch",
            },
        }
        await monitor._on_fetch_request_paused(response_paused_msg, mock_cdp_session)
        assert monitor.req_meta["fetch-flow"]["status"] == 201
        assert 42 in monitor.fetch_get_body_wait

        # Step 3: Body reply
        body_reply_msg = {
            "id": 42,
            "result": {
                "body": '{"id": 123, "created": true}',
                "base64Encoded": False,
            },
        }
        await monitor._on_fetch_get_body_reply(42, body_reply_msg, mock_cdp_session)

        # Verify event was emitted
        mock_event_callback.assert_called_once()
        _, event = mock_event_callback.call_args[0]
        assert event.status == 201
        assert event.method == "POST"
        assert "123" in event.response_body
        # Cleanup
        assert "fetch-flow" not in monitor.req_meta
        assert 42 not in monitor.fetch_get_body_wait


# =============================================================================
# AsyncWindowPropertyMonitor CDP Message Handling Tests
# =============================================================================


class TestAsyncWindowPropertyMonitorMessageHandling:
    """Tests for window property CDP message handling."""

    @pytest.mark.asyncio
    async def test_handle_execution_contexts_cleared(
        self, mock_event_callback: AsyncMock, mock_cdp_session: AsyncMock
    ) -> None:
        """Runtime.executionContextsCleared resets page_ready and sets navigation flag."""
        monitor = AsyncWindowPropertyMonitor(event_callback_fn=mock_event_callback)
        monitor.page_ready = True
        monitor.navigation_detected = False

        msg = {"method": "Runtime.executionContextsCleared", "params": {}}

        result = await monitor.handle_window_property_message(msg, mock_cdp_session)

        assert result is True
        assert monitor.page_ready is False
        assert monitor.navigation_detected is True

    @pytest.mark.asyncio
    async def test_handle_frame_navigated(
        self, mock_event_callback: AsyncMock, mock_cdp_session: AsyncMock
    ) -> None:
        """Page.frameNavigated sets page_ready and navigation_detected."""
        monitor = AsyncWindowPropertyMonitor(event_callback_fn=mock_event_callback)
        monitor.page_ready = False
        monitor.navigation_detected = False

        msg = {
            "method": "Page.frameNavigated",
            "params": {"frame": {"url": "https://example.com"}},
        }

        result = await monitor.handle_window_property_message(msg, mock_cdp_session)

        assert result is True
        assert monitor.page_ready is True
        assert monitor.navigation_detected is True

    @pytest.mark.asyncio
    async def test_handle_dom_content_event_fired(
        self, mock_event_callback: AsyncMock, mock_cdp_session: AsyncMock
    ) -> None:
        """Page.domContentEventFired sets page_ready and navigation_detected."""
        monitor = AsyncWindowPropertyMonitor(event_callback_fn=mock_event_callback)
        monitor.page_ready = False

        msg = {"method": "Page.domContentEventFired", "params": {}}

        result = await monitor.handle_window_property_message(msg, mock_cdp_session)

        assert result is True
        assert monitor.page_ready is True
        assert monitor.navigation_detected is True

    @pytest.mark.asyncio
    async def test_handle_load_event_fired(
        self, mock_event_callback: AsyncMock, mock_cdp_session: AsyncMock
    ) -> None:
        """Page.loadEventFired sets page_ready and navigation_detected."""
        monitor = AsyncWindowPropertyMonitor(event_callback_fn=mock_event_callback)
        monitor.page_ready = False

        msg = {"method": "Page.loadEventFired", "params": {}}

        result = await monitor.handle_window_property_message(msg, mock_cdp_session)

        assert result is True
        assert monitor.page_ready is True
        assert monitor.navigation_detected is True

    @pytest.mark.asyncio
    async def test_navigation_during_collection_sets_pending(
        self, mock_event_callback: AsyncMock, mock_cdp_session: AsyncMock
    ) -> None:
        """Navigation during collection sets pending_navigation and abort flags."""
        import asyncio

        monitor = AsyncWindowPropertyMonitor(event_callback_fn=mock_event_callback)
        # Simulate a running collection task
        async def fake_collection():
            await asyncio.sleep(10)
        monitor.collection_task = asyncio.create_task(fake_collection())
        monitor.page_ready = True

        # Runtime.executionContextsCleared during collection
        msg = {"method": "Runtime.executionContextsCleared", "params": {}}

        result = await monitor.handle_window_property_message(msg, mock_cdp_session)

        assert result is True
        assert monitor.abort_collection is True
        assert monitor.pending_navigation is True

        # Cleanup
        monitor.collection_task.cancel()
        try:
            await monitor.collection_task
        except asyncio.CancelledError:
            pass

    def test_handle_unknown_message_returns_false(
        self, mock_event_callback: AsyncMock
    ) -> None:
        """Unknown method returns False."""
        import asyncio

        monitor = AsyncWindowPropertyMonitor(event_callback_fn=mock_event_callback)
        msg = {"method": "Unknown.method", "params": {}}
        mock_cdp_session = AsyncMock()

        # Run the async function synchronously
        result = asyncio.get_event_loop().run_until_complete(
            monitor.handle_window_property_message(msg, mock_cdp_session)
        )

        assert result is False


class TestAsyncWindowPropertyMonitorStateTracking:
    """Tests for window property state tracking logic."""

    def test_is_application_object_filters_native(self) -> None:
        """_is_application_object correctly filters native browser objects."""
        # Native objects should return False
        assert AsyncWindowPropertyMonitor._is_application_object("HTMLElement", "div") is False
        assert AsyncWindowPropertyMonitor._is_application_object("Window", "window") is False
        assert AsyncWindowPropertyMonitor._is_application_object(None, "document") is False
        assert AsyncWindowPropertyMonitor._is_application_object(None, "navigator") is False

        # Application objects should return True
        assert AsyncWindowPropertyMonitor._is_application_object("Object", "myApp") is True
        assert AsyncWindowPropertyMonitor._is_application_object(None, "customState") is True
        assert AsyncWindowPropertyMonitor._is_application_object("", "appConfig") is True


# =============================================================================
# AsyncInteractionMonitor CDP Message Handling Tests
# =============================================================================


class TestAsyncInteractionMonitorMessageHandling:
    """Tests for interaction monitor CDP message handling."""

    @pytest.mark.asyncio
    async def test_handle_binding_called_valid_payload(
        self, mock_event_callback: AsyncMock
    ) -> None:
        """Runtime.bindingCalled with valid payload emits interaction event."""
        monitor = AsyncInteractionMonitor(event_callback_fn=mock_event_callback)
        msg = {
            "method": "Runtime.bindingCalled",
            "params": {
                "name": "__webHackerInteractionLog",
                "payload": json.dumps({
                    "type": "click",
                    "timestamp": 1704067200000,
                    "url": "https://example.com",
                    "event": {
                        "mouse_button": 0,
                        "mouse_x_viewport": 100,
                        "mouse_y_viewport": 200,
                    },
                    "element": {
                        "tag_name": "button",
                        "id": "submit-btn",
                        "text": "Submit",
                        "bounding_box": {"x": 10, "y": 20, "width": 100, "height": 40},
                    },
                }),
            },
        }

        result = await monitor._on_binding_called(msg)

        assert result is True
        mock_event_callback.assert_called_once()
        call_args = mock_event_callback.call_args
        category, data = call_args[0]
        assert category == "AsyncInteractionMonitor"
        # Data should be the parsed model dump
        assert data["type"] == "click"

    @pytest.mark.asyncio
    async def test_handle_binding_called_updates_counters(
        self, mock_event_callback: AsyncMock
    ) -> None:
        """Runtime.bindingCalled updates interaction statistics."""
        monitor = AsyncInteractionMonitor(event_callback_fn=mock_event_callback)
        assert monitor.interaction_count == 0

        msg = {
            "method": "Runtime.bindingCalled",
            "params": {
                "name": "__webHackerInteractionLog",
                "payload": json.dumps({
                    "type": "keydown",
                    "timestamp": 1704067200000,
                    "url": "https://example.com",
                    "event": {"key_value": "Enter"},
                    "element": {"tag_name": "input", "id": "search"},
                }),
            },
        }

        await monitor._on_binding_called(msg)

        assert monitor.interaction_count == 1
        assert monitor.interaction_types["keydown"] == 1
        assert monitor.interactions_by_url["https://example.com"] == 1

    @pytest.mark.asyncio
    async def test_handle_binding_called_ignores_other_bindings(
        self, mock_event_callback: AsyncMock
    ) -> None:
        """Runtime.bindingCalled ignores bindings with different names."""
        monitor = AsyncInteractionMonitor(event_callback_fn=mock_event_callback)
        msg = {
            "method": "Runtime.bindingCalled",
            "params": {
                "name": "someOtherBinding",
                "payload": "{}",
            },
        }

        result = await monitor._on_binding_called(msg)

        assert result is False
        mock_event_callback.assert_not_called()

    @pytest.mark.asyncio
    async def test_handle_binding_called_handles_malformed_json(
        self, mock_event_callback: AsyncMock
    ) -> None:
        """Runtime.bindingCalled handles malformed JSON payload gracefully."""
        monitor = AsyncInteractionMonitor(event_callback_fn=mock_event_callback)
        msg = {
            "method": "Runtime.bindingCalled",
            "params": {
                "name": "__webHackerInteractionLog",
                "payload": "not valid json{",
            },
        }

        result = await monitor._on_binding_called(msg)

        assert result is False
        mock_event_callback.assert_not_called()

    @pytest.mark.asyncio
    async def test_handle_interaction_message_dispatches_binding_called(
        self, mock_event_callback: AsyncMock, mock_cdp_session: AsyncMock
    ) -> None:
        """handle_interaction_message routes Runtime.bindingCalled."""
        monitor = AsyncInteractionMonitor(event_callback_fn=mock_event_callback)
        msg = {
            "method": "Runtime.bindingCalled",
            "params": {
                "name": "__webHackerInteractionLog",
                "payload": json.dumps({
                    "type": "click",
                    "timestamp": 1000,
                    "url": "https://test.com",
                    "event": {},
                    "element": {"tag_name": "div"},
                }),
            },
        }

        result = await monitor.handle_interaction_message(msg, mock_cdp_session)

        assert result is True
        mock_event_callback.assert_called_once()

    def test_handle_unknown_message_returns_false(
        self, mock_event_callback: AsyncMock, mock_cdp_session: AsyncMock
    ) -> None:
        """Unknown method returns False."""
        import asyncio

        monitor = AsyncInteractionMonitor(event_callback_fn=mock_event_callback)
        msg = {"method": "Unknown.method", "params": {}}

        result = asyncio.get_event_loop().run_until_complete(
            monitor.handle_interaction_message(msg, mock_cdp_session)
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_handle_interaction_message_page_navigated_does_not_swallow(
        self, mock_event_callback: AsyncMock, mock_cdp_session: AsyncMock
    ) -> None:
        """Page.frameNavigated returns False to allow other handlers."""
        monitor = AsyncInteractionMonitor(event_callback_fn=mock_event_callback)
        msg = {
            "method": "Page.frameNavigated",
            "params": {"frame": {"url": "https://example.com"}},
        }

        result = await monitor.handle_interaction_message(msg, mock_cdp_session)

        assert result is False  # Don't swallow, let other monitors handle
