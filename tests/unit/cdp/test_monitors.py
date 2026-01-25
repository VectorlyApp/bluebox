"""
tests/unit/test_cdp_monitors.py

Tests for CDP monitors: AbstractAsyncMonitor, AsyncNetworkMonitor,
AsyncStorageMonitor, AsyncWindowPropertyMonitor, AsyncInteractionMonitor.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock

from web_hacker.cdp.monitors.abstract_async_monitor import AbstractAsyncMonitor
from web_hacker.cdp.monitors.async_network_monitor import AsyncNetworkMonitor
from web_hacker.cdp.monitors.async_storage_monitor import AsyncStorageMonitor
from web_hacker.cdp.monitors.async_window_property_monitor import AsyncWindowPropertyMonitor
from web_hacker.cdp.monitors.async_interaction_monitor import AsyncInteractionMonitor


class TestAbstractAsyncMonitor:
    """Tests for AbstractAsyncMonitor base class."""

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
    """Tests for AsyncNetworkMonitor static methods."""

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
    """Tests for AsyncNetworkMonitor.get_ws_event_summary."""

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


class TestAsyncNetworkMonitorFileOperations:
    """Tests for AsyncNetworkMonitor file operations."""

    def test_consolidate_transactions(self, tmp_path: Path) -> None:
        """Reads JSONL and returns dict."""
        events_file = tmp_path / "events.jsonl"
        events_file.write_text(
            '{"request_id": "1", "url": "https://a.com", "method": "GET"}\n'
            '{"request_id": "2", "url": "https://b.com", "method": "POST"}\n'
        )

        result = AsyncNetworkMonitor.consolidate_transactions(str(events_file))

        assert "1" in result
        assert "2" in result
        assert result["1"]["url"] == "https://a.com"
        assert result["2"]["method"] == "POST"

    def test_consolidate_transactions_missing_file(self, tmp_path: Path) -> None:
        """Missing file should return empty dict."""
        missing_file = tmp_path / "nonexistent.jsonl"
        result = AsyncNetworkMonitor.consolidate_transactions(str(missing_file))
        assert result == {}

    def test_consolidate_transactions_with_output(self, tmp_path: Path) -> None:
        """Writes to output file when provided."""
        events_file = tmp_path / "events.jsonl"
        output_file = tmp_path / "output.json"
        events_file.write_text('{"request_id": "1", "url": "https://a.com", "method": "GET"}\n')

        AsyncNetworkMonitor.consolidate_transactions(str(events_file), str(output_file))

        assert output_file.exists()
        content = json.loads(output_file.read_text())
        assert "1" in content

    def test_generate_har_from_transactions(self, tmp_path: Path) -> None:
        """Generates valid HAR structure."""
        events_file = tmp_path / "events.jsonl"
        har_file = tmp_path / "network.har"
        events_file.write_text(
            '{"request_id": "1", "url": "https://a.com", "method": "GET", "status": 200}\n'
        )

        result = AsyncNetworkMonitor.generate_har_from_transactions(
            str(events_file), str(har_file), title="Test"
        )

        assert "log" in result
        assert result["log"]["version"] == "1.2"
        assert len(result["log"]["entries"]) == 1
        assert har_file.exists()

    def test_generate_har_missing_file(self, tmp_path: Path) -> None:
        """Missing file should create empty HAR."""
        missing_file = tmp_path / "nonexistent.jsonl"
        har_file = tmp_path / "network.har"

        result = AsyncNetworkMonitor.generate_har_from_transactions(
            str(missing_file), str(har_file)
        )

        assert result["log"]["entries"] == []
        assert har_file.exists()

    def test_create_har_entry_from_event(self) -> None:
        """Creates proper HAR entry from event dict."""
        event = {
            "url": "https://example.com/api?foo=bar",
            "method": "POST",
            "status": 200,
            "status_text": "OK",
            "request_headers": {"Content-Type": "application/json", "Cookie": "a=1; b=2"},
            "response_headers": {"Content-Type": "application/json"},
            "post_data": '{"key": "value"}',
            "response_body": '{"result": "ok"}',
            "mime_type": "application/json",
        }

        entry = AsyncNetworkMonitor._create_har_entry_from_event(event)

        assert entry is not None
        assert entry["request"]["method"] == "POST"
        assert entry["request"]["url"] == "https://example.com/api?foo=bar"
        assert entry["response"]["status"] == 200
        # query string parsing
        assert any(q["name"] == "foo" for q in entry["request"]["queryString"])
        # cookie parsing
        assert any(c["name"] == "a" for c in entry["request"]["cookies"])


class TestAsyncStorageMonitorWsEventSummary:
    """Tests for AsyncStorageMonitor.get_ws_event_summary."""

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
    """Tests for AsyncStorageMonitor state management methods."""

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
        """getAllCookies reply sets initial cookie state."""
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
        mock_event_callback.assert_called_once()
        call_args = mock_event_callback.call_args
        assert call_args[0][1].type == "initialCookies"

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
    """Tests for AsyncWindowPropertyMonitor static methods."""

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
    """Tests for AsyncWindowPropertyMonitor.get_ws_event_summary."""

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
    """Tests for AsyncInteractionMonitor.get_ws_event_summary."""

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
    """Tests for AsyncInteractionMonitor._parse_interaction_event."""

    def test_parse_interaction_event(self, mock_event_callback: AsyncMock) -> None:
        """Parses raw JS data into UiInteractionEvent."""
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
    """Tests for AsyncInteractionMonitor file operations."""

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
