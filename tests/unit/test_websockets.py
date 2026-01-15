"""
tests/unit/test_websockets.py

Unit tests for websockets data models.
"""

import time

import pytest
from pydantic import ValidationError

from web_hacker.data_models.websockets import (
    ParsedWebSocketClientCommand,
    WebSocketClientCommandType,
    WebSocketCurrentUrlResponse,
    WebSocketErrorResponse,
    WebSocketMessageResponse,
    WebSocketPongResponse,
    WebSocketSessionEndedResponse,
    WebSocketSnapshotResponse,
    WebSocketStateResponse,
    WebSocketStatsResponse,
    WebSocketSubscribedResponse,
    WebSocketSuccessResponse,
    WebSocketToolInvocationRequestResponse,
    WebSocketToolInvocationResultResponse,
    WebSocketUpdateEvent,
    WebSocketUpdateResponse,
    WebSocketWarningResponse,
)


class TestParseWebsocketCommand:
    """
    Parsing behavior for client -> server websocket commands.
    """

    def test_ping(self) -> None:
        parsed = ParsedWebSocketClientCommand.parse_raw_websocket_command("ping")
        assert parsed.is_valid
        assert parsed.command == WebSocketClientCommandType.PING
        assert parsed.args is None

    def test_whitespace_and_case_normalization(self) -> None:
        parsed = ParsedWebSocketClientCommand.parse_raw_websocket_command("  PING  ")
        assert parsed.is_valid
        assert parsed.command == WebSocketClientCommandType.PING

    @pytest.mark.parametrize(
        "raw,expected_args",
        [
            ("subscribe", "all"),
            ("subscribe:", "all"),
            ("subscribe:all", "all"),
            (" SUBSCRIBE:ALL ", "all"),
        ],
    )
    def test_subscribe_all_variants(self, raw: str, expected_args: str) -> None:
        parsed = ParsedWebSocketClientCommand.parse_raw_websocket_command(raw)
        assert parsed.is_valid
        assert parsed.command == WebSocketClientCommandType.SUBSCRIBE
        assert parsed.args == expected_args
        assert parsed.subscribe_categories == set()

    def test_subscribe_specific_categories(self) -> None:
        parsed = ParsedWebSocketClientCommand.parse_raw_websocket_command("subscribe:AsyncNetworkMonitor, AsyncStorageMonitor")
        assert parsed.is_valid
        assert parsed.command == WebSocketClientCommandType.SUBSCRIBE
        # args are normalized to lowercase because the parser lowercases the message
        assert parsed.args == "asyncnetworkmonitor, asyncstoragemonitor"
        assert parsed.subscribe_categories == {"asyncnetworkmonitor", "asyncstoragemonitor"}

    def test_subscribe_missing_categories_after_colon(self) -> None:
        parsed = ParsedWebSocketClientCommand.parse_raw_websocket_command("subscribe:   ")
        # stripping makes this equivalent to "subscribe:", so it is treated as "all"
        assert parsed.is_valid
        assert parsed.command == WebSocketClientCommandType.SUBSCRIBE
        assert parsed.args == "all"
        assert parsed.subscribe_categories == set()

    def test_subscribe_invalid_format_without_colon(self) -> None:
        parsed = ParsedWebSocketClientCommand.parse_raw_websocket_command("subscribenotvalid")
        assert not parsed.is_valid
        assert parsed.command is None
        assert "must be of the form" in (parsed.error or "").lower()

    def test_navigate_success(self) -> None:
        parsed = ParsedWebSocketClientCommand.parse_raw_websocket_command("navigate:https://example.com")
        assert parsed.is_valid
        assert parsed.command == WebSocketClientCommandType.NAVIGATE
        assert parsed.navigate_url == "https://example.com"

    def test_navigate_missing_url(self) -> None:
        parsed = ParsedWebSocketClientCommand.parse_raw_websocket_command("navigate:")
        assert not parsed.is_valid
        assert parsed.command is None
        assert "missing url" in (parsed.error or "").lower()

    def test_navigate_missing_colon(self) -> None:
        parsed = ParsedWebSocketClientCommand.parse_raw_websocket_command("navigatehttps://example.com")
        assert not parsed.is_valid
        assert parsed.command is None
        assert "must be of the form" in (parsed.error or "").lower()

    def test_unknown_command(self) -> None:
        parsed = ParsedWebSocketClientCommand.parse_raw_websocket_command("not_a_command")
        assert not parsed.is_valid
        assert parsed.command is None
        assert "unknown command" in (parsed.error or "").lower()

    def test_get_stats(self) -> None:
        parsed = ParsedWebSocketClientCommand.parse_raw_websocket_command("get_stats")
        assert parsed.is_valid
        assert parsed.command == WebSocketClientCommandType.GET_STATS
        assert parsed.args is None

    def test_get_stats_case_insensitive(self) -> None:
        parsed = ParsedWebSocketClientCommand.parse_raw_websocket_command("  GET_STATS  ")
        assert parsed.is_valid
        assert parsed.command == WebSocketClientCommandType.GET_STATS

    def test_back(self) -> None:
        parsed = ParsedWebSocketClientCommand.parse_raw_websocket_command("back")
        assert parsed.is_valid
        assert parsed.command == WebSocketClientCommandType.BACK
        assert parsed.args is None

    def test_back_case_insensitive(self) -> None:
        parsed = ParsedWebSocketClientCommand.parse_raw_websocket_command("  BACK  ")
        assert parsed.is_valid
        assert parsed.command == WebSocketClientCommandType.BACK

    def test_get_current_url(self) -> None:
        parsed = ParsedWebSocketClientCommand.parse_raw_websocket_command("get_current_url")
        assert parsed.is_valid
        assert parsed.command == WebSocketClientCommandType.GET_CURRENT_URL
        assert parsed.args is None

    def test_get_current_url_case_insensitive(self) -> None:
        parsed = ParsedWebSocketClientCommand.parse_raw_websocket_command("  GET_CURRENT_URL  ")
        assert parsed.is_valid
        assert parsed.command == WebSocketClientCommandType.GET_CURRENT_URL

    def test_subscribe_categories_property_returns_none_for_non_subscribe(self) -> None:
        parsed = ParsedWebSocketClientCommand.parse_raw_websocket_command("ping")
        assert parsed.subscribe_categories is None

    def test_navigate_url_property_returns_none_for_non_navigate(self) -> None:
        parsed = ParsedWebSocketClientCommand.parse_raw_websocket_command("ping")
        assert parsed.navigate_url is None


class TestResponseModels:
    """
    Validation behavior for server -> client websocket responses.
    """

    def test_pong_response_has_timestamp(self) -> None:
        before = time.time()
        resp = WebSocketPongResponse()
        after = time.time()
        assert before <= resp.timestamp <= after
        assert resp.type == resp.type.PONG  # enum literal check

    def test_subscribed_response_defaults(self) -> None:
        resp = WebSocketSubscribedResponse()
        assert resp.type == resp.type.SUBSCRIBED
        assert resp.categories == "all"

    def test_error_response_round_trip(self) -> None:
        resp = WebSocketErrorResponse(message="oops")
        dumped = resp.model_dump()
        loaded = WebSocketErrorResponse(**dumped)
        assert loaded.message == "oops"
        assert loaded.type == resp.type

    def test_stats_response_requires_stats(self) -> None:
        with pytest.raises(ValidationError):
            WebSocketStatsResponse()  # type: ignore[call-arg]

    def test_success_response_requires_message(self) -> None:
        with pytest.raises(ValidationError):
            WebSocketSuccessResponse()  # type: ignore[call-arg]

    def test_snapshot_response_validation(self) -> None:
        resp = WebSocketSnapshotResponse(
            cdp_captures_id="abc123",
            stats={"total_events": 1},
        )
        assert resp.type == resp.type.SNAPSHOT
        assert resp.cdp_captures_id == "abc123"
        assert resp.stats["total_events"] == 1

    def test_update_response_requires_events(self) -> None:
        with pytest.raises(ValidationError):
            WebSocketUpdateResponse(stats={"total_events": 0})  # type: ignore[call-arg]

    def test_update_event_validation(self) -> None:
        event = WebSocketUpdateEvent(
            category="AsyncNetworkMonitor",
            summary={"foo": "bar"},
            timestamp=123.4,
        )
        assert event.category == "AsyncNetworkMonitor"
        assert event.summary == {"foo": "bar"}

    def test_update_event_missing_summary(self) -> None:
        with pytest.raises(ValidationError):
            WebSocketUpdateEvent(
                category="AsyncNetworkMonitor",
                timestamp=123.4,
            )  # type: ignore[call-arg]

    def test_update_response_round_trip(self) -> None:
        event = WebSocketUpdateEvent(
            category="AsyncNetworkMonitor",
            summary={"foo": "bar"},
            timestamp=123.4,
        )
        resp = WebSocketUpdateResponse(
            stats={"total_events": 1},
            events=[event],
        )
        dumped = resp.model_dump()
        loaded = WebSocketUpdateResponse(**dumped)
        assert loaded.events[0].category == "AsyncNetworkMonitor"
        assert loaded.stats["total_events"] == 1

    def test_warning_response_round_trip(self) -> None:
        resp = WebSocketWarningResponse(message="something went wrong")
        dumped = resp.model_dump()
        loaded = WebSocketWarningResponse(**dumped)
        assert loaded.message == "something went wrong"
        assert loaded.type == resp.type

    def test_warning_response_requires_message(self) -> None:
        with pytest.raises(ValidationError):
            WebSocketWarningResponse()  # type: ignore[call-arg]

    def test_session_ended_response_validation(self) -> None:
        resp = WebSocketSessionEndedResponse(reason="Timeout reached")
        assert resp.type == resp.type.SESSION_ENDED
        assert resp.reason == "Timeout reached"
        assert resp.error_message is None

    def test_session_ended_response_with_error(self) -> None:
        resp = WebSocketSessionEndedResponse(
            reason="Error occurred",
            error_message="Connection lost",
        )
        assert resp.reason == "Error occurred"
        assert resp.error_message == "Connection lost"

    def test_session_ended_response_requires_reason(self) -> None:
        with pytest.raises(ValidationError):
            WebSocketSessionEndedResponse()  # type: ignore[call-arg]

    def test_tool_invocation_request_response_validation(self) -> None:
        resp = WebSocketToolInvocationRequestResponse(
            invocation_id="abc-123",
            tool_name="navigate",
            tool_arguments={"url": "https://example.com"},
            explanation="Navigate to example.com",
        )
        assert resp.type == resp.type.TOOL_INVOCATION_REQUEST
        assert resp.invocation_id == "abc-123"
        assert resp.tool_name == "navigate"
        assert resp.tool_arguments == {"url": "https://example.com"}
        assert resp.explanation == "Navigate to example.com"

    def test_tool_invocation_request_response_requires_all_fields(self) -> None:
        with pytest.raises(ValidationError):
            WebSocketToolInvocationRequestResponse(
                invocation_id="abc-123",
                tool_name="navigate",
            )  # type: ignore[call-arg]

    def test_current_url_response_validation(self) -> None:
        resp = WebSocketCurrentUrlResponse(url="https://example.com/page")
        assert resp.type == resp.type.CURRENT_URL
        assert resp.url == "https://example.com/page"

    def test_current_url_response_requires_url(self) -> None:
        with pytest.raises(ValidationError):
            WebSocketCurrentUrlResponse()  # type: ignore[call-arg]


class TestParseWebsocketCommandAgentCommands:
    """
    Parsing behavior for agent-specific websocket commands (get_state, reset, send_message, etc.).
    """

    def test_ping(self) -> None:
        parsed = ParsedWebSocketClientCommand.parse_raw_websocket_command("ping")
        assert parsed.is_valid
        assert parsed.command == WebSocketClientCommandType.PING
        assert parsed.args is None

    def test_ping_case_insensitive(self) -> None:
        parsed = ParsedWebSocketClientCommand.parse_raw_websocket_command("  PING  ")
        assert parsed.is_valid
        assert parsed.command == WebSocketClientCommandType.PING

    def test_get_state(self) -> None:
        parsed = ParsedWebSocketClientCommand.parse_raw_websocket_command("get_state")
        assert parsed.is_valid
        assert parsed.command == WebSocketClientCommandType.GET_STATE
        assert parsed.args is None

    def test_get_state_case_insensitive(self) -> None:
        parsed = ParsedWebSocketClientCommand.parse_raw_websocket_command("  GET_STATE  ")
        assert parsed.is_valid
        assert parsed.command == WebSocketClientCommandType.GET_STATE

    def test_reset(self) -> None:
        parsed = ParsedWebSocketClientCommand.parse_raw_websocket_command("reset")
        assert parsed.is_valid
        assert parsed.command == WebSocketClientCommandType.RESET
        assert parsed.args is None

    def test_reset_case_insensitive(self) -> None:
        parsed = ParsedWebSocketClientCommand.parse_raw_websocket_command("  RESET  ")
        assert parsed.is_valid
        assert parsed.command == WebSocketClientCommandType.RESET

    def test_send_message_without_content(self) -> None:
        parsed = ParsedWebSocketClientCommand.parse_raw_websocket_command("send_message")
        assert parsed.is_valid
        assert parsed.command == WebSocketClientCommandType.SEND_MESSAGE
        assert parsed.args is None

    def test_send_message_with_content(self) -> None:
        parsed = ParsedWebSocketClientCommand.parse_raw_websocket_command("send_message:Hello World")
        assert parsed.is_valid
        assert parsed.command == WebSocketClientCommandType.SEND_MESSAGE
        assert parsed.args == "Hello World"

    def test_send_message_preserves_case(self) -> None:
        parsed = ParsedWebSocketClientCommand.parse_raw_websocket_command("SEND_MESSAGE:Hello World!")
        assert parsed.is_valid
        assert parsed.command == WebSocketClientCommandType.SEND_MESSAGE
        assert parsed.args == "Hello World!"

    def test_send_message_with_empty_content_after_colon(self) -> None:
        parsed = ParsedWebSocketClientCommand.parse_raw_websocket_command("send_message:")
        assert parsed.is_valid
        assert parsed.command == WebSocketClientCommandType.SEND_MESSAGE
        assert parsed.args is None

    def test_send_message_with_whitespace_content(self) -> None:
        parsed = ParsedWebSocketClientCommand.parse_raw_websocket_command("send_message:   ")
        assert parsed.is_valid
        assert parsed.command == WebSocketClientCommandType.SEND_MESSAGE
        assert parsed.args is None

    def test_send_message_invalid_format(self) -> None:
        parsed = ParsedWebSocketClientCommand.parse_raw_websocket_command("send_messageHello")
        assert not parsed.is_valid
        assert parsed.command is None
        assert "send_message" in (parsed.error or "").lower()

    def test_confirm_tool_success(self) -> None:
        parsed = ParsedWebSocketClientCommand.parse_raw_websocket_command("confirm_tool:abc-123")
        assert parsed.is_valid
        assert parsed.command == WebSocketClientCommandType.CONFIRM_TOOL
        assert parsed.args == "abc-123"

    def test_confirm_tool_preserves_id_case(self) -> None:
        parsed = ParsedWebSocketClientCommand.parse_raw_websocket_command("CONFIRM_TOOL:ABC-123-XYZ")
        assert parsed.is_valid
        assert parsed.command == WebSocketClientCommandType.CONFIRM_TOOL
        assert parsed.args == "ABC-123-XYZ"

    def test_confirm_tool_missing_id(self) -> None:
        parsed = ParsedWebSocketClientCommand.parse_raw_websocket_command("confirm_tool")
        assert not parsed.is_valid
        assert parsed.command is None
        assert "invocation_id" in (parsed.error or "").lower()

    def test_confirm_tool_empty_id(self) -> None:
        parsed = ParsedWebSocketClientCommand.parse_raw_websocket_command("confirm_tool:")
        assert not parsed.is_valid
        assert parsed.command is None
        assert "invocation_id" in (parsed.error or "").lower()

    def test_confirm_tool_invalid_format(self) -> None:
        parsed = ParsedWebSocketClientCommand.parse_raw_websocket_command("confirm_toolabc123")
        assert not parsed.is_valid
        assert parsed.command is None
        assert "confirm_tool" in (parsed.error or "").lower()

    def test_deny_tool_success(self) -> None:
        parsed = ParsedWebSocketClientCommand.parse_raw_websocket_command("deny_tool:abc-123")
        assert parsed.is_valid
        assert parsed.command == WebSocketClientCommandType.DENY_TOOL
        assert parsed.args == "abc-123"

    def test_deny_tool_with_reason(self) -> None:
        parsed = ParsedWebSocketClientCommand.parse_raw_websocket_command("deny_tool:abc-123:User declined")
        assert parsed.is_valid
        assert parsed.command == WebSocketClientCommandType.DENY_TOOL
        assert parsed.args == "abc-123:User declined"

    def test_deny_tool_preserves_case(self) -> None:
        parsed = ParsedWebSocketClientCommand.parse_raw_websocket_command("DENY_TOOL:ABC-123:Some Reason")
        assert parsed.is_valid
        assert parsed.command == WebSocketClientCommandType.DENY_TOOL
        assert parsed.args == "ABC-123:Some Reason"

    def test_deny_tool_missing_id(self) -> None:
        parsed = ParsedWebSocketClientCommand.parse_raw_websocket_command("deny_tool")
        assert not parsed.is_valid
        assert parsed.command is None
        assert "invocation_id" in (parsed.error or "").lower()

    def test_deny_tool_empty_id(self) -> None:
        parsed = ParsedWebSocketClientCommand.parse_raw_websocket_command("deny_tool:")
        assert not parsed.is_valid
        assert parsed.command is None
        assert "invocation_id" in (parsed.error or "").lower()

    def test_deny_tool_invalid_format(self) -> None:
        parsed = ParsedWebSocketClientCommand.parse_raw_websocket_command("deny_toolabc123")
        assert not parsed.is_valid
        assert parsed.command is None
        assert "deny_tool" in (parsed.error or "").lower()

    def test_unknown_command(self) -> None:
        parsed = ParsedWebSocketClientCommand.parse_raw_websocket_command("unknown_cmd")
        assert not parsed.is_valid
        assert parsed.command is None
        assert "unknown command" in (parsed.error or "").lower()


class TestWebsocketCommandAgentProperties:
    """
    Property behavior for agent-specific command properties (message_content, invocation_id, deny_reason).
    """

    def test_message_content_for_send_message(self) -> None:
        parsed = ParsedWebSocketClientCommand.parse_raw_websocket_command("send_message:Hello")
        assert parsed.message_content == "Hello"

    def test_message_content_empty_for_send_message_no_content(self) -> None:
        parsed = ParsedWebSocketClientCommand.parse_raw_websocket_command("send_message")
        assert parsed.message_content == ""

    def test_message_content_none_for_other_commands(self) -> None:
        parsed = ParsedWebSocketClientCommand.parse_raw_websocket_command("ping")
        assert parsed.message_content is None

    def test_invocation_id_for_confirm_tool(self) -> None:
        parsed = ParsedWebSocketClientCommand.parse_raw_websocket_command("confirm_tool:abc-123")
        assert parsed.invocation_id == "abc-123"

    def test_invocation_id_for_deny_tool(self) -> None:
        parsed = ParsedWebSocketClientCommand.parse_raw_websocket_command("deny_tool:xyz-789")
        assert parsed.invocation_id == "xyz-789"

    def test_invocation_id_for_deny_tool_with_reason(self) -> None:
        parsed = ParsedWebSocketClientCommand.parse_raw_websocket_command("deny_tool:abc-123:reason here")
        assert parsed.invocation_id == "abc-123"

    def test_invocation_id_none_for_other_commands(self) -> None:
        parsed = ParsedWebSocketClientCommand.parse_raw_websocket_command("ping")
        assert parsed.invocation_id is None

    def test_invocation_id_none_when_args_empty(self) -> None:
        # Create directly to test edge case
        parsed = ParsedWebSocketClientCommand(
            command=WebSocketClientCommandType.CONFIRM_TOOL,
            args=None,
        )
        assert parsed.invocation_id is None

    def test_deny_reason_extracted(self) -> None:
        parsed = ParsedWebSocketClientCommand.parse_raw_websocket_command("deny_tool:abc-123:User declined action")
        assert parsed.deny_reason == "User declined action"

    def test_deny_reason_none_when_no_reason(self) -> None:
        parsed = ParsedWebSocketClientCommand.parse_raw_websocket_command("deny_tool:abc-123")
        assert parsed.deny_reason is None

    def test_deny_reason_none_for_confirm_tool(self) -> None:
        parsed = ParsedWebSocketClientCommand.parse_raw_websocket_command("confirm_tool:abc-123")
        assert parsed.deny_reason is None

    def test_deny_reason_none_for_other_commands(self) -> None:
        parsed = ParsedWebSocketClientCommand.parse_raw_websocket_command("ping")
        assert parsed.deny_reason is None


class TestAgentResponseModels:
    """
    Validation behavior for agent-specific response models (message, state, tool invocation result).
    """

    def test_message_response_validation(self) -> None:
        before = time.time()
        resp = WebSocketMessageResponse(content="Hello from agent")
        after = time.time()
        assert resp.type == resp.type.MESSAGE
        assert resp.content == "Hello from agent"
        assert before <= resp.timestamp <= after

    def test_message_response_requires_content(self) -> None:
        with pytest.raises(ValidationError):
            WebSocketMessageResponse()  # type: ignore[call-arg]

    def test_message_response_round_trip(self) -> None:
        resp = WebSocketMessageResponse(content="Test message")
        dumped = resp.model_dump()
        loaded = WebSocketMessageResponse(**dumped)
        assert loaded.content == "Test message"
        assert loaded.type == resp.type

    def test_state_response_validation(self) -> None:
        resp = WebSocketStateResponse(
            guide_chat_id="session-abc-123",
            message_count=5,
            has_pending_tool_invocation=False,
        )
        assert resp.type == resp.type.STATE
        assert resp.guide_chat_id == "session-abc-123"
        assert resp.message_count == 5
        assert resp.has_pending_tool_invocation is False
        assert resp.pending_tool_invocation_id is None

    def test_state_response_with_pending_invocation(self) -> None:
        resp = WebSocketStateResponse(
            guide_chat_id="session-abc-123",
            message_count=3,
            has_pending_tool_invocation=True,
            pending_tool_invocation_id="inv-xyz-789",
        )
        assert resp.has_pending_tool_invocation is True
        assert resp.pending_tool_invocation_id == "inv-xyz-789"

    def test_state_response_requires_fields(self) -> None:
        with pytest.raises(ValidationError):
            WebSocketStateResponse(guide_chat_id="abc")  # type: ignore[call-arg]

    def test_state_response_round_trip(self) -> None:
        resp = WebSocketStateResponse(
            guide_chat_id="session-123",
            message_count=10,
            has_pending_tool_invocation=True,
            pending_tool_invocation_id="inv-456",
        )
        dumped = resp.model_dump()
        loaded = WebSocketStateResponse(**dumped)
        assert loaded.guide_chat_id == "session-123"
        assert loaded.message_count == 10
        assert loaded.has_pending_tool_invocation is True
        assert loaded.pending_tool_invocation_id == "inv-456"

    def test_tool_invocation_result_response_confirmed(self) -> None:
        resp = WebSocketToolInvocationResultResponse(
            invocation_id="inv-123",
            status="confirmed",
        )
        assert resp.type == resp.type.TOOL_INVOCATION_RESULT
        assert resp.invocation_id == "inv-123"
        assert resp.status == "confirmed"
        assert resp.result is None
        assert resp.error is None

    def test_tool_invocation_result_response_executed(self) -> None:
        resp = WebSocketToolInvocationResultResponse(
            invocation_id="inv-123",
            status="executed",
            result={"output": "success", "data": [1, 2, 3]},
        )
        assert resp.status == "executed"
        assert resp.result == {"output": "success", "data": [1, 2, 3]}

    def test_tool_invocation_result_response_denied(self) -> None:
        resp = WebSocketToolInvocationResultResponse(
            invocation_id="inv-123",
            status="denied",
        )
        assert resp.status == "denied"

    def test_tool_invocation_result_response_failed(self) -> None:
        resp = WebSocketToolInvocationResultResponse(
            invocation_id="inv-123",
            status="failed",
            error="Tool execution timed out",
        )
        assert resp.status == "failed"
        assert resp.error == "Tool execution timed out"

    def test_tool_invocation_result_response_requires_fields(self) -> None:
        with pytest.raises(ValidationError):
            WebSocketToolInvocationResultResponse(invocation_id="inv-123")  # type: ignore[call-arg]

    def test_tool_invocation_result_response_round_trip(self) -> None:
        resp = WebSocketToolInvocationResultResponse(
            invocation_id="inv-123",
            status="executed",
            result={"key": "value"},
        )
        dumped = resp.model_dump()
        loaded = WebSocketToolInvocationResultResponse(**dumped)
        assert loaded.invocation_id == "inv-123"
        assert loaded.status == "executed"
        assert loaded.result == {"key": "value"}
