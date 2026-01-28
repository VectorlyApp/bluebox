"""
tests/unit/test_async_cdp_session.py

Tests for AsyncCDPSession.
"""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock

from bluebox.cdp.async_cdp_session import AsyncCDPSession


class TestAsyncCDPSessionInit:
    """
    Tests for AsyncCDPSession initialization.
    """

    def test_init_creates_monitors(self, mock_event_callback: AsyncMock) -> None:
        """All 4 monitors should be instantiated."""
        session = AsyncCDPSession(
            ws_url="ws://localhost:9222/devtools/page/123",
            session_start_dtm="2024-01-01T00:00:00Z",
            event_callback_fn=mock_event_callback,
        )

        assert session.network_monitor is not None
        assert session.storage_monitor is not None
        assert session.window_property_monitor is not None
        assert session.interaction_monitor is not None

    def test_init_default_state(self, mock_event_callback: AsyncMock) -> None:
        """Session should have correct initial state."""
        session = AsyncCDPSession(
            ws_url="ws://localhost:9222/devtools/page/123",
            session_start_dtm="2024-01-01T00:00:00Z",
            event_callback_fn=mock_event_callback,
        )

        assert session.seq == 0
        assert session.ws is None
        assert session.page_session_id is None
        assert session.pending_responses == {}
        assert session._enabled_domains == set()


class TestAsyncCDPSessionSend:
    """
    Tests for AsyncCDPSession.send method.
    """

    @pytest.mark.asyncio
    async def test_send_increments_seq(self, mock_event_callback: AsyncMock) -> None:
        """Sequence ID should increment on each send."""
        session = AsyncCDPSession(
            ws_url="ws://localhost:9222/devtools/page/123",
            session_start_dtm="2024-01-01T00:00:00Z",
            event_callback_fn=mock_event_callback,
        )
        session.ws = AsyncMock()

        cmd_id_1 = await session.send("Test.method1")
        cmd_id_2 = await session.send("Test.method2")
        cmd_id_3 = await session.send("Test.method3")

        assert cmd_id_1 == 1
        assert cmd_id_2 == 2
        assert cmd_id_3 == 3
        assert session.seq == 3

    @pytest.mark.asyncio
    async def test_send_includes_session_id(self, mock_event_callback: AsyncMock) -> None:
        """SessionId should be included when available."""
        session = AsyncCDPSession(
            ws_url="ws://localhost:9222/devtools/page/123",
            session_start_dtm="2024-01-01T00:00:00Z",
            event_callback_fn=mock_event_callback,
        )
        session.ws = AsyncMock()
        session.page_session_id = "test-session-123"

        await session.send("Page.navigate", {"url": "https://example.com"})

        # verify ws.send was called with sessionId in message
        session.ws.send.assert_called_once()

        sent_msg = json.loads(session.ws.send.call_args[0][0])
        assert sent_msg["sessionId"] == "test-session-123"

    @pytest.mark.asyncio
    async def test_send_without_session_id(self, mock_event_callback: AsyncMock) -> None:
        """Message without sessionId when not available."""
        session = AsyncCDPSession(
            ws_url="ws://localhost:9222/devtools/page/123",
            session_start_dtm="2024-01-01T00:00:00Z",
            event_callback_fn=mock_event_callback,
        )
        session.ws = AsyncMock()
        session.page_session_id = None

        await session.send("Target.getTargets")

        sent_msg = json.loads(session.ws.send.call_args[0][0])
        assert "sessionId" not in sent_msg

    @pytest.mark.asyncio
    async def test_send_raises_without_ws(self, mock_event_callback: AsyncMock) -> None:
        """Should raise RuntimeError when WebSocket not connected."""
        session = AsyncCDPSession(
            ws_url="ws://localhost:9222/devtools/page/123",
            session_start_dtm="2024-01-01T00:00:00Z",
            event_callback_fn=mock_event_callback,
        )

        with pytest.raises(RuntimeError, match="WebSocket not connected"):
            await session.send("Test.method")


class TestAsyncCDPSessionEnableDomain:
    """
    Tests for AsyncCDPSession.enable_domain method.
    """

    @pytest.mark.asyncio
    async def test_enable_domain_idempotent(self, mock_event_callback: AsyncMock) -> None:
        """Second call to enable_domain should skip enable."""
        session = AsyncCDPSession(
            ws_url="ws://localhost:9222/devtools/page/123",
            session_start_dtm="2024-01-01T00:00:00Z",
            event_callback_fn=mock_event_callback,
        )
        session.ws = AsyncMock()

        # mock send_and_wait to return success
        session.send_and_wait = AsyncMock(return_value={})

        # first call should enable
        await session.enable_domain("Page")
        assert "Page" in session._enabled_domains
        assert session.send_and_wait.call_count == 1

        # second call should skip
        await session.enable_domain("Page")
        assert session.send_and_wait.call_count == 1  # still 1

    @pytest.mark.asyncio
    async def test_enable_domain_with_params(self, mock_event_callback: AsyncMock) -> None:
        """enable_domain should pass params to enable call."""
        session = AsyncCDPSession(
            ws_url="ws://localhost:9222/devtools/page/123",
            session_start_dtm="2024-01-01T00:00:00Z",
            event_callback_fn=mock_event_callback,
        )
        session.ws = AsyncMock()
        session.send_and_wait = AsyncMock(return_value={})

        await session.enable_domain("Network", params={"maxTotalBufferSize": 1000})

        session.send_and_wait.assert_called_once()
        call_args = session.send_and_wait.call_args
        assert call_args[1]["method"] == "Network.enable"
        assert call_args[1]["params"] == {"maxTotalBufferSize": 1000}


class TestAsyncCDPSessionHandleMessage:
    """
    Tests for AsyncCDPSession.handle_message routing.
    """

    @pytest.mark.asyncio
    async def test_handle_message_routes_to_network(
        self, mock_event_callback: AsyncMock
    ) -> None:
        """Network events should be routed to network monitor."""
        session = AsyncCDPSession(
            ws_url="ws://localhost:9222/devtools/page/123",
            session_start_dtm="2024-01-01T00:00:00Z",
            event_callback_fn=mock_event_callback,
        )

        # mock network monitor
        session.network_monitor.handle_network_message = AsyncMock(return_value=True)
        session.storage_monitor.handle_storage_message = AsyncMock(return_value=False)

        msg = {"method": "Network.requestWillBeSent", "params": {}}
        await session.handle_message(msg)

        session.network_monitor.handle_network_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_message_routes_to_storage(
        self, mock_event_callback: AsyncMock
    ) -> None:
        """Storage events should be routed to storage monitor."""
        session = AsyncCDPSession(
            ws_url="ws://localhost:9222/devtools/page/123",
            session_start_dtm="2024-01-01T00:00:00Z",
            event_callback_fn=mock_event_callback,
        )

        # mock monitors
        session.network_monitor.handle_network_message = AsyncMock(return_value=False)
        session.storage_monitor.handle_storage_message = AsyncMock(return_value=True)

        msg = {"method": "DOMStorage.domStorageItemAdded", "params": {}}
        await session.handle_message(msg)

        session.storage_monitor.handle_storage_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_message_captures_session_id(
        self, mock_event_callback: AsyncMock
    ) -> None:
        """Target.attachedToTarget should capture sessionId."""
        session = AsyncCDPSession(
            ws_url="ws://localhost:9222/devtools/page/123",
            session_start_dtm="2024-01-01T00:00:00Z",
            event_callback_fn=mock_event_callback,
        )

        # mock all monitors to not handle
        session.network_monitor.handle_network_message = AsyncMock(return_value=False)
        session.storage_monitor.handle_storage_message = AsyncMock(return_value=False)
        session.window_property_monitor.handle_window_property_message = AsyncMock(
            return_value=False
        )
        session.interaction_monitor.handle_interaction_message = AsyncMock(return_value=False)

        msg = {
            "method": "Target.attachedToTarget",
            "params": {
                "sessionId": "captured-session-id",
                "targetInfo": {"type": "page"},
            },
        }
        await session.handle_message(msg)

        assert session.page_session_id == "captured-session-id"


class TestAsyncCDPSessionCommandReply:
    """
    Tests for AsyncCDPSession command reply handling.
    """

    @pytest.mark.asyncio
    async def test_handle_command_reply_resolves_future(
        self, mock_event_callback: AsyncMock
    ) -> None:
        """Command reply should resolve pending future."""
        session = AsyncCDPSession(
            ws_url="ws://localhost:9222/devtools/page/123",
            session_start_dtm="2024-01-01T00:00:00Z",
            event_callback_fn=mock_event_callback,
        )

        # create pending future
        future = asyncio.Future()
        session.pending_responses[123] = future

        # mock monitors to not handle
        session.network_monitor.handle_network_command_reply = AsyncMock(return_value=False)
        session.storage_monitor.handle_storage_command_reply = AsyncMock(return_value=False)
        session.interaction_monitor.handle_interaction_command_reply = AsyncMock(
            return_value=False
        )

        # handle reply
        await session._handle_command_reply({"id": 123, "result": {"data": "test"}})

        assert future.done()
        assert future.result() == {"data": "test"}
        assert 123 not in session.pending_responses


class TestAsyncCDPSessionGetMonitoringSummary:
    """
    Tests for AsyncCDPSession.get_monitoring_summary method.
    """

    def test_get_monitoring_summary(self, mock_event_callback: AsyncMock) -> None:
        """Should aggregate all monitor summaries."""
        session = AsyncCDPSession(
            ws_url="ws://localhost:9222/devtools/page/123",
            session_start_dtm="2024-01-01T00:00:00Z",
            event_callback_fn=mock_event_callback,
        )

        summary = session.get_monitoring_summary()

        assert "network" in summary
        assert "storage" in summary
        assert "window_properties" in summary
        assert "interactions" in summary

        # verify structure from each monitor
        assert "requests_tracked" in summary["network"]
        assert "cookies_count" in summary["storage"]
        assert "total_keys" in summary["window_properties"]
        assert "interactions_logged" in summary["interactions"]


class TestAsyncCDPSessionSendAndWait:
    """
    Tests for AsyncCDPSession.send_and_wait method.
    """

    @pytest.mark.asyncio
    async def test_send_and_wait_returns_result(
        self, mock_event_callback: AsyncMock
    ) -> None:
        """send_and_wait should return result from response."""
        session = AsyncCDPSession(
            ws_url="ws://localhost:9222/devtools/page/123",
            session_start_dtm="2024-01-01T00:00:00Z",
            event_callback_fn=mock_event_callback,
        )
        session.ws = AsyncMock()

        async def resolve_future():
            # wait for pending response to be added
            await asyncio.sleep(0.01)
            cmd_id = 1
            if cmd_id in session.pending_responses:
                session.pending_responses[cmd_id].set_result({"test": "data"})

        asyncio.create_task(resolve_future())

        result = await session.send_and_wait("Test.method", timeout=1.0)
        assert result == {"test": "data"}

    @pytest.mark.asyncio
    async def test_send_and_wait_timeout(self, mock_event_callback: AsyncMock) -> None:
        """send_and_wait should raise TimeoutError on timeout."""
        session = AsyncCDPSession(
            ws_url="ws://localhost:9222/devtools/page/123",
            session_start_dtm="2024-01-01T00:00:00Z",
            event_callback_fn=mock_event_callback,
        )
        session.ws = AsyncMock()

        with pytest.raises(TimeoutError, match="timed out"):
            await session.send_and_wait("Test.method", timeout=0.01)


class TestAsyncCDPSessionWaitForPageSessionId:
    """
    Tests for AsyncCDPSession.wait_for_page_session_id method.
    """

    @pytest.mark.asyncio
    async def test_wait_for_page_session_id_already_set(
        self, mock_event_callback: AsyncMock
    ) -> None:
        """Should return immediately if session_id already set."""
        session = AsyncCDPSession(
            ws_url="ws://localhost:9222/devtools/page/123",
            session_start_dtm="2024-01-01T00:00:00Z",
            event_callback_fn=mock_event_callback,
        )
        session.page_session_id = "existing-session-id"

        result = await session.wait_for_page_session_id(timeout=0.1)
        assert result == "existing-session-id"

    @pytest.mark.asyncio
    async def test_wait_for_page_session_id_timeout(
        self, mock_event_callback: AsyncMock
    ) -> None:
        """Should return None on timeout."""
        session = AsyncCDPSession(
            ws_url="ws://localhost:9222/devtools/page/123",
            session_start_dtm="2024-01-01T00:00:00Z",
            event_callback_fn=mock_event_callback,
        )

        result = await session.wait_for_page_session_id(timeout=0.01)
        assert result is None

    @pytest.mark.asyncio
    async def test_wait_for_page_session_id_wait_and_receive(
        self, mock_event_callback: AsyncMock
    ) -> None:
        """Should return session_id when it becomes available."""
        session = AsyncCDPSession(
            ws_url="ws://localhost:9222/devtools/page/123",
            session_start_dtm="2024-01-01T00:00:00Z",
            event_callback_fn=mock_event_callback,
        )

        async def set_session_id():
            await asyncio.sleep(0.01)
            session.page_session_id = "new-session-id"
            session._session_id_event.set()

        asyncio.create_task(set_session_id())

        result = await session.wait_for_page_session_id(timeout=1.0)
        assert result == "new-session-id"
