"""
tests/unit/llms/test_anthropic_client.py

Unit tests for Anthropic client, including retry behavior with exponential backoff.
"""

from collections.abc import Generator
from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
from anthropic import APIStatusError, RateLimitError

from bluebox.data_models.llms.vendors import AnthropicModel
from bluebox.llms.anthropic_client import (
    AnthropicClient,
    _is_retryable_error,
    _calculate_backoff,
    MAX_RETRIES,
    BASE_DELAY,
    MAX_DELAY,
)


# --- Helper function tests ---


class TestIsRetryableError:
    """Tests for _is_retryable_error helper function."""

    def test_rate_limit_error_is_retryable(self) -> None:
        """RateLimitError should be retryable."""
        error = MagicMock(spec=RateLimitError)
        error.__class__ = RateLimitError
        # Need to actually create a RateLimitError-like object
        with patch("bluebox.llms.anthropic_client.RateLimitError", RateLimitError):
            # Create a proper mock that isinstance checks work on
            class MockRateLimitError(RateLimitError):
                def __init__(self) -> None:
                    pass

            mock_error = MockRateLimitError()
            assert _is_retryable_error(mock_error) is True

    def test_overloaded_error_is_retryable(self) -> None:
        """APIStatusError with overloaded_error type should be retryable."""
        error = MagicMock(spec=APIStatusError)
        error.body = {"error": {"type": "overloaded_error", "message": "Overloaded"}}
        error.status_code = 529
        # Make isinstance check work
        error.__class__ = APIStatusError
        assert _is_retryable_error(error) is True

    def test_api_error_is_retryable(self) -> None:
        """APIStatusError with api_error type should be retryable."""
        error = MagicMock(spec=APIStatusError)
        error.body = {"error": {"type": "api_error", "message": "Internal error"}}
        error.status_code = 500
        error.__class__ = APIStatusError
        assert _is_retryable_error(error) is True

    def test_status_429_is_retryable(self) -> None:
        """APIStatusError with status 429 should be retryable."""
        error = MagicMock(spec=APIStatusError)
        error.body = {}
        error.status_code = 429
        error.__class__ = APIStatusError
        assert _is_retryable_error(error) is True

    def test_status_500_is_retryable(self) -> None:
        """APIStatusError with status 500 should be retryable."""
        error = MagicMock(spec=APIStatusError)
        error.body = {}
        error.status_code = 500
        error.__class__ = APIStatusError
        assert _is_retryable_error(error) is True

    def test_status_502_is_retryable(self) -> None:
        """APIStatusError with status 502 should be retryable."""
        error = MagicMock(spec=APIStatusError)
        error.body = {}
        error.status_code = 502
        error.__class__ = APIStatusError
        assert _is_retryable_error(error) is True

    def test_status_503_is_retryable(self) -> None:
        """APIStatusError with status 503 should be retryable."""
        error = MagicMock(spec=APIStatusError)
        error.body = {}
        error.status_code = 503
        error.__class__ = APIStatusError
        assert _is_retryable_error(error) is True

    def test_status_529_is_retryable(self) -> None:
        """APIStatusError with status 529 (overloaded) should be retryable."""
        error = MagicMock(spec=APIStatusError)
        error.body = {}
        error.status_code = 529
        error.__class__ = APIStatusError
        assert _is_retryable_error(error) is True

    def test_status_400_is_not_retryable(self) -> None:
        """APIStatusError with status 400 should NOT be retryable."""
        error = MagicMock(spec=APIStatusError)
        error.body = {"error": {"type": "invalid_request_error"}}
        error.status_code = 400
        error.__class__ = APIStatusError
        assert _is_retryable_error(error) is False

    def test_status_401_is_not_retryable(self) -> None:
        """APIStatusError with status 401 should NOT be retryable."""
        error = MagicMock(spec=APIStatusError)
        error.body = {"error": {"type": "authentication_error"}}
        error.status_code = 401
        error.__class__ = APIStatusError
        assert _is_retryable_error(error) is False

    def test_status_404_is_not_retryable(self) -> None:
        """APIStatusError with status 404 should NOT be retryable."""
        error = MagicMock(spec=APIStatusError)
        error.body = {"error": {"type": "not_found_error"}}
        error.status_code = 404
        error.__class__ = APIStatusError
        assert _is_retryable_error(error) is False

    def test_generic_exception_is_not_retryable(self) -> None:
        """Generic exceptions should NOT be retryable."""
        error = ValueError("Something went wrong")
        assert _is_retryable_error(error) is False


class TestCalculateBackoff:
    """Tests for _calculate_backoff helper function."""

    def test_first_attempt_base_delay(self) -> None:
        """First attempt (0) should have delay around BASE_DELAY."""
        delay = _calculate_backoff(0)
        # With jitter, delay should be between BASE_DELAY and BASE_DELAY * 1.5
        assert BASE_DELAY <= delay <= BASE_DELAY * 1.5

    def test_second_attempt_doubled_delay(self) -> None:
        """Second attempt (1) should have delay around 2 * BASE_DELAY."""
        delay = _calculate_backoff(1)
        expected_base = BASE_DELAY * 2
        assert expected_base <= delay <= expected_base * 1.5

    def test_third_attempt_quadrupled_delay(self) -> None:
        """Third attempt (2) should have delay around 4 * BASE_DELAY."""
        delay = _calculate_backoff(2)
        expected_base = BASE_DELAY * 4
        assert expected_base <= delay <= expected_base * 1.5

    def test_max_delay_cap(self) -> None:
        """Delay should be capped at MAX_DELAY (plus jitter)."""
        # Very high attempt number
        delay = _calculate_backoff(100)
        assert delay <= MAX_DELAY * 1.5

    def test_delay_increases_with_attempts(self) -> None:
        """Delay should generally increase with attempt number."""
        delays = [_calculate_backoff(i) for i in range(5)]
        # Due to jitter, we can't guarantee strict increase, but average should increase
        # Just check that later delays are generally larger
        assert sum(delays[3:]) / 2 > sum(delays[:2]) / 2


# --- Anthropic Client tests ---


class TestAnthropicClientRetry:
    """Tests for AnthropicClient retry behavior."""

    @pytest.fixture
    def mock_anthropic(self) -> Generator[MagicMock, None, None]:
        """Mock the Anthropic client."""
        with patch("bluebox.llms.anthropic_client.Anthropic") as mock_cls:
            with patch("bluebox.llms.anthropic_client.AsyncAnthropic"):
                mock_instance = MagicMock()
                mock_cls.return_value = mock_instance
                yield mock_instance

    @pytest.fixture
    def client(self, mock_anthropic: MagicMock) -> AnthropicClient:
        """Create an AnthropicClient instance for testing."""
        with patch("bluebox.llms.anthropic_client.Config") as mock_config:
            mock_config.ANTHROPIC_API_KEY = "test-key"
            return AnthropicClient(model=AnthropicModel.CLAUDE_SONNET_4_5)

    def _create_mock_response(self, content: str = "Hello!") -> MagicMock:
        """Create a mock Anthropic response."""
        mock_response = MagicMock()
        mock_text_block = MagicMock()
        mock_text_block.type = "text"
        mock_text_block.text = content
        mock_response.content = [mock_text_block]
        mock_response.id = "msg_123"
        return mock_response

    def _create_overloaded_error(self) -> Exception:
        """Create a mock overloaded error."""
        error = MagicMock(spec=APIStatusError)
        error.body = {"error": {"type": "overloaded_error", "message": "Overloaded"}}
        error.status_code = 529
        error.__class__ = APIStatusError
        return error

    # --- call_sync tests ---

    def test_call_sync_success_no_retry(
        self,
        client: AnthropicClient,
        mock_anthropic: MagicMock,
    ) -> None:
        """Successful call should not retry."""
        mock_anthropic.messages.create.return_value = self._create_mock_response()

        response = client.call_sync(input="Hello")

        assert response.content == "Hello!"
        assert mock_anthropic.messages.create.call_count == 1

    def test_call_sync_retries_on_overloaded_error(
        self,
        client: AnthropicClient,
        mock_anthropic: MagicMock,
    ) -> None:
        """Should retry on overloaded error."""
        # Create a real-ish error for isinstance checks
        overloaded_error = APIStatusError(
            message="Overloaded",
            response=MagicMock(status_code=529),
            body={"error": {"type": "overloaded_error", "message": "Overloaded"}},
        )

        # Fail twice, then succeed
        mock_anthropic.messages.create.side_effect = [
            overloaded_error,
            overloaded_error,
            self._create_mock_response("Success after retries"),
        ]

        with patch("bluebox.llms.anthropic_client.time.sleep"):
            response = client.call_sync(input="Hello")

        assert response.content == "Success after retries"
        assert mock_anthropic.messages.create.call_count == 3

    def test_call_sync_retries_with_exponential_backoff(
        self,
        client: AnthropicClient,
        mock_anthropic: MagicMock,
    ) -> None:
        """Should use exponential backoff between retries."""
        overloaded_error = APIStatusError(
            message="Overloaded",
            response=MagicMock(status_code=529),
            body={"error": {"type": "overloaded_error", "message": "Overloaded"}},
        )

        mock_anthropic.messages.create.side_effect = [
            overloaded_error,
            overloaded_error,
            self._create_mock_response(),
        ]

        sleep_calls: list[float] = []
        with patch("bluebox.llms.anthropic_client.time.sleep", side_effect=lambda x: sleep_calls.append(x)):
            client.call_sync(input="Hello")

        # Should have slept twice (after 1st and 2nd failures)
        assert len(sleep_calls) == 2
        # Second delay should be larger than first (exponential)
        assert sleep_calls[1] > sleep_calls[0]

    def test_call_sync_max_retries_exceeded(
        self,
        client: AnthropicClient,
        mock_anthropic: MagicMock,
    ) -> None:
        """Should raise after max retries exceeded."""
        overloaded_error = APIStatusError(
            message="Overloaded",
            response=MagicMock(status_code=529),
            body={"error": {"type": "overloaded_error", "message": "Overloaded"}},
        )

        # Always fail
        mock_anthropic.messages.create.side_effect = overloaded_error

        with patch("bluebox.llms.anthropic_client.time.sleep"):
            with pytest.raises(APIStatusError):
                client.call_sync(input="Hello")

        assert mock_anthropic.messages.create.call_count == MAX_RETRIES

    def test_call_sync_no_retry_on_non_retryable_error(
        self,
        client: AnthropicClient,
        mock_anthropic: MagicMock,
    ) -> None:
        """Should not retry on non-retryable errors like 400."""
        bad_request_error = APIStatusError(
            message="Bad request",
            response=MagicMock(status_code=400),
            body={"error": {"type": "invalid_request_error", "message": "Bad request"}},
        )

        mock_anthropic.messages.create.side_effect = bad_request_error

        with pytest.raises(APIStatusError):
            client.call_sync(input="Hello")

        # Should fail immediately without retrying
        assert mock_anthropic.messages.create.call_count == 1


class TestAnthropicClientStreamRetry:
    """Tests for AnthropicClient streaming retry behavior."""

    @pytest.fixture
    def mock_anthropic(self) -> Generator[MagicMock, None, None]:
        """Mock the Anthropic client."""
        with patch("bluebox.llms.anthropic_client.Anthropic") as mock_cls:
            with patch("bluebox.llms.anthropic_client.AsyncAnthropic"):
                mock_instance = MagicMock()
                mock_cls.return_value = mock_instance
                yield mock_instance

    @pytest.fixture
    def client(self, mock_anthropic: MagicMock) -> AnthropicClient:
        """Create an AnthropicClient instance for testing."""
        with patch("bluebox.llms.anthropic_client.Config") as mock_config:
            mock_config.ANTHROPIC_API_KEY = "test-key"
            return AnthropicClient(model=AnthropicModel.CLAUDE_SONNET_4_5)

    def _create_mock_stream_events(self, text: str = "Hello!") -> list[MagicMock]:
        """Create mock streaming events."""
        events = []

        # Text delta event
        text_delta = MagicMock()
        text_delta.type = "content_block_delta"
        text_delta.delta = MagicMock()
        text_delta.delta.type = "text_delta"
        text_delta.delta.text = text
        events.append(text_delta)

        return events

    @contextmanager
    def _create_mock_stream_context(self, events: list[MagicMock]) -> Generator[MagicMock, None, None]:
        """Create a mock stream context manager."""
        mock_stream = MagicMock()
        mock_stream.__iter__ = lambda self: iter(events)
        yield mock_stream

    def test_call_stream_sync_success_no_retry(
        self,
        client: AnthropicClient,
        mock_anthropic: MagicMock,
    ) -> None:
        """Successful streaming call should not retry."""
        events = self._create_mock_stream_events("Hello!")

        @contextmanager
        def mock_stream_context(**kwargs: Any) -> Generator[MagicMock, None, None]:
            mock_stream = MagicMock()
            mock_stream.__iter__ = lambda self: iter(events)
            yield mock_stream

        mock_anthropic.messages.stream = mock_stream_context

        results = list(client.call_stream_sync(input="Hello"))

        # Should have text chunk and final response
        assert len(results) == 2
        assert results[0] == "Hello!"

    def test_call_stream_sync_retries_on_error(
        self,
        client: AnthropicClient,
        mock_anthropic: MagicMock,
    ) -> None:
        """Streaming should retry on transient errors."""
        overloaded_error = APIStatusError(
            message="Overloaded",
            response=MagicMock(status_code=529),
            body={"error": {"type": "overloaded_error", "message": "Overloaded"}},
        )

        call_count = 0
        events = self._create_mock_stream_events("Success!")

        @contextmanager
        def mock_stream_context(**kwargs: Any) -> Generator[MagicMock, None, None]:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise overloaded_error
            mock_stream = MagicMock()
            mock_stream.__iter__ = lambda self: iter(events)
            yield mock_stream

        mock_anthropic.messages.stream = mock_stream_context

        with patch("bluebox.llms.anthropic_client.time.sleep"):
            results = list(client.call_stream_sync(input="Hello"))

        assert call_count == 3
        assert results[0] == "Success!"

    def test_call_stream_sync_max_retries_exceeded(
        self,
        client: AnthropicClient,
        mock_anthropic: MagicMock,
    ) -> None:
        """Streaming should raise after max retries."""
        overloaded_error = APIStatusError(
            message="Overloaded",
            response=MagicMock(status_code=529),
            body={"error": {"type": "overloaded_error", "message": "Overloaded"}},
        )

        call_count = 0

        @contextmanager
        def mock_stream_context(**kwargs: Any) -> Generator[MagicMock, None, None]:
            nonlocal call_count
            call_count += 1
            raise overloaded_error
            yield  # type: ignore[misc]  # Never reached

        mock_anthropic.messages.stream = mock_stream_context

        with patch("bluebox.llms.anthropic_client.time.sleep"):
            with pytest.raises(APIStatusError):
                list(client.call_stream_sync(input="Hello"))

        assert call_count == MAX_RETRIES


class TestAnthropicClientToolRegistration:
    """Tests for AnthropicClient tool registration."""

    @pytest.fixture
    def mock_anthropic(self) -> Generator[MagicMock, None, None]:
        """Mock the Anthropic client."""
        with patch("bluebox.llms.anthropic_client.Anthropic") as mock_cls:
            with patch("bluebox.llms.anthropic_client.AsyncAnthropic"):
                mock_instance = MagicMock()
                mock_cls.return_value = mock_instance
                yield mock_instance

    @pytest.fixture
    def client(self, mock_anthropic: MagicMock) -> AnthropicClient:
        """Create an AnthropicClient instance for testing."""
        with patch("bluebox.llms.anthropic_client.Config") as mock_config:
            mock_config.ANTHROPIC_API_KEY = "test-key"
            return AnthropicClient(model=AnthropicModel.CLAUDE_SONNET_4_5)

    def test_register_tool(self, client: AnthropicClient) -> None:
        """Test that tools are registered in Anthropic format."""
        client.register_tool(
            name="test_tool",
            description="A test tool",
            parameters={"type": "object", "properties": {"arg": {"type": "string"}}},
        )

        assert len(client.tools) == 1
        assert client.tools[0]["name"] == "test_tool"
        assert client.tools[0]["description"] == "A test tool"
        assert "input_schema" in client.tools[0]

    def test_clear_tools(self, client: AnthropicClient) -> None:
        """Test that tools can be cleared."""
        client.register_tool(
            name="test_tool",
            description="A test tool",
            parameters={"type": "object", "properties": {}},
        )
        assert len(client.tools) == 1

        client.clear_tools()
        assert len(client.tools) == 0

    def test_register_tool_upsert_behavior(self, client: AnthropicClient) -> None:
        """Test that registering a tool with the same name updates instead of duplicating."""
        client.register_tool(
            name="test_tool",
            description="Original description",
            parameters={"type": "object", "properties": {"arg1": {"type": "string"}}},
        )
        assert len(client.tools) == 1
        assert client.tools[0]["description"] == "Original description"

        # Register same tool name with updated description
        client.register_tool(
            name="test_tool",
            description="Updated description",
            parameters={"type": "object", "properties": {"arg2": {"type": "integer"}}},
        )

        # Should still have only 1 tool, but with updated content
        assert len(client.tools) == 1
        assert client.tools[0]["description"] == "Updated description"
        assert "arg2" in client.tools[0]["input_schema"]["properties"]


class TestToolChoiceNormalization:
    """Tests for tool_choice normalization."""

    @pytest.fixture
    def mock_anthropic(self) -> Generator[MagicMock, None, None]:
        """Mock the Anthropic client."""
        with patch("bluebox.llms.anthropic_client.Anthropic") as mock_cls:
            with patch("bluebox.llms.anthropic_client.AsyncAnthropic"):
                mock_instance = MagicMock()
                mock_cls.return_value = mock_instance
                yield mock_instance

    @pytest.fixture
    def client(self, mock_anthropic: MagicMock) -> AnthropicClient:
        """Create an AnthropicClient instance for testing."""
        with patch("bluebox.llms.anthropic_client.Config") as mock_config:
            mock_config.ANTHROPIC_API_KEY = "test-key"
            return AnthropicClient(model=AnthropicModel.CLAUDE_SONNET_4_5)

    def test_normalize_auto(self, client: AnthropicClient) -> None:
        """Test that 'auto' normalizes to {'type': 'auto'}."""
        result = client._normalize_tool_choice("auto")
        assert result == {"type": "auto"}

    def test_normalize_required(self, client: AnthropicClient) -> None:
        """Test that 'required' normalizes to {'type': 'any'}."""
        result = client._normalize_tool_choice("required")
        assert result == {"type": "any"}

    def test_normalize_tool_name(self, client: AnthropicClient) -> None:
        """Test that tool name normalizes to tool dict."""
        result = client._normalize_tool_choice("get_weather")
        assert result == {"type": "tool", "name": "get_weather"}

    def test_normalize_none(self, client: AnthropicClient) -> None:
        """Test that None normalizes to None."""
        result = client._normalize_tool_choice(None)
        assert result is None

    def test_normalize_other_tool_name(self, client: AnthropicClient) -> None:
        """Test that other tool names normalize correctly."""
        result = client._normalize_tool_choice("search_docs")
        assert result == {"type": "tool", "name": "search_docs"}


class TestAnthropicClientMessageConversion:
    """Tests for message format conversion."""

    @pytest.fixture
    def mock_anthropic(self) -> Generator[MagicMock, None, None]:
        """Mock the Anthropic client."""
        with patch("bluebox.llms.anthropic_client.Anthropic") as mock_cls:
            with patch("bluebox.llms.anthropic_client.AsyncAnthropic"):
                mock_instance = MagicMock()
                mock_cls.return_value = mock_instance
                yield mock_instance

    @pytest.fixture
    def client(self, mock_anthropic: MagicMock) -> AnthropicClient:
        """Create an AnthropicClient instance for testing."""
        with patch("bluebox.llms.anthropic_client.Config") as mock_config:
            mock_config.ANTHROPIC_API_KEY = "test-key"
            return AnthropicClient(model=AnthropicModel.CLAUDE_SONNET_4_5)

    def test_convert_simple_messages(self, client: AnthropicClient) -> None:
        """Test conversion of simple user/assistant messages."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        converted = client._convert_messages_for_anthropic(messages)

        assert len(converted) == 2
        assert converted[0]["role"] == "user"
        assert converted[0]["content"] == "Hello"
        assert converted[1]["role"] == "assistant"
        assert converted[1]["content"] == "Hi there!"

    def test_convert_system_message_filtered(self, client: AnthropicClient) -> None:
        """Test that system messages are filtered out."""
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
        ]

        converted = client._convert_messages_for_anthropic(messages)

        assert len(converted) == 1
        assert converted[0]["role"] == "user"

    def test_convert_tool_calls(self, client: AnthropicClient) -> None:
        """Test conversion of assistant messages with tool calls."""
        messages = [
            {
                "role": "assistant",
                "content": "Let me check that.",
                "tool_calls": [
                    {
                        "call_id": "call_123",
                        "name": "search",
                        "arguments": {"query": "weather"},
                    }
                ],
            }
        ]

        converted = client._convert_messages_for_anthropic(messages)

        assert len(converted) == 1
        assert converted[0]["role"] == "assistant"
        assert len(converted[0]["content"]) == 2
        assert converted[0]["content"][0]["type"] == "text"
        assert converted[0]["content"][1]["type"] == "tool_use"
        assert converted[0]["content"][1]["id"] == "call_123"
        assert converted[0]["content"][1]["name"] == "search"

    def test_convert_tool_result(self, client: AnthropicClient) -> None:
        """Test conversion of tool result messages."""
        messages = [
            {
                "role": "tool",
                "tool_call_id": "call_123",
                "content": "The weather is sunny.",
            }
        ]

        converted = client._convert_messages_for_anthropic(messages)

        assert len(converted) == 1
        assert converted[0]["role"] == "user"
        assert converted[0]["content"][0]["type"] == "tool_result"
        assert converted[0]["content"][0]["tool_use_id"] == "call_123"
        assert converted[0]["content"][0]["content"] == "The weather is sunny."


class TestToolChoiceIntegration:
    """Integration tests for tool_choice in API calls."""

    @pytest.fixture
    def mock_anthropic(self) -> Generator[MagicMock, None, None]:
        """Mock the Anthropic client."""
        with patch("bluebox.llms.anthropic_client.Anthropic") as mock_cls:
            with patch("bluebox.llms.anthropic_client.AsyncAnthropic"):
                mock_instance = MagicMock()
                mock_cls.return_value = mock_instance
                yield mock_instance

    @pytest.fixture
    def client(self, mock_anthropic: MagicMock) -> AnthropicClient:
        """Create an AnthropicClient instance for testing."""
        with patch("bluebox.llms.anthropic_client.Config") as mock_config:
            mock_config.ANTHROPIC_API_KEY = "test-key"
            return AnthropicClient(model=AnthropicModel.CLAUDE_SONNET_4_5)

    def _create_mock_response(self, content: str = "Hello!") -> MagicMock:
        """Create a mock Anthropic response."""
        mock_response = MagicMock()
        mock_text_block = MagicMock()
        mock_text_block.type = "text"
        mock_text_block.text = content
        mock_response.content = [mock_text_block]
        mock_response.id = "msg_123"
        return mock_response

    def test_tool_choice_auto_passed_to_api(self, client: AnthropicClient, mock_anthropic: MagicMock) -> None:
        """Test that 'auto' tool_choice is normalized and passed correctly to API."""
        mock_anthropic.messages.create.return_value = self._create_mock_response()

        client.register_tool(
            name="test_tool",
            description="A test tool",
            parameters={"type": "object", "properties": {}},
        )

        client.call_sync(input="Hello", tool_choice="auto")

        # Verify tool_choice was normalized to {"type": "auto"}
        call_kwargs = mock_anthropic.messages.create.call_args[1]
        assert call_kwargs.get("tool_choice") == {"type": "auto"}

    def test_tool_choice_required_passed_to_api(self, client: AnthropicClient, mock_anthropic: MagicMock) -> None:
        """Test that 'required' tool_choice is normalized and passed correctly to API."""
        mock_anthropic.messages.create.return_value = self._create_mock_response()

        client.register_tool(
            name="test_tool",
            description="A test tool",
            parameters={"type": "object", "properties": {}},
        )

        client.call_sync(input="Hello", tool_choice="required")

        # Verify tool_choice was normalized to {"type": "any"}
        call_kwargs = mock_anthropic.messages.create.call_args[1]
        assert call_kwargs.get("tool_choice") == {"type": "any"}

    def test_tool_choice_tool_name_passed_to_api(self, client: AnthropicClient, mock_anthropic: MagicMock) -> None:
        """Test that tool name tool_choice is normalized and passed correctly to API."""
        mock_anthropic.messages.create.return_value = self._create_mock_response()

        client.register_tool(
            name="test_tool",
            description="A test tool",
            parameters={"type": "object", "properties": {}},
        )

        client.call_sync(input="Hello", tool_choice="test_tool")

        # Verify tool_choice was normalized to tool dict
        call_kwargs = mock_anthropic.messages.create.call_args[1]
        assert call_kwargs.get("tool_choice") == {"type": "tool", "name": "test_tool"}
