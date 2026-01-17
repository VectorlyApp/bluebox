"""
tests/unit/test_openai_client.py

Unit tests for OpenAI client validation logic.
"""

import pytest

from web_hacker.data_models.llms.vendors import OpenAIAPIType, OpenAIModel
from web_hacker.llms.openai_client import OpenAIClient


class TestValidateAndResolveAPIType:
    """Tests for _validate_and_resolve_api_type method."""

    @pytest.fixture
    def client(self) -> OpenAIClient:
        """Create an OpenAIClient instance for testing."""
        return OpenAIClient(model=OpenAIModel.GPT_5_MINI)

    # Happy path tests - valid combinations

    def test_default_resolves_to_chat_completions(self, client: OpenAIClient) -> None:
        """Default (no special params) should resolve to Chat Completions API."""
        result = client._validate_and_resolve_api_type(
            api_type=None,
            extended_reasoning=False,
            previous_response_id=None,
        )
        assert result == OpenAIAPIType.CHAT_COMPLETIONS

    def test_extended_reasoning_auto_resolves_to_responses(self, client: OpenAIClient) -> None:
        """extended_reasoning=True should auto-resolve to Responses API."""
        result = client._validate_and_resolve_api_type(
            api_type=None,
            extended_reasoning=True,
            previous_response_id=None,
        )
        assert result == OpenAIAPIType.RESPONSES

    def test_previous_response_id_auto_resolves_to_responses(self, client: OpenAIClient) -> None:
        """previous_response_id should auto-resolve to Responses API."""
        result = client._validate_and_resolve_api_type(
            api_type=None,
            extended_reasoning=False,
            previous_response_id="resp_123",
        )
        assert result == OpenAIAPIType.RESPONSES

    def test_explicit_chat_completions_api_type(self, client: OpenAIClient) -> None:
        """Explicit Chat Completions API type should be honored."""
        result = client._validate_and_resolve_api_type(
            api_type=OpenAIAPIType.CHAT_COMPLETIONS,
            extended_reasoning=False,
            previous_response_id=None,
        )
        assert result == OpenAIAPIType.CHAT_COMPLETIONS

    def test_explicit_responses_api_type(self, client: OpenAIClient) -> None:
        """Explicit Responses API type should be honored."""
        result = client._validate_and_resolve_api_type(
            api_type=OpenAIAPIType.RESPONSES,
            extended_reasoning=False,
            previous_response_id=None,
        )
        assert result == OpenAIAPIType.RESPONSES

    def test_extended_reasoning_with_responses_api_valid(self, client: OpenAIClient) -> None:
        """extended_reasoning=True with explicit Responses API should work."""
        result = client._validate_and_resolve_api_type(
            api_type=OpenAIAPIType.RESPONSES,
            extended_reasoning=True,
            previous_response_id=None,
        )
        assert result == OpenAIAPIType.RESPONSES

    def test_previous_response_id_with_responses_api_valid(self, client: OpenAIClient) -> None:
        """previous_response_id with explicit Responses API should work."""
        result = client._validate_and_resolve_api_type(
            api_type=OpenAIAPIType.RESPONSES,
            extended_reasoning=False,
            previous_response_id="resp_123",
        )
        assert result == OpenAIAPIType.RESPONSES

    # Error cases - invalid combinations

    def test_extended_reasoning_with_chat_completions_raises_error(self, client: OpenAIClient) -> None:
        """extended_reasoning=True with Chat Completions API should raise ValueError."""
        with pytest.raises(ValueError, match="extended_reasoning=True requires Responses API"):
            client._validate_and_resolve_api_type(
                api_type=OpenAIAPIType.CHAT_COMPLETIONS,
                extended_reasoning=True,
                previous_response_id=None,
            )

    def test_previous_response_id_with_chat_completions_raises_error(self, client: OpenAIClient) -> None:
        """previous_response_id with Chat Completions API should raise ValueError."""
        with pytest.raises(ValueError, match="previous_response_id requires Responses API"):
            client._validate_and_resolve_api_type(
                api_type=OpenAIAPIType.CHAT_COMPLETIONS,
                extended_reasoning=False,
                previous_response_id="resp_123",
            )

    def test_both_extended_reasoning_and_previous_response_id_with_chat_completions_raises_error(
        self,
        client: OpenAIClient,
    ) -> None:
        """Both extended_reasoning and previous_response_id with Chat Completions should raise ValueError."""
        with pytest.raises(ValueError):
            client._validate_and_resolve_api_type(
                api_type=OpenAIAPIType.CHAT_COMPLETIONS,
                extended_reasoning=True,
                previous_response_id="resp_123",
            )


class TestToolRegistration:
    """Tests for tool registration."""

    @pytest.fixture
    def client(self) -> OpenAIClient:
        """Create an OpenAIClient instance for testing."""
        return OpenAIClient(model=OpenAIModel.GPT_5_MINI)

    def test_register_tool(self, client: OpenAIClient) -> None:
        """Test that tools are registered correctly."""
        client.register_tool(
            name="test_tool",
            description="A test tool",
            parameters={"type": "object", "properties": {}},
        )

        assert len(client.tools) == 1
        assert client.tools[0]["type"] == "function"
        assert client.tools[0]["function"]["name"] == "test_tool"
        assert client.tools[0]["function"]["description"] == "A test tool"

    def test_clear_tools(self, client: OpenAIClient) -> None:
        """Test that tools can be cleared."""
        client.register_tool(
            name="test_tool",
            description="A test tool",
            parameters={"type": "object", "properties": {}},
        )
        assert len(client.tools) == 1

        client.clear_tools()
        assert len(client.tools) == 0


class TestCallSyncValidation:
    """Tests for call_sync parameter validation."""

    @pytest.fixture
    def client(self) -> OpenAIClient:
        """Create an OpenAIClient instance for testing."""
        return OpenAIClient(model=OpenAIModel.GPT_5_MINI)

    def test_messages_required_for_chat_completions(self, client: OpenAIClient) -> None:
        """Test that messages is required for Chat Completions API."""
        with pytest.raises(ValueError, match="messages is required for Chat Completions API"):
            client.call_sync(
                messages=None,
                input="Hello",
                api_type=OpenAIAPIType.CHAT_COMPLETIONS,
            )

    def test_extended_reasoning_with_chat_completions_raises_error(self, client: OpenAIClient) -> None:
        """Test that extended_reasoning with Chat Completions raises ValueError."""
        with pytest.raises(ValueError, match="extended_reasoning=True requires Responses API"):
            client.call_sync(
                messages=[{"role": "user", "content": "Hello"}],
                extended_reasoning=True,
                api_type=OpenAIAPIType.CHAT_COMPLETIONS,
            )


class TestLLMChatResponseFields:
    """Tests for LLMChatResponse new fields."""

    def test_response_id_field_exists(self) -> None:
        """Test that response_id field exists on LLMChatResponse."""
        from web_hacker.data_models.llms.interaction import LLMChatResponse

        response = LLMChatResponse(content="test", response_id="resp_123")
        assert response.response_id == "resp_123"

    def test_reasoning_content_field_exists(self) -> None:
        """Test that reasoning_content field exists on LLMChatResponse."""
        from web_hacker.data_models.llms.interaction import LLMChatResponse

        response = LLMChatResponse(content="test", reasoning_content="I thought about this...")
        assert response.reasoning_content == "I thought about this..."

    def test_default_values_are_none(self) -> None:
        """Test that new fields default to None."""
        from web_hacker.data_models.llms.interaction import LLMChatResponse

        response = LLMChatResponse(content="test")
        assert response.response_id is None
        assert response.reasoning_content is None
