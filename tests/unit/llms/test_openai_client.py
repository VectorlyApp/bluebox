"""
tests/unit/test_openai_client.py

Unit tests for OpenAI client.
"""

from unittest.mock import MagicMock, patch

import pytest

from bluebox.data_models.llms.vendors import OpenAIModel
from bluebox.llms.openai_client import OpenAIClient


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

    def test_register_tool_upsert_behavior(self, client: OpenAIClient) -> None:
        """Test that registering a tool with the same name updates instead of duplicating."""
        client.register_tool(
            name="test_tool",
            description="Original description",
            parameters={"type": "object", "properties": {"arg1": {"type": "string"}}},
        )
        assert len(client.tools) == 1
        assert client.tools[0]["function"]["description"] == "Original description"

        # Register same tool name with updated description
        client.register_tool(
            name="test_tool",
            description="Updated description",
            parameters={"type": "object", "properties": {"arg2": {"type": "integer"}}},
        )

        # Should still have only 1 tool, but with updated content
        assert len(client.tools) == 1
        assert client.tools[0]["function"]["description"] == "Updated description"
        assert "arg2" in client.tools[0]["function"]["parameters"]["properties"]


class TestToolChoiceIntegration:
    """Integration tests for tool_choice in API calls."""

    @pytest.fixture
    def mock_openai(self) -> MagicMock:
        """Mock the OpenAI client."""
        with patch("bluebox.llms.openai_client.OpenAI") as mock_cls:
            with patch("bluebox.llms.openai_client.AsyncOpenAI"):
                mock_instance = MagicMock()
                mock_cls.return_value = mock_instance
                yield mock_instance

    @pytest.fixture
    def client(self, mock_openai: MagicMock) -> OpenAIClient:
        """Create an OpenAIClient instance for testing."""
        with patch("bluebox.llms.openai_client.Config") as mock_config:
            mock_config.OPENAI_API_KEY = "test-key"
            return OpenAIClient(model=OpenAIModel.GPT_5_MINI)

    def test_tool_choice_auto_passed_to_api(self, client: OpenAIClient, mock_openai: MagicMock) -> None:
        """Test that 'auto' tool_choice is passed correctly to API."""
        mock_response = MagicMock()
        mock_response.id = "resp_123"
        mock_response.output = []
        mock_openai.responses.create.return_value = mock_response

        client.register_tool(
            name="test_tool",
            description="A test tool",
            parameters={"type": "object", "properties": {}},
        )

        client.call_sync(input="Hello", tool_choice="auto")

        # Verify tool_choice was passed as "auto"
        call_kwargs = mock_openai.responses.create.call_args[1]
        assert call_kwargs.get("tool_choice") == "auto"

    def test_tool_choice_required_passed_to_api(self, client: OpenAIClient, mock_openai: MagicMock) -> None:
        """Test that 'required' tool_choice is passed correctly to API."""
        mock_response = MagicMock()
        mock_response.id = "resp_123"
        mock_response.output = []
        mock_openai.responses.create.return_value = mock_response

        client.register_tool(
            name="test_tool",
            description="A test tool",
            parameters={"type": "object", "properties": {}},
        )

        client.call_sync(input="Hello", tool_choice="required")

        # Verify tool_choice was passed as "required"
        call_kwargs = mock_openai.responses.create.call_args[1]
        assert call_kwargs.get("tool_choice") == "required"

    def test_tool_choice_tool_name_passed_to_api(self, client: OpenAIClient, mock_openai: MagicMock) -> None:
        """Test that tool name tool_choice is normalized and passed correctly to API."""
        mock_response = MagicMock()
        mock_response.id = "resp_123"
        mock_response.output = []
        mock_openai.responses.create.return_value = mock_response

        client.register_tool(
            name="test_tool",
            description="A test tool",
            parameters={"type": "object", "properties": {}},
        )

        client.call_sync(input="Hello", tool_choice="test_tool")

        # Verify tool_choice was normalized to function dict
        call_kwargs = mock_openai.responses.create.call_args[1]
        assert call_kwargs.get("tool_choice") == {"type": "function", "name": "test_tool"}


class TestToolChoiceNormalization:
    """Tests for tool_choice normalization."""

    @pytest.fixture
    def client(self) -> OpenAIClient:
        """Create an OpenAIClient instance for testing."""
        return OpenAIClient(model=OpenAIModel.GPT_5_MINI)

    def test_normalize_auto(self, client: OpenAIClient) -> None:
        """Test that 'auto' normalizes to 'auto'."""
        result = client._normalize_tool_choice("auto")
        assert result == "auto"

    def test_normalize_required(self, client: OpenAIClient) -> None:
        """Test that 'required' normalizes to 'required'."""
        result = client._normalize_tool_choice("required")
        assert result == "required"

    def test_normalize_tool_name(self, client: OpenAIClient) -> None:
        """Test that tool name normalizes to function dict."""
        result = client._normalize_tool_choice("get_weather")
        assert result == {"type": "function", "name": "get_weather"}

    def test_normalize_none(self, client: OpenAIClient) -> None:
        """Test that None normalizes to None."""
        result = client._normalize_tool_choice(None)
        assert result is None

    def test_normalize_other_tool_name(self, client: OpenAIClient) -> None:
        """Test that other tool names normalize correctly."""
        result = client._normalize_tool_choice("search_docs")
        assert result == {"type": "function", "name": "search_docs"}


class TestLLMChatResponseFields:
    """Tests for LLMChatResponse new fields."""

    def test_response_id_field_exists(self) -> None:
        """Test that response_id field exists on LLMChatResponse."""
        from bluebox.data_models.llms.interaction import LLMChatResponse

        response = LLMChatResponse(content="test", response_id="resp_123")
        assert response.response_id == "resp_123"

    def test_reasoning_content_field_exists(self) -> None:
        """Test that reasoning_content field exists on LLMChatResponse."""
        from bluebox.data_models.llms.interaction import LLMChatResponse

        response = LLMChatResponse(content="test", reasoning_content="I thought about this...")
        assert response.reasoning_content == "I thought about this..."

    def test_default_values_are_none(self) -> None:
        """Test that new fields default to None."""
        from bluebox.data_models.llms.interaction import LLMChatResponse

        response = LLMChatResponse(content="test")
        assert response.response_id is None
        assert response.reasoning_content is None
