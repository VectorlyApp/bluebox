"""
tests/unit/test_openai_client.py

Unit tests for OpenAI client.
"""

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
