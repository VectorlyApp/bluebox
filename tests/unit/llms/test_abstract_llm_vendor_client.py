"""
tests/unit/llms/test_abstract_llm_vendor_client.py

Unit tests for AbstractLLMVendorClient, including vendor discovery and client factory.
"""

from bluebox.data_models.llms.vendors import (
    AnthropicModel,
    LLMVendor,
    OpenAIModel,
)
from bluebox.llms.abstract_llm_vendor_client import AbstractLLMVendorClient
from bluebox.llms.anthropic_client import AnthropicClient
from bluebox.llms.openai_client import OpenAIClient


class TestVendorClassAttribute:
    """Tests for _vendor class attribute on vendor clients."""

    def test_openai_client_has_openai_vendor(self) -> None:
        """OpenAIClient should declare OPENAI as its vendor."""
        assert OpenAIClient._vendor == LLMVendor.OPENAI

    def test_anthropic_client_has_anthropic_vendor(self) -> None:
        """AnthropicClient should declare ANTHROPIC as its vendor."""
        assert AnthropicClient._vendor == LLMVendor.ANTHROPIC


class TestGetLLMVendorClient:
    """Tests for AbstractLLMVendorClient.get_llm_vendor_client factory method."""

    def test_returns_openai_client_for_openai_model(self) -> None:
        """Should return OpenAIClient instance for OpenAI models."""
        client = AbstractLLMVendorClient.get_llm_vendor_client(OpenAIModel.GPT_5_MINI)
        assert isinstance(client, OpenAIClient)
        assert client.model == OpenAIModel.GPT_5_MINI

    def test_returns_anthropic_client_for_anthropic_model(self) -> None:
        """Should return AnthropicClient instance for Anthropic models."""
        client = AbstractLLMVendorClient.get_llm_vendor_client(AnthropicModel.CLAUDE_HAIKU_4_5)
        assert isinstance(client, AnthropicClient)
        assert client.model == AnthropicModel.CLAUDE_HAIKU_4_5

    def test_all_openai_models_return_openai_client(self) -> None:
        """All OpenAI models should return OpenAIClient."""
        for model in OpenAIModel:
            client = AbstractLLMVendorClient.get_llm_vendor_client(model)
            assert isinstance(client, OpenAIClient)

    def test_all_anthropic_models_return_anthropic_client(self) -> None:
        """All Anthropic models should return AnthropicClient."""
        for model in AnthropicModel:
            client = AbstractLLMVendorClient.get_llm_vendor_client(model)
            assert isinstance(client, AnthropicClient)
