"""
tests/unit/data_models/llms/test_vendors.py

Unit tests for LLM vendor models and helper functions.
"""

import pytest

from bluebox.data_models.llms.vendors import (
    AnthropicModel,
    LLMVendor,
    OpenAIModel,
    get_all_model_values,
    get_model_by_value,
    get_model_vendor,
)


class TestGetModelByValue:
    """Tests for get_model_by_value function."""

    # OpenAI models

    def test_get_openai_gpt5(self) -> None:
        """Test resolving gpt-5 model."""
        result = get_model_by_value("gpt-5")
        assert result is not None
        model, vendor = result
        assert model == OpenAIModel.GPT_5
        assert vendor == LLMVendor.OPENAI

    def test_get_openai_gpt5_1(self) -> None:
        """Test resolving gpt-5.1 model."""
        result = get_model_by_value("gpt-5.1")
        assert result is not None
        model, vendor = result
        assert model == OpenAIModel.GPT_5_1
        assert vendor == LLMVendor.OPENAI

    def test_get_openai_gpt5_2(self) -> None:
        """Test resolving gpt-5.2 model."""
        result = get_model_by_value("gpt-5.2")
        assert result is not None
        model, vendor = result
        assert model == OpenAIModel.GPT_5_2
        assert vendor == LLMVendor.OPENAI

    def test_get_openai_gpt5_mini(self) -> None:
        """Test resolving gpt-5-mini model."""
        result = get_model_by_value("gpt-5-mini")
        assert result is not None
        model, vendor = result
        assert model == OpenAIModel.GPT_5_MINI
        assert vendor == LLMVendor.OPENAI

    def test_get_openai_gpt5_nano(self) -> None:
        """Test resolving gpt-5-nano model."""
        result = get_model_by_value("gpt-5-nano")
        assert result is not None
        model, vendor = result
        assert model == OpenAIModel.GPT_5_NANO
        assert vendor == LLMVendor.OPENAI

    # Anthropic models

    def test_get_anthropic_opus(self) -> None:
        """Test resolving claude-opus-4-5 model."""
        result = get_model_by_value("claude-opus-4-5")
        assert result is not None
        model, vendor = result
        assert model == AnthropicModel.CLAUDE_OPUS_4_5
        assert vendor == LLMVendor.ANTHROPIC

    def test_get_anthropic_sonnet(self) -> None:
        """Test resolving claude-sonnet-4-5 model."""
        result = get_model_by_value("claude-sonnet-4-5")
        assert result is not None
        model, vendor = result
        assert model == AnthropicModel.CLAUDE_SONNET_4_5
        assert vendor == LLMVendor.ANTHROPIC

    def test_get_anthropic_haiku(self) -> None:
        """Test resolving claude-haiku-4-5 model."""
        result = get_model_by_value("claude-haiku-4-5")
        assert result is not None
        model, vendor = result
        assert model == AnthropicModel.CLAUDE_HAIKU_4_5
        assert vendor == LLMVendor.ANTHROPIC

    # Invalid models

    def test_unknown_model_returns_none(self) -> None:
        """Test that unknown model returns None."""
        result = get_model_by_value("unknown-model")
        assert result is None

    def test_empty_string_returns_none(self) -> None:
        """Test that empty string returns None."""
        result = get_model_by_value("")
        assert result is None

    def test_partial_match_returns_none(self) -> None:
        """Test that partial match returns None."""
        result = get_model_by_value("gpt")
        assert result is None


class TestGetAllModelValues:
    """Tests for get_all_model_values function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        result = get_all_model_values()
        assert isinstance(result, list)

    def test_contains_all_openai_models(self) -> None:
        """Test that all OpenAI models are included."""
        result = get_all_model_values()
        for model in OpenAIModel:
            assert model.value in result

    def test_contains_all_anthropic_models(self) -> None:
        """Test that all Anthropic models are included."""
        result = get_all_model_values()
        for model in AnthropicModel:
            assert model.value in result

    def test_total_count_matches_enums(self) -> None:
        """Test that total count matches sum of enum members."""
        result = get_all_model_values()
        expected_count = len(OpenAIModel) + len(AnthropicModel)
        assert len(result) == expected_count


class TestGetModelVendor:
    """Tests for get_model_vendor function."""

    def test_openai_model_returns_openai_vendor(self) -> None:
        """Test that OpenAI models return OpenAI vendor."""
        for model in OpenAIModel:
            vendor = get_model_vendor(model)
            assert vendor == LLMVendor.OPENAI

    def test_anthropic_model_returns_anthropic_vendor(self) -> None:
        """Test that Anthropic models return Anthropic vendor."""
        for model in AnthropicModel:
            vendor = get_model_vendor(model)
            assert vendor == LLMVendor.ANTHROPIC
