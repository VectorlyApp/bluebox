"""
bluebox/data_models/llms/vendors.py

This module contains the LLM vendor models.
"""

from enum import StrEnum


class LLMVendor(StrEnum):
    """
    Represents the vendor of an LLM.
    """
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class OpenAIAPIType(StrEnum):
    """
    OpenAI API type.
    """
    CHAT_COMPLETIONS = "chat_completions"
    RESPONSES = "responses"


class OpenAIModel(StrEnum):
    """
    OpenAI models.
    """
    GPT_5 = "gpt-5"
    GPT_5_1 = "gpt-5.1"
    GPT_5_2 = "gpt-5.2"
    GPT_5_MINI = "gpt-5-mini"
    GPT_5_NANO = "gpt-5-nano"


class AnthropicModel(StrEnum):
    """Anthropic models."""
    CLAUDE_OPUS_4 = "claude-opus-4-20250514"
    CLAUDE_SONNET_4_5 = "claude-sonnet-4-5-20250929"
    CLAUDE_SONNET_4 = "claude-sonnet-4-20250514"
    CLAUDE_HAIKU_3_5 = "claude-3-5-haiku-20241022"


# LLMModel type: union of all vendor models
type LLMModel = OpenAIModel | AnthropicModel


# Build model to vendor lookup from all vendor models
_model_to_vendor: dict[str, LLMVendor] = {}
_all_models: dict[str, str] = {}

for model in OpenAIModel:
    _model_to_vendor[model.value] = LLMVendor.OPENAI
    _all_models[model.name] = model.value

for model in AnthropicModel:
    _model_to_vendor[model.value] = LLMVendor.ANTHROPIC
    _all_models[model.name] = model.value


def get_model_vendor(model: LLMModel) -> LLMVendor:
    """
    Returns the vendor of the LLM model.
    """
    return _model_to_vendor[model.value]
