"""
web_hacker/data_models/llms/vendors.py

This module contains the LLM vendor models.
"""

from enum import StrEnum


class LLMVendor(StrEnum):
    """
    Represents the vendor of an LLM.
    """
    OPENAI = "openai"
    #TODO:ANTHROPIC = "anthropic"


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


# class AnthropicModel(StrEnum):
#     """Anthropic models."""
#     CLAUDE_OPUS_4_5 = "claude-opus-4-5-20251101"
#     CLAUDE_SONNET_4_5 = "claude-sonnet-4-5-20250929"
#     CLAUDE_HAIKU_4_5 = "claude-haiku-4-5-20251001"


# LLMModel type; only OpenAI models for now
type LLMModel = OpenAIModel


# Build model to vendor lookup from OpenAI models only
_model_to_vendor: dict[str, LLMVendor] = {}
_all_models: dict[str, str] = {}

for model in OpenAIModel:
    _model_to_vendor[model.value] = LLMVendor.OPENAI
    _all_models[model.name] = model.value


def get_model_vendor(model: LLMModel) -> LLMVendor:
    """
    Returns the vendor of the LLM model.
    """
    return _model_to_vendor[model.value]
