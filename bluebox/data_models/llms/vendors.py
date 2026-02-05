"""
bluebox/data_models/llms/vendors.py

This module contains the LLM vendor models.
"""

from enum import StrEnum
from typing import ClassVar


class LLMVendor(StrEnum):
    """
    Represents the vendor of an LLM.
    """
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class VendorModel(StrEnum):
    """Base for model enums. Subclasses must define _vendor."""
    _vendor: ClassVar[LLMVendor]

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if not hasattr(cls, '_vendor'):
            raise TypeError(f"{cls.__name__} must define _vendor class attribute")

    @property
    def vendor(self) -> LLMVendor:
        return self.__class__._vendor


class OpenAIModel(VendorModel):
    """OpenAI models."""
    _vendor = LLMVendor.OPENAI

    GPT_5 = "gpt-5"
    GPT_5_1 = "gpt-5.1"
    GPT_5_2 = "gpt-5.2"
    GPT_5_MINI = "gpt-5-mini"
    GPT_5_NANO = "gpt-5-nano"


class AnthropicModel(VendorModel):
    """Anthropic models."""
    _vendor = LLMVendor.ANTHROPIC

    CLAUDE_OPUS_4_6 = "claude-opus-4-6"
    CLAUDE_OPUS_4_5 = "claude-opus-4-5"
    CLAUDE_SONNET_4_5 = "claude-sonnet-4-5"
    CLAUDE_HAIKU_4_5 = "claude-haiku-4-5"


# LLMModel type: union of all vendor models
LLMModel = OpenAIModel | AnthropicModel


def get_model_by_value(model_value: str) -> LLMModel | None:
    """
    Get model enum by value string.

    Args:
        model_value: The model value string (e.g., "gpt-5.1", "claude-opus-4-5").

    Returns:
        LLMModel if found, None otherwise. Use model.vendor to get the vendor.
    """
    for model_cls in VendorModel.__subclasses__():
        try:
            return model_cls(model_value)
        except ValueError:
            continue
    return None


def get_all_model_values() -> list[str]:
    """Get all available model value strings."""
    return [m.value for cls in VendorModel.__subclasses__() for m in cls]
