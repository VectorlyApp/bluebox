"""
bluebox/llms/abstract_llm_vendor_client.py

Abstract base class for LLM vendor clients.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Generator
from typing import Any, ClassVar, TypeVar

from pydantic import BaseModel

from bluebox.data_models.llms.interaction import LLMChatResponse
from bluebox.data_models.llms.vendors import LLMModel, LLMVendor

T = TypeVar("T", bound=BaseModel)


class AbstractLLMVendorClient(ABC):
    """
    Abstract base class defining the interface for LLM vendor clients.

    All vendor-specific clients must implement this interface to ensure
    consistent behavior across the LLMClient.
    """

    # Class attributes ____________________________________________________________________________________________________

    _vendor: ClassVar[LLMVendor]
    DEFAULT_MAX_TOKENS: ClassVar[int] = 4_096
    DEFAULT_TEMPERATURE: ClassVar[float] = 0.7
    DEFAULT_STRUCTURED_TEMPERATURE: ClassVar[float] = 0.0

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if not hasattr(cls, '_vendor'):
            raise TypeError(f"{cls.__name__} must define _vendor class attribute")

    @classmethod
    def get_llm_vendor_client(cls, model: LLMModel) -> AbstractLLMVendorClient:
        """Create the appropriate vendor client for the given model."""
        for subclass in cls.__subclasses__():
            if subclass._vendor == model.vendor:
                return subclass(model=model)
        raise ValueError(f"No client found for vendor: {model.vendor}")

    # Magic methods ________________________________________________________________________________________________________

    def __init__(self, model: LLMModel) -> None:
        """
        Initialize the vendor client.

        Args:
            model: The LLM model to use.
        """
        self.model = model
        self._tools: list[dict[str, Any]] = []

    # Protected methods ____________________________________________________________________________________________________

    def _resolve_max_tokens(self, max_tokens: int | None) -> int:
        """Resolve max_tokens, using default if None."""
        return max_tokens if max_tokens is not None else self.DEFAULT_MAX_TOKENS

    def _resolve_temperature(
        self,
        temperature: float | None,
        structured: bool = False,
    ) -> float:
        """Resolve temperature, using appropriate default if None."""
        if temperature is not None:
            return temperature
        return self.DEFAULT_STRUCTURED_TEMPERATURE if structured else self.DEFAULT_TEMPERATURE

    # Tool management ______________________________________________________________________________________________________

    @abstractmethod
    def register_tool(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
    ) -> None:
        """
        Register a tool for function calling.

        Args:
            name: The name of the tool/function.
            description: Description of what the tool does.
            parameters: JSON Schema describing the tool's parameters.
        """
        pass

    def clear_tools(self) -> None:
        """Clear all registered tools."""
        self._tools = []

    @property
    def tools(self) -> list[dict[str, Any]]:
        """Return the list of registered tools."""
        return self._tools

    # Unified API methods __________________________________________________________________________________________________

    @abstractmethod
    def call_sync(
        self,
        messages: list[dict[str, str]] | None = None,
        input: str | None = None,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        response_model: type[T] | None = None,
        extended_reasoning: bool = False,
        previous_response_id: str | None = None,
        tool_choice: str | None = None,
    ) -> LLMChatResponse | T:
        """
        Unified sync call to the LLM.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            input: Input string (shorthand for simple prompts).
            system_prompt: Optional system prompt for context.
            max_tokens: Maximum tokens in the response.
            temperature: Sampling temperature (0.0-1.0).
            response_model: Pydantic model class for structured response.
            extended_reasoning: Enable extended reasoning (if supported).
            previous_response_id: Previous response ID for chaining (if supported).
            tool_choice: Tool selection mode ("auto", "none", "required", or specific tool).

        Returns:
            LLMChatResponse or parsed Pydantic model if response_model is provided.
        """
        pass

    @abstractmethod
    async def call_async(
        self,
        messages: list[dict[str, str]] | None = None,
        input: str | None = None,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        response_model: type[T] | None = None,
        extended_reasoning: bool = False,
        previous_response_id: str | None = None,
        tool_choice: str | None = None,
    ) -> LLMChatResponse | T:
        """
        Unified async call to the LLM.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            input: Input string (shorthand for simple prompts).
            system_prompt: Optional system prompt for context.
            max_tokens: Maximum tokens in the response.
            temperature: Sampling temperature (0.0-1.0).
            response_model: Pydantic model class for structured response.
            extended_reasoning: Enable extended reasoning (if supported).
            previous_response_id: Previous response ID for chaining (if supported).
            tool_choice: Tool selection mode ("auto", "none", "required", or specific tool).

        Returns:
            LLMChatResponse or parsed Pydantic model if response_model is provided.
        """
        pass

    @abstractmethod
    def call_stream_sync(
        self,
        messages: list[dict[str, str]] | None = None,
        input: str | None = None,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        extended_reasoning: bool = False,
        previous_response_id: str | None = None,
        tool_choice: str | None = None,
    ) -> Generator[str | LLMChatResponse, None, None]:
        """
        Unified streaming call to the LLM.

        Yields text chunks as they arrive, then yields the final LLMChatResponse.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            input: Input string (shorthand for simple prompts).
            system_prompt: Optional system prompt for context.
            max_tokens: Maximum tokens in the response.
            temperature: Sampling temperature (0.0-1.0).
            extended_reasoning: Enable extended reasoning (if supported).
            previous_response_id: Previous response ID for chaining (if supported).
            tool_choice: Tool selection mode ("auto", "none", "required", or specific tool).

        Yields:
            str: Text chunks as they arrive.
            LLMChatResponse: Final response with complete content and optional tool call.
        """
        pass
