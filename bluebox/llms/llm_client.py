"""
bluebox/llms/llm_client.py

Unified LLM client for OpenAI and Anthropic models.

Contains:
- LLMClient: High-level interface for chat completions
- Tool registration and execution
- Streaming and non-streaming response handling
- Structured output parsing with Pydantic
"""

from collections.abc import Generator
from typing import Any, Callable, TypeVar

from pydantic import BaseModel

from bluebox.data_models.llms.interaction import LLMChatResponse
from bluebox.data_models.llms.vendors import (
    AnthropicModel,
    LLMModel,
    LLMVendor,
    OpenAIAPIType,
    OpenAIModel,
    get_model_vendor,
)
from bluebox.llms.abstract_llm_vendor_client import AbstractLLMVendorClient
from bluebox.llms.anthropic_client import AnthropicClient
from bluebox.llms.openai_client import OpenAIClient
from bluebox.llms.tools.tool_utils import extract_description_from_docstring, generate_parameters_schema
from bluebox.utils.logger import get_logger

logger = get_logger(name=__name__)


T = TypeVar("T", bound=BaseModel)


class LLMClient:
    """
    Unified LLM client class for interacting with OpenAI and Anthropic APIs.

    This is a facade that delegates to the appropriate vendor client
    and provides a convenient interface for tool registration.

    Supports:
    - Sync and async API calls
    - Streaming responses
    - Structured responses using Pydantic models
    - Tool/function registration
    - OpenAI Chat Completions and Responses APIs
    - Anthropic Messages API
    """

    # Magic methods ________________________________________________________________________________________________________

    def __init__(self, llm_model: LLMModel) -> None:
        self.llm_model = llm_model
        self._client: AbstractLLMVendorClient = self._create_vendor_client(llm_model)
        logger.info("Instantiated LLMClient with model: %s", llm_model)

    def _create_vendor_client(self, model: LLMModel) -> AbstractLLMVendorClient:
        """Create the appropriate vendor client based on the model."""
        vendor = get_model_vendor(model)
        if vendor == LLMVendor.OPENAI:
            return OpenAIClient(model=model)  # type: ignore[arg-type]
        elif vendor == LLMVendor.ANTHROPIC:
            return AnthropicClient(model=model)  # type: ignore[arg-type]
        else:
            raise ValueError(f"Unsupported vendor: {vendor}")

    # Public methods _______________________________________________________________________________________________________

    ## Tools

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
        logger.debug("Registering tool %s (description: %s) with parameters: %s", name, description, parameters)
        self._client.register_tool(name, description, parameters)

    def register_tool_from_function(self, func: Callable[..., Any]) -> None:
        """
        Register a tool from a Python function, extracting metadata automatically.

        Extracts:
        - name: from func.__name__
        - description: from the docstring (first paragraph)
        - parameters: JSON Schema generated from type hints via pydantic

        Args:
            func: The function to register as a tool. Must have type hints.
        """
        name = func.__name__
        description = extract_description_from_docstring(func.__doc__)
        parameters = generate_parameters_schema(func)
        logger.info("Registering tool %s with schema: %s", name, parameters)
        self.register_tool(name, description, parameters)

    def clear_tools(self) -> None:
        """Clear all registered tools."""
        self._client.clear_tools()
        logger.debug("Cleared all registered tools")

    def set_file_search_vectorstores(
        self,
        vector_store_ids: list[str] | None,
        filters: dict | None = None,
    ) -> None:
        """
        Set vectorstore IDs for file_search tool.

        When set, the file_search tool will be automatically included in LLM calls.
        This enables the LLM to search through the specified vectorstores.

        Note: This feature is only supported by OpenAI. Other vendors will log a warning.

        Args:
            vector_store_ids: List of vectorstore IDs to search, or None to disable.
            filters: Optional filters for file_search (e.g., {"type": "eq", "key": "uuid", "value": ["..."]}).
        """
        if hasattr(self._client, "set_file_search_vectorstores"):
            self._client.set_file_search_vectorstores(vector_store_ids, filters)
        else:
            logger.warning(
                "set_file_search_vectorstores is not supported by %s",
                type(self._client).__name__,
            )

    ## Unified API methods

    def call_sync(
        self,
        messages: list[dict[str, str]] | None = None,
        input: str | None = None,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        response_model: type[T] | None = None,
        extended_reasoning: bool = False,
        stateful: bool = False,
        previous_response_id: str | None = None,
        api_type: OpenAIAPIType | None = None,
        tool_choice: str | dict | None = None,
    ) -> LLMChatResponse:
        """
        Unified sync call to OpenAI.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            input: Input string (Responses API shorthand).
            system_prompt: Optional system prompt for context.
            max_tokens: Maximum tokens in the response.
            temperature: Sampling temperature (0.0-1.0).
            response_model: Pydantic model class for structured response.
            extended_reasoning: Enable extended reasoning (Responses API only).
            stateful: Enable stateful conversation (Responses API only).
            previous_response_id: Previous response ID for chaining (Responses API only).
            api_type: Explicit API type, or None for auto-resolution.
            tool_choice: Tool choice for the API call (e.g., "auto", "required", or specific tool).

        Returns:
            LLMChatResponse. If response_model is provided, the parsed model is in response.parsed.
        """
        return self._client.call_sync(
            messages=messages,
            input=input,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            response_model=response_model,
            extended_reasoning=extended_reasoning,
            stateful=stateful,
            previous_response_id=previous_response_id,
            api_type=api_type,
            tool_choice=tool_choice,
        )

    async def call_async(
        self,
        messages: list[dict[str, str]] | None = None,
        input: str | None = None,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        response_model: type[T] | None = None,
        extended_reasoning: bool = False,
        stateful: bool = False,
        previous_response_id: str | None = None,
        api_type: OpenAIAPIType | None = None,
        tool_choice: str | dict | None = None,
    ) -> LLMChatResponse:
        """
        Unified async call to OpenAI.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            input: Input string (Responses API shorthand).
            system_prompt: Optional system prompt for context.
            max_tokens: Maximum tokens in the response.
            temperature: Sampling temperature (0.0-1.0).
            response_model: Pydantic model class for structured response.
            extended_reasoning: Enable extended reasoning (Responses API only).
            stateful: Enable stateful conversation (Responses API only).
            previous_response_id: Previous response ID for chaining (Responses API only).
            api_type: Explicit API type, or None for auto-resolution.
            tool_choice: Tool choice for the API call (e.g., "auto", "required", or specific tool).

        Returns:
            LLMChatResponse. If response_model is provided, the parsed model is in response.parsed.
        """
        return await self._client.call_async(
            messages=messages,
            input=input,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            response_model=response_model,
            extended_reasoning=extended_reasoning,
            stateful=stateful,
            previous_response_id=previous_response_id,
            api_type=api_type,
            tool_choice=tool_choice,
        )

    def call_stream_sync(
        self,
        messages: list[dict[str, str]] | None = None,
        input: str | None = None,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        extended_reasoning: bool = False,
        stateful: bool = False,
        previous_response_id: str | None = None,
        api_type: OpenAIAPIType | None = None,
        tool_choice: str | dict | None = None,
    ) -> Generator[str | LLMChatResponse, None, None]:
        """
        Unified streaming call to OpenAI.

        Yields text chunks as they arrive, then yields the final LLMChatResponse.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            input: Input string (Responses API shorthand).
            system_prompt: Optional system prompt for context.
            max_tokens: Maximum tokens in the response.
            temperature: Sampling temperature (0.0-1.0).
            extended_reasoning: Enable extended reasoning (Responses API only).
            stateful: Enable stateful conversation (Responses API only).
            previous_response_id: Previous response ID for chaining (Responses API only).
            api_type: Explicit API type, or None for auto-resolution.
            tool_choice: Tool choice for the API call (e.g., "auto", "required", or specific tool).

        Yields:
            str: Text chunks as they arrive.
            LLMChatResponse: Final response with complete content and optional tool call.
        """
        yield from self._client.call_stream_sync(
            messages=messages,
            input=input,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            extended_reasoning=extended_reasoning,
            stateful=stateful,
            previous_response_id=previous_response_id,
            api_type=api_type,
            tool_choice=tool_choice,
        )
