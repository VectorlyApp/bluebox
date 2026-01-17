"""
web_hacker/llms/llm_client.py

Unified LLM client supporting OpenAI models.
"""

from collections.abc import Generator
from typing import Any, Callable, TypeVar

from pydantic import BaseModel

from web_hacker.data_models.llms.interaction import LLMChatResponse
from web_hacker.data_models.llms.vendors import OpenAIAPIType, OpenAIModel
from web_hacker.llms.openai_client import OpenAIClient
from web_hacker.llms.tools.tool_utils import extract_description_from_docstring, generate_parameters_schema
from web_hacker.utils.logger import get_logger

logger = get_logger(name=__name__)


T = TypeVar("T", bound=BaseModel)


class LLMClient:
    """
    Unified LLM client class for interacting with OpenAI APIs.

    This is a facade that delegates to OpenAIClient and provides
    a convenient interface for tool registration.

    Supports:
    - Sync and async API calls
    - Streaming responses
    - Structured responses using Pydantic models
    - Tool/function registration
    - Both Chat Completions and Responses APIs
    """

    # Magic methods ________________________________________________________________________________________________________

    def __init__(self, llm_model: OpenAIModel) -> None:
        self.llm_model = llm_model
        self._client = OpenAIClient(model=llm_model)
        logger.info("Instantiated LLMClient with model: %s", llm_model)

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
        self.register_tool(name, description, parameters)

    def clear_tools(self) -> None:
        """Clear all registered tools."""
        self._client.clear_tools()
        logger.debug("Cleared all registered tools")

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
    ) -> LLMChatResponse | T:
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

        Returns:
            LLMChatResponse or parsed Pydantic model if response_model is provided.
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
    ) -> LLMChatResponse | T:
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

        Returns:
            LLMChatResponse or parsed Pydantic model if response_model is provided.
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
        )
