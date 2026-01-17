"""
web_hacker/llms/openai_client.py

OpenAI-specific LLM client implementation with unified API supporting
both Chat Completions and Responses APIs.
"""

import json
from collections.abc import Generator
from typing import Any, TypeVar

from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel

from web_hacker.config import Config
from web_hacker.data_models.llms.interaction import LLMChatResponse, LLMToolCall
from web_hacker.data_models.llms.vendors import OpenAIAPIType, OpenAIModel
from web_hacker.llms.abstract_llm_vendor_client import AbstractLLMVendorClient
from web_hacker.utils.logger import get_logger

logger = get_logger(name=__name__)


T = TypeVar("T", bound=BaseModel)


class OpenAIClient(AbstractLLMVendorClient):
    """
    OpenAI-specific LLM client with unified API.

    Supports both Chat Completions API and Responses API with automatic
    API type resolution based on parameters.
    """

    # Magic methods ________________________________________________________________________________________________________

    def __init__(self, model: OpenAIModel) -> None:
        super().__init__(model)
        self._client = OpenAI(api_key=Config.OPENAI_API_KEY)
        self._async_client = AsyncOpenAI(api_key=Config.OPENAI_API_KEY)
        logger.debug("Initialized OpenAIClient with model: %s", model)

    # Private methods ______________________________________________________________________________________________________

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

    def _prepend_system_prompt(
        self,
        messages: list[dict[str, str]],
        system_prompt: str | None,
    ) -> list[dict[str, str]]:
        """Prepend system prompt to messages if provided."""
        if system_prompt:
            return [{"role": "system", "content": system_prompt}] + messages
        return messages

    def _validate_and_resolve_api_type(
        self,
        api_type: OpenAIAPIType | None,
        extended_reasoning: bool,
        previous_response_id: str | None,
    ) -> OpenAIAPIType:
        """
        Validate params and resolve API type. Raises ValueError for invalid combos.

        Args:
            api_type: Explicit API type, or None for auto-resolution.
            extended_reasoning: Whether extended reasoning is requested.
            previous_response_id: Previous response ID for chaining.

        Returns:
            The resolved API type.

        Raises:
            ValueError: If incompatible parameters are combined.
        """
        if extended_reasoning and api_type == OpenAIAPIType.CHAT_COMPLETIONS:
            raise ValueError("extended_reasoning=True requires Responses API")
        if previous_response_id and api_type == OpenAIAPIType.CHAT_COMPLETIONS:
            raise ValueError("previous_response_id requires Responses API")

        # Auto-resolve
        if api_type is None:
            if extended_reasoning or previous_response_id:
                resolved = OpenAIAPIType.RESPONSES
            else:
                resolved = OpenAIAPIType.CHAT_COMPLETIONS
            logger.debug("Auto-resolved API type to: %s", resolved.value)
            return resolved

        logger.debug("Using explicit API type: %s", api_type.value)
        return api_type

    def _build_chat_completions_kwargs(
        self,
        messages: list[dict[str, str]],
        system_prompt: str | None,
        max_tokens: int | None,
        response_model: type[T] | None,
        stream: bool = False,
    ) -> dict[str, Any]:
        """Build kwargs for Chat Completions API call."""
        all_messages = self._prepend_system_prompt(messages, system_prompt)

        kwargs: dict[str, Any] = {
            "model": self.model.value,
            "messages": all_messages,
            "max_completion_tokens": self._resolve_max_tokens(max_tokens),
        }

        if stream:
            kwargs["stream"] = True

        if self._tools and response_model is None:
            kwargs["tools"] = self._tools

        return kwargs

    def _build_responses_api_kwargs(
        self,
        messages: list[dict[str, str]] | None,
        input_text: str | None,
        system_prompt: str | None,
        max_tokens: int | None,
        extended_reasoning: bool,
        previous_response_id: str | None,
        response_model: type[T] | None,
        stream: bool = False,
    ) -> dict[str, Any]:
        """Build kwargs for Responses API call."""
        kwargs: dict[str, Any] = {
            "model": self.model.value,
            "max_output_tokens": self._resolve_max_tokens(max_tokens),
        }

        # Handle input: either input string or messages array
        if previous_response_id:
            kwargs["previous_response_id"] = previous_response_id
            # When chaining, input is the new user message
            if input_text:
                kwargs["input"] = input_text
            elif messages:
                # Use the last user message as input for chaining
                user_messages = [m for m in messages if m.get("role") == "user"]
                if user_messages:
                    kwargs["input"] = user_messages[-1]["content"]
        elif input_text:
            kwargs["input"] = input_text
        elif messages:
            # Convert messages to Responses API format
            all_messages = self._prepend_system_prompt(messages, system_prompt)
            kwargs["input"] = all_messages
        else:
            raise ValueError("Either messages or input must be provided")

        # Add system instructions if provided and not using messages
        if system_prompt and input_text and not messages:
            kwargs["instructions"] = system_prompt

        if stream:
            kwargs["stream"] = True

        if extended_reasoning:
            kwargs["reasoning"] = {"effort": "medium"}

        if self._tools and response_model is None:
            kwargs["tools"] = self._tools

        return kwargs

    def _parse_chat_completions_response(
        self,
        response: Any,
        response_model: type[T] | None,
    ) -> LLMChatResponse | T:
        """Parse response from Chat Completions API."""
        message = response.choices[0].message

        # Handle structured response
        if response_model is not None:
            parsed = getattr(message, "parsed", None)
            if parsed is None:
                raise ValueError("Failed to parse structured response from OpenAI")
            return parsed

        # Extract tool call if present
        tool_call: LLMToolCall | None = None
        if message.tool_calls and len(message.tool_calls) > 0:
            tc = message.tool_calls[0]
            tool_call = LLMToolCall(
                tool_name=tc.function.name,
                tool_arguments=json.loads(tc.function.arguments),
            )

        return LLMChatResponse(
            content=message.content,
            tool_call=tool_call,
        )

    def _parse_responses_api_response(
        self,
        response: Any,
        response_model: type[T] | None,
    ) -> LLMChatResponse | T:
        """Parse response from Responses API."""
        # Handle structured response
        if response_model is not None:
            # Responses API returns structured output differently
            output = response.output
            if output and len(output) > 0:
                for item in output:
                    if hasattr(item, "content") and item.content:
                        for content_block in item.content:
                            if hasattr(content_block, "parsed") and content_block.parsed:
                                return content_block.parsed
            raise ValueError("Failed to parse structured response from OpenAI Responses API")

        # Extract content and tool calls
        content: str | None = None
        tool_call: LLMToolCall | None = None
        reasoning_content: str | None = None

        output = response.output
        if output:
            for item in output:
                # Handle reasoning content
                if item.type == "reasoning":
                    if hasattr(item, "summary") and item.summary:
                        reasoning_parts = []
                        for summary_item in item.summary:
                            if hasattr(summary_item, "text"):
                                reasoning_parts.append(summary_item.text)
                        if reasoning_parts:
                            reasoning_content = "".join(reasoning_parts)

                # Handle message content
                if item.type == "message":
                    if hasattr(item, "content") and item.content:
                        text_parts = []
                        for content_block in item.content:
                            if content_block.type == "output_text":
                                text_parts.append(content_block.text)
                        if text_parts:
                            content = "".join(text_parts)

                # Handle function calls
                if item.type == "function_call":
                    tool_call = LLMToolCall(
                        tool_name=item.name,
                        tool_arguments=json.loads(item.arguments) if isinstance(item.arguments, str) else item.arguments,
                    )

        return LLMChatResponse(
            content=content,
            tool_call=tool_call,
            response_id=response.id,
            reasoning_content=reasoning_content,
        )

    # Public methods _______________________________________________________________________________________________________

    ## Tool management

    def register_tool(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
    ) -> None:
        """Register a tool in OpenAI's function calling format."""
        logger.debug("Registering OpenAI tool: %s", name)
        self._tools.append({
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters,
            }
        })

    ## Unified API methods

    def call_sync(
        self,
        messages: list[dict[str, str]] | None = None,
        input: str | None = None,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,  # noqa: ARG002 - reserved for future use
        response_model: type[T] | None = None,
        extended_reasoning: bool = False,
        stateful: bool = False,  # noqa: ARG002 - reserved for future use
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

        Raises:
            ValueError: If incompatible parameters are combined.
        """
        resolved_api_type = self._validate_and_resolve_api_type(
            api_type, extended_reasoning, previous_response_id
        )

        if resolved_api_type == OpenAIAPIType.CHAT_COMPLETIONS:
            if messages is None:
                raise ValueError("messages is required for Chat Completions API")

            if response_model is not None:
                # Use beta.chat.completions.parse for structured output
                kwargs = self._build_chat_completions_kwargs(
                    messages, system_prompt, max_tokens, response_model
                )
                response = self._client.beta.chat.completions.parse(
                    **kwargs,
                    response_format=response_model,
                )
            else:
                kwargs = self._build_chat_completions_kwargs(
                    messages, system_prompt, max_tokens, response_model
                )
                response = self._client.chat.completions.create(**kwargs)

            return self._parse_chat_completions_response(response, response_model)

        else:  # Responses API
            kwargs = self._build_responses_api_kwargs(
                messages, input, system_prompt, max_tokens,
                extended_reasoning, previous_response_id, response_model
            )

            if response_model is not None:
                # Add structured output format
                kwargs["text"] = {"format": {"type": "json_schema", "schema": response_model.model_json_schema()}}

            response = self._client.responses.create(**kwargs)
            return self._parse_responses_api_response(response, response_model)

    async def call_async(
        self,
        messages: list[dict[str, str]] | None = None,
        input: str | None = None,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,  # noqa: ARG002 - reserved for future use
        response_model: type[T] | None = None,
        extended_reasoning: bool = False,
        stateful: bool = False,  # noqa: ARG002 - reserved for future use
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

        Raises:
            ValueError: If incompatible parameters are combined.
        """
        resolved_api_type = self._validate_and_resolve_api_type(
            api_type, extended_reasoning, previous_response_id
        )

        if resolved_api_type == OpenAIAPIType.CHAT_COMPLETIONS:
            if messages is None:
                raise ValueError("messages is required for Chat Completions API")

            if response_model is not None:
                kwargs = self._build_chat_completions_kwargs(
                    messages, system_prompt, max_tokens, response_model
                )
                response = await self._async_client.beta.chat.completions.parse(
                    **kwargs,
                    response_format=response_model,
                )
            else:
                kwargs = self._build_chat_completions_kwargs(
                    messages, system_prompt, max_tokens, response_model
                )
                response = await self._async_client.chat.completions.create(**kwargs)

            return self._parse_chat_completions_response(response, response_model)

        else:  # Responses API
            kwargs = self._build_responses_api_kwargs(
                messages, input, system_prompt, max_tokens,
                extended_reasoning, previous_response_id, response_model
            )

            if response_model is not None:
                kwargs["text"] = {"format": {"type": "json_schema", "schema": response_model.model_json_schema()}}

            response = await self._async_client.responses.create(**kwargs)
            return self._parse_responses_api_response(response, response_model)

    def call_stream_sync(
        self,
        messages: list[dict[str, str]] | None = None,
        input: str | None = None,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,  # noqa: ARG002 - reserved for future use
        extended_reasoning: bool = False,
        stateful: bool = False,  # noqa: ARG002 - reserved for future use
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
        resolved_api_type = self._validate_and_resolve_api_type(
            api_type, extended_reasoning, previous_response_id
        )

        if resolved_api_type == OpenAIAPIType.CHAT_COMPLETIONS:
            if messages is None:
                raise ValueError("messages is required for Chat Completions API")

            kwargs = self._build_chat_completions_kwargs(
                messages, system_prompt, max_tokens, response_model=None, stream=True
            )
            stream = self._client.chat.completions.create(**kwargs)

            # Accumulate content and tool call data
            full_content: list[str] = []
            tool_call_name: str | None = None
            tool_call_args: list[str] = []

            for chunk in stream:
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta is None:
                    continue

                # Handle text content
                if delta.content:
                    full_content.append(delta.content)
                    yield delta.content

                # Handle tool calls (streamed in chunks)
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        if tc.function:
                            if tc.function.name:
                                tool_call_name = tc.function.name
                            if tc.function.arguments:
                                tool_call_args.append(tc.function.arguments)

            # Build final response
            tool_call: LLMToolCall | None = None
            if tool_call_name:
                tool_call = LLMToolCall(
                    tool_name=tool_call_name,
                    tool_arguments=json.loads("".join(tool_call_args)) if tool_call_args else {},
                )

            yield LLMChatResponse(
                content="".join(full_content) if full_content else None,
                tool_call=tool_call,
            )

        else:  # Responses API streaming
            kwargs = self._build_responses_api_kwargs(
                messages, input, system_prompt, max_tokens,
                extended_reasoning, previous_response_id, response_model=None, stream=True
            )

            stream = self._client.responses.create(**kwargs)

            full_content: list[str] = []
            tool_call_name: str | None = None
            tool_call_args: list[str] = []
            reasoning_content: str | None = None
            response_id: str | None = None

            for event in stream:
                # Handle different event types from Responses API streaming
                if hasattr(event, "type"):
                    if event.type == "response.created":
                        response_id = event.response.id

                    elif event.type == "response.output_text.delta":
                        if hasattr(event, "delta"):
                            full_content.append(event.delta)
                            yield event.delta

                    elif event.type == "response.function_call_arguments.delta":
                        if hasattr(event, "delta"):
                            tool_call_args.append(event.delta)

                    elif event.type == "response.output_item.added":
                        if hasattr(event, "item") and event.item.type == "function_call":
                            tool_call_name = event.item.name

            # Build final response
            tool_call: LLMToolCall | None = None
            if tool_call_name:
                tool_call = LLMToolCall(
                    tool_name=tool_call_name,
                    tool_arguments=json.loads("".join(tool_call_args)) if tool_call_args else {},
                )

            yield LLMChatResponse(
                content="".join(full_content) if full_content else None,
                tool_call=tool_call,
                response_id=response_id,
                reasoning_content=reasoning_content,
            )
