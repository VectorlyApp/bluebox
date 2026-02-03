"""
bluebox/llms/anthropic_client.py

Anthropic-specific LLM client implementation using the Messages API.
"""

import asyncio
import json
import random
import time
from collections.abc import Generator
from typing import Any, TypeVar

from anthropic import Anthropic, AsyncAnthropic, APIStatusError, RateLimitError
from pydantic import BaseModel

from bluebox.config import Config
from bluebox.data_models.llms.interaction import LLMChatResponse, LLMToolCall
from bluebox.data_models.llms.vendors import AnthropicModel
from bluebox.llms.abstract_llm_vendor_client import AbstractLLMVendorClient
from bluebox.utils.logger import get_logger

logger = get_logger(name=__name__)


T = TypeVar("T", bound=BaseModel)

# Retry configuration
MAX_RETRIES = 5
BASE_DELAY = 1.0  # seconds
MAX_DELAY = 60.0  # seconds
JITTER_FACTOR = 0.5  # Add randomness to avoid thundering herd


def _is_retryable_error(error: Exception) -> bool:
    """Check if an error is retryable (transient)."""
    if isinstance(error, RateLimitError):
        return True
    if isinstance(error, APIStatusError):
        # Check for overloaded, timeout, or server errors
        error_type = getattr(error, "body", {})
        if isinstance(error_type, dict):
            error_info = error_type.get("error", {})
            if isinstance(error_info, dict):
                err_type = error_info.get("type", "")
                if err_type in ("overloaded_error", "api_error"):
                    return True
        # Also check status codes: 429 (rate limit), 500+, 529 (overloaded)
        status = getattr(error, "status_code", 0)
        if status in (429, 500, 502, 503, 504, 529):
            return True
    return False


def _calculate_backoff(attempt: int) -> float:
    """Calculate backoff delay with exponential growth and jitter."""
    delay = min(BASE_DELAY * (2 ** attempt), MAX_DELAY)
    jitter = delay * JITTER_FACTOR * random.random()
    return delay + jitter


def _clean_schema_for_anthropic(schema: dict[str, Any]) -> dict[str, Any]:
    """
    Recursively clean a JSON schema for Anthropic's tool parameters.
    Removes unsupported keys like '$defs' and resolves $ref references.
    """
    schema = schema.copy()

    # Store $defs for reference resolution before removing
    defs = schema.pop("$defs", {})

    def resolve_refs(obj: dict[str, Any]) -> dict[str, Any]:
        """Recursively resolve $ref references."""
        if not isinstance(obj, dict):
            return obj

        obj = obj.copy()

        # If $ref is present, resolve it
        if "$ref" in obj:
            ref_path = obj["$ref"]
            # Extract definition name from "#/$defs/DefinitionName"
            if ref_path.startswith("#/$defs/"):
                def_name = ref_path[len("#/$defs/"):]
                if def_name in defs:
                    # Replace with the resolved definition
                    resolved = resolve_refs(defs[def_name].copy())
                    return resolved
            # If can't resolve, return empty object
            return {"type": "object"}

        # Recursively process nested structures
        if "properties" in obj:
            obj["properties"] = {
                k: resolve_refs(v) for k, v in obj["properties"].items()
            }

        if "items" in obj:
            obj["items"] = resolve_refs(obj["items"])

        for key in ("anyOf", "oneOf", "allOf"):
            if key in obj:
                obj[key] = [resolve_refs(s) for s in obj[key]]

        return obj

    return resolve_refs(schema)


class AnthropicClient(AbstractLLMVendorClient):
    """
    Anthropic-specific LLM client using the Messages API.

    Supports:
    - Sync and async API calls
    - Streaming responses
    - Tool/function calling
    - Structured responses using Pydantic models
    - Extended thinking (reasoning)
    """

    # Magic methods ________________________________________________________________________________________________________

    def __init__(self, model: AnthropicModel) -> None:
        super().__init__(model)
        self._client = Anthropic(api_key=Config.ANTHROPIC_API_KEY)
        self._async_client = AsyncAnthropic(api_key=Config.ANTHROPIC_API_KEY)
        logger.debug("Initialized AnthropicClient with model: %s", model)

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

    def _convert_messages_for_anthropic(
        self,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Convert messages from generic format to Anthropic Messages API format.

        Handles:
        - System messages are filtered out (passed separately to API)
        - Assistant messages with tool_calls become content with tool_use blocks
        - Tool role messages become user messages with tool_result content

        Args:
            messages: Messages in generic format

        Returns:
            Messages converted to Anthropic format
        """
        converted: list[dict[str, Any]] = []

        for msg in messages:
            role = msg.get("role")

            # Skip system messages - they're passed separately
            if role == "system":
                continue

            # Handle tool result messages
            if role == "tool":
                tool_call_id = msg.get("tool_call_id")
                content = msg.get("content", "")

                # Tool results are sent as user messages with tool_result content
                converted.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_call_id,
                            "content": content,
                        }
                    ],
                })

            # Handle assistant messages with tool calls
            elif role == "assistant" and msg.get("tool_calls"):
                content_blocks: list[dict[str, Any]] = []

                # Add text content if present
                if msg.get("content"):
                    content_blocks.append({
                        "type": "text",
                        "text": msg["content"],
                    })

                # Add tool_use blocks for each tool call
                # Handle both generic format (name, arguments) and LLMToolCall format (tool_name, tool_arguments)
                for tc in msg["tool_calls"]:
                    tool_name = tc.get("name") or tc.get("tool_name")
                    tool_args = tc.get("arguments") or tc.get("tool_arguments", {})
                    call_id = tc.get("call_id")
                    content_blocks.append({
                        "type": "tool_use",
                        "id": call_id,
                        "name": tool_name,
                        "input": tool_args,
                    })

                converted.append({
                    "role": "assistant",
                    "content": content_blocks,
                })

            # Handle regular messages
            else:
                # Convert string content to content blocks if needed
                content = msg.get("content", "")
                if isinstance(content, str):
                    converted.append({
                        "role": role,
                        "content": content,
                    })
                else:
                    converted.append({
                        "role": role,
                        "content": content,
                    })

        return converted

    def _extract_system_prompt(
        self,
        messages: list[dict[str, Any]],
        system_prompt: str | None,
    ) -> str | None:
        """
        Extract system prompt from messages or use provided one.

        Anthropic expects system prompt as a separate parameter, not in messages.
        """
        if system_prompt:
            return system_prompt

        # Look for system message in messages list
        for msg in messages:
            if msg.get("role") == "system":
                return msg.get("content")

        return None

    def _build_messages_api_kwargs(
        self,
        messages: list[dict[str, Any]],
        system_prompt: str | None,
        max_tokens: int | None,
        temperature: float | None,
        response_model: type[T] | None,
        extended_thinking: bool = False,
        stream: bool = False,
    ) -> dict[str, Any]:
        """Build kwargs for Anthropic Messages API call."""
        converted_messages = self._convert_messages_for_anthropic(messages)
        resolved_system = self._extract_system_prompt(messages, system_prompt)

        kwargs: dict[str, Any] = {
            "model": self.model.value,
            "messages": converted_messages,
            "max_tokens": self._resolve_max_tokens(max_tokens),
        }

        if resolved_system:
            kwargs["system"] = resolved_system

        # Temperature is not compatible with extended thinking
        if not extended_thinking:
            kwargs["temperature"] = self._resolve_temperature(
                temperature, structured=response_model is not None
            )

        if stream:
            kwargs["stream"] = True

        # Add extended thinking (reasoning) if requested
        if extended_thinking:
            kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": 10000,
            }

        # Add tools if registered and no response_model
        if self._tools and response_model is None:
            kwargs["tools"] = self._tools.copy()

        # Add structured output tool if response_model is provided
        if response_model is not None:
            schema = response_model.model_json_schema()
            cleaned_schema = _clean_schema_for_anthropic(schema)
            kwargs["tools"] = [
                {
                    "name": "structured_output",
                    "description": f"Return a structured response matching the {response_model.__name__} schema.",
                    "input_schema": cleaned_schema,
                }
            ]
            kwargs["tool_choice"] = {"type": "tool", "name": "structured_output"}

        return kwargs

    def _parse_response(
        self,
        response: Any,
        response_model: type[T] | None,
    ) -> LLMChatResponse:
        """Parse response from Anthropic Messages API."""
        content: str | None = None
        tool_calls: list[LLMToolCall] = []
        reasoning_content: str | None = None
        parsed = None

        # Extract content from response
        for block in response.content:
            if block.type == "text":
                content = block.text
            elif block.type == "thinking":
                reasoning_content = block.thinking
            elif block.type == "tool_use":
                # Handle structured output response
                if response_model is not None and block.name == "structured_output":
                    try:
                        parsed = response_model.model_validate(block.input)
                    except Exception as e:
                        logger.error(
                            "Failed to parse structured output into %s: %s",
                            response_model.__name__,
                            e,
                        )
                        raise ValueError(
                            f"Failed to parse structured response from Anthropic: {e}"
                        )
                else:
                    # Regular tool call
                    tool_calls.append(LLMToolCall(
                        tool_name=block.name,
                        tool_arguments=block.input,
                        call_id=block.id,
                    ))

        # NOTE: We intentionally do NOT return response_id for Anthropic.
        # Anthropic doesn't support response ID chaining like OpenAI's Responses API.
        # Returning an ID would cause NetworkSpyAgent to skip messages (via _previous_response_id),
        # which breaks Anthropic's requirement that every tool_result has a preceding tool_use.
        return LLMChatResponse(
            content=content,
            tool_calls=tool_calls,
            response_id=None,
            reasoning_content=reasoning_content,
            parsed=parsed,
        )

    # Public methods _______________________________________________________________________________________________________

    ## Tool management

    def register_tool(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
    ) -> None:
        """Register a tool in Anthropic's tool format."""
        logger.debug("Registering Anthropic tool: %s", name)
        cleaned_schema = _clean_schema_for_anthropic(parameters)
        self._tools.append({
            "name": name,
            "description": description,
            "input_schema": cleaned_schema,
        })

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
        stateful: bool = False,  # noqa: ARG002 - not supported by Anthropic
        previous_response_id: str | None = None,  # noqa: ARG002 - not supported by Anthropic
        api_type: Any = None,  # noqa: ARG002 - not applicable for Anthropic
        tool_choice: str | dict | None = None,  # noqa: ARG002 - TODO: implement
    ) -> LLMChatResponse:
        """
        Unified sync call to Anthropic.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            input: Input string (shorthand for simple prompts).
            system_prompt: Optional system prompt for context.
            max_tokens: Maximum tokens in the response.
            temperature: Sampling temperature (0.0-1.0).
            response_model: Pydantic model class for structured response.
            extended_reasoning: Enable extended thinking.
            stateful: Not supported by Anthropic (ignored).
            previous_response_id: Not supported by Anthropic (ignored).
            api_type: Not applicable for Anthropic (ignored).
            tool_choice: Tool choice configuration (TODO: implement).

        Returns:
            LLMChatResponse. If response_model is provided, the parsed model is in response.parsed.

        Raises:
            ValueError: If neither messages nor input is provided.
        """
        # Handle input shorthand
        if messages is None:
            if input is None:
                raise ValueError("Either messages or input must be provided")
            messages = [{"role": "user", "content": input}]

        kwargs = self._build_messages_api_kwargs(
            messages, system_prompt, max_tokens, temperature, response_model,
            extended_reasoning,
        )

        # Retry loop with exponential backoff
        last_error: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                response = self._client.messages.create(**kwargs)
                return self._parse_response(response, response_model)
            except (APIStatusError, RateLimitError) as e:
                last_error = e
                if not _is_retryable_error(e) or attempt == MAX_RETRIES - 1:
                    raise
                delay = _calculate_backoff(attempt)
                logger.warning(
                    "Anthropic API error (attempt %d/%d), retrying in %.1fs: %s",
                    attempt + 1, MAX_RETRIES, delay, e,
                )
                time.sleep(delay)

        # Should not reach here, but raise last error if we do
        raise last_error  # type: ignore[misc]

    async def call_async(
        self,
        messages: list[dict[str, str]] | None = None,
        input: str | None = None,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        response_model: type[T] | None = None,
        extended_reasoning: bool = False,
        stateful: bool = False,  # noqa: ARG002 - not supported by Anthropic
        previous_response_id: str | None = None,  # noqa: ARG002 - not supported by Anthropic
        api_type: Any = None,  # noqa: ARG002 - not applicable for Anthropic
        tool_choice: str | dict | None = None,  # noqa: ARG002 - TODO: implement
    ) -> LLMChatResponse:
        """
        Unified async call to Anthropic.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            input: Input string (shorthand for simple prompts).
            system_prompt: Optional system prompt for context.
            max_tokens: Maximum tokens in the response.
            temperature: Sampling temperature (0.0-1.0).
            response_model: Pydantic model class for structured response.
            extended_reasoning: Enable extended thinking.
            stateful: Not supported by Anthropic (ignored).
            previous_response_id: Not supported by Anthropic (ignored).
            api_type: Not applicable for Anthropic (ignored).
            tool_choice: Tool choice configuration (TODO: implement).

        Returns:
            LLMChatResponse. If response_model is provided, the parsed model is in response.parsed.

        Raises:
            ValueError: If neither messages nor input is provided.
        """
        # Handle input shorthand
        if messages is None:
            if input is None:
                raise ValueError("Either messages or input must be provided")
            messages = [{"role": "user", "content": input}]

        kwargs = self._build_messages_api_kwargs(
            messages, system_prompt, max_tokens, temperature, response_model,
            extended_reasoning,
        )

        # Retry loop with exponential backoff
        last_error: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                response = await self._async_client.messages.create(**kwargs)
                return self._parse_response(response, response_model)
            except (APIStatusError, RateLimitError) as e:
                last_error = e
                if not _is_retryable_error(e) or attempt == MAX_RETRIES - 1:
                    raise
                delay = _calculate_backoff(attempt)
                logger.warning(
                    "Anthropic API error (attempt %d/%d), retrying in %.1fs: %s",
                    attempt + 1, MAX_RETRIES, delay, e,
                )
                await asyncio.sleep(delay)

        # Should not reach here, but raise last error if we do
        raise last_error  # type: ignore[misc]

    def call_stream_sync(
        self,
        messages: list[dict[str, str]] | None = None,
        input: str | None = None,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        extended_reasoning: bool = False,
        stateful: bool = False,  # noqa: ARG002 - not supported by Anthropic
        previous_response_id: str | None = None,  # noqa: ARG002 - not supported by Anthropic
        api_type: Any = None,  # noqa: ARG002 - not applicable for Anthropic
        tool_choice: str | dict | None = None,  # noqa: ARG002 - TODO: implement
    ) -> Generator[str | LLMChatResponse, None, None]:
        """
        Unified streaming call to Anthropic.

        Yields text chunks as they arrive, then yields the final LLMChatResponse.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            input: Input string (shorthand for simple prompts).
            system_prompt: Optional system prompt for context.
            max_tokens: Maximum tokens in the response.
            temperature: Sampling temperature (0.0-1.0).
            extended_reasoning: Enable extended thinking.
            stateful: Not supported by Anthropic (ignored).
            previous_response_id: Not supported by Anthropic (ignored).
            api_type: Not applicable for Anthropic (ignored).
            tool_choice: Tool choice configuration (TODO: implement).

        Yields:
            str: Text chunks as they arrive.
            LLMChatResponse: Final response with complete content and optional tool call.
        """
        # Handle input shorthand
        if messages is None:
            if input is None:
                raise ValueError("Either messages or input must be provided")
            messages = [{"role": "user", "content": input}]

        # Build kwargs without stream parameter - messages.stream() handles it implicitly
        kwargs = self._build_messages_api_kwargs(
            messages, system_prompt, max_tokens, temperature, response_model=None,
            extended_thinking=extended_reasoning,
        )

        # Retry loop with exponential backoff
        last_error: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                full_content: list[str] = []
                tool_calls: list[LLMToolCall] = []
                reasoning_content: str | None = None

                # Track current tool being streamed
                current_tool: dict[str, Any] | None = None
                current_tool_input: list[str] = []

                with self._client.messages.stream(**kwargs) as stream:
                    for event in stream:
                        # Handle content block start
                        if event.type == "content_block_start":
                            if event.content_block.type == "tool_use":
                                current_tool = {
                                    "id": event.content_block.id,
                                    "name": event.content_block.name,
                                }
                                current_tool_input = []

                        # Handle content block delta
                        elif event.type == "content_block_delta":
                            delta = event.delta
                            if delta.type == "text_delta":
                                full_content.append(delta.text)
                                yield delta.text
                            elif delta.type == "thinking_delta":
                                if reasoning_content is None:
                                    reasoning_content = ""
                                reasoning_content += delta.thinking
                            elif delta.type == "input_json_delta":
                                if current_tool is not None:
                                    current_tool_input.append(delta.partial_json)

                        # Handle content block stop
                        elif event.type == "content_block_stop":
                            if current_tool is not None:
                                # Parse accumulated JSON input
                                raw_input = "".join(current_tool_input) if current_tool_input else "{}"
                                try:
                                    parsed_input = json.loads(raw_input)
                                except json.JSONDecodeError as e:
                                    logger.error(
                                        "Failed to parse tool input for %s: %s. Raw: %s",
                                        current_tool["name"],
                                        e,
                                        raw_input[:500],
                                    )
                                    raise

                                tool_calls.append(LLMToolCall(
                                    tool_name=current_tool["name"],
                                    tool_arguments=parsed_input,
                                    call_id=current_tool["id"],
                                ))
                                current_tool = None
                                current_tool_input = []

                # Yield final response
                # NOTE: We intentionally do NOT return response_id for Anthropic.
                # See _parse_response for explanation.
                yield LLMChatResponse(
                    content="".join(full_content) if full_content else None,
                    tool_calls=tool_calls,
                    response_id=None,
                    reasoning_content=reasoning_content,
                )
                return  # Success, exit retry loop

            except (APIStatusError, RateLimitError) as e:
                last_error = e
                if not _is_retryable_error(e) or attempt == MAX_RETRIES - 1:
                    raise
                delay = _calculate_backoff(attempt)
                logger.warning(
                    "Anthropic API error during streaming (attempt %d/%d), retrying in %.1fs: %s",
                    attempt + 1, MAX_RETRIES, delay, e,
                )
                time.sleep(delay)

        # Should not reach here, but raise last error if we do
        raise last_error  # type: ignore[misc]
