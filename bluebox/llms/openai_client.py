"""
bluebox/llms/openai_client.py

OpenAI-specific LLM client implementation with unified API supporting
both Chat Completions and Responses APIs.
"""

import json
from collections.abc import Generator
from typing import Any, TypeVar

from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel

from bluebox.config import Config
from bluebox.data_models.llms.interaction import LLMChatResponse, LLMToolCall
from bluebox.data_models.llms.vendors import OpenAIAPIType, OpenAIModel
from bluebox.llms.abstract_llm_vendor_client import AbstractLLMVendorClient
from bluebox.utils.logger import get_logger

logger = get_logger(name=__name__)


T = TypeVar("T", bound=BaseModel)


def _clean_schema_for_openai(schema: dict[str, Any]) -> dict[str, Any]:
    """
    Recursively clean a JSON schema for OpenAI's structured output requirements:
    1. Add 'additionalProperties': false to all object types
    2. Remove extra keywords when $ref is present (OpenAI doesn't allow $ref with other keys)
    3. Ensure all properties are in the 'required' array (OpenAI strict mode requirement)
    """
    schema = schema.copy()

    # If $ref is present, remove all other keys except $ref (OpenAI requirement)
    if "$ref" in schema:
        return {"$ref": schema["$ref"]}

    # Add to root if it's an object type
    if schema.get("type") == "object" or "properties" in schema:
        schema["additionalProperties"] = False
        # OpenAI strict mode requires all properties to be in 'required'
        if "properties" in schema:
            schema["required"] = list(schema["properties"].keys())

    # Process properties
    if "properties" in schema:
        schema["properties"] = {
            k: _clean_schema_for_openai(v) for k, v in schema["properties"].items()
        }

    # Process $defs (Pydantic uses this for nested models)
    if "$defs" in schema:
        schema["$defs"] = {
            k: _clean_schema_for_openai(v) for k, v in schema["$defs"].items()
        }

    # Process array items
    if "items" in schema:
        schema["items"] = _clean_schema_for_openai(schema["items"])

    # Process anyOf, oneOf, allOf
    for key in ("anyOf", "oneOf", "allOf"):
        if key in schema:
            schema[key] = [_clean_schema_for_openai(s) for s in schema[key]]

    return schema


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
        self._file_search_vectorstores: list[str] | None = None
        self._file_search_filters: dict | None = None
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

    def _convert_messages_for_chat_completions(
        self,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Convert messages from generic format to Chat Completions API format.

        Handles tool_calls conversion from generic format:
        - Generic: {"call_id": ..., "name": ..., "arguments": {...}}
        - Chat Completions: {"id": ..., "type": "function", "function": {"name": ..., "arguments": "..."}}

        Args:
            messages: Messages in generic format

        Returns:
            Messages converted to Chat Completions API format
        """
        converted: list[dict[str, Any]] = []
        for msg in messages:
            if msg.get("tool_calls"):
                # Convert tool_calls from generic to Chat Completions format
                converted_msg = {k: v for k, v in msg.items() if k != "tool_calls"}
                converted_msg["tool_calls"] = [
                    {
                        "id": tc.get("call_id"),
                        "type": "function",
                        "function": {
                            "name": tc.get("name"),
                            "arguments": json.dumps(tc.get("arguments", {})) if isinstance(tc.get("arguments"), dict) else tc.get("arguments", "{}"),
                        },
                    }
                    for tc in msg["tool_calls"]
                ]
                converted.append(converted_msg)
            else:
                converted.append(msg)
        return converted

    def _has_file_search_tools(self) -> bool:
        """Check if file_search vectorstores are configured (Responses API only)."""
        return bool(self._file_search_vectorstores)

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
        has_file_search = self._has_file_search_tools()

        if extended_reasoning and api_type == OpenAIAPIType.CHAT_COMPLETIONS:
            raise ValueError("extended_reasoning=True requires Responses API")
        if previous_response_id and api_type == OpenAIAPIType.CHAT_COMPLETIONS:
            raise ValueError("previous_response_id requires Responses API")
        if has_file_search and api_type == OpenAIAPIType.CHAT_COMPLETIONS:
            raise ValueError("file_search tools require Responses API")

        # Auto-resolve
        if api_type is None:
            if extended_reasoning or previous_response_id or has_file_search:
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
        # Convert messages from generic format to Chat Completions format
        converted_messages = self._convert_messages_for_chat_completions(messages)
        all_messages = self._prepend_system_prompt(converted_messages, system_prompt)

        kwargs: dict[str, Any] = {
            "model": self.model.value,
            "messages": all_messages,
            "max_completion_tokens": self._resolve_max_tokens(max_tokens),
        }

        if stream:
            kwargs["stream"] = True

        if self._tools and response_model is None:
            kwargs["tools"] = self._tools.copy()

        return kwargs

    def _convert_tool_to_responses_api_format(self, tool: dict[str, Any]) -> dict[str, Any]:
        """
        Convert a tool from Chat Completions format to Responses API format if needed.

        Chat Completions format: {"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}
        Responses API format: {"type": "function", "name": ..., "description": ..., "parameters": ...}
        """
        if tool.get("type") == "function" and "function" in tool:
            # Convert from Chat Completions format to Responses API format
            func_def = tool["function"]
            return {
                "type": "function",
                "name": func_def.get("name"),
                "description": func_def.get("description"),
                "parameters": func_def.get("parameters"),
            }
        # Already in Responses API format (file_search, or already flat function)
        return tool

    def _convert_messages_for_responses_api(
        self,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Convert messages to Responses API format.

        Handles the difference between Chat Completions API and Responses API:
        - Chat Completions uses role: "tool" with tool_call_id
        - Responses API uses type: "function_call_output" with call_id
        - Assistant tool_calls become separate function_call items

        Args:
            messages: Messages in Chat Completions format

        Returns:
            Messages converted to Responses API format
        """
        converted: list[dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role")
            if role == "tool":
                # Convert tool message to function_call_output item
                call_id = msg.get("tool_call_id")
                if call_id:
                    converted.append({
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": msg.get("content", ""),
                    })
                else:
                    # Fallback: include as user message if no call_id
                    logger.warning("Tool message without call_id, converting to user message")
                    converted.append({
                        "role": "user",
                        "content": f"[Tool result]: {msg.get('content', '')}",
                    })
            elif role == "assistant" and msg.get("tool_calls"):
                # Assistant message with tool calls
                # First add the message content (if any)
                if msg.get("content"):
                    converted.append({
                        "role": "assistant",
                        "content": msg["content"],
                    })
                # Then add function_call items for each tool call
                for tc in msg["tool_calls"]:
                    call_id = tc.get("call_id")
                    if call_id:
                        converted.append({
                            "type": "function_call",
                            "call_id": call_id,
                            "name": tc.get("name", ""),
                            "arguments": json.dumps(tc.get("arguments", {})) if isinstance(tc.get("arguments"), dict) else tc.get("arguments", "{}"),
                        })
            else:
                # Keep other messages as-is (but remove tool_calls field if present)
                clean_msg = {k: v for k, v in msg.items() if k != "tool_calls"}
                converted.append(clean_msg)
        return converted

    def _build_responses_api_kwargs(
        self,
        messages: list[dict[str, Any]] | None,
        input_text: str | None,
        system_prompt: str | None,
        max_tokens: int | None,
        extended_reasoning: bool,
        previous_response_id: str | None,
        response_model: type[T] | None,
        stream: bool = False,
        tool_choice: str | dict | None = None,
    ) -> dict[str, Any]:
        """Build kwargs for Responses API call."""
        kwargs: dict[str, Any] = {
            "model": self.model.value,
            "max_output_tokens": self._resolve_max_tokens(max_tokens),
        }

        # Handle previous_response_id for conversation chaining
        if previous_response_id:
            kwargs["previous_response_id"] = previous_response_id

        # Handle input
        if input_text:
            kwargs["input"] = input_text
        elif messages:
            converted_messages = self._convert_messages_for_responses_api(messages)
            kwargs["input"] = converted_messages
        elif not previous_response_id:
            raise ValueError("Either messages or input must be provided")

        # Always pass system prompt as instructions
        if system_prompt:
            kwargs["instructions"] = system_prompt

        if stream:
            kwargs["stream"] = True

        if extended_reasoning:
            kwargs["reasoning"] = {"effort": "medium"}

        # Build tools list: registered function tools + file_search if configured
        all_tools: list[dict[str, Any]] = []
        for tool in self._tools:
            all_tools.append(self._convert_tool_to_responses_api_format(tool))
        if self._file_search_vectorstores:
            file_search_tool: dict[str, Any] = {
                "type": "file_search",
                "vector_store_ids": self._file_search_vectorstores,
            }
            if self._file_search_filters:
                file_search_tool["filters"] = self._file_search_filters
            all_tools.append(file_search_tool)

        if all_tools and response_model is None:
            kwargs["tools"] = all_tools
            if tool_choice:
                kwargs["tool_choice"] = tool_choice

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

        # Extract all tool calls
        tool_calls: list[LLMToolCall] = []
        if message.tool_calls:
            for tc in message.tool_calls:
                tool_calls.append(LLMToolCall(
                    tool_name=tc.function.name,
                    tool_arguments=json.loads(tc.function.arguments),
                    call_id=tc.id,
                ))

        return LLMChatResponse(
            content=message.content,
            tool_calls=tool_calls,
        )

    def _parse_responses_api_response(
        self,
        response: Any,
        response_model: type[T] | None,
    ) -> LLMChatResponse:
        """Parse response from Responses API."""
        # Extract content, tool calls, and parsed model
        content: str | None = None
        tool_calls: list[LLMToolCall] = []
        reasoning_content: str | None = None
        parsed = None

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

                # Handle message content and parsed structured output
                if item.type == "message":
                    if hasattr(item, "content") and item.content:
                        text_parts = []
                        for content_block in item.content:
                            if content_block.type == "output_text":
                                text_parts.append(content_block.text)
                            # Extract parsed model if available
                            if response_model is not None and hasattr(content_block, "parsed") and content_block.parsed:
                                parsed = content_block.parsed
                        if text_parts:
                            content = "".join(text_parts)

                # Handle function calls - collect all of them
                if item.type == "function_call":
                    tool_calls.append(LLMToolCall(
                        tool_name=item.name,
                        tool_arguments=json.loads(item.arguments) if isinstance(item.arguments, str) else item.arguments,
                        call_id=getattr(item, "call_id", None),
                    ))

        # If response_model provided but not auto-parsed, manually parse from JSON content
        if response_model is not None and parsed is None and content:
            try:
                parsed = response_model.model_validate_json(content)
            except Exception as e:
                logger.error("Failed to parse JSON content into %s: %s", response_model.__name__, e)
                raise ValueError(f"Failed to parse structured response from OpenAI Responses API: {e}")

        # Validate parsed model was found if response_model was provided
        if response_model is not None and parsed is None:
            raise ValueError("Failed to parse structured response from OpenAI Responses API: no content returned")

        return LLMChatResponse(
            content=content,
            tool_calls=tool_calls,
            response_id=response.id,
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

    def set_file_search_vectorstores(
        self,
        vector_store_ids: list[str] | None,
        filters: dict | None = None,
    ) -> None:
        """
        Set vectorstore IDs for file_search tool.

        Args:
            vector_store_ids: List of vectorstore IDs to search, or None to disable.
            filters: Optional filters for file_search (e.g., {"type": "eq", "key": "uuid", "value": ["..."]}).
        """
        self._file_search_vectorstores = vector_store_ids
        self._file_search_filters = filters
        if vector_store_ids:
            logger.info("Set file_search vectorstores: %s (filters: %s)", vector_store_ids, filters)
        else:
            logger.info("Cleared file_search vectorstores")

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
                    messages, system_prompt, max_tokens, response_model,
                )
                response = self._client.beta.chat.completions.parse(
                    **kwargs,
                    response_format=response_model,
                )
            else:
                kwargs = self._build_chat_completions_kwargs(
                    messages, system_prompt, max_tokens, response_model,
                )
                response = self._client.chat.completions.create(**kwargs)

            return self._parse_chat_completions_response(response, response_model)

        else:  # Responses API
            kwargs = self._build_responses_api_kwargs(
                messages, input, system_prompt, max_tokens,
                extended_reasoning, previous_response_id, response_model,
                tool_choice=tool_choice,
            )

            if response_model is not None:
                # Use responses.parse() with text_format for automatic schema handling
                response = self._client.responses.parse(
                    **kwargs,
                    text_format=response_model,
                )
            else:
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
                    messages, system_prompt, max_tokens, response_model,
                )
                response = await self._async_client.beta.chat.completions.parse(
                    **kwargs,
                    response_format=response_model,
                )
            else:
                kwargs = self._build_chat_completions_kwargs(
                    messages, system_prompt, max_tokens, response_model,
                )
                response = await self._async_client.chat.completions.create(**kwargs)

            return self._parse_chat_completions_response(response, response_model)

        else:  # Responses API
            kwargs = self._build_responses_api_kwargs(
                messages, input, system_prompt, max_tokens,
                extended_reasoning, previous_response_id, response_model,
                tool_choice=tool_choice,
            )

            if response_model is not None:
                # Use responses.parse() with text_format for automatic schema handling
                response = await self._async_client.responses.parse(
                    **kwargs,
                    text_format=response_model,
                )
            else:
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
        resolved_api_type = self._validate_and_resolve_api_type(
            api_type, extended_reasoning, previous_response_id
        )

        if resolved_api_type == OpenAIAPIType.CHAT_COMPLETIONS:
            if messages is None:
                raise ValueError("messages is required for Chat Completions API")

            kwargs = self._build_chat_completions_kwargs(
                messages, system_prompt, max_tokens, response_model=None, stream=True,
            )
            stream = self._client.chat.completions.create(**kwargs)

            # Accumulate content and tool call data
            full_content: list[str] = []
            # Track tool calls by index: {index: {"name": str | None, "args": list[str]}}
            tool_calls_by_index: dict[int, dict[str, Any]] = {}

            for chunk in stream:
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta is None:
                    continue

                # Handle text content
                if delta.content:
                    full_content.append(delta.content)
                    yield delta.content

                # Handle tool calls (streamed in chunks) - track by index
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index
                        if idx not in tool_calls_by_index:
                            tool_calls_by_index[idx] = {"name": None, "args": [], "call_id": None}
                        if tc.id:  # Tool call ID comes in first chunk
                            tool_calls_by_index[idx]["call_id"] = tc.id
                        if tc.function:
                            if tc.function.name:
                                tool_calls_by_index[idx]["name"] = tc.function.name
                            if tc.function.arguments:
                                tool_calls_by_index[idx]["args"].append(tc.function.arguments)

            # Build final response with all tool calls
            tool_calls: list[LLMToolCall] = []
            for idx in sorted(tool_calls_by_index.keys()):
                tc_data = tool_calls_by_index[idx]
                if tc_data["name"]:
                    raw_args = "".join(tc_data["args"]) if tc_data["args"] else "{}"
                    tool_calls.append(LLMToolCall(
                        tool_name=tc_data["name"],
                        tool_arguments=json.loads(raw_args),
                        call_id=tc_data.get("call_id"),
                    ))

            yield LLMChatResponse(
                content="".join(full_content) if full_content else None,
                tool_calls=tool_calls,
            )

        else:  # Responses API streaming
            kwargs = self._build_responses_api_kwargs(
                messages, input, system_prompt, max_tokens,
                extended_reasoning, previous_response_id, response_model=None, stream=True,
                tool_choice=tool_choice,
            )

            stream = self._client.responses.create(**kwargs)

            full_content: list[str] = []
            # Track tool calls by output_index: {index: {"name": str | None, "args": list[str]}}
            tool_calls_by_index: dict[int, dict[str, Any]] = {}
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
                        # Track arguments by output_index
                        if hasattr(event, "delta") and hasattr(event, "output_index"):
                            idx = event.output_index
                            if idx not in tool_calls_by_index:
                                tool_calls_by_index[idx] = {"name": None, "args": [], "call_id": None}
                            tool_calls_by_index[idx]["args"].append(event.delta)

                    elif event.type == "response.output_item.added":
                        # Track function call name and call_id by output_index
                        if hasattr(event, "item") and event.item.type == "function_call":
                            idx = event.output_index if hasattr(event, "output_index") else 0
                            if idx not in tool_calls_by_index:
                                tool_calls_by_index[idx] = {"name": None, "args": [], "call_id": None}
                            tool_calls_by_index[idx]["name"] = event.item.name
                            tool_calls_by_index[idx]["call_id"] = getattr(event.item, "call_id", None)
                            # Also capture arguments if already present (not streamed via delta)
                            if hasattr(event.item, "arguments") and event.item.arguments:
                                tool_calls_by_index[idx]["args"].append(event.item.arguments)

            # Build final response with all tool calls
            tool_calls: list[LLMToolCall] = []
            for idx in sorted(tool_calls_by_index.keys()):
                tc_data = tool_calls_by_index[idx]
                if tc_data["name"]:
                    raw_args = "".join(tc_data["args"]) if tc_data["args"] else "{}"
                    try:
                        parsed_args = json.loads(raw_args)
                    except json.JSONDecodeError as e:
                        logger.error(
                            "Failed to parse tool call arguments for %s (index %d): %s. Raw args: %s",
                            tc_data["name"],
                            idx,
                            e,
                            raw_args[:500],
                        )
                        raise
                    logger.debug(
                        "Parsed tool call %s (index %d): raw_args=%s, parsed=%s",
                        tc_data["name"],
                        idx,
                        raw_args[:200],
                        parsed_args,
                    )
                    tool_calls.append(LLMToolCall(
                        tool_name=tc_data["name"],
                        tool_arguments=parsed_args,
                        call_id=tc_data.get("call_id"),
                    ))

            yield LLMChatResponse(
                content="".join(full_content) if full_content else None,
                tool_calls=tool_calls,
                response_id=response_id,
                reasoning_content=reasoning_content,
            )
