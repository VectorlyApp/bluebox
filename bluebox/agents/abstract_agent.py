"""
bluebox/agents/abstract_agent.py

Abstract base class for LLM-powered agents.

This module provides the foundational infrastructure for agents that interact with LLMs:
  - Chat/thread management and persistence
  - Tool registration and execution via the @agent_tool decorator
  - LLM calling with streaming support
  - Message emission

Subclasses implement domain-specific behavior by:
  - Overriding _get_system_prompt() to provide context
  - Decorating methods with @agent_tool to expose tools to the LLM
  - Implementing their own run loop using the provided infrastructure
"""

from __future__ import annotations

import json
import functools
from abc import ABC, abstractmethod
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, ClassVar, get_type_hints

from pydantic import TypeAdapter, ValidationError

from bluebox.data_models.llms.interaction import (
    Chat,
    ChatRole,
    ChatThread,
    EmittedMessage,
    ChatResponseEmittedMessage,
    ErrorEmittedMessage,
    LLMChatResponse,
    LLMToolCall,
    ToolInvocationResultEmittedMessage,
    PendingToolInvocation,
    ToolInvocationStatus,
)
from bluebox.data_models.llms.vendors import LLMModel, OpenAIModel
from bluebox.llms.data_loaders.documentation_data_loader import DocumentationDataLoader, FileType
from bluebox.llms.llm_client import LLMClient
from bluebox.llms.tools.tool_utils import extract_description_from_docstring, generate_parameters_schema
from bluebox.utils.data_utils import format_bytes
from bluebox.utils.llm_utils import token_optimized
from bluebox.utils.logger import get_logger

logger = get_logger(name=__name__)


@dataclass(frozen=True)
class AgentCard:
    """
    Self-describing metadata for an agent.

    Every concrete (non-abstract) AbstractAgent subclass must define an AGENT_CARD
    class variable. Orchestrator agents use these cards to discover subagent
    capabilities at runtime — e.g. to auto-generate delegation prompts.
    """
    description: str  # 1-2 sentence summary for orchestrator prompts


@dataclass(frozen=True)
class _ToolMeta:
    """Metadata attached to a handler method by @agent_tool."""
    name: str                                           # tool name registered with the LLM client
    description: str                                    # tool description shown to the LLM
    parameters: dict[str, Any]                          # JSON Schema for tool parameters
    availability: bool | Callable[..., bool]            # whether the tool should be registered right now


def agent_tool(
    description: str | Callable | None = None,
    parameters: dict[str, Any] | None = None,
    *,
    availability: bool | Callable[..., bool] = True,
) -> Callable:
    """
    Decorator that marks a method as an agent tool handler.

    The tool name is derived from the method name by stripping leading
    underscores. For example, ``_get_dom_snapshot`` becomes ``get_dom_snapshot``.

    Can be used with or without parentheses:
        @agent_tool
        def _my_tool(self, x: str) -> dict: ...

        @agent_tool()
        def _my_tool(self, x: str) -> dict: ...

        @agent_tool(description="Custom description")
        def _my_tool(self, x: str) -> dict: ...

    Args:
        description: Tool description for the LLM. If None, extracted from
            the method's docstring (text before Args/Returns sections).
        parameters: JSON Schema for tool parameters. If None, auto-generated
            from the method's type hints and docstring.
        availability: Controls when the tool is available to the LLM.
            - True (default): always available.
            - Callable[[self], bool]: evaluated before each LLM call;
              tool is available only when it returns True. Use this for tools
              gated behind lifecycle state or dynamic conditions (e.g.
              ``availability=lambda self: self.can_finalize``).
    """
    def decorator(method: Callable, desc: str | None = None) -> Callable:
        tool_name = method.__name__.lstrip("_")

        # auto-extract description from docstring if not provided
        if desc is None:
            final_description = extract_description_from_docstring(docstring=method.__doc__)
            if not final_description:
                raise ValueError(f"Tool {tool_name} has no description and no docstring")
        else:
            final_description = desc

        # auto-generate parameters schema from method signature if not provided
        if parameters is None:
            final_parameters = generate_parameters_schema(func=method)
        else:
            final_parameters = parameters

        method._tool_meta = _ToolMeta(
            name=tool_name,
            description=final_description,
            parameters=final_parameters,
            availability=availability,
        )
        return method

    # Support @agent_tool without parentheses - description will be the function
    if callable(description):
        return decorator(description, desc=None)

    # Support @agent_tool() or @agent_tool(description="...")
    return lambda method: decorator(method, desc=description)


class AbstractAgent(ABC):
    """
    Abstract base class for LLM-powered agents.

    Subclasses implement domain-specific logic by overriding:
      - _get_system_prompt(): Return the system prompt for the agent

    Tools are defined declaratively via the @agent_tool decorator on handler
    methods. Each tool's ``availability`` controls when it is registered: True
    (always), or a callable evaluated before each LLM call.

    The base class handles all LLM conversation mechanics:
      - Chat history, threading, persistence
      - Streaming support
      - Tool registration and execution
      - Message emission
    """

    # Class-level configuration (can be overridden by subclasses)
    AGENT_LOOP_MAX_ITERATIONS: int = 10
    AGENT_CARD: ClassVar[AgentCard]  # must be defined by every concrete subclass

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Validate that concrete subclasses define AGENT_CARD."""
        super().__init_subclass__(**kwargs)
        # skip abstract classes (matches existing naming convention in AbstractSpecialist)
        if cls.__name__.startswith("Abstract"):
            return
        # validate that the subclass defines an AGENT_CARD class variable of type AgentCard
        if not hasattr(cls, "AGENT_CARD") or not isinstance(cls.AGENT_CARD, AgentCard):
            raise TypeError(
                f"{cls.__name__} must define an AGENT_CARD class variable of type AgentCard"
            )

    ## Abstract methods

    @abstractmethod
    def _get_system_prompt(self) -> str:
        """Return the system prompt for the agent."""

    ## Magic methods

    def __init__(
        self,
        emit_message_callable: Callable[[EmittedMessage], None],
        persist_chat_callable: Callable[[Chat], Chat] | None = None,
        persist_chat_thread_callable: Callable[[ChatThread], ChatThread] | None = None,
        stream_chunk_callable: Callable[[str], None] | None = None,
        llm_model: LLMModel = OpenAIModel.GPT_5_1,
        chat_thread: ChatThread | None = None,
        existing_chats: list[Chat] | None = None,
        documentation_data_loader: DocumentationDataLoader | None = None,
    ) -> None:
        """
        Initialize the agent.

        Args:
            emit_message_callable: Callback to emit messages to the host.
            persist_chat_callable: Optional callback to persist Chat objects.
            persist_chat_thread_callable: Optional callback to persist ChatThread.
            stream_chunk_callable: Optional callback for streaming text chunks.
            llm_model: The LLM model to use.
            chat_thread: Existing ChatThread to continue, or None for new.
            existing_chats: Existing Chat messages if loading from persistence.
            documentation_data_loader: Optional DocumentationDataLoader for docs/code search tools.
        """
        self._emit_message_callable = emit_message_callable
        self._persist_chat_callable = persist_chat_callable
        self._persist_chat_thread_callable = persist_chat_thread_callable
        self._stream_chunk_callable = stream_chunk_callable
        self._documentation_data_loader = documentation_data_loader
        self._previous_response_id: str | None = None
        self._response_id_to_chat_index: dict[str, int] = {}

        self.llm_model = llm_model
        self.llm_client = LLMClient(llm_model)

        # Track which tools are currently registered
        self._registered_tool_names: set[str] = set()

        # Initial tool sync
        self._sync_tools()

        # Conversation state
        self._thread = chat_thread or ChatThread()
        self._chats: dict[str, Chat] = {}
        if existing_chats:
            for chat in existing_chats:
                self._chats[chat.id] = chat

        # Persist initial thread if callback provided
        if self._persist_chat_thread_callable and chat_thread is None:
            self._thread = self._persist_chat_thread_callable(self._thread)

    ## Properties

    @property
    def chat_thread_id(self) -> str:
        """Return the current thread ID."""
        return self._thread.id

    ## Public API

    def get_thread(self) -> ChatThread:
        """Get the current conversation thread."""
        return self._thread

    def get_chats(self) -> list[Chat]:
        """Get all Chat messages in order."""
        return [self._chats[chat_id] for chat_id in self._thread.chat_ids if chat_id in self._chats]

    def reset(self) -> None:
        """Reset the conversation to a fresh state."""
        old_chat_thread_id = self._thread.id
        self._thread = ChatThread()
        self._chats = {}
        self._previous_response_id = None
        self._response_id_to_chat_index = {}

        # Sync tools for fresh state
        self._sync_tools()

        if self._persist_chat_thread_callable:
            self._thread = self._persist_chat_thread_callable(self._thread)

        logger.debug("Reset conversation from %s to %s", old_chat_thread_id, self._thread.id)

    ## Tool registration and dispatch

    def _sync_tools(self) -> None:
        """
        Synchronize tools registered with the LLM client based on current availability.

        Clears all existing tools and re-registers only those whose ``availability``
        evaluates to True. Called automatically before each LLM call to ensure the
        tool set reflects current state (mode, iteration, etc.).
        """
        self.llm_client.clear_tools()
        self._registered_tool_names = set()
        collected_tools = self._collect_tools()
        if not collected_tools:
            logger.debug("No tools to sync")
            return

        for tool_meta, _ in collected_tools:
            available = tool_meta.availability(self) if callable(tool_meta.availability) else tool_meta.availability
            if not available:
                continue
            self.llm_client.register_tool(
                name=tool_meta.name,
                description=tool_meta.description,
                parameters=tool_meta.parameters,
            )
            self._registered_tool_names.add(tool_meta.name)
        logger.debug("Synced %s total tools: %s", len(collected_tools), self._registered_tool_names)

    def _execute_tool(self, tool_name: str, tool_arguments: dict[str, Any]) -> dict[str, Any]:
        """
        Dispatch a tool call to the appropriate decorated handler.

        Validates required parameters and types based on the handler's signature,
        then dispatches with **kwargs.
        """
        dispatch: dict[str, tuple[_ToolMeta, Callable]] = {
            tool_meta.name: (tool_meta, method) for tool_meta, method in self._collect_tools()
        }
        entry = dispatch.get(tool_name)
        if entry is None:
            logger.warning("Unknown tool: %s", tool_name)
            return {"error": f"Unknown tool: {tool_name}"}

        tool_meta, handler = entry

        # check availability at execution time (tool set may have changed since LLM saw it)
        available = tool_meta.availability(self) if callable(tool_meta.availability) else tool_meta.availability
        if not available:
            logger.warning("Invoked tool '%s' is not currently available", tool_name)
            return {"error": f"Tool '{tool_name}' is not currently available"}

        # validate required parameters
        required = tool_meta.parameters.get("required", [])
        missing = [p for p in required if p not in tool_arguments or tool_arguments[p] is None]
        if missing:
            return {"error": f"Missing required parameter(s): {', '.join(missing)}"}

        # validate no extra parameters
        valid_params = set(tool_meta.parameters.get("properties", {}).keys())
        extra = set(tool_arguments.keys()) - valid_params
        if extra:
            return {"error": f"Unknown parameter(s) for '{tool_name}': {', '.join(sorted(extra))}"}

        # validate and coerce types using the handler's type hints
        # (e.g. a dict from the LLM becomes a Pydantic model instance)
        validated_arguments: dict[str, Any] = {}
        try:
            hints = get_type_hints(obj=handler)
            for param_name, value in tool_arguments.items():
                if param_name in hints and value is not None:
                    expected_type = hints[param_name]
                    validated_arguments[param_name] = TypeAdapter(expected_type).validate_python(value)
                else:
                    validated_arguments[param_name] = value
        except ValidationError as e:
            # extract readable error message
            errors = e.errors()
            if errors:
                err = errors[0]
                msg = f"{param_name}: expected {err.get('type', 'valid type')}, got {type(value).__name__}"
            else:
                msg = str(e)
            return {"error": f"Invalid argument type: {msg}"}

        logger.debug("Executing tool %s with arguments: %s", tool_name, tool_arguments)
        # handler is unbound (from cls, not self) so pass self explicitly
        return handler(self, **validated_arguments)

    @classmethod
    @functools.lru_cache
    def _collect_tools(cls) -> tuple[tuple[_ToolMeta, Callable], ...]:
        """
        Collect all methods decorated with @agent_tool.

        NOTE: lru_cache memoizes the result per subclass, avoiding repeated dir(cls) traversal
        on every _sync_tools/_execute_tool call. Safe because decorated tools are fixed at class definition.
        """
        results: list[tuple[_ToolMeta, Callable]] = []
        for attr_name in dir(cls):
            method = getattr(cls, attr_name, None)
            tool_meta = getattr(method, "_tool_meta", None)
            if isinstance(tool_meta, _ToolMeta):
                results.append((tool_meta, method))
        return tuple(results)

    ## Documentation tools & prompt section

    def _get_documentation_prompt_section(self) -> str:
        """
        Build a system prompt addendum describing available documentation files.

        Lists the indexed file inventory so the LLM knows what docs/code are searchable.
        Tool names are NOT listed here — they come from _get_tool_availability_prompt_section().
        Appended automatically to system prompts when a documentation_data_loader is present.
        """
        if not self._documentation_data_loader:
            return ""

        stats = self._documentation_data_loader.stats
        lines = [
            "\n\n## Documentation",
            f"You have {stats.total_files} indexed files ({stats.total_docs} docs, {stats.total_code} code, {format_bytes(stats.total_bytes)}).",
        ]

        doc_index = self._documentation_data_loader.get_documentation_index()
        if doc_index:
            lines.append("\nDoc files:")
            for doc in doc_index:
                title = doc.get("title", "")
                if title:
                    if len(title) > 80:
                        title = title[:80] + "..."
                    lines.append(f"- `{doc['filename']}`: {title}")
                else:
                    lines.append(f"- `{doc['filename']}`")

        code_index = self._documentation_data_loader.get_code_index()
        if code_index:
            lines.append("\nCode files:")
            for code in code_index:
                docstring = code.get("docstring", "")
                if docstring:
                    if len(docstring) > 80:
                        docstring = docstring[:80] + "..."
                    lines.append(f"- `{code['filename']}`: {docstring}")
                else:
                    lines.append(f"- `{code['filename']}`")

        return "\n".join(lines)

    @agent_tool(
        availability=lambda self: self._documentation_data_loader is not None,
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The exact string to search for.",
                },
                "file_type": {
                    "type": "string",
                    "enum": ["documentation", "code"],
                    "description": "Optional filter by file type.",
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Whether the search should be case-sensitive. Defaults to false.",
                },
            },
            "required": ["query"],
        },
    )
    @token_optimized
    def _search_docs(
        self,
        query: str,
        file_type: str | None = None,
        case_sensitive: bool = False,
    ) -> dict[str, Any]:
        """
        Search documentation/code file contents for an exact query string (like Cmd+F).

        Returns line numbers where matches are found. Use get_doc_file to read around those lines.

        Args:
            query: The exact string to search for.
            file_type: Optional filter: 'documentation' for docs, 'code' for source files.
            case_sensitive: Whether the search should be case-sensitive. Defaults to false.
        """
        if not query:
            return {"error": "query is required"}

        file_type_enum = FileType(file_type) if file_type else None

        results = self._documentation_data_loader.search_content_with_lines(
            query=query,
            file_type=file_type_enum,
            case_sensitive=case_sensitive,
            max_matches_per_file=10,
        )

        if not results:
            return {"message": f"No matches found for '{query}'", "case_sensitive": case_sensitive}

        return {
            "query": query,
            "case_sensitive": case_sensitive,
            "files_with_matches": len(results),
            "results": results[:20],
        }

    @agent_tool(availability=lambda self: self._documentation_data_loader is not None)
    @token_optimized
    def _get_doc_file(
        self,
        path: str,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> dict[str, Any]:
        """
        Read documentation/code file content by path.

        Supports optional line range. Use start_line/end_line to read around matches from search_docs.

        Args:
            path: The file path (can be partial, will match).
            start_line: Starting line number (1-indexed, inclusive). Omit for beginning.
            end_line: Ending line number (1-indexed, inclusive). Omit to read to end.
        """
        if not path:
            return {"error": "path is required"}

        if start_line is not None or end_line is not None:
            result = self._documentation_data_loader.get_file_lines(
                path=path, start_line=start_line, end_line=end_line,
            )
            if result is None:
                return {"error": f"File '{path}' not found"}

            content, total_lines = result
            return {
                "path": path,
                "lines_shown": f"{start_line or 1}-{end_line or total_lines}",
                "total_lines": total_lines,
                "content": content,
            }

        entry = self._documentation_data_loader.get_file_by_path(path)
        if entry is None:
            return {"error": f"File '{path}' not found"}

        content = entry.content
        total_lines = content.count("\n") + 1

        if len(content) > 10000:
            content = content[:10000] + f"\n... (truncated, {len(entry.content)} total chars)"

        return {
            "path": str(entry.path),
            "file_type": entry.file_type,
            "title": entry.title,
            "summary": entry.summary,
            "total_lines": total_lines,
            "content": content,
        }

    @agent_tool(
        availability=lambda self: self._documentation_data_loader is not None,
        parameters={
            "type": "object",
            "properties": {
                "terms": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of search terms (case-insensitive).",
                },
                "top_n": {
                    "type": "integer",
                    "description": "Number of top results to return. Defaults to 20.",
                },
            },
            "required": ["terms"],
        },
    )
    @token_optimized
    def _search_docs_by_terms(self, terms: list[str], top_n: int = 20) -> dict[str, Any]:
        """
        Search documentation files by multiple terms with relevance scoring.

        Ranks files by how many terms match and total hit count. Good for broad topic searches.

        Args:
            terms: List of search terms (case-insensitive).
            top_n: Number of top results to return. Defaults to 20.
        """
        if not terms:
            return {"error": "terms list is required"}

        results = self._documentation_data_loader.search_by_terms(terms=terms, top_n=top_n)
        return {"terms": terms, "results_count": len(results), "results": results}

    @agent_tool(
        availability=lambda self: self._documentation_data_loader is not None,
        parameters={
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regex pattern to search for.",
                },
                "top_n": {
                    "type": "integer",
                    "description": "Max entries to return. Defaults to 20.",
                },
            },
            "required": ["pattern"],
        },
    )
    @token_optimized
    def _search_docs_by_regex(self, pattern: str, top_n: int = 20) -> dict[str, Any]:
        """
        Search documentation files by regex pattern with timeout protection.

        Returns matching snippets with context. Useful for pattern-based searches.

        Args:
            pattern: Regex pattern to search for.
            top_n: Max entries to return. Defaults to 20.
        """
        if not pattern:
            return {"error": "pattern is required"}

        return self._documentation_data_loader.search_by_regex(pattern=pattern, top_n=top_n)

    ## Tool availability prompt section

    def _get_tool_availability_prompt_section(self) -> str:
        """
        Build a system prompt section listing currently available tools.

        Only lists tools whose availability evaluates to True right now.
        Called automatically after _sync_tools() so the tool set is current.
        Injected in _call_llm() so tools are automatically listed in the system prompt.
        """
        if not self._registered_tool_names:
            return ""

        collected = self._collect_tools()
        if not collected:
            return ""

        lines = ["\n\n## Tools"]
        for tool_meta, _ in collected:
            if tool_meta.name in self._registered_tool_names:
                # Truncate long descriptions to first sentence
                short = tool_meta.description.split(". ")[0].split("\n")[0]
                lines.append(f"- `{tool_meta.name}` — {short}")

        return "\n".join(lines)

    ## LLMs and streaming

    def _call_llm(
        self,
        messages: list[dict[str, Any]],
        system_prompt: str,
        tool_choice: str | None = None,
    ) -> LLMChatResponse:
        """
        Call the LLM, using streaming if a chunk callback is configured.

        Args:
            messages: List of message dicts for the LLM.
            system_prompt: System prompt for the LLM.
            tool_choice: Tool choice mode. Can be:
                - None or "auto": Let the model decide (default)
                - "required": Force the model to use at least one tool
                - Tool name string: Force the model to use a specific tool
        """
        # ensure tool availability reflects current state
        self._sync_tools()
    
        # append tool availability (injected here so subclasses can't accidentally omit it)
        system_prompt = system_prompt + self._get_tool_availability_prompt_section()

        # append documentation context (injected here so subclasses can't accidentally omit it)
        docs_section = self._get_documentation_prompt_section()
        if docs_section:
            system_prompt = system_prompt + docs_section

        if self._stream_chunk_callable:
            return self._process_streaming_response(messages, system_prompt, tool_choice)

        return self.llm_client.call_sync(
            messages=messages,
            system_prompt=system_prompt,
            previous_response_id=self._previous_response_id,
            tool_choice=tool_choice,
        )

    def _process_streaming_response(
        self,
        messages: list[dict[str, Any]],
        system_prompt: str,
        tool_choice: str | None = None,
    ) -> LLMChatResponse:
        """Process LLM response with streaming, calling chunk callback for each chunk."""
        response: LLMChatResponse | None = None

        for item in self.llm_client.call_stream_sync(
            messages=messages,
            system_prompt=system_prompt,
            previous_response_id=self._previous_response_id,
            tool_choice=tool_choice,
        ):
            if isinstance(item, str):
                if self._stream_chunk_callable:
                    self._stream_chunk_callable(item)
            elif isinstance(item, LLMChatResponse):
                response = item

        if response is None:
            raise ValueError("No final response received from streaming LLM")

        return response

    ## Chat helpers

    def _emit_message(self, message: EmittedMessage) -> None:
        """Emit a message via the callback."""
        self._emit_message_callable(message)

    def _add_chat(
        self,
        role: ChatRole,
        content: str,
        tool_call_id: str | None = None,
        tool_calls: list[LLMToolCall] | None = None,
        llm_provider_response_id: str | None = None,
    ) -> Chat:
        """Create and store a new Chat, update thread, persist if callbacks set."""
        chat = Chat(
            chat_thread_id=self._thread.id,
            role=role,
            content=content,
            tool_call_id=tool_call_id,
            tool_calls=tool_calls or [],
            llm_provider_response_id=llm_provider_response_id,
        )

        if self._persist_chat_callable:
            chat = self._persist_chat_callable(chat)

        self._chats[chat.id] = chat
        self._thread.chat_ids.append(chat.id)
        self._thread.updated_at = int(datetime.now().timestamp())

        # Track response_id → chat index for O(1) lookup (ASSISTANT messages only)
        if llm_provider_response_id and role == ChatRole.ASSISTANT:
            self._response_id_to_chat_index[llm_provider_response_id] = len(self._thread.chat_ids) - 1

        if self._persist_chat_thread_callable:
            self._thread = self._persist_chat_thread_callable(self._thread)

        return chat

    def _build_messages_for_llm(self) -> list[dict[str, Any]]:
        """Build messages list for LLM from Chat objects."""
        messages: list[dict[str, Any]] = []

        # Only include chats after the last response_id (for response chaining)
        chats_to_include = self._thread.chat_ids
        if self._previous_response_id is not None:
            index = self._response_id_to_chat_index.get(self._previous_response_id)
            if index is not None:
                chats_to_include = self._thread.chat_ids[index + 1:]

        for chat_id in chats_to_include:
            chat = self._chats.get(chat_id)
            if not chat:
                continue
            msg: dict[str, Any] = {
                "role": chat.role.value,
                "content": chat.content,
            }
            if chat.tool_call_id:
                msg["tool_call_id"] = chat.tool_call_id
            if chat.tool_calls:
                msg["tool_calls"] = [
                    {
                        "call_id": tc.call_id if tc.call_id else f"call_{idx}_{chat_id[:8]}",
                        "name": tc.tool_name,
                        "arguments": tc.tool_arguments,
                    }
                    for idx, tc in enumerate(chat.tool_calls)
                ]
            messages.append(msg)
        return messages

    ## Tool execution helpers

    def _auto_execute_tool(self, tool_name: str, tool_arguments: dict[str, Any]) -> str:
        """Auto-execute a tool, emit result message, return JSON string."""
        invocation = PendingToolInvocation(
            invocation_id="",
            tool_name=tool_name,
            tool_arguments=tool_arguments,
            status=ToolInvocationStatus.CONFIRMED,
        )
        try:
            result = self._execute_tool(tool_name, tool_arguments)
            invocation.status = ToolInvocationStatus.EXECUTED

            self._emit_message(
                ToolInvocationResultEmittedMessage(
                    tool_invocation=invocation,
                    tool_result=result,
                )
            )

            logger.debug("Auto-executed tool %s successfully", tool_name)
            return json.dumps(result)

        except Exception as e:
            invocation.status = ToolInvocationStatus.FAILED

            self._emit_message(
                ToolInvocationResultEmittedMessage(
                    tool_invocation=invocation,
                    tool_result={"error": str(e)},
                )
            )

            logger.error("Auto-executed tool %s failed: %s", tool_name, e)
            return json.dumps({"error": str(e)})

    def _process_tool_calls(self, tool_calls: list[LLMToolCall]) -> None:
        """Execute a list of tool calls in parallel and add results to chat history."""
        if len(tool_calls) == 1:
            # Single tool call - no need for threading overhead
            tool_call = tool_calls[0]
            logger.debug("Executing tool %s", tool_call.tool_name)
            result_str = self._auto_execute_tool(tool_call.tool_name, tool_call.tool_arguments)
            self._add_chat(
                ChatRole.TOOL,
                f"Tool '{tool_call.tool_name}' result: {result_str}",
                tool_call_id=tool_call.call_id,
            )
            return

        # multiple tool calls, execute in parallel
        logger.debug("Executing %d tool calls in parallel", len(tool_calls))

        # warn if any call_ids are None or duplicated (would break result reordering)
        call_ids = [tc.call_id for tc in tool_calls]
        if None in call_ids:
            logger.warning("One or more tool calls have call_id=None — result ordering may be incorrect")
        if len(set(call_ids)) != len(call_ids):
            logger.warning("Duplicate call_ids detected among tool calls — result ordering may be incorrect")

        def execute_one(tc: LLMToolCall) -> tuple[LLMToolCall, str]:
            result_str = self._auto_execute_tool(tc.tool_name, tc.tool_arguments)
            return tc, result_str

        results: list[tuple[LLMToolCall, str]] = []
        with ThreadPoolExecutor(max_workers=len(tool_calls)) as executor:
            futures: dict[Future[tuple[LLMToolCall, str]], LLMToolCall] = {
                executor.submit(execute_one, tc): tc for tc in tool_calls
            }
            for future in as_completed(futures):
                results.append(future.result())

        # add results to chat history in original order
        call_id_to_result = {tc.call_id: result_str for tc, result_str in results}
        for tool_call in tool_calls:
            result_str = call_id_to_result[tool_call.call_id]
            self._add_chat(
                ChatRole.TOOL,
                f"Tool '{tool_call.tool_name}' result: {result_str}",
                tool_call_id=tool_call.call_id,
            )

    ## agent loop (basic implementation, can be overridden)

    def _run_agent_loop(self) -> None:
        """Run a basic agent loop: call LLM → execute tools → repeat."""
        max_iterations = self.AGENT_LOOP_MAX_ITERATIONS
        for iteration in range(max_iterations):
            logger.debug("Agent loop iteration %d", iteration + 1)

            messages = self._build_messages_for_llm()
            try:
                response = self._call_llm(messages, self._get_system_prompt())

                if response.response_id:
                    self._previous_response_id = response.response_id

                if response.content or response.tool_calls:
                    chat = self._add_chat(
                        ChatRole.ASSISTANT,
                        response.content or "",
                        tool_calls=response.tool_calls if response.tool_calls else None,
                        llm_provider_response_id=response.response_id,
                    )
                    if response.content:
                        self._emit_message(
                            ChatResponseEmittedMessage(
                                content=response.content,
                                chat_id=chat.id,
                                chat_thread_id=self._thread.id,
                            )
                        )

                if not response.tool_calls:
                    logger.debug("Agent loop complete - no more tool calls")
                    return

                self._process_tool_calls(response.tool_calls)

            except Exception as e:
                logger.exception("Error in agent loop: %s", e)
                self._emit_message(ErrorEmittedMessage(error=str(e)))
                return

        logger.warning("Agent loop hit max iterations (%d)", max_iterations)

    def process_new_message(self, content: str, role: ChatRole = ChatRole.USER) -> None:
        """
        Process a new message and emit responses via callback.

        Args:
            content: The message content.
            role: The role of the message sender.
        """
        self._add_chat(role, content)
        self._run_agent_loop()
