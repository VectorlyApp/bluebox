"""
bluebox/agents/docs_digger_agent.py

# NOTE: THIS AGENT IS IN BETA AND NOT READY FOR PRODUCTION YET

Agent specialized in searching through documentation and code files.

Contains:
- DocsDiggerAgent: Conversational interface for documentation/code analysis
- DocumentSearchResult: Result model for autonomous documentation discovery
- Uses: LLMClient with tools for documentation searching
- Maintains: ChatThread for multi-turn conversation
"""

import json
import textwrap
from datetime import datetime
from typing import Any, Callable

from pydantic import BaseModel, Field

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
from bluebox.data_models.llms.vendors import OpenAIModel
from bluebox.llms.llm_client import LLMClient
from bluebox.llms.infra.documentation_data_store import (
    DocumentationDataStore,
    FileType,
)
from bluebox.utils.llm_utils import token_optimized
from bluebox.utils.logger import get_logger


logger = get_logger(name=__name__)


class DiscoveredDocument(BaseModel):
    """A single discovered documentation or code file."""

    path: str = Field(
        description="Path to the file"
    )
    file_type: str = Field(
        description="Type of file: 'documentation' or 'code'"
    )
    relevance_reason: str = Field(
        description="Brief explanation of why this file is relevant to the task"
    )
    key_content: str = Field(
        description="Key content or sections found in this file relevant to the task"
    )


class DocumentSearchResult(BaseModel):
    """
    Result of autonomous documentation search.

    Contains one or more discovered documents relevant to the user's query.
    """

    documents: list[DiscoveredDocument] = Field(
        description="List of discovered documents relevant to the query"
    )
    summary: str = Field(
        description="Brief summary answering the user's question based on the found documents"
    )


class DocumentSearchFailureResult(BaseModel):
    """
    Result when autonomous documentation search fails.

    Returned when the agent cannot find relevant documentation after exhaustive search.
    """

    reason: str = Field(
        description="Explanation of why relevant documentation could not be found"
    )
    searched_terms: list[str] = Field(
        default_factory=list,
        description="List of search terms that were tried"
    )
    closest_matches: list[str] = Field(
        default_factory=list,
        description="Paths of files that came closest to matching (if any)"
    )


class DocsDiggerAgent:
    """
    Documentation digger agent that helps analyze documentation and code files.

    The agent maintains a ChatThread with Chat messages and uses LLM with tools
    to search and analyze documentation and code.

    Usage:
        def handle_message(message: EmittedMessage) -> None:
            print(f"[{message.type}] {message.content}")

        docs_store = DocumentationDataStore(
            documentation_paths=["docs/**/*.md"],
            code_paths=["src/**/*.py"],
        )
        agent = DocsDiggerAgent(
            emit_message_callable=handle_message,
            documentation_data_store=docs_store,
        )
        agent.process_new_message("How do I configure the SDK?", ChatRole.USER)
    """

    SYSTEM_PROMPT: str = textwrap.dedent("""
        You are a documentation and code analyst specializing in helping users find information.

        ## Your Role

        You help users find and understand documentation and code in their project. Your main job is to:
        - Find documentation files that answer the user's questions
        - Locate relevant code files and understand their structure
        - Explain how different parts of the codebase work together

        ## Finding Relevant Information

        When the user asks about specific topics:

        1. Generate 15-25 relevant search terms that might appear in documentation or code
           - Include variations: singular/plural, different casings, synonyms
           - Include technical terms: function names, class names, config keys
           - Include domain-specific terms related to the project

        2. Use the `search_docs_by_terms` tool with your terms

        3. Analyze the top results - examine files with highest scores first

        4. Use `get_file_content` to read promising files and extract relevant information

        ## Available Tools

        - **`search_docs_by_terms`**: Search documentation and code by a list of terms.
          - Pass 15-25 search terms for best results
          - Returns top files ranked by relevance
          - Can filter by file_type: "documentation" or "code"

        - **`search_content`**: Search for exact content in files.
          - Useful for finding specific strings, function names, or error messages
          - Returns matches with surrounding context

        - **`get_file_content`**: Get the full content of a file by path.
          - Use after finding a relevant file to read its contents
          - Supports partial path matching

        - **`get_file_index`**: Get a list of all indexed files.
          - Shows paths, titles, and summaries
          - Useful for understanding project structure

        - **`search_functions`**: Search for function definitions in code.
          - Can filter by function name pattern
          - Returns function signatures and locations

        - **`search_classes`**: Search for class definitions in code.
          - Can filter by class name pattern
          - Returns class names, bases, and locations

        - **`search_by_pattern`**: Search files by path pattern (glob-style).
          - Useful for finding files in specific directories
          - Example: "**/test_*.py" for test files

        ## Guidelines

        - Be concise and direct in your responses
        - When you find relevant documentation, summarize the key points
        - Quote relevant code snippets when helpful
        - Always use search_docs_by_terms first when looking for information
        - Verify information by reading the actual file content
    """).strip()

    AUTONOMOUS_SYSTEM_PROMPT: str = textwrap.dedent("""
        You are a documentation analyst that autonomously finds information in documentation and code.

        ## Your Mission

        Given a user query, find the documentation and/or code files that best answer their question.
        Provide a comprehensive summary based on the found documents.

        ## Process

        1. **Search**: Use `search_docs_by_terms` with 15-25 relevant terms
        2. **Analyze**: Look at top results, examine their content with `get_file_content`
        3. **Verify**: Read the actual files to confirm they contain relevant information
        4. **Finalize**: Once confident, call `finalize_result` with your findings

        ## Strategy

        - Search for both documentation (.md) and code (.py) files
        - Look at file titles and summaries in search results
        - Read promising files to extract key information
        - Consider multiple related files for comprehensive answers

        ## When finalize tools are available

        After sufficient exploration, the `finalize_result` and `finalize_failure` tools become available.

        ### finalize_result - Use when relevant documentation IS found
        Call it with:
        - documents: List of relevant files with paths, types, reasons, and key content
        - summary: A comprehensive answer to the user's question based on found documents

        ### finalize_failure - Use when documentation is NOT found
        If after exhaustive search you determine the information does NOT exist in the indexed files:
        - Call `finalize_failure` with a clear reason explaining what was searched
        - Include the search terms you tried and any files that came close
        - Only use this after thoroughly searching - don't give up too early!
    """).strip()

    def __init__(
        self,
        emit_message_callable: Callable[[EmittedMessage], None],
        documentation_data_store: DocumentationDataStore,
        persist_chat_callable: Callable[[Chat], Chat] | None = None,
        persist_chat_thread_callable: Callable[[ChatThread], ChatThread] | None = None,
        stream_chunk_callable: Callable[[str], None] | None = None,
        llm_model: OpenAIModel = OpenAIModel.GPT_5_1,
        chat_thread: ChatThread | None = None,
        existing_chats: list[Chat] | None = None,
    ) -> None:
        """
        Initialize the documentation digger agent.

        Args:
            emit_message_callable: Callback function to emit messages to the host.
            documentation_data_store: The DocumentationDataStore containing indexed files.
            persist_chat_callable: Optional callback to persist Chat objects.
            persist_chat_thread_callable: Optional callback to persist ChatThread.
            stream_chunk_callable: Optional callback for streaming text chunks.
            llm_model: The LLM model to use for conversation.
            chat_thread: Existing ChatThread to continue, or None for new conversation.
            existing_chats: Existing Chat messages if loading from persistence.
        """
        self._emit_message_callable = emit_message_callable
        self._persist_chat_callable = persist_chat_callable
        self._persist_chat_thread_callable = persist_chat_thread_callable
        self._stream_chunk_callable = stream_chunk_callable
        self._documentation_data_store = documentation_data_store
        self._previous_response_id: str | None = None
        self._response_id_to_chat_index: dict[str, int] = {}

        self.llm_model = llm_model
        self.llm_client = LLMClient(llm_model)

        # Register tools
        self._register_tools()

        # Initialize or load conversation state
        self._thread = chat_thread or ChatThread()
        self._chats: dict[str, Chat] = {}
        if existing_chats:
            for chat in existing_chats:
                self._chats[chat.id] = chat

        # Persist initial thread if callback provided
        if self._persist_chat_thread_callable and chat_thread is None:
            self._thread = self._persist_chat_thread_callable(self._thread)

        # Autonomous mode state
        self._autonomous_mode: bool = False
        self._autonomous_iteration: int = 0
        self._autonomous_max_iterations: int = 10
        self._search_result: DocumentSearchResult | None = None
        self._search_failure: DocumentSearchFailureResult | None = None
        self._finalize_tool_registered: bool = False

        logger.debug(
            "Instantiated DocsDiggerAgent with model: %s, chat_thread_id: %s, files: %d",
            llm_model,
            self._thread.id,
            len(documentation_data_store.entries),
        )

    def _register_tools(self) -> None:
        """Register tools for documentation analysis."""
        # search_docs_by_terms
        self.llm_client.register_tool(
            name="search_docs_by_terms",
            description=(
                "Search documentation and code files by a list of terms. "
                "Returns top files ranked by relevance score. "
                "Pass 15-25 search terms for best results."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "terms": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "List of 15-25 search terms to look for in file contents. "
                            "Include variations, related terms, and technical names."
                        ),
                    },
                    "file_type": {
                        "type": "string",
                        "enum": ["documentation", "code"],
                        "description": (
                            "Optional filter to search only documentation or code files."
                        ),
                    },
                },
                "required": ["terms"],
            },
        )

        # search_content
        self.llm_client.register_tool(
            name="search_content",
            description=(
                "Search file contents for an exact query string. "
                "Returns matches with surrounding context. "
                "Useful for finding specific strings, function names, or error messages."
            ),
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

        # get_file_content
        self.llm_client.register_tool(
            name="get_file_content",
            description=(
                "Get the full content of a file by its path. "
                "Supports exact path or partial path matching."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The file path (can be partial, will match).",
                    },
                },
                "required": ["path"],
            },
        )

        # get_file_index
        self.llm_client.register_tool(
            name="get_file_index",
            description=(
                "Get a list of all indexed files with their metadata. "
                "Shows paths, file types, titles, and summaries."
            ),
            parameters={
                "type": "object",
                "properties": {},
            },
        )

        # search_functions
        self.llm_client.register_tool(
            name="search_functions",
            description=(
                "Search for function/method definitions in code files. "
                "Returns function signatures and line numbers."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "name_pattern": {
                        "type": "string",
                        "description": (
                            "Regex pattern to match function names. "
                            "Example: 'get_.*' matches functions starting with 'get_'."
                        ),
                    },
                    "file_pattern": {
                        "type": "string",
                        "description": (
                            "Glob pattern to filter files. "
                            "Example: '**/test_*.py' for test files."
                        ),
                    },
                },
            },
        )

        # search_classes
        self.llm_client.register_tool(
            name="search_classes",
            description=(
                "Search for class definitions in code files. "
                "Returns class names, base classes, and line numbers."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "name_pattern": {
                        "type": "string",
                        "description": (
                            "Regex pattern to match class names. "
                            "Example: '.*Agent' matches classes ending with 'Agent'."
                        ),
                    },
                    "file_pattern": {
                        "type": "string",
                        "description": (
                            "Glob pattern to filter files. "
                            "Example: '**/models/*.py' for model files."
                        ),
                    },
                },
            },
        )

        # search_by_pattern
        self.llm_client.register_tool(
            name="search_by_pattern",
            description=(
                "Search files by path pattern (glob-style). "
                "Useful for finding files in specific directories or with specific names."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": (
                            "Glob pattern to match paths. "
                            "Examples: '**/test_*.py', 'docs/**/*.md', '**/README*'."
                        ),
                    },
                    "file_type": {
                        "type": "string",
                        "enum": ["documentation", "code"],
                        "description": "Optional filter by file type.",
                    },
                },
                "required": ["pattern"],
            },
        )

    def _register_finalize_tool(self) -> None:
        """Register the finalize_result tool for autonomous mode (available after iteration 2)."""
        if self._finalize_tool_registered:
            return

        self.llm_client.register_tool(
            name="finalize_result",
            description=(
                "Finalize the documentation search with your findings. "
                "Call this when you have found documentation that answers the user's question. "
                "Provide a list of relevant documents and a summary."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "documents": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "path": {
                                    "type": "string",
                                    "description": "Path to the file.",
                                },
                                "file_type": {
                                    "type": "string",
                                    "enum": ["documentation", "code"],
                                    "description": "Type of file.",
                                },
                                "relevance_reason": {
                                    "type": "string",
                                    "description": "Why this file is relevant.",
                                },
                                "key_content": {
                                    "type": "string",
                                    "description": "Key content from this file.",
                                },
                            },
                            "required": ["path", "file_type", "relevance_reason", "key_content"],
                        },
                        "description": "List of relevant documents found.",
                    },
                    "summary": {
                        "type": "string",
                        "description": "Comprehensive answer based on found documents.",
                    },
                },
                "required": ["documents", "summary"],
            },
        )

        # Also register the failure tool
        self.llm_client.register_tool(
            name="finalize_failure",
            description=(
                "Signal that the documentation search has failed. "
                "Call this ONLY when you have exhaustively searched and are confident "
                "that the information does NOT exist in the indexed files."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Explanation of why the information could not be found.",
                    },
                    "searched_terms": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of key search terms that were tried.",
                    },
                    "closest_matches": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Paths of files that came closest to matching (if any).",
                    },
                },
                "required": ["reason"],
            },
        )

        self._finalize_tool_registered = True
        logger.debug("Registered finalize_result and finalize_failure tools")

    @property
    def chat_thread_id(self) -> str:
        """Return the current thread ID."""
        return self._thread.id

    @property
    def autonomous_iteration(self) -> int:
        """Return the current/final autonomous iteration count."""
        return self._autonomous_iteration

    def _get_system_prompt(self) -> str:
        """Get system prompt with documentation stats context."""
        stats = self._documentation_data_store.stats
        stats_context = (
            f"\n\n## Documentation Context\n"
            f"- Total Files: {stats.total_files}\n"
            f"- Documentation Files: {stats.total_docs}\n"
            f"- Code Files: {stats.total_code}\n"
            f"- Total Size: {stats._format_bytes(stats.total_bytes)}\n"
        )

        # Add extension breakdown
        if stats.extensions:
            ext_lines = []
            for ext, count in sorted(stats.extensions.items(), key=lambda x: -x[1])[:10]:
                ext_lines.append(f"  - {ext}: {count} files")
            stats_context += "\n- Extensions:\n" + "\n".join(ext_lines)

        # Add documentation index (titles and summaries)
        doc_index = self._documentation_data_store.get_documentation_index()
        if doc_index:
            doc_lines = []
            for doc in doc_index[:20]:  # Limit to 20
                if doc["title"]:
                    doc_lines.append(f"- `{doc['filename']}`: {doc['title']}")
                else:
                    doc_lines.append(f"- `{doc['filename']}`")
            doc_context = (
                f"\n\n## Documentation Files Overview\n"
                + "\n".join(doc_lines)
            )
            if len(doc_index) > 20:
                doc_context += f"\n... and {len(doc_index) - 20} more documentation files"
        else:
            doc_context = ""

        return self.SYSTEM_PROMPT + stats_context + doc_context

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
        """
        Create and store a new Chat, update thread, persist if callbacks set.
        """
        chat = Chat(
            chat_thread_id=self._thread.id,
            role=role,
            content=content,
            tool_call_id=tool_call_id,
            tool_calls=tool_calls or [],
            llm_provider_response_id=llm_provider_response_id,
        )

        # Persist chat first if callback provided (may assign new ID)
        if self._persist_chat_callable:
            chat = self._persist_chat_callable(chat)

        # Store with final ID
        self._chats[chat.id] = chat
        self._thread.chat_ids.append(chat.id)
        self._thread.updated_at = int(datetime.now().timestamp())

        # Track response_id to chat index for O(1) lookup (only for ASSISTANT messages)
        if llm_provider_response_id and role == ChatRole.ASSISTANT:
            self._response_id_to_chat_index[llm_provider_response_id] = len(self._thread.chat_ids) - 1

        # Persist thread if callback provided
        if self._persist_chat_thread_callable:
            self._thread = self._persist_chat_thread_callable(self._thread)

        return chat

    def _build_messages_for_llm(self) -> list[dict[str, Any]]:
        """Build messages list for LLM from Chat objects."""
        messages: list[dict[str, Any]] = []

        # Determine which chats to include based on the previous response id
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
            # Include tool_call_id for TOOL role messages
            if chat.tool_call_id:
                msg["tool_call_id"] = chat.tool_call_id
            # Include tool_calls for ASSISTANT role messages
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

    @token_optimized
    def _tool_search_docs_by_terms(self, tool_arguments: dict[str, Any]) -> dict[str, Any]:
        """Execute search_docs_by_terms tool."""
        terms = tool_arguments.get("terms", [])
        if not terms:
            return {"error": "No search terms provided"}

        file_type_str = tool_arguments.get("file_type")
        file_type = FileType(file_type_str) if file_type_str else None

        results = self._documentation_data_store.search_by_terms(
            terms=terms,
            file_type=file_type,
            top_n=20,
        )

        if not results:
            return {
                "message": "No matching files found",
                "terms_searched": len(terms),
            }

        return {
            "terms_searched": len(terms),
            "results_found": len(results),
            "results": results,
        }

    @token_optimized
    def _tool_search_content(self, tool_arguments: dict[str, Any]) -> dict[str, Any]:
        """Execute search_content tool."""
        query = tool_arguments.get("query", "")
        if not query:
            return {"error": "query is required"}

        file_type_str = tool_arguments.get("file_type")
        file_type = FileType(file_type_str) if file_type_str else None
        case_sensitive = tool_arguments.get("case_sensitive", False)

        results = self._documentation_data_store.search_content(
            query=query,
            file_type=file_type,
            case_sensitive=case_sensitive,
        )

        if not results:
            return {
                "message": f"No matches found for '{query}'",
                "case_sensitive": case_sensitive,
            }

        return {
            "query": query,
            "case_sensitive": case_sensitive,
            "results_found": len(results),
            "results": results[:20],  # Top 20
        }

    @token_optimized
    def _tool_get_file_content(self, tool_arguments: dict[str, Any]) -> dict[str, Any]:
        """Execute get_file_content tool."""
        path = tool_arguments.get("path", "")
        if not path:
            return {"error": "path is required"}

        entry = self._documentation_data_store.get_file_by_path(path)
        if entry is None:
            return {"error": f"File '{path}' not found"}

        # Truncate large content
        content = entry.content
        if len(content) > 10000:
            content = content[:10000] + f"\n... (truncated, {len(entry.content)} total chars)"

        return {
            "path": str(entry.path),
            "file_type": entry.file_type,
            "title": entry.title,
            "summary": entry.summary,
            "size_bytes": entry.size_bytes,
            "content": content,
        }

    @token_optimized
    def _tool_get_file_index(self, tool_arguments: dict[str, Any]) -> dict[str, Any]:
        """Execute get_file_index tool."""
        index = self._documentation_data_store.get_file_index()
        return {
            "total_files": len(index),
            "files": index,
        }

    @token_optimized
    def _tool_search_functions(self, tool_arguments: dict[str, Any]) -> dict[str, Any]:
        """Execute search_functions tool."""
        name_pattern = tool_arguments.get("name_pattern")
        file_pattern = tool_arguments.get("file_pattern")

        results = self._documentation_data_store.search_functions(
            name_pattern=name_pattern,
            file_pattern=file_pattern,
        )

        if not results:
            return {"message": "No matching functions found"}

        return {
            "results_found": len(results),
            "results": results[:50],  # Top 50
        }

    @token_optimized
    def _tool_search_classes(self, tool_arguments: dict[str, Any]) -> dict[str, Any]:
        """Execute search_classes tool."""
        name_pattern = tool_arguments.get("name_pattern")
        file_pattern = tool_arguments.get("file_pattern")

        results = self._documentation_data_store.search_classes(
            name_pattern=name_pattern,
            file_pattern=file_pattern,
        )

        if not results:
            return {"message": "No matching classes found"}

        return {
            "results_found": len(results),
            "results": results[:50],  # Top 50
        }

    @token_optimized
    def _tool_search_by_pattern(self, tool_arguments: dict[str, Any]) -> dict[str, Any]:
        """Execute search_by_pattern tool."""
        pattern = tool_arguments.get("pattern", "")
        if not pattern:
            return {"error": "pattern is required"}

        file_type_str = tool_arguments.get("file_type")
        file_type = FileType(file_type_str) if file_type_str else None

        entries = self._documentation_data_store.search_by_pattern(
            pattern=pattern,
            file_type=file_type,
        )

        if not entries:
            return {"message": f"No files matching pattern '{pattern}'"}

        results = [
            {
                "path": str(e.path),
                "file_type": e.file_type,
                "title": e.title,
                "summary": e.summary,
            }
            for e in entries[:30]  # Top 30
        ]

        return {
            "pattern": pattern,
            "results_found": len(entries),
            "results": results,
        }

    @token_optimized
    def _tool_finalize_result(self, tool_arguments: dict[str, Any]) -> dict[str, Any]:
        """Handle finalize_result tool call in autonomous mode."""
        documents_data = tool_arguments.get("documents", [])
        summary = tool_arguments.get("summary", "")

        if not documents_data:
            return {"error": "documents list is required and cannot be empty"}
        if not summary:
            return {"error": "summary is required"}

        # Build document objects
        discovered_docs: list[DiscoveredDocument] = []
        for i, doc in enumerate(documents_data):
            path = doc.get("path", "")
            file_type = doc.get("file_type", "")
            relevance_reason = doc.get("relevance_reason", "")
            key_content = doc.get("key_content", "")

            if not path:
                return {"error": f"documents[{i}].path is required"}
            if not file_type:
                return {"error": f"documents[{i}].file_type is required"}
            if not relevance_reason:
                return {"error": f"documents[{i}].relevance_reason is required"}
            if not key_content:
                return {"error": f"documents[{i}].key_content is required"}

            # Validate that the file exists in the data store
            entry = self._documentation_data_store.get_file_by_path(path)
            if entry is None:
                return {
                    "error": f"documents[{i}].path '{path}' not found in data store",
                    "hint": "Use get_file_index to see available files.",
                }

            discovered_docs.append(DiscoveredDocument(
                path=path,
                file_type=file_type,
                relevance_reason=relevance_reason,
                key_content=key_content,
            ))

        # Store the result
        self._search_result = DocumentSearchResult(
            documents=discovered_docs,
            summary=summary,
        )

        logger.info(
            "Finalized documentation search: %d document(s) found",
            len(discovered_docs),
        )

        return {
            "status": "success",
            "message": f"Documentation search finalized with {len(discovered_docs)} document(s)",
            "result": self._search_result.model_dump(),
        }

    @token_optimized
    def _tool_finalize_failure(self, tool_arguments: dict[str, Any]) -> dict[str, Any]:
        """Handle finalize_failure tool call when documentation search fails."""
        reason = tool_arguments.get("reason", "")
        searched_terms = tool_arguments.get("searched_terms", [])
        closest_matches = tool_arguments.get("closest_matches", [])

        if not reason:
            return {"error": "reason is required"}

        # Store the failure result
        self._search_failure = DocumentSearchFailureResult(
            reason=reason,
            searched_terms=searched_terms,
            closest_matches=closest_matches,
        )

        logger.info("Documentation search failed: %s", reason)

        return {
            "status": "failure",
            "message": "Documentation search marked as failed",
            "result": self._search_failure.model_dump(),
        }

    def _execute_tool(self, tool_name: str, tool_arguments: dict[str, Any]) -> dict[str, Any]:
        """Execute a tool and return the result."""
        logger.debug("Executing tool %s with arguments: %s", tool_name, tool_arguments)

        if tool_name == "search_docs_by_terms":
            return self._tool_search_docs_by_terms(tool_arguments)

        if tool_name == "search_content":
            return self._tool_search_content(tool_arguments)

        if tool_name == "get_file_content":
            return self._tool_get_file_content(tool_arguments)

        if tool_name == "get_file_index":
            return self._tool_get_file_index(tool_arguments)

        if tool_name == "search_functions":
            return self._tool_search_functions(tool_arguments)

        if tool_name == "search_classes":
            return self._tool_search_classes(tool_arguments)

        if tool_name == "search_by_pattern":
            return self._tool_search_by_pattern(tool_arguments)

        if tool_name == "finalize_result":
            return self._tool_finalize_result(tool_arguments)

        if tool_name == "finalize_failure":
            return self._tool_finalize_failure(tool_arguments)

        return {"error": f"Unknown tool: {tool_name}"}

    def _auto_execute_tool(self, tool_name: str, tool_arguments: dict[str, Any]) -> str:
        """Auto-execute a tool and emit the result."""
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

    def process_new_message(self, content: str, role: ChatRole = ChatRole.USER) -> None:
        """
        Process a new message and emit responses via callback.

        Args:
            content: The message content
            role: The role of the message sender (USER or SYSTEM)
        """
        # Add message to history
        self._add_chat(role, content)

        # Run the agent loop
        self._run_agent_loop()

    def _run_agent_loop(self) -> None:
        """Run the agent loop: call LLM, execute tools, feed results back, repeat."""
        max_iterations = 10

        for iteration in range(max_iterations):
            logger.debug("Agent loop iteration %d", iteration + 1)

            messages = self._build_messages_for_llm()

            try:
                # Use streaming if chunk callback is set
                if self._stream_chunk_callable:
                    response = self._process_streaming_response(messages)
                else:
                    response = self.llm_client.call_sync(
                        messages=messages,
                        system_prompt=self._get_system_prompt(),
                        previous_response_id=self._previous_response_id,
                    )

                # Update previous_response_id for response chaining
                if response.response_id:
                    self._previous_response_id = response.response_id

                # Handle response - add assistant message if there's content or tool calls
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

                # If no tool calls, we're done
                if not response.tool_calls:
                    logger.debug("Agent loop complete - no more tool calls")
                    return

                # Process tool calls
                for tool_call in response.tool_calls:
                    tool_name = tool_call.tool_name
                    tool_arguments = tool_call.tool_arguments
                    call_id = tool_call.call_id

                    # Auto-execute tool
                    logger.debug("Auto-executing tool %s", tool_name)
                    result_str = self._auto_execute_tool(tool_name, tool_arguments)

                    # Add tool result to conversation history
                    self._add_chat(
                        ChatRole.TOOL,
                        f"Tool '{tool_name}' result: {result_str}",
                        tool_call_id=call_id,
                    )

            except Exception as e:
                logger.exception("Error in agent loop: %s", e)
                self._emit_message(
                    ErrorEmittedMessage(
                        error=str(e),
                    )
                )
                return

        logger.warning("Agent loop hit max iterations (%d)", max_iterations)

    def _process_streaming_response(self, messages: list[dict[str, str]]) -> LLMChatResponse:
        """Process LLM response with streaming, calling chunk callback for each chunk."""
        response: LLMChatResponse | None = None

        for item in self.llm_client.call_stream_sync(
            messages=messages,
            system_prompt=self._get_system_prompt(),
            previous_response_id=self._previous_response_id,
        ):
            if isinstance(item, str):
                # Text chunk - call the callback
                if self._stream_chunk_callable:
                    self._stream_chunk_callable(item)
            elif isinstance(item, LLMChatResponse):
                # Final response
                response = item

        if response is None:
            raise ValueError("No final response received from streaming LLM")

        return response

    def get_thread(self) -> ChatThread:
        """Get the current conversation thread."""
        return self._thread

    def get_chats(self) -> list[Chat]:
        """Get all Chat messages in order."""
        return [self._chats[chat_id] for chat_id in self._thread.chat_ids if chat_id in self._chats]

    def run_autonomous(
        self,
        query: str,
        min_iterations: int = 3,
        max_iterations: int = 10,
    ) -> DocumentSearchResult | DocumentSearchFailureResult | None:
        """
        Run the agent autonomously to find documentation for a query.

        The agent will:
        1. Search through documentation and code files
        2. Analyze and verify the content
        3. After iteration 2, the finalize tools become available
        4. Return when finalize_result/finalize_failure is called or max_iterations reached

        Args:
            query: User query (e.g., "How do I configure authentication?")
            min_iterations: Minimum iterations before allowing finalize (default 3)
            max_iterations: Maximum iterations before stopping (default 10)

        Returns:
            DocumentSearchResult if documentation was found,
            DocumentSearchFailureResult if agent determined info doesn't exist,
            None if max iterations reached without finalization.

        Example:
            result = agent.run_autonomous(
                query="How do I set up the SDK?"
            )
            if isinstance(result, DocumentSearchResult):
                print(f"Found {len(result.documents)} document(s)")
                print(result.summary)
            elif isinstance(result, DocumentSearchFailureResult):
                print(f"Failed: {result.reason}")
            else:
                print("Reached max iterations without conclusion")
        """
        # Enable autonomous mode
        self._autonomous_mode = True
        self._autonomous_iteration = 0
        self._autonomous_max_iterations = max_iterations
        self._search_result = None
        self._search_failure = None
        self._finalize_tool_registered = False

        # Add the query as initial message
        initial_message = (
            f"QUERY: {query}\n\n"
            "Find documentation and/or code that answers this question. "
            "Search, analyze, and when confident, use finalize_result to report your findings. "
            "If after thorough search you determine the information does not exist, "
            "use finalize_failure to report why."
        )
        self._add_chat(ChatRole.USER, initial_message)

        logger.info("Starting autonomous documentation search for: %s", query)

        # Run the autonomous agent loop
        self._run_autonomous_loop(min_iterations, max_iterations)

        # Reset autonomous mode
        self._autonomous_mode = False

        # Return result or failure
        if self._search_result is not None:
            return self._search_result
        if self._search_failure is not None:
            return self._search_failure
        return None

    def _run_autonomous_loop(self, min_iterations: int, max_iterations: int) -> None:
        """
        Run the autonomous agent loop with iteration tracking and finalize tool gating.

        Args:
            min_iterations: Minimum iterations before finalize_result is available
            max_iterations: Maximum iterations before stopping
        """
        for iteration in range(max_iterations):
            self._autonomous_iteration = iteration + 1
            logger.debug("Autonomous loop iteration %d/%d", self._autonomous_iteration, max_iterations)

            # After min_iterations-1, register the finalize_result tool
            if self._autonomous_iteration >= min_iterations - 1 and not self._finalize_tool_registered:
                self._register_finalize_tool()
                logger.info(
                    "finalize_result tool now available (iteration %d)",
                    self._autonomous_iteration,
                )

            messages = self._build_messages_for_llm()

            try:
                # Use streaming if chunk callback is set
                if self._stream_chunk_callable:
                    response = self._process_streaming_response_autonomous(messages)
                else:
                    response = self.llm_client.call_sync(
                        messages=messages,
                        system_prompt=self._get_autonomous_system_prompt(),
                        previous_response_id=self._previous_response_id,
                    )

                # Update previous_response_id for response chaining
                if response.response_id:
                    self._previous_response_id = response.response_id

                # Handle response
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

                # If no tool calls, we're done
                if not response.tool_calls:
                    logger.warning("Autonomous loop: no tool calls in iteration %d", self._autonomous_iteration)
                    return

                # Process tool calls
                for tool_call in response.tool_calls:
                    tool_name = tool_call.tool_name
                    tool_arguments = tool_call.tool_arguments
                    call_id = tool_call.call_id

                    # Auto-execute tool
                    logger.debug("Auto-executing tool %s", tool_name)
                    result_str = self._auto_execute_tool(tool_name, tool_arguments)

                    # Add tool result to conversation history
                    self._add_chat(
                        ChatRole.TOOL,
                        f"Tool '{tool_name}' result: {result_str}",
                        tool_call_id=call_id,
                    )

                    # Check if finalize_result was called successfully
                    if tool_name == "finalize_result" and self._search_result is not None:
                        logger.info(
                            "Autonomous documentation search completed at iteration %d",
                            self._autonomous_iteration,
                        )
                        return

                    # Check if finalize_failure was called
                    if tool_name == "finalize_failure" and self._search_failure is not None:
                        logger.info(
                            "Autonomous documentation search failed at iteration %d: %s",
                            self._autonomous_iteration,
                            self._search_failure.reason,
                        )
                        return

            except Exception as e:
                logger.exception("Error in autonomous loop: %s", e)
                self._emit_message(
                    ErrorEmittedMessage(
                        error=str(e),
                    )
                )
                return

        logger.warning(
            "Autonomous loop hit max iterations (%d) without finalize_result",
            max_iterations,
        )

    def _get_autonomous_system_prompt(self) -> str:
        """Get system prompt for autonomous mode with documentation context."""
        stats = self._documentation_data_store.stats
        stats_context = (
            f"\n\n## Documentation Context\n"
            f"- Total Files: {stats.total_files}\n"
            f"- Documentation Files: {stats.total_docs}\n"
            f"- Code Files: {stats.total_code}\n"
        )

        # Add documentation index
        doc_index = self._documentation_data_store.get_documentation_index()
        if doc_index:
            doc_lines = []
            for doc in doc_index[:15]:
                if doc["title"]:
                    doc_lines.append(f"- `{doc['filename']}`: {doc['title']}")
                else:
                    doc_lines.append(f"- `{doc['filename']}`")
            doc_context = (
                f"\n\n## Documentation Files\n"
                + "\n".join(doc_lines)
            )
        else:
            doc_context = ""

        # Add finalize tool availability notice
        if self._finalize_tool_registered:
            remaining_iterations = self._autonomous_max_iterations - self._autonomous_iteration
            if remaining_iterations <= 2:
                finalize_notice = (
                    f"\n\n## CRITICAL: YOU MUST CALL finalize_result NOW!\n"
                    f"Only {remaining_iterations} iterations remaining. "
                    f"You MUST call `finalize_result` with your best findings immediately."
                )
            elif remaining_iterations <= 4:
                finalize_notice = (
                    f"\n\n## URGENT: Call finalize_result soon!\n"
                    f"Only {remaining_iterations} iterations remaining. "
                    f"If you have found relevant documentation, finalize now."
                )
            else:
                finalize_notice = (
                    "\n\n## IMPORTANT: finalize_result is now available!\n"
                    "You can now call `finalize_result` to complete the search."
                )
        else:
            finalize_notice = (
                f"\n\n## Note: Continue exploring\n"
                f"The `finalize_result` tool will become available after more exploration. "
                f"Currently on iteration {self._autonomous_iteration}."
            )

        return self.AUTONOMOUS_SYSTEM_PROMPT + stats_context + doc_context + finalize_notice

    def _process_streaming_response_autonomous(self, messages: list[dict[str, str]]) -> LLMChatResponse:
        """Process LLM response with streaming for autonomous mode."""
        response: LLMChatResponse | None = None

        for item in self.llm_client.call_stream_sync(
            messages=messages,
            system_prompt=self._get_autonomous_system_prompt(),
            previous_response_id=self._previous_response_id,
        ):
            if isinstance(item, str):
                if self._stream_chunk_callable:
                    self._stream_chunk_callable(item)
            elif isinstance(item, LLMChatResponse):
                response = item

        if response is None:
            raise ValueError("No final response received from streaming LLM")

        return response

    def reset(self) -> None:
        """Reset the conversation to a fresh state."""
        old_chat_thread_id = self._thread.id
        self._thread = ChatThread()
        self._chats = {}
        self._previous_response_id = None
        self._response_id_to_chat_index = {}

        # Reset autonomous mode state
        self._autonomous_mode = False
        self._autonomous_iteration = 0
        self._autonomous_max_iterations = 10
        self._search_result = None
        self._search_failure = None
        self._finalize_tool_registered = False

        if self._persist_chat_thread_callable:
            self._thread = self._persist_chat_thread_callable(self._thread)

        logger.debug(
            "Reset conversation from %s to %s",
            old_chat_thread_id,
            self._thread.id,
        )
