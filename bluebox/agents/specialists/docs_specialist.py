"""
bluebox/agents/specialists/docs_specialist.py

# NOTE: THIS AGENT IS IN BETA AND NOT READY FOR PRODUCTION YET

Agent specialized in searching through documentation and code files.

Contains:
- DocsSpecialist: Specialist for documentation/code analysis
- Uses: AbstractSpecialist base class for all agent plumbing
"""

from __future__ import annotations

import textwrap
from typing import Any, Callable

from bluebox.agents.abstract_agent import agent_tool
from bluebox.agents.specialists.abstract_specialist import AbstractSpecialist, RunMode
from bluebox.data_models.llms.interaction import (
    Chat,
    ChatThread,
    EmittedMessage,
)
from bluebox.data_models.llms.vendors import LLMModel, OpenAIModel
from bluebox.llms.data_loaders.documentation_data_loader import (
    DocumentationDataLoader,
    FileType,
)
from bluebox.utils.data_utils import format_bytes
from bluebox.utils.llm_utils import token_optimized
from bluebox.utils.logger import get_logger


logger = get_logger(name=__name__)


class DocsSpecialist(AbstractSpecialist):
    """
    Documentation digger agent that helps analyze documentation and code files.

    The agent uses AbstractSpecialist as its base and provides tools to search
    and analyze documentation and code.
    """

    SYSTEM_PROMPT: str = textwrap.dedent("""
        You are a documentation and code analyst. You have access to a small documentation set
        and can see the full file index below.

        ## Your Role

        Help users find information in the documentation and code. The file index is in the
        system prompt - you can see all file titles and summaries.

        ## Available Tools

        - **`search_content`**: Search for exact strings (like Cmd+F).
          - Returns LINE NUMBERS where matches are found
          - Use this to find specific terms, function names, or error messages
          - Then use get_file_content to read around those lines

        - **`get_file_content`**: Read file content.
          - Supports OPTIONAL line range: use start_line and end_line to read specific sections
          - Example: After search_content finds a match on line 45, read lines 40-60

        ## Workflow

        1. **Look at the file index below** - find files with relevant titles/summaries
        2. **Use search_content** to find specific terms and get line numbers
        3. **Use get_file_content** to read the relevant lines/files

        ## Guidelines

        - Be concise and direct
        - Summarize key points from documentation
        - Quote relevant code snippets when helpful
        - Use line ranges to focus on relevant sections
    """).strip()

    AUTONOMOUS_SYSTEM_PROMPT: str = textwrap.dedent("""
        You are a documentation analyst. You have access to a small documentation set
        and can see the full file index below.

        ## Your Mission

        Given a user query, find the documentation and/or code files that best answer their question.
        Provide structured output matching the orchestrator's expected schema.

        ## Available Tools

        - **`search_content`**: Search for exact strings (like Cmd+F).
          - Returns LINE NUMBERS where matches are found
          - Use this to locate specific content

        - **`get_file_content`**: Read file content.
          - Supports line range: use start_line/end_line for focused reading

        ## Process

        1. **Look at the file index below** - identify promising files by title/summary
        2. **Search**: Use `search_content` to find specific terms
        3. **Read**: Use `get_file_content` to examine relevant sections
        4. **Finalize**: Call `finalize_with_output` with your findings matching the expected schema

        ## When finalize tools are available

        After exploration, `finalize_with_output` and `finalize_with_failure` tools become available.

        - **finalize_with_output**: Submit your findings matching the expected output schema
        - **finalize_with_failure**: Report that the task could not be completed

        Use `add_note()` before finalizing to record any notes, complaints, warnings, or errors.
    """).strip()

    ## Magic methods

    def __init__(
        self,
        emit_message_callable: Callable[[EmittedMessage], None],
        documentation_data_store: DocumentationDataLoader,
        persist_chat_callable: Callable[[Chat], Chat] | None = None,
        persist_chat_thread_callable: Callable[[ChatThread], ChatThread] | None = None,
        stream_chunk_callable: Callable[[str], None] | None = None,
        llm_model: LLMModel = OpenAIModel.GPT_5_1,
        run_mode: RunMode = RunMode.CONVERSATIONAL,
        chat_thread: ChatThread | None = None,
        existing_chats: list[Chat] | None = None,
    ) -> None:
        """
        Initialize the documentation digger agent.

        Args:
            emit_message_callable: Callback function to emit messages to the host.
            documentation_data_store: The DocumentationDataLoader containing indexed files.
            persist_chat_callable: Optional callback to persist Chat objects.
            persist_chat_thread_callable: Optional callback to persist ChatThread.
            stream_chunk_callable: Optional callback for streaming text chunks.
            llm_model: The LLM model to use for conversation.
            run_mode: How the specialist will be run (conversational or autonomous).
            chat_thread: Existing ChatThread to continue, or None for new conversation.
            existing_chats: Existing Chat messages if loading from persistence.
        """
        self._documentation_data_store = documentation_data_store

        super().__init__(
            emit_message_callable=emit_message_callable,
            persist_chat_callable=persist_chat_callable,
            persist_chat_thread_callable=persist_chat_thread_callable,
            stream_chunk_callable=stream_chunk_callable,
            llm_model=llm_model,
            run_mode=run_mode,
            chat_thread=chat_thread,
            existing_chats=existing_chats,
        )

        logger.debug(
            "DocsSpecialist initialized with model: %s, chat_thread_id: %s, files: %d",
            llm_model,
            self._thread.id,
            len(documentation_data_store.entries),
        )

    ## Abstract method implementations

    def _get_system_prompt(self) -> str:
        """Get system prompt with full file index."""
        stats = self._documentation_data_store.stats
        stats_context = (
            f"\n\n## File Index ({stats.total_files} files, {format_bytes(stats.total_bytes)})\n"
        )

        # Add FULL documentation index with titles and summaries
        doc_index = self._documentation_data_store.get_documentation_index()
        if doc_index:
            doc_lines = ["\n### Documentation Files"]
            for doc in doc_index:
                title = doc.get("title", "")
                summary = doc.get("summary", "")
                if title and summary:
                    # Truncate long summaries
                    if len(summary) > 100:
                        summary = summary[:100] + "..."
                    doc_lines.append(f"- `{doc['filename']}`: **{title}** - {summary}")
                elif title:
                    doc_lines.append(f"- `{doc['filename']}`: **{title}**")
                else:
                    doc_lines.append(f"- `{doc['filename']}`")
            stats_context += "\n".join(doc_lines)

        # Add FULL code index with docstrings
        code_index = self._documentation_data_store.get_code_index()
        if code_index:
            code_lines = ["\n\n### Code Files"]
            for code in code_index:
                docstring = code.get("docstring", "")
                if docstring:
                    # Truncate long docstrings
                    if len(docstring) > 100:
                        docstring = docstring[:100] + "..."
                    code_lines.append(f"- `{code['filename']}`: {docstring}")
                else:
                    code_lines.append(f"- `{code['filename']}`")
            stats_context += "\n".join(code_lines)

        return self.SYSTEM_PROMPT + stats_context

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

        # Include output schema if set by orchestrator
        schema_section = self._get_output_schema_prompt_section()

        # Add finalize tool availability notice
        if self.can_finalize:
            remaining_iterations = self._autonomous_config.max_iterations - self._autonomous_iteration
            if remaining_iterations <= 2:
                finalize_notice = (
                    f"\n\n## CRITICAL: YOU MUST CALL finalize_with_output NOW!\n"
                    f"Only {remaining_iterations} iterations remaining."
                )
            elif remaining_iterations <= 4:
                finalize_notice = (
                    f"\n\n## URGENT: Call finalize_with_output soon!\n"
                    f"Only {remaining_iterations} iterations remaining."
                )
            else:
                finalize_notice = (
                    "\n\n## IMPORTANT: finalize_with_output is now available!\n"
                    "Call it when you have your findings."
                )
        else:
            finalize_notice = (
                f"\n\n## Note: Continue exploring\n"
                f"Finalize tools will become available after more exploration. "
                f"Currently on iteration {self._autonomous_iteration}."
            )

        return self.AUTONOMOUS_SYSTEM_PROMPT + stats_context + doc_context + schema_section + finalize_notice

    def _get_autonomous_initial_message(self, task: str) -> str:
        return (
            f"QUERY: {task}\n\n"
            "Find documentation and/or code that answers this question. "
            "Search, analyze, and when confident, use finalize_result to report your findings. "
            "If after thorough search you determine the information does not exist, "
            "use finalize_failure to report why."
        )

    def _check_autonomous_completion(self, tool_name: str) -> bool:
        # Delegate to base class (handles generic finalize tools)
        return super()._check_autonomous_completion(tool_name)

    def _get_autonomous_result(self):
        # Delegate to base class (returns wrapped result)
        return super()._get_autonomous_result()

    def _reset_autonomous_state(self) -> None:
        # Call base class to reset generic state
        super()._reset_autonomous_state()

    ## Tool handlers

    @agent_tool(
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
                    "description": "Optional filter by file type: 'documentation' for docs, 'code' for source files.",
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Whether the search should be case-sensitive. Defaults to false.",
                },
            },
            "required": ["query"],
        }
    )
    @token_optimized
    def _search_content(
        self,
        query: str,
        file_type: str | None = None,
        case_sensitive: bool = False,
    ) -> dict[str, Any]:
        """
        Search file contents for an exact query string (like Cmd+F).

        Returns LINE NUMBERS where matches are found. Use this to find specific terms,
        then use get_file_content to read around those lines.

        Args:
            query: The exact string to search for.
            file_type: Optional filter: 'documentation' for docs, 'code' for source files.
            case_sensitive: Whether the search should be case-sensitive. Defaults to false.
        """
        if not query:
            return {"error": "query is required"}

        file_type_enum = FileType(file_type) if file_type else None

        results = self._documentation_data_store.search_content_with_lines(
            query=query,
            file_type=file_type_enum,
            case_sensitive=case_sensitive,
            max_matches_per_file=10,
        )

        if not results:
            return {
                "message": f"No matches found for '{query}'",
                "case_sensitive": case_sensitive,
            }

        return {
            "query": query,
            "case_sensitive": case_sensitive,
            "files_with_matches": len(results),
            "results": results[:20],  # Top 20 files
        }

    @agent_tool()
    @token_optimized
    def _get_file_content(
        self,
        path: str,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> dict[str, Any]:
        """
        Get file content by path.

        Can read the full file or specific LINE RANGE. Use start_line/end_line
        to read around matches from search_content.

        Args:
            path: The file path (can be partial, will match).
            start_line: Starting line number (1-indexed, inclusive). Omit for beginning.
            end_line: Ending line number (1-indexed, inclusive). Omit to read to end.
        """
        if not path:
            return {"error": "path is required"}

        # If line range specified, use get_file_lines
        if start_line is not None or end_line is not None:
            result = self._documentation_data_store.get_file_lines(
                path=path,
                start_line=start_line,
                end_line=end_line,
            )
            if result is None:
                return {"error": f"File '{path}' not found"}

            content, total_lines = result
            actual_start = start_line or 1
            actual_end = end_line or total_lines

            return {
                "path": path,
                "lines_shown": f"{actual_start}-{actual_end}",
                "total_lines": total_lines,
                "content": content,
            }

        # Otherwise, read full file
        entry = self._documentation_data_store.get_file_by_path(path)
        if entry is None:
            return {"error": f"File '{path}' not found"}

        content = entry.content
        total_lines = content.count("\n") + 1

        # Truncate large content
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
