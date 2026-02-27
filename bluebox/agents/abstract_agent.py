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
import re
from abc import ABC, abstractmethod
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from textwrap import dedent
from typing import Any, Callable, ClassVar, NamedTuple, get_type_hints

import jsonschema
from pydantic import BaseModel, TypeAdapter, ValidationError

from bluebox.workspace import AgentWorkspace
from bluebox.data_models.llms.interaction import (
    BrowserAgentStepEmittedMessage,
    Chat,
    ChatRole,
    ChatThread,
    EmittedMessage,
    ChatResponseEmittedMessage,
    ErrorEmittedMessage,
    LLMChatResponse,
    LLMToolCall,
    StatusUpdateEmittedMessage,
    ToolInvocationResultEmittedMessage,
    PendingToolInvocation,
    ToolInvocationStatus,
)
from bluebox.data_models.llms.vendors import LLMModel, OpenAIModel
from bluebox.data_models.orchestration.result import SpecialistResultWrapper
from bluebox.llms.data_loaders.documentation_data_loader import DocumentationDataLoader, FileType
from bluebox.llms.llm_client import LLMClient
from bluebox.llms.tools.tool_utils import extract_description_from_docstring, generate_parameters_schema
from bluebox.utils.code_execution_sandbox import (
    BLOCKED_MODULES,
    BLOCKED_PATTERNS,
    execute_python_sandboxed,
    get_active_sandbox_mode,
    get_workaround_for_error,
)
from bluebox.utils.data_utils import format_bytes
from bluebox.utils.llm_utils import token_optimized as token_optimized_decorator
from bluebox.utils.logger import get_logger

logger = get_logger(name=__name__)


class ToolResultPersistMode(StrEnum):
    NEVER = "never"
    ALWAYS = "always"
    OVERFLOW = "overflow"


# Keep persisted tool previews small so iterative runs don't blow context.
PERSISTED_TOOL_PREVIEW_MAX_CHARS = 800


class AgentExecutionMode(StrEnum):
    """Execution mode for agent loops."""
    CONVERSATIONAL = "conversational"
    AUTONOMOUS = "autonomous"


class AutonomousRunConfig(NamedTuple):
    """
    Configuration for autonomous agent runs.

    min_iterations controls when finalize tools become available.
    max_iterations controls when the autonomous loop gives up.
    """
    min_iterations: int = 3
    max_iterations: int = 10


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
    persist: ToolResultPersistMode = ToolResultPersistMode.NEVER
    max_characters: int = 10_000
    token_optimized: bool = False


def _serialize_tool_result(tool_result: Any) -> tuple[str, str]:
    try:
        return json.dumps(tool_result, ensure_ascii=False, default=str, indent=2), "json"
    except (TypeError, ValueError):
        return str(tool_result), "text"


def _normalize_file_scope(scope: str) -> str:
    """Normalize and validate file tool scope."""
    normalized_scope = scope.strip().lower()
    if normalized_scope not in {"workspace", "docs"}:
        raise ValueError("scope must be 'workspace' or 'docs'")
    return normalized_scope


def _parse_search_terms(query: str) -> list[str]:
    """Split query text into distinct terms for terms-mode search."""
    seen: set[str] = set()
    terms: list[str] = []
    for token in re.split(r"[,\s]+", query):
        term = token.strip()
        if term and term not in seen:
            seen.add(term)
            terms.append(term)
    return terms


def agent_tool(
    description: str | Callable | None = None,
    parameters: dict[str, Any] | None = None,
    *,
    availability: bool | Callable[..., bool] = True,
    persist: ToolResultPersistMode = ToolResultPersistMode.NEVER,
    max_characters: int = 10_000,
    token_optimized: bool = False,
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
        persist: Tool-result persistence policy.
            - ToolResultPersistMode.NEVER (default): never persist.
            - ToolResultPersistMode.ALWAYS: always persist.
            - ToolResultPersistMode.OVERFLOW: persist only if result exceeds max_characters.
        max_characters: Character threshold for OVERFLOW mode.
        token_optimized: If True, encode tool output with toon for token efficiency.
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

        if not isinstance(persist, ToolResultPersistMode):
            raise ValueError(
                f"Tool {tool_name} has invalid persist value: {persist!r}. "
                "Use ToolResultPersistMode values.",
            )
        if max_characters <= 0:
            raise ValueError(f"Tool {tool_name} must have max_characters > 0")

        method._tool_meta = _ToolMeta(
            name=tool_name,
            description=final_description,
            parameters=final_parameters,
            availability=availability,
            persist=persist,
            max_characters=max_characters,
            token_optimized=token_optimized,
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
    _subclasses: ClassVar[list[type[AbstractAgent]]] = []
    WORKSPACE_USAGE_SECTION: ClassVar[str] = dedent("""\
        ## Workspace
        - Use `raw/` (read-only) for tool-call artifacts (inputs/results), not deliverables.
        - Write generated deliverables to `output/`.
        - Store reusable notes/context in `context/`.
        - `meta/` (read-only) is system-managed and not editable.
    """)

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Validate that concrete subclasses define AGENT_CARD."""
        super().__init_subclass__(**kwargs)
        # skip abstract classes
        if cls.__name__.startswith("Abstract"):
            return
        # validate that the subclass defines an AGENT_CARD class variable of type AgentCard
        if not hasattr(cls, "AGENT_CARD") or not isinstance(cls.AGENT_CARD, AgentCard):
            raise TypeError(
                f"{cls.__name__} must define an AGENT_CARD class variable of type AgentCard"
            )
        if cls not in cls._subclasses:
            cls._subclasses.append(cls)

    @classmethod
    def get_all_subclasses(cls) -> list[type[AbstractAgent]]:
        """Return a copy of all registered concrete AbstractAgent subclasses."""
        return cls._subclasses.copy()

    ## Abstract methods

    @abstractmethod
    def _get_system_prompt(self) -> str:
        """Return the system prompt for the agent."""

    ## Magic methods

    def __init__(
        self,
        emit_message_callable: Callable[[EmittedMessage], None],
        workspace: AgentWorkspace | None = None,
        persist_chat_callable: Callable[[Chat], Chat] | None = None,
        persist_chat_thread_callable: Callable[[ChatThread], ChatThread] | None = None,
        stream_chunk_callable: Callable[[str], None] | None = None,
        llm_model: LLMModel = OpenAIModel.GPT_5_2,
        chat_thread: ChatThread | None = None,
        existing_chats: list[Chat] | None = None,
        documentation_data_loader: DocumentationDataLoader | None = None,
        on_llm_response: Callable[[LLMChatResponse], None] | None = None,
        execution_mode: AgentExecutionMode = AgentExecutionMode.CONVERSATIONAL,
        allow_code_execution: bool = False,
        code_execution_globals: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize the agent.

        Args:
            emit_message_callable: Callback to emit messages to the host.
            workspace: Optional workspace used for agent file operations.
                When omitted, workspace-scoped tools/features are unavailable.
            persist_chat_callable: Optional callback to persist Chat objects.
            persist_chat_thread_callable: Optional callback to persist ChatThread.
            stream_chunk_callable: Optional callback for streaming text chunks.
            llm_model: The LLM model to use.
            chat_thread: Existing ChatThread to continue, or None for new.
            existing_chats: Existing Chat messages if loading from persistence.
            documentation_data_loader: Optional DocumentationDataLoader for docs/code search tools.
            on_llm_response: Optional callback invoked after each LLM call with the response (for token tracking).
            execution_mode: Agent execution mode (conversational or autonomous).
            allow_code_execution: Whether to expose the generic execute_python tool.
            code_execution_globals: Globals injected into execute_python sandbox runs.
                Must be empty when allow_code_execution is False.
        """
        normalized_globals = dict(code_execution_globals or {})
        if not allow_code_execution and normalized_globals:
            raise ValueError(
                "code_execution_globals must be empty when allow_code_execution is False",
            )

        self._emit_message_callable = emit_message_callable
        self._persist_chat_callable = persist_chat_callable
        self._persist_chat_thread_callable = persist_chat_thread_callable
        self._stream_chunk_callable = stream_chunk_callable
        self._documentation_data_loader = documentation_data_loader
        self._on_llm_response = on_llm_response
        self._on_chat_added: Callable[[Chat], None] | None = None
        self.execution_mode = execution_mode
        self._autonomous_iteration: int = 0
        self._autonomous_config: AutonomousRunConfig = AutonomousRunConfig()
        self._task_output_schema: dict[str, Any] | None = None
        self._task_output_description: str | None = None
        self._notes: list[str] = []
        self._wrapped_result: BaseModel | None = None
        self._finalize_with_output_failed = False
        self._allow_code_execution = allow_code_execution
        self._code_execution_globals = normalized_globals if allow_code_execution else {}
        self._sandbox_mode = (
            get_active_sandbox_mode(work_dir_set=workspace is not None)
            if allow_code_execution
            else "blocklist"
        )
        self._previous_response_id: str | None = None
        self._response_id_to_chat_index: dict[str, int] = {}

        if workspace is not None and not isinstance(workspace, AgentWorkspace):
            raise TypeError("workspace must implement AgentWorkspace when provided")
        self._workspace = workspace

        self.llm_model = llm_model
        self.llm_client = LLMClient(llm_model)

        # Track which tools are currently registered
        self._registered_tool_names: set[str] = set()

        # Initial tool sync
        self._sync_tools()

        # Conversation state
        self._thread = chat_thread or ChatThread()
        self._thread_persisted = chat_thread is not None
        self._chats: dict[str, Chat] = {}
        self._current_assistant_chat_id: str | None = None
        if existing_chats:
            for chat in existing_chats:
                self._chats[chat.id] = chat

    @agent_tool(
        availability=lambda self: self._allow_code_execution,
        persist=ToolResultPersistMode.OVERFLOW,
    )
    def _execute_python(self, code: str) -> dict[str, Any]:
        """
        Execute Python code in a sandbox.

        When a workspace is configured, code runs with `work_dir` set to the
        workspace root so file I/O is scoped to that directory.  All files
        created or modified anywhere in the workspace are tracked and
        reported in the response as ``files_created``.
        Without a workspace, execution is compute-only and file I/O (open/Path)
        remains blocked by sandbox policy.

        Globals passed via `code_execution_globals` are always available.

        Args:
            code: Python code to execute.
        """
        files_created: list[str] = []

        if self.has_workspace:
            workspace = self._require_workspace()
            workspace.ensure_dirs()

            # Snapshot entire workspace before execution for file-tracking
            files_before = workspace.snapshot_paths(["."])

            sandbox_result = execute_python_sandboxed(
                code=code,
                extra_globals=self._code_execution_globals,
                work_dir=str(workspace.root_path.resolve()),
                read_only_paths=[
                    str((workspace.root_path / "raw").resolve()),
                    str((workspace.root_path / "meta").resolve()),
                ],
            )

            # Diff entire workspace to detect created/modified files
            files_after = workspace.snapshot_paths(["."])
            delta = workspace.diff_snapshot(files_before, files_after)
            changed_states = delta.created + delta.modified
            files_created = [
                str(workspace.root_path / state.relative_path)
                for state in changed_states
            ]
        else:
            sandbox_result = execute_python_sandboxed(
                code=code,
                extra_globals=self._code_execution_globals,
            )

        # Build response
        result: dict[str, Any] = {}

        if "error" in sandbox_result:
            result["error"] = sandbox_result["error"]
            workaround = get_workaround_for_error(sandbox_result["error"])
            if workaround:
                result["_hint"] = (
                    f"Sandbox restriction: {workaround} "
                    "Fix the code and call execute_python again."
                )
            elif self.has_workspace:
                result["_hint"] = (
                    "Code failed. Read the error and stdout carefully, then fix and retry."
                )

        output = sandbox_result.get("output", "")
        if output and output != "(no output)":
            result["output"] = output

        if files_created:
            result["files_created"] = files_created
            result["output_file"] = files_created[0]

        if not result:
            result["output"] = "(no output)"

        return result

    def _generate_code_execution_prompt(self) -> str:
        """Generate a prompt section describing the code execution environment.

        Covers sandbox mode (blocklist restrictions vs docker/lambda permissiveness)
        and any pre-loaded globals available in the sandbox.

        Subclasses can override to add domain-specific guidance, but should call
        super() to include the base sandbox information.
        """
        if not self._allow_code_execution:
            return ""

        lines: list[str] = ["\n## Code Execution Environment"]

        if not self.has_workspace:
            lines.append(
                "\nNo workspace is configured for this agent."
                "\nExecution is compute-only: filesystem access is disabled."
                "\n`open()` and `Path` file operations are unavailable."
            )

        # Globals section
        if self._code_execution_globals:
            global_names = ", ".join(f"`{k}`" for k in sorted(self._code_execution_globals))
            lines.append(
                f"\nPre-loaded globals available in `execute_python`: {global_names}."
                "\nUse these directly — do NOT re-import or re-define them."
            )

        # Sandbox-specific guidance
        if self._sandbox_mode == "blocklist":
            blocked_modules_str = ", ".join(sorted(BLOCKED_MODULES))
            blocked_patterns_str = ", ".join(
                f"`{p}`" for p, _ in BLOCKED_PATTERNS if p != "open("
            )
            path_rule = (
                "\n- `Path` is already pre-loaded — use it directly, do NOT `import pathlib`"
                if self.has_workspace
                else "\n- `Path` is not available without workspace-backed file scope"
            )
            open_rule = (
                "\n- `open()` is already pre-loaded — use it directly for all file I/O"
                if self.has_workspace
                else "\n- `open()` is blocked when no workspace is configured"
            )
            lines.append(
                "\n### Sandbox Restrictions (IMPORTANT — read before writing any Python code)"
                "\nYou are running in restricted sandbox mode."
                "\n"
                "\n**Blocked imports** — do NOT import any of these modules:"
                f"\n{blocked_modules_str}"
                "\n"
                "\n**Blocked code patterns** — do NOT use any of these in your code:"
                f"\n{blocked_patterns_str}"
                "\n"
                "\n**Safe imports you CAN use:**"
                "\n`collections`, `re`, `datetime`, `math`, `itertools`, `functools`,"
                " `operator`, `string`, `textwrap`, `decimal`, `fractions`,"
                " `statistics`, `urllib.parse`, `hashlib`, `hmac`, `base64`,"
                " `copy`, `pprint`, `dataclasses`, `enum`, `typing`"
                "\n"
                "\n**Key rules to avoid errors:**"
                "\n- Do NOT `import os`, `import pathlib`, `import sys`, or any blocked module"
                f"{path_rule}"
                f"{open_rule}"
                "\n- Do NOT use `getattr()` — use dict access: `obj[\"key\"]` or `obj.get(\"key\")`"
            )
        else:
            # Docker or Lambda — more permissive
            lines.append(
                f"\nSandbox mode: **{self._sandbox_mode}** (full Python environment available)."
                "\nYou have access to the standard library and common packages."
                + (
                    "\nFile I/O is scoped to the workspace directory."
                    if self.has_workspace
                    else "\nNo workspace attached: file I/O is disabled for this run."
                )
            )

        return "\n".join(lines)

    def _maybe_persist_tool_result(
        self,
        tool_name: str,
        tool_meta: _ToolMeta,
        tool_result: Any,
    ) -> Any:
        persist_mode = tool_meta.persist
        if persist_mode == ToolResultPersistMode.NEVER:
            return tool_result

        serialized, content_type = _serialize_tool_result(tool_result)
        char_count = len(serialized)

        if persist_mode == ToolResultPersistMode.OVERFLOW and char_count <= tool_meta.max_characters:
            return tool_result

        safe_tool_name = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in tool_name)
        extension = ".json" if content_type == "json" else ".txt"
        filename = f"{datetime.now().strftime('%Y-%m-%d-%H%M%S-%f')}-{safe_tool_name}_result{extension}"
        is_truncated = char_count > tool_meta.max_characters
        preview_limit = min(tool_meta.max_characters, PERSISTED_TOOL_PREVIEW_MAX_CHARS)
        preview = serialized[:preview_limit]
        if char_count > preview_limit:
            preview += f"\n... (preview truncated, {char_count} total chars)"

        workspace = self._workspace
        if workspace is None:
            logger.debug(
                "Skipping persistence for '%s': no workspace configured",
                tool_name,
            )
            return tool_result

        try:
            workspace.ensure_dirs()
            ref = workspace.save_artifact(
                source="raw",
                filename=filename,
                content=serialized,
                tool_name=tool_name,
                content_type=content_type,
                metadata={
                    "persist_mode": persist_mode.value,
                    "char_count": char_count,
                    "max_characters": tool_meta.max_characters,
                    "preview_max_characters": preview_limit,
                },
            )
            logger.debug(
                "Persisted tool result for '%s' as artifact %s (%s chars)",
                tool_name,
                ref.artifact_id,
                char_count,
            )
            return {
                "tool_name": tool_name,
                "persist_mode": persist_mode.value,
                "artifact_id": ref.artifact_id,
                "artifact_path": ref.relative_path,
                "truncated": is_truncated,
                "preview": preview,
                "_hint": (
                    "Full tool result saved to workspace raw artifacts. "
                    f"Read artifact_id={ref.artifact_id} (path: {ref.relative_path}) to inspect complete output."
                ),
            }
        except Exception as e:
            logger.warning("Failed to persist tool result for '%s': %s", tool_name, e)
            return tool_result

    ## Properties

    @property
    def chat_thread_id(self) -> str:
        """Return the current thread ID."""
        return self._thread.id

    @property
    def has_workspace(self) -> bool:
        """Whether this agent has an attached workspace."""
        return self._workspace is not None

    @property
    def autonomous_iteration(self) -> int:
        """Return the current/final autonomous iteration count."""
        return self._autonomous_iteration

    @property
    def can_finalize(self) -> bool:
        """
        Whether finalize tools should be available.

        Finalize tools are available only in autonomous mode after min_iterations.
        """
        return (
            self.execution_mode == AgentExecutionMode.AUTONOMOUS
            and self._autonomous_iteration >= self._autonomous_config.min_iterations
        )

    @property
    def has_output_schema(self) -> bool:
        """Whether an orchestrator-defined output schema has been set."""
        return self._task_output_schema is not None

    ## Autonomous extension points

    def _require_workspace(self) -> AgentWorkspace:
        """Return workspace or raise a clear runtime error when unavailable."""
        if self._workspace is None:
            raise RuntimeError(
                "This agent has no workspace configured. "
                "Workspace-scoped tools are unavailable.",
            )
        return self._workspace

    def _get_autonomous_system_prompt(self) -> str:
        """
        Return system prompt for autonomous mode.

        Subclasses can override this for custom autonomous behavior.
        """
        return (
            self._get_system_prompt()
            + dedent("""

                ## Autonomous Execution
                - Operate independently and use tools to complete the task.
                - Keep reasoning concise and tool-driven.
                - Use `add_note()` for warnings, assumptions, and blockers.
                - Finalize only via the designated finalize tool.
            """)
            + self._get_output_schema_prompt_section()
            + self._get_urgency_notice()
        )

    def _get_autonomous_initial_message(self, task: str) -> str:
        """
        Build initial USER message for autonomous mode.

        Subclasses can override this for custom autonomous task framing.
        """
        finalize_call = (
            "finalize_with_output(output={...})"
            if self.has_output_schema
            else "finalize_result(output={...})"
        )
        return dedent(
            f"""
            Task: {task}

            Run autonomously until complete.
            Use available tools to gather evidence and produce the best possible output.
            When done, call `{finalize_call}`.
            """
        ).strip()

    def _check_autonomous_completion(self, tool_name: str) -> bool:
        """
        Check whether a tool call signals autonomous completion.

        Default implementation checks generic finalize tool names and
        requires a wrapped result to be present.
        """
        finalize_tools = (
            "finalize_with_output",
            "finalize_with_failure",
            "finalize_result",
            "finalize_failure",
        )
        if tool_name in finalize_tools:
            return self._wrapped_result is not None
        return False

    def _get_autonomous_result(self) -> BaseModel | None:
        """
        Return autonomous run result.

        Subclasses may override this to map to specialized result objects.
        """
        return self._wrapped_result

    def _reset_autonomous_state(self) -> None:
        """
        Reset autonomous-mode state before a new run.

        Subclasses can extend this to clear their own autonomous fields.
        """
        self._task_output_schema = None
        self._task_output_description = None
        self._notes = []
        self._wrapped_result = None
        self._finalize_with_output_failed = False

    ## Output schema helpers for autonomous runs

    def set_output_schema(
        self,
        schema: dict[str, Any],
        description: str | None = None,
    ) -> None:
        """Set expected output schema for the current autonomous task."""
        self._task_output_schema = schema
        self._task_output_description = description

    def _get_output_schema_prompt_section(self) -> str:
        """Return formatted output schema prompt section for autonomous mode."""
        if not self._task_output_schema:
            return ""

        parts = ["\n\n## Expected Output Schema\n"]
        if self._task_output_description:
            parts.append(f"**Description:** {self._task_output_description}\n\n")
        parts.append("**Schema:**\n```json\n")
        parts.append(json.dumps(self._task_output_schema, indent=2))
        parts.append("\n```\n")
        parts.append(
            "\nWhen ready, call `finalize_with_output(output={...})` with data matching this schema. "
            "Use `add_note()` before finalizing to record notes, warnings, or errors."
        )
        return "".join(parts)

    def _get_urgency_notice(self) -> str:
        """Iteration-aware urgency notice for autonomous prompts."""
        finalize_tool = (
            "`finalize_with_output(output={...})`"
            if self.has_output_schema
            else "`finalize_result(output={...})`"
        )
        if self.can_finalize:
            remaining = self._autonomous_config.max_iterations - self._autonomous_iteration
            if remaining <= 2:
                return f"\n\n## URGENT: Only {remaining} iteration(s) left — call {finalize_tool} NOW."
            if remaining <= 4:
                return f"\n\n## Finalize soon — {remaining} iterations remaining."
            return f"\n\n## {finalize_tool} is now available."
        return f"\n\n## Continue exploring (iteration {self._autonomous_iteration})."

    @agent_tool(
        availability=lambda self: (
            self.execution_mode == AgentExecutionMode.AUTONOMOUS
        )
    )
    def add_note(self, note: str) -> dict[str, Any]:
        """
        Add a note to the autonomous result wrapper.

        Args:
            note: Note, warning, complaint, or error to include in final output.
        """
        self._notes.append(note)
        return {"status": "ok", "total_notes": len(self._notes)}

    @agent_tool(availability=lambda self: self.can_finalize and self.has_output_schema, token_optimized=True)
    def _finalize_with_output(self, output: dict[str, Any]) -> dict[str, Any]:
        """
        Finalize with output matching the orchestrator's expected schema.

        Args:
            output: Result data matching the configured JSON schema.
        """
        if not self._task_output_schema:
            return {"error": "No output schema defined for this task"}

        try:
            jsonschema.validate(instance=output, schema=self._task_output_schema)
        except jsonschema.ValidationError as e:
            self._finalize_with_output_failed = True
            return {
                "error": "VALIDATION FAILED — output does not match the expected schema.",
                "validation_error": str(e.message),
                "schema_path": list(e.absolute_schema_path),
                "hint": "Fix the output structure and call finalize_with_output again.",
            }

        self._finalize_with_output_failed = False
        self._wrapped_result = SpecialistResultWrapper(
            output=output,
            success=True,
            notes=self._notes.copy(),
            failure_reason=None,
        )
        return {
            "status": "success",
            "message": "Output validated and stored successfully",
            "notes_count": len(self._notes),
        }

    @agent_tool(availability=lambda self: self.can_finalize and self.has_output_schema, token_optimized=True)
    def _finalize_with_failure(self, reason: str) -> dict[str, Any]:
        """
        Finalize with failure when a schema-based task cannot be completed.

        Args:
            reason: Why the task could not be completed.
        """
        if self._finalize_with_output_failed:
            return {
                "error": (
                    "REJECTED — finalize_with_output already failed validation. "
                    "Fix output and retry finalize_with_output instead of giving up."
                ),
            }

        self._wrapped_result = SpecialistResultWrapper(
            output=None,
            success=False,
            notes=self._notes.copy(),
            failure_reason=reason,
        )
        return {
            "status": "failure",
            "message": "Task marked as failed",
            "reason": reason,
        }

    @agent_tool(availability=lambda self: self.can_finalize and not self.has_output_schema, token_optimized=True)
    def _finalize_result(self, output: dict[str, Any]) -> dict[str, Any]:
        """
        Finalize and submit result data for tasks without a predefined schema.

        Args:
            output: Result payload.
        """
        self._wrapped_result = SpecialistResultWrapper(
            output=output,
            success=True,
            notes=self._notes.copy(),
            failure_reason=None,
        )
        return {
            "status": "success",
            "message": "Result submitted successfully",
            "notes_count": len(self._notes),
        }

    @agent_tool(availability=lambda self: self.can_finalize and not self.has_output_schema, token_optimized=True)
    def _finalize_failure(self, reason: str) -> dict[str, Any]:
        """
        Finalize with failure for tasks without a predefined schema.

        Args:
            reason: Why the task could not be completed.
        """
        self._wrapped_result = SpecialistResultWrapper(
            output=None,
            success=False,
            notes=self._notes.copy(),
            failure_reason=reason,
        )
        return {
            "status": "failure",
            "message": "Task marked as failed",
            "reason": reason,
        }

    ## Public API

    def get_thread(self) -> ChatThread:
        """Get the current conversation thread."""
        return self._thread

    def get_chats(self) -> list[Chat]:
        """Get all Chat messages in order."""
        return [self._chats[chat_id] for chat_id in self._thread.chat_ids if chat_id in self._chats]

    def reset(self) -> None:
        """Reset the conversation to a fresh state."""
        # Reset autonomous mode state
        self.execution_mode = AgentExecutionMode.CONVERSATIONAL
        self._autonomous_iteration = 0
        self._reset_autonomous_state()

        old_chat_thread_id = self._thread.id
        self._thread = ChatThread()
        self._thread_persisted = False
        self._chats = {}
        self._previous_response_id = None
        self._response_id_to_chat_index = {}

        # Sync tools for fresh state
        self._sync_tools()

        logger.debug("Reset conversation from %s to %s", old_chat_thread_id, self._thread.id)

    ## Tool registration and dispatch

    def _get_tool_registration_payload(self, tool_meta: _ToolMeta) -> tuple[str, dict[str, Any]]:
        """
        Return (description, parameters) used when registering a tool with the LLM.

        Subclasses can override to dynamically customize registration metadata
        without reimplementing _sync_tools.
        """
        description, parameters = tool_meta.description, tool_meta.parameters
        if tool_meta.name == "finalize_result":
            description = (
                "Finalize and submit result data. "
                "You MUST call this with a non-empty `output` object: "
                "`finalize_result(output={...})`. Do NOT call with empty arguments."
            )
            return description, parameters

        if tool_meta.name != "finalize_with_output" or not self._task_output_schema:
            return description, parameters

        parameters = {
            "type": "object",
            "properties": {
                "output": self._task_output_schema,
            },
            "required": ["output"],
        }
        desc_suffix = (
            f" Output description: {self._task_output_description}"
            if self._task_output_description
            else ""
        )
        description = (
            "Finalize with output matching the expected schema. "
            "The output parameter MUST include all required fields "
            "defined in the schema — do NOT call with empty arguments."
            + desc_suffix
        )
        return description, parameters

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
            description, parameters = self._get_tool_registration_payload(tool_meta)
            self.llm_client.register_tool(
                name=tool_meta.name,
                description=description,
                parameters=parameters,
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
        valid_params = set(tool_meta.parameters.get("properties", {}).keys())
        missing = [p for p in required if p not in tool_arguments or tool_arguments[p] is None]
        extra = set(tool_arguments.keys()) - valid_params

        # Auto-wrap: if the tool expects a single "output" dict parameter and the LLM
        # passed the dict contents as top-level kwargs instead, wrap them automatically.
        # This is the #1 cause of finalize_with_output / finalize_result failures.
        if (
            missing
            and valid_params == {"output"}
            and "output" not in tool_arguments
            and tool_arguments  # LLM passed *something*
        ):
            logger.info(
                "Auto-wrapping %d top-level arg(s) into 'output' for tool '%s'",
                len(tool_arguments), tool_name,
            )
            tool_arguments = {"output": dict(tool_arguments)}
            missing = []
            extra = set()

        if missing:
            return {"error": f"Missing required parameter(s): {', '.join(missing)}. "
                    f"Expected parameters: {', '.join(sorted(valid_params))}"}

        # validate no extra parameters
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
            # extract readable error message with actionable guidance
            errors = e.errors()
            if errors:
                err = errors[0]
                got_type = type(value).__name__
                expected = err.get("type", "valid type")
                msg = f"{param_name}: expected {expected}, got {got_type}"
                # Add actionable hint for common dict vs string confusion
                if got_type == "str" and "dict" in expected:
                    msg += (
                        ". You passed a string but this parameter requires a JSON object. "
                        'Example: {"key": "value", "nested": {"a": 1}} — NOT a string.'
                    )
            else:
                msg = str(e)
            return {"error": f"Invalid argument type: {msg}"}

        logger.debug("Executing tool %s with arguments: %s", tool_name, tool_arguments)
        # handler is unbound (from cls, not self) so pass self explicitly
        raw_result = handler(self, **validated_arguments)
        result_for_llm = self._maybe_persist_tool_result(
            tool_name=tool_name,
            tool_meta=tool_meta,
            tool_result=raw_result,
        )
        if tool_meta.token_optimized:
            if isinstance(result_for_llm, dict) and "artifact_id" in result_for_llm:
                result_for_llm = {
                    **result_for_llm,
                    "_token_optimized_note": (
                        "This chat output is token-optimized; the saved artifact contains raw output."
                    ),
                }
            return token_optimized_decorator(lambda: result_for_llm)()
        return result_for_llm

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

    def _list_workspace_files_scoped(self, path: str) -> dict[str, Any]:
        """List files under a workspace subpath."""
        if not self.has_workspace:
            return {"error": "workspace scope unavailable: no workspace configured"}
        normalized_path = path.strip() or "."
        if normalized_path in {".", "./"}:
            workspace = self._require_workspace()
            result = workspace.list_files()
            return {"scope": "workspace", "path": ".", **result}

        workspace_root = self._require_workspace().root_path.resolve()
        resolved = (workspace_root / normalized_path).resolve()
        try:
            resolved.relative_to(workspace_root)
        except ValueError:
            return {"error": f"Access denied: '{path}' is outside the workspace directory"}
        if not resolved.exists():
            return {"error": f"Path not found: {path}"}
        if resolved.is_file():
            return {
                "scope": "workspace",
                "path": normalized_path,
                "type": "file",
                "total_files": 1,
                "files": [{
                    "path": normalized_path,
                    "size_bytes": resolved.stat().st_size,
                }],
            }

        tree_lines: list[str] = [f"{Path(normalized_path).name or normalized_path}/"]
        total_files = 0
        for dirpath, dirnames, filenames in sorted(resolved.walk()):
            rel_dir = dirpath.relative_to(resolved)
            depth = len(rel_dir.parts)
            indent = "  " * depth
            if depth > 0:
                tree_lines.append(f"{indent}{rel_dir.name}/")
            dirnames.sort()
            for filename in sorted(filenames):
                tree_lines.append(f"{indent}  {filename}")
                total_files += 1

        return {
            "scope": "workspace",
            "path": normalized_path,
            "tree": "\n".join(tree_lines),
            "total_files": total_files,
        }

    def _list_docs_files(self, file_type: str | None, top_n: int) -> dict[str, Any]:
        """List indexed documentation/code files."""
        if self._documentation_data_loader is None:
            return {"error": "docs scope unavailable: documentation_data_loader is not configured"}

        normalized_file_type = file_type.strip().lower() if file_type else None
        if normalized_file_type and normalized_file_type not in {"documentation", "code"}:
            return {"error": "file_type must be 'documentation' or 'code'"}

        rows: list[dict[str, Any]] = []
        for entry in self._documentation_data_loader.entries:
            if normalized_file_type and entry.file_type.value != normalized_file_type:
                continue
            rows.append({
                "path": str(entry.path),
                "file_type": entry.file_type.value,
                "title": entry.title,
                "summary": entry.summary,
                "size_bytes": entry.size_bytes,
            })

        rows.sort(key=lambda row: row["path"])
        return {
            "scope": "docs",
            "file_type": normalized_file_type,
            "total_files": len(rows),
            "files": rows[:max(1, top_n)],
        }

    def _search_workspace_files(
        self,
        query: str,
        mode: str,
        case_sensitive: bool,
        top_n: int,
    ) -> dict[str, Any]:
        """Search text files in the workspace."""
        if not self.has_workspace:
            return {"error": "workspace scope unavailable: no workspace configured"}
        if not query:
            return {"error": "query is required"}
        if mode not in {"exact", "terms", "regex"}:
            return {"error": "mode must be 'exact', 'terms', or 'regex'"}

        workspace_root = self._require_workspace().root_path.resolve()
        compiled_regex: re.Pattern[str] | None = None
        if mode == "regex":
            flags = 0 if case_sensitive else re.IGNORECASE
            try:
                compiled_regex = re.compile(query, flags)
            except re.error as e:
                return {"error": f"Invalid regex pattern: {e}"}

        search_query = query if case_sensitive else query.lower()
        terms = _parse_search_terms(query)
        normalized_terms = terms if case_sensitive else [t.lower() for t in terms]

        results: list[dict[str, Any]] = []
        for file_path in workspace_root.rglob("*"):
            if not file_path.is_file():
                continue
            try:
                if file_path.stat().st_size > 512_000:
                    continue
                content = file_path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            if "\x00" in content:
                continue

            score = 0
            matches: list[dict[str, Any]] = []
            for line_number, line in enumerate(content.splitlines(), start=1):
                search_line = line if case_sensitive else line.lower()
                line_hits = 0
                if mode == "exact":
                    line_hits = search_line.count(search_query)
                elif mode == "terms":
                    for term in normalized_terms:
                        line_hits += search_line.count(term)
                else:
                    assert compiled_regex is not None
                    line_hits = len(list(compiled_regex.finditer(line)))

                if line_hits > 0:
                    score += line_hits
                    matches.append({
                        "line_number": line_number,
                        "line_content": line.strip(),
                        "hits": line_hits,
                    })
                if len(matches) >= 10:
                    break

            if score == 0:
                continue
            results.append({
                "path": str(file_path.relative_to(workspace_root)),
                "score": score,
                "matches": matches,
            })

        results.sort(key=lambda row: row["score"], reverse=True)
        capped_results = results[:max(1, top_n)]
        if not capped_results:
            return {"scope": "workspace", "mode": mode, "query": query, "message": f"No matches found for '{query}'"}
        return {
            "scope": "workspace",
            "mode": mode,
            "query": query,
            "files_with_matches": len(capped_results),
            "results": capped_results,
        }

    @agent_tool(
        availability=lambda self: self.has_workspace or self._documentation_data_loader is not None,
        persist=ToolResultPersistMode.NEVER,
        token_optimized=True,
        parameters={
            "type": "object",
            "properties": {
                "scope": {
                    "type": "string",
                    "enum": ["workspace", "docs"],
                    "description": "Where to list files from.",
                },
                "path": {
                    "type": "string",
                    "description": "Workspace path to list (workspace scope only). Defaults to '.'.",
                },
                "file_type": {
                    "type": "string",
                    "enum": ["documentation", "code"],
                    "description": "Optional docs-only file type filter.",
                },
                "top_n": {
                    "type": "integer",
                    "description": "Max docs files to return. Defaults to 200.",
                },
            },
            "required": ["scope"],
        },
    )
    def _list_files(
        self,
        scope: str,
        path: str = ".",
        file_type: str | None = None,
        top_n: int = 200,
    ) -> dict[str, Any]:
        """
        List files from workspace or docs.

        Args:
            scope: Either 'workspace' or 'docs'.
            path: Workspace subpath to list when scope='workspace'. Defaults to '.'.
            file_type: Optional docs-only filter.
            top_n: Max docs entries to return.
        """
        try:
            normalized_scope = _normalize_file_scope(scope)
        except ValueError as e:
            return {"error": str(e)}

        if normalized_scope == "workspace":
            return self._list_workspace_files_scoped(path)
        return self._list_docs_files(file_type=file_type, top_n=top_n)

    @agent_tool(
        availability=lambda self: self.has_workspace or self._documentation_data_loader is not None,
        persist=ToolResultPersistMode.NEVER,
        token_optimized=True,
        parameters={
            "type": "object",
            "properties": {
                "scope": {
                    "type": "string",
                    "enum": ["workspace", "docs"],
                    "description": "Where to read the file from.",
                },
                "path": {
                    "type": "string",
                    "description": "Path to file. Workspace paths are relative to workspace root.",
                },
                "start_line": {
                    "type": "integer",
                    "description": "Optional 1-based start line number.",
                },
                "end_line": {
                    "type": "integer",
                    "description": "Optional 1-based end line number (inclusive).",
                },
            },
            "required": ["scope", "path"],
        },
    )
    def _read_file(
        self,
        scope: str,
        path: str,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> dict[str, Any]:
        """
        Read a file from workspace or docs.

        Args:
            scope: Either 'workspace' or 'docs'.
            path: Path to file (workspace-relative when scope='workspace').
            start_line: Optional 1-based start line number.
            end_line: Optional 1-based end line number (inclusive).
        """
        if not path:
            return {"error": "path is required"}
        try:
            normalized_scope = _normalize_file_scope(scope)
        except ValueError as e:
            return {"error": str(e)}

        if normalized_scope == "workspace":
            if not self.has_workspace:
                return {"error": "workspace scope unavailable: no workspace configured"}
            workspace = self._require_workspace()
            return {"scope": "workspace", **workspace.read_file(path, start_line=start_line, end_line=end_line)}

        if self._documentation_data_loader is None:
            return {"error": "docs scope unavailable: documentation_data_loader is not configured"}

        if start_line is not None or end_line is not None:
            result = self._documentation_data_loader.get_file_lines(path, start_line=start_line, end_line=end_line)
            if result is None:
                return {"error": f"File '{path}' not found"}
            content, total_lines = result
            content_lines = content.count("\n") + (1 if content else 0)
            read_start = start_line or 1
            read_end = read_start + max(content_lines - 1, 0)
            if len(content) > 5_000:
                content = (
                    content[:5_000]
                    + f"\n... [output too large... read lines {read_start} - {read_end}]"
                )
            return {
                "scope": "docs",
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
        if len(content) > 5_000:
            content = content[:5_000] + f"\n... [output too large... read lines 1 - {total_lines}]"
        return {
            "scope": "docs",
            "path": str(entry.path),
            "file_type": entry.file_type.value,
            "title": entry.title,
            "summary": entry.summary,
            "total_lines": total_lines,
            "content": content,
        }

    @agent_tool(
        availability=lambda self: self.has_workspace or self._documentation_data_loader is not None,
        persist=ToolResultPersistMode.OVERFLOW,
        token_optimized=True,
        parameters={
            "type": "object",
            "properties": {
                "scope": {
                    "type": "string",
                    "enum": ["workspace", "docs"],
                    "description": "Where to search files.",
                },
                "query": {
                    "type": "string",
                    "description": "Search query.",
                },
                "mode": {
                    "type": "string",
                    "enum": ["exact", "terms", "regex"],
                    "description": "Search mode. Defaults to exact.",
                },
                "file_type": {
                    "type": "string",
                    "enum": ["documentation", "code"],
                    "description": "Optional docs-only file type filter.",
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Whether matching is case-sensitive. Defaults to false.",
                },
                "top_n": {
                    "type": "integer",
                    "description": "Maximum number of results to return. Defaults to 20.",
                },
            },
            "required": ["scope", "query"],
        },
    )
    def _search_files(
        self,
        scope: str,
        query: str,
        mode: str = "exact",
        file_type: str | None = None,
        case_sensitive: bool = False,
        top_n: int = 20,
    ) -> dict[str, Any]:
        """
        Search files in workspace or docs.

        Args:
            scope: Either 'workspace' or 'docs'.
            query: Query string.
            mode: Search mode: exact, terms, or regex.
            file_type: Optional docs-only file type filter.
            case_sensitive: Whether to match case-sensitively.
            top_n: Maximum number of results to return.
        """
        try:
            normalized_scope = _normalize_file_scope(scope)
        except ValueError as e:
            return {"error": str(e)}
        normalized_mode = mode.strip().lower()

        if normalized_scope == "workspace":
            return self._search_workspace_files(
                query=query,
                mode=normalized_mode,
                case_sensitive=case_sensitive,
                top_n=top_n,
            )

        if self._documentation_data_loader is None:
            return {"error": "docs scope unavailable: documentation_data_loader is not configured"}
        if not query:
            return {"error": "query is required"}

        normalized_file_type = file_type.strip().lower() if file_type else None
        if normalized_file_type and normalized_file_type not in {"documentation", "code"}:
            return {"error": "file_type must be 'documentation' or 'code'"}
        file_type_enum = FileType(normalized_file_type) if normalized_file_type else None

        if normalized_mode == "exact":
            results = self._documentation_data_loader.search_content_with_lines(
                query=query,
                file_type=file_type_enum,
                case_sensitive=case_sensitive,
                max_matches_per_file=10,
            )
            if not results:
                return {"scope": "docs", "mode": "exact", "query": query, "message": f"No matches found for '{query}'"}
            return {
                "scope": "docs",
                "mode": "exact",
                "query": query,
                "case_sensitive": case_sensitive,
                "files_with_matches": len(results),
                "results": results[:max(1, top_n)],
            }

        if normalized_mode == "terms":
            terms = _parse_search_terms(query)
            if not terms:
                return {"error": "query must contain at least one term for terms mode"}
            results = self._documentation_data_loader.search_by_terms(terms=terms, top_n=top_n)
            if normalized_file_type:
                filtered: list[dict[str, Any]] = []
                for row in results:
                    entry_id = row.get("id")
                    if not entry_id:
                        continue
                    entry = self._documentation_data_loader.get_file_by_path(entry_id)
                    if entry and entry.file_type.value == normalized_file_type:
                        filtered.append(row)
                results = filtered
            return {
                "scope": "docs",
                "mode": "terms",
                "query": query,
                "terms": terms,
                "results_count": len(results),
                "results": results,
            }

        if normalized_mode == "regex":
            regex_results = self._documentation_data_loader.search_by_regex(pattern=query, top_n=top_n)
            if normalized_file_type and "matches" in regex_results:
                matches = regex_results.get("matches", [])
                filtered_matches: list[dict[str, Any]] = []
                for match in matches:
                    entry_id = match.get("id")
                    if not entry_id:
                        continue
                    entry = self._documentation_data_loader.get_file_by_path(entry_id)
                    if entry and entry.file_type.value == normalized_file_type:
                        filtered_matches.append(match)
                regex_results["matches"] = filtered_matches
            return {"scope": "docs", "mode": "regex", "query": query, **regex_results}

        return {"error": "mode must be 'exact', 'terms', or 'regex'"}

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

    def _get_workspace_usage_prompt_section(self) -> str:
        """
        Build a concise workspace usage guide for the system prompt.

        Uses class-level WORKSPACE_USAGE_SECTION so specialized agents can
        override the full section text.
        """
        if not self.has_workspace:
            return ""
        section = self.WORKSPACE_USAGE_SECTION.strip()
        mounted_section = self._get_mounted_inputs_prompt_section()
        if not section and not mounted_section:
            return ""

        parts: list[str] = []
        if section:
            parts.append(section)
        if mounted_section:
            parts.append(mounted_section)

        return "\n\n" + "\n\n".join(parts)

    def _get_mounted_inputs_prompt_section(self) -> str:
        """Build a system prompt section listing currently mounted input files."""
        if not self.has_workspace:
            return ""
        workspace = self._require_workspace()
        mounted_inputs = workspace.list_mounted_inputs()
        if not mounted_inputs:
            return ""

        lines = [
            "## Mounted Input Files (read-only)",
            "The following files are mounted under `raw/` and can be read directly without tools:",
            "```python",
            "text = open('raw/<file>', encoding='utf-8').read()",
            "blob = open('raw/<file>', 'rb').read()",
            "```",
        ]
        for item in mounted_inputs:
            lines.append(
                f"- `{item.relative_path}` — {item.size_bytes} bytes (source: `{item.source_path}`)"
            )
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

        # append workspace usage guide (injected centrally for all agents)
        workspace_section = self._get_workspace_usage_prompt_section()
        if workspace_section:
            system_prompt = system_prompt + workspace_section

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
        # Persist status updates as Chat objects
        if isinstance(message, (StatusUpdateEmittedMessage, BrowserAgentStepEmittedMessage)):
            metadata: dict[str, Any] = {}
            if self._current_assistant_chat_id:
                metadata["associated_chat_id"] = self._current_assistant_chat_id
            if isinstance(message, BrowserAgentStepEmittedMessage):
                metadata["source"] = "browser_agent"
                metadata["step_number"] = message.step_number
            self._add_chat(
                role=ChatRole.SYSTEM_STATUS_UPDATE,
                content=message.content,
                metadata=metadata or None,
            )

    def _add_chat(
        self,
        role: ChatRole,
        content: str,
        tool_call_id: str | None = None,
        tool_calls: list[LLMToolCall] | None = None,
        llm_provider_response_id: str | None = None,
        metadata: dict | None = None,
    ) -> Chat:
        """Create and store a new Chat, update thread, persist if callbacks set."""
        # Lazy-persist thread on first chat (avoids creating empty threads on connect)
        if not self._thread_persisted and self._persist_chat_thread_callable:
            self._thread = self._persist_chat_thread_callable(self._thread)
            self._thread_persisted = True

        chat = Chat(
            chat_thread_id=self._thread.id,
            role=role,
            content=content,
            tool_call_id=tool_call_id,
            tool_calls=tool_calls or [],
            llm_provider_response_id=llm_provider_response_id,
            metadata=metadata,
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

        if self._on_chat_added is not None:
            self._on_chat_added(chat)

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
            if chat.role == ChatRole.SYSTEM_STATUS_UPDATE:
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

    ## autonomous loop

    def run_autonomous(
        self,
        task: str,
        config: AutonomousRunConfig | None = None,
        output_schema: dict[str, Any] | None = None,
        output_description: str | None = None,
    ) -> BaseModel | None:
        """
        Run the agent autonomously to completion.

        Subclasses may override autonomous extension hooks for domain-specific
        prompting and completion behavior.
        """
        self.execution_mode = AgentExecutionMode.AUTONOMOUS
        try:
            self._autonomous_iteration = 0
            self._autonomous_config = config or AutonomousRunConfig()

            # Subclasses should clear specialized result fields in overrides.
            self._reset_autonomous_state()

            # Set output schema after reset so it is retained for this run.
            if output_schema:
                self.set_output_schema(output_schema, output_description)

            initial_message = self._get_autonomous_initial_message(task)
            self._add_chat(ChatRole.USER, initial_message)

            logger.info(
                "Starting %s autonomous run for task: %s",
                self.__class__.__name__, task,
            )
            self._run_autonomous_loop()
            return self._get_autonomous_result()
        finally:
            self.execution_mode = AgentExecutionMode.CONVERSATIONAL

    def _run_autonomous_loop(self) -> None:
        """Run the autonomous loop with iteration tracking and finalize gating."""
        max_iterations = self._autonomous_config.max_iterations
        for iteration in range(max_iterations):
            self._autonomous_iteration = iteration + 1
            logger.debug(
                "Autonomous loop iteration %d/%d",
                self._autonomous_iteration,
                max_iterations,
            )

            messages = self._build_messages_for_llm()
            try:
                response = self._call_llm(
                    messages,
                    self._get_autonomous_system_prompt(),
                    tool_choice="required",
                )

                if response.response_id:
                    self._previous_response_id = response.response_id

                if response.content or response.tool_calls:
                    chat = self._add_chat(
                        role=ChatRole.ASSISTANT,
                        content=response.content or "",
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
                    logger.warning(
                        "Autonomous loop: no tool calls in iteration %d (unexpected with tool_choice=required)",
                        self._autonomous_iteration,
                    )
                    return

                for tool_call in response.tool_calls:
                    result_str = self._auto_execute_tool(
                        tool_call.tool_name,
                        tool_call.tool_arguments,
                    )

                    self._add_chat(
                        role=ChatRole.TOOL,
                        content=f"Tool '{tool_call.tool_name}' result: {result_str}",
                        tool_call_id=tool_call.call_id,
                    )

                    if self._check_autonomous_completion(tool_call.tool_name):
                        logger.debug(
                            "Autonomous run completed at iteration %d",
                            self._autonomous_iteration,
                        )
                        return

            except Exception as e:
                logger.exception("Error in autonomous loop: %s", e)
                self._emit_message(ErrorEmittedMessage(error=str(e)))
                return

        logger.warning(
            "Autonomous loop hit max iterations (%d) without finalization",
            max_iterations,
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

                if self._on_llm_response and response.usage:
                    self._on_llm_response(response)

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

                self._current_assistant_chat_id = chat.id
                try:
                    self._process_tool_calls(response.tool_calls)
                finally:
                    self._current_assistant_chat_id = None

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
