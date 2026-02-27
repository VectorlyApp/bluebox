"""
tests/unit/agents/test_abstract_agent.py

Comprehensive unit tests for AbstractAgent base class.

Covers:
  - Initialization and default state
  - Chat management (_add_chat, get_chats, _build_messages_for_llm)
  - Tool infrastructure (_collect_tools, _sync_tools, _execute_tool)
  - Tool execution helpers (_auto_execute_tool, _process_tool_calls)
  - Documentation tools and prompt section
  - _call_llm system prompt injection
  - reset()
  - process_new_message()
  - Response chaining (_previous_response_id)
"""

import json
import tempfile
from pathlib import Path
from typing import Annotated, Any
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel, Field, computed_field

from bluebox.agents.abstract_agent import (
    AbstractAgent,
    AgentCard,
    ToolResultPersistMode,
    agent_tool,
    _ToolMeta,
)
from bluebox.utils.llm_serialization import LLMExclude
from bluebox.workspace import LocalAgentWorkspace
from bluebox.data_models.llms.interaction import (
    Chat,
    ChatRole,
    ChatThread,
    LLMChatResponse,
    LLMToolCall,
    ChatResponseEmittedMessage,
    ErrorEmittedMessage,
    ToolInvocationResultEmittedMessage,
)
from bluebox.data_models.llms.vendors import OpenAIModel
from bluebox.llms.data_loaders.documentation_data_loader import DocumentationDataLoader


# =============================================================================
# Concrete subclass for testing
# =============================================================================


class SearchParams(BaseModel):
    """A Pydantic model used as an agent tool parameter."""
    query: str = Field(description="The search query.")
    max_results: int = Field(default=10, description="Maximum number of results.")
    tags: list[str] = Field(default_factory=list, description="Tags to filter by.")


class TransactionModel(BaseModel):
    """Model with LLMExclude fields for integration testing."""
    url: str
    method: str
    status_code: int
    response_body: Annotated[str, LLMExclude()]
    raw_headers: Annotated[dict, LLMExclude()]


class ComputedModel(BaseModel):
    """Model with a computed field to verify it's preserved."""
    first: str
    last: str
    internal: Annotated[str, LLMExclude()]

    @computed_field
    @property
    def full_name(self) -> str:
        return f"{self.first} {self.last}"


class CleanModel(BaseModel):
    """Model with no LLMExclude annotations."""
    x: str
    y: int


class ConcreteAgent(AbstractAgent):
    """Minimal concrete AbstractAgent for testing."""

    AGENT_CARD = AgentCard(description="Test agent for unit tests.")

    def __init__(self, **kwargs: Any) -> None:
        if "workspace" not in kwargs:
            workspace_dir = Path(tempfile.mkdtemp(prefix="bluebox-abstract-agent-test-"))
            kwargs["workspace"] = LocalAgentWorkspace.from_directory_path(workspace_dir)
        super().__init__(**kwargs)

    def _get_system_prompt(self) -> str:
        return "You are a test agent."

    @agent_tool
    def _echo(self, message: str) -> dict[str, Any]:
        """
        Echo the message back.

        Args:
            message: The message to echo.
        """
        return {"echoed": message}

    @agent_tool
    def _add_numbers(self, a: int, b: int) -> dict[str, Any]:
        """
        Add two numbers.

        Args:
            a: First number.
            b: Second number.
        """
        return {"sum": a + b}

    @agent_tool(availability=False)
    def _disabled_tool(self) -> dict[str, Any]:
        """A permanently disabled tool."""
        return {"should": "never run"}

    @agent_tool(availability=lambda self: getattr(self, "_feature_flag", False))
    def _gated_tool(self) -> dict[str, Any]:
        """A tool gated by a feature flag."""
        return {"gated": True}

    @agent_tool
    def _no_params(self) -> dict[str, Any]:
        """A tool with no parameters."""
        return {"status": "ok"}

    @agent_tool
    def _optional_params(self, required: str, opt: int = 5) -> dict[str, Any]:
        """
        Tool with mixed required and optional params.

        Args:
            required: A required string.
            opt: An optional integer.
        """
        return {"required": required, "opt": opt}

    @agent_tool
    def _raises_error(self) -> dict[str, Any]:
        """A tool that raises an exception."""
        raise RuntimeError("intentional test error")

    @agent_tool
    def _search(self, params: SearchParams) -> dict[str, Any]:
        """
        Search with structured params.

        Args:
            params: The search parameters.
        """
        # params should arrive as a SearchParams instance, not a raw dict
        return {
            "is_model": isinstance(params, SearchParams),
            "query": params.query,
            "max_results": params.max_results,
            "tags": params.tags,
        }

    @agent_tool(persist=ToolResultPersistMode.ALWAYS)
    def _persist_always(self, text: str = "ok") -> dict[str, Any]:
        """
        Tool that always persists results.

        Args:
            text: Value to return in the payload.
        """
        return {"text": text}

    @agent_tool(
        persist=ToolResultPersistMode.OVERFLOW,
        max_characters=50,
    )
    def _persist_overflow(self, text: str) -> dict[str, Any]:
        """
        Tool that persists only when output exceeds max_characters.

        Args:
            text: Payload content.
        """
        return {"text": text}

    @agent_tool(persist=ToolResultPersistMode.NEVER)
    def _persist_never(self, text: str = "ok") -> dict[str, Any]:
        """
        Tool that never persists results.

        Args:
            text: Value to return in the payload.
        """
        return {"text": text}

    @agent_tool(
        persist=ToolResultPersistMode.ALWAYS,
        token_optimized=True,
    )
    def _persist_always_token_optimized(self) -> dict[str, Any]:
        """Always persists raw result, returns token-optimized payload."""
        return {"status": "ok", "items": [1, 2, 3]}

    @agent_tool(token_optimized=True)
    def _token_optimized_no_persist(self) -> dict[str, Any]:
        """Token-optimized tool with no persistence."""
        return {"status": "ok"}

    @agent_tool
    def _get_transaction(self) -> dict[str, Any]:
        """
        Return a transaction model with LLMExclude fields.
        """
        tx = TransactionModel(
            url="https://api.example.com/data",
            method="GET",
            status_code=200,
            response_body='{"huge": "blob"}',
            raw_headers={"x-internal": "secret"},
        )
        return {"transaction": tx, "count": 1}

    @agent_tool
    def _get_transaction_direct(self) -> TransactionModel:
        """
        Return a TransactionModel directly (not wrapped in a dict).
        """
        return TransactionModel(
            url="https://example.com",
            method="POST",
            status_code=201,
            response_body="big",
            raw_headers={"h": "v"},
        )

    @agent_tool
    def _get_computed(self) -> ComputedModel:
        """
        Return a model with a computed field.
        """
        return ComputedModel(first="Jane", last="Doe", internal="hidden")

    @agent_tool
    def _get_transaction_list(self) -> dict[str, Any]:
        """
        Return a list of models with LLMExclude fields.
        """
        txs = [
            TransactionModel(
                url=f"https://api.example.com/{i}",
                method="GET",
                status_code=200,
                response_body=f"body_{i}",
                raw_headers={"h": str(i)},
            )
            for i in range(3)
        ]
        return {"transactions": txs}

    @agent_tool
    def _get_clean_model(self) -> CleanModel:
        """
        Return a model with no LLMExclude annotations.
        """
        return CleanModel(x="visible", y=42)


@pytest.fixture
def mock_emit() -> MagicMock:
    return MagicMock()


@pytest.fixture
def agent(mock_emit: MagicMock) -> ConcreteAgent:
    """Agent without documentation data loader."""
    return ConcreteAgent(emit_message_callable=mock_emit)


@pytest.fixture
def agent_with_docs(mock_emit: MagicMock, tmp_path: Path) -> ConcreteAgent:
    """Agent with a documentation data loader containing real files."""
    # Create some test documentation files
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "guide.md").write_text(
        "# User Guide\n\n> A guide for users.\n\nThis guide covers setup and usage.\n\n"
        "## Installation\n\nRun `pip install bluebox`.\n\n"
        "## Configuration\n\nSet your API key in `.env`.\n"
    )
    (docs_dir / "api.md").write_text(
        "# API Reference\n\n> API docs for developers.\n\n"
        "## Endpoints\n\n### GET /health\n\nReturns health status.\n\n"
        "### POST /search\n\nSearch for routines.\n"
    )

    # Create some test code files
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    (code_dir / "main.py").write_text(
        '"""Main module for the application."""\n\n'
        "def hello() -> str:\n"
        '    return "Hello, World!"\n\n'
        "def search(query: str) -> list[str]:\n"
        '    """Search for items matching query."""\n'
        "    return []\n"
    )

    loader = DocumentationDataLoader(
        documentation_paths=[str(docs_dir)],
        code_paths=[str(code_dir)],
    )
    return ConcreteAgent(
        emit_message_callable=mock_emit,
        documentation_data_loader=loader,
    )


@pytest.fixture
def agent_docs_only(mock_emit: MagicMock, tmp_path: Path) -> ConcreteAgent:
    """Agent with docs loader but no workspace configured."""
    docs_dir = tmp_path / "docs_only"
    docs_dir.mkdir()
    (docs_dir / "guide.md").write_text("# Docs Only\n\ncontent here\n")
    loader = DocumentationDataLoader(documentation_paths=[str(docs_dir)])
    return ConcreteAgent(
        emit_message_callable=mock_emit,
        workspace=None,
        documentation_data_loader=loader,
    )


# =============================================================================
# Initialization
# =============================================================================


class TestInitialization:
    """Tests for AbstractAgent.__init__."""

    def test_default_state(self, agent: ConcreteAgent) -> None:
        """Agent initializes with expected default state."""
        assert agent._previous_response_id is None
        assert agent._response_id_to_chat_index == {}
        assert agent.llm_model == OpenAIModel.GPT_5_2
        assert agent._documentation_data_loader is None
        assert isinstance(agent._thread, ChatThread)
        assert agent._chats == {}

    def test_empty_chat_list_initially(self, agent: ConcreteAgent) -> None:
        assert agent.get_chats() == []

    def test_thread_id_is_string(self, agent: ConcreteAgent) -> None:
        assert isinstance(agent.chat_thread_id, str)
        assert len(agent.chat_thread_id) > 0

    def test_existing_chats_loaded(self, mock_emit: MagicMock) -> None:
        """Existing chats are indexed on init."""
        thread = ChatThread()
        chat = Chat(chat_thread_id=thread.id, role=ChatRole.USER, content="hello")
        thread.chat_ids.append(chat.id)

        agent = ConcreteAgent(
            emit_message_callable=mock_emit,
            chat_thread=thread,
            existing_chats=[chat],
        )
        chats = agent.get_chats()
        assert len(chats) == 1
        assert chats[0].content == "hello"

    def test_persist_thread_not_called_on_init(self, mock_emit: MagicMock) -> None:
        """persist_chat_thread_callable is NOT called during __init__ (lazy persist)."""
        mock_persist = MagicMock(side_effect=lambda t: t)
        ConcreteAgent(
            emit_message_callable=mock_emit,
            persist_chat_thread_callable=mock_persist,
        )
        assert mock_persist.call_count == 0

    def test_persist_thread_called_on_first_chat(self, mock_emit: MagicMock) -> None:
        """persist_chat_thread_callable is called lazily on the first _add_chat."""
        mock_persist = MagicMock(side_effect=lambda t: t)
        agent = ConcreteAgent(
            emit_message_callable=mock_emit,
            persist_chat_thread_callable=mock_persist,
        )
        agent._add_chat(role=ChatRole.USER, content="hello")
        # Called twice: once for lazy thread create, once for thread update after chat
        assert mock_persist.call_count == 2

    def test_persist_thread_not_called_with_existing_thread(self, mock_emit: MagicMock) -> None:
        """persist_chat_thread_callable is NOT called when thread is provided."""
        mock_persist = MagicMock(side_effect=lambda t: t)
        ConcreteAgent(
            emit_message_callable=mock_emit,
            persist_chat_thread_callable=mock_persist,
            chat_thread=ChatThread(),
        )
        assert mock_persist.call_count == 0

    def test_documentation_data_loader_stored(self, agent_with_docs: ConcreteAgent) -> None:
        assert agent_with_docs._documentation_data_loader is not None

    def test_workspace_attached_on_init(self, agent: ConcreteAgent) -> None:
        assert isinstance(agent._workspace, LocalAgentWorkspace)

    def test_workspace_can_be_omitted(self, mock_emit: MagicMock) -> None:
        agent = ConcreteAgent(emit_message_callable=mock_emit, workspace=None)
        assert agent._workspace is None
        assert agent.has_workspace is False

    def test_code_execution_disabled_by_default(self, agent: ConcreteAgent) -> None:
        assert agent._allow_code_execution is False
        assert agent._code_execution_globals == {}

    def test_code_execution_globals_require_enabled(self, mock_emit: MagicMock) -> None:
        with pytest.raises(ValueError, match="code_execution_globals must be empty"):
            ConcreteAgent(
                emit_message_callable=mock_emit,
                allow_code_execution=False,
                code_execution_globals={"x": 1},
            )

    def test_code_execution_globals_stored_when_enabled(self, mock_emit: MagicMock) -> None:
        configured = {"items": [1, 2, 3], "name": "demo"}
        agent = ConcreteAgent(
            emit_message_callable=mock_emit,
            allow_code_execution=True,
            code_execution_globals=configured,
        )
        assert agent._allow_code_execution is True
        assert agent._code_execution_globals == configured

    def test_code_execution_prompt_includes_compute_only_notice_without_workspace(
        self,
        mock_emit: MagicMock,
    ) -> None:
        agent = ConcreteAgent(
            emit_message_callable=mock_emit,
            workspace=None,
            allow_code_execution=True,
            code_execution_globals={},
        )
        section = agent._generate_code_execution_prompt()
        assert "Code Execution Environment" in section
        assert "compute-only" in section
        assert "open()" in section


# =============================================================================
# Chat management
# =============================================================================


class TestChatManagement:
    """Tests for _add_chat, get_chats, get_thread."""

    def test_add_chat_creates_chat(self, agent: ConcreteAgent) -> None:
        chat = agent._add_chat(ChatRole.USER, "hello")
        assert chat.role == ChatRole.USER
        assert chat.content == "hello"
        assert chat.chat_thread_id == agent.chat_thread_id

    def test_add_chat_updates_thread(self, agent: ConcreteAgent) -> None:
        chat = agent._add_chat(ChatRole.USER, "hello")
        assert chat.id in agent._thread.chat_ids

    def test_get_chats_returns_ordered(self, agent: ConcreteAgent) -> None:
        agent._add_chat(ChatRole.USER, "first")
        agent._add_chat(ChatRole.ASSISTANT, "second")
        agent._add_chat(ChatRole.USER, "third")

        chats = agent.get_chats()
        assert len(chats) == 3
        assert [c.content for c in chats] == ["first", "second", "third"]

    def test_add_chat_with_tool_call_id(self, agent: ConcreteAgent) -> None:
        chat = agent._add_chat(ChatRole.TOOL, "result", tool_call_id="call_123")
        assert chat.tool_call_id == "call_123"

    def test_add_chat_with_tool_calls(self, agent: ConcreteAgent) -> None:
        tool_calls = [LLMToolCall(tool_name="echo", tool_arguments={"message": "hi"}, call_id="c1")]
        chat = agent._add_chat(ChatRole.ASSISTANT, "thinking...", tool_calls=tool_calls)
        assert len(chat.tool_calls) == 1
        assert chat.tool_calls[0].tool_name == "echo"

    def test_persist_chat_called(self, mock_emit: MagicMock) -> None:
        mock_persist = MagicMock(side_effect=lambda c: c)
        agent = ConcreteAgent(
            emit_message_callable=mock_emit,
            persist_chat_callable=mock_persist,
        )
        agent._add_chat(ChatRole.USER, "test")
        mock_persist.assert_called_once()

    def test_response_id_tracking(self, agent: ConcreteAgent) -> None:
        """Assistant messages with response_id are tracked for response chaining."""
        agent._add_chat(
            ChatRole.ASSISTANT, "reply",
            llm_provider_response_id="resp_123",
        )
        assert "resp_123" in agent._response_id_to_chat_index

    def test_non_assistant_response_id_not_tracked(self, agent: ConcreteAgent) -> None:
        """Only ASSISTANT messages have response_id tracked."""
        agent._add_chat(
            ChatRole.USER, "hello",
            llm_provider_response_id="resp_456",
        )
        assert "resp_456" not in agent._response_id_to_chat_index

    def test_get_thread_returns_thread(self, agent: ConcreteAgent) -> None:
        thread = agent.get_thread()
        assert isinstance(thread, ChatThread)
        assert thread.id == agent.chat_thread_id


# =============================================================================
# _build_messages_for_llm
# =============================================================================


class TestBuildMessagesForLLM:
    """Tests for _build_messages_for_llm."""

    def test_empty_when_no_chats(self, agent: ConcreteAgent) -> None:
        assert agent._build_messages_for_llm() == []

    def test_includes_all_chats(self, agent: ConcreteAgent) -> None:
        agent._add_chat(ChatRole.USER, "hello")
        agent._add_chat(ChatRole.ASSISTANT, "hi there")

        messages = agent._build_messages_for_llm()
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "hello"
        assert messages[1]["role"] == "assistant"

    def test_includes_tool_call_id(self, agent: ConcreteAgent) -> None:
        agent._add_chat(ChatRole.TOOL, "result", tool_call_id="call_abc")
        messages = agent._build_messages_for_llm()
        assert messages[0]["tool_call_id"] == "call_abc"

    def test_includes_tool_calls(self, agent: ConcreteAgent) -> None:
        tool_calls = [LLMToolCall(tool_name="echo", tool_arguments={"message": "x"}, call_id="c1")]
        agent._add_chat(ChatRole.ASSISTANT, "", tool_calls=tool_calls)

        messages = agent._build_messages_for_llm()
        assert "tool_calls" in messages[0]
        assert messages[0]["tool_calls"][0]["name"] == "echo"

    def test_response_chaining_truncates_old_messages(self, agent: ConcreteAgent) -> None:
        """When _previous_response_id is set, only messages after that response are included."""
        agent._add_chat(ChatRole.USER, "first")
        agent._add_chat(ChatRole.ASSISTANT, "first reply", llm_provider_response_id="resp_1")

        # Set the response id (simulates LLM response chaining)
        agent._previous_response_id = "resp_1"

        agent._add_chat(ChatRole.USER, "second")
        agent._add_chat(ChatRole.ASSISTANT, "second reply", llm_provider_response_id="resp_2")

        messages = agent._build_messages_for_llm()
        # Only chats after resp_1's index should be included
        contents = [m["content"] for m in messages]
        assert "first" not in contents
        assert "first reply" not in contents
        assert "second" in contents

    def test_tool_call_id_auto_generated_when_missing(self, agent: ConcreteAgent) -> None:
        """Tool calls without call_id get auto-generated IDs."""
        tool_calls = [LLMToolCall(tool_name="echo", tool_arguments={"message": "x"})]
        agent._add_chat(ChatRole.ASSISTANT, "", tool_calls=tool_calls)

        messages = agent._build_messages_for_llm()
        call_id = messages[0]["tool_calls"][0]["call_id"]
        assert call_id.startswith("call_0_")


# =============================================================================
# _collect_tools
# =============================================================================


class TestCollectTools:
    """Tests for _collect_tools classmethod."""

    def test_finds_all_decorated_methods(self) -> None:
        tools = ConcreteAgent._collect_tools()
        tool_names = {meta.name for meta, _ in tools}

        expected = {
            "echo", "add_numbers", "disabled_tool", "gated_tool",
            "no_params", "optional_params", "raises_error", "search",
            "persist_always", "persist_overflow", "persist_never", "persist_always_token_optimized",
            "token_optimized_no_persist",
            # LLMExclude integration test tools
            "get_transaction", "get_transaction_direct", "get_computed",
            "get_transaction_list", "get_clean_model",
            "add_note", "finalize_with_output", "finalize_with_failure", "finalize_result", "finalize_failure",
            "execute_python",
            # Unified file tools from AbstractAgent
            "list_files", "read_file", "search_files",
        }
        assert tool_names == expected

    def test_result_is_cached_tuple(self) -> None:
        ConcreteAgent._collect_tools.cache_clear()
        tools1 = ConcreteAgent._collect_tools()
        tools2 = ConcreteAgent._collect_tools()
        assert tools1 is tools2
        assert isinstance(tools1, tuple)

    def test_subclass_has_separate_cache(self) -> None:
        """A subclass with additional tools has its own cache entry."""

        class ExtendedAgent(ConcreteAgent):
            @agent_tool
            def _extra(self) -> dict[str, Any]:
                """An extra tool."""
                return {}

        parent_names = {m.name for m, _ in ConcreteAgent._collect_tools()}
        child_names = {m.name for m, _ in ExtendedAgent._collect_tools()}
        assert "extra" not in parent_names
        assert "extra" in child_names


# =============================================================================
# _sync_tools
# =============================================================================


class TestSyncTools:
    """Tests for _sync_tools."""

    def test_registers_available_tools(self, agent: ConcreteAgent) -> None:
        agent._sync_tools()
        assert "echo" in agent._registered_tool_names
        assert "add_numbers" in agent._registered_tool_names
        assert "no_params" in agent._registered_tool_names

    def test_skips_unavailable_tools(self, agent: ConcreteAgent) -> None:
        agent._sync_tools()
        assert "disabled_tool" not in agent._registered_tool_names

    def test_callable_availability_false(self, agent: ConcreteAgent) -> None:
        """Gated tool not registered when feature flag is off."""
        agent._feature_flag = False
        agent._sync_tools()
        assert "gated_tool" not in agent._registered_tool_names

    def test_callable_availability_true(self, agent: ConcreteAgent) -> None:
        """Gated tool registered when feature flag is on."""
        agent._feature_flag = True
        agent._sync_tools()
        assert "gated_tool" in agent._registered_tool_names

    def test_file_tools_registered_without_loader(self, agent: ConcreteAgent) -> None:
        """Unified file tools are registered when workspace is available."""
        agent._sync_tools()
        assert "list_files" in agent._registered_tool_names
        assert "read_file" in agent._registered_tool_names
        assert "search_files" in agent._registered_tool_names

    def test_file_tools_registered_with_loader(self, agent_with_docs: ConcreteAgent) -> None:
        """Unified file tools are also registered when docs loader is available."""
        agent_with_docs._sync_tools()
        assert "list_files" in agent_with_docs._registered_tool_names
        assert "read_file" in agent_with_docs._registered_tool_names
        assert "search_files" in agent_with_docs._registered_tool_names

    def test_file_tools_not_registered_without_workspace_or_loader(self, mock_emit: MagicMock) -> None:
        no_workspace_agent = ConcreteAgent(emit_message_callable=mock_emit, workspace=None)
        no_workspace_agent._sync_tools()
        assert "list_files" not in no_workspace_agent._registered_tool_names
        assert "read_file" not in no_workspace_agent._registered_tool_names
        assert "search_files" not in no_workspace_agent._registered_tool_names

    def test_file_tools_registered_with_loader_and_no_workspace(self, agent_docs_only: ConcreteAgent) -> None:
        agent_docs_only._sync_tools()
        assert "list_files" in agent_docs_only._registered_tool_names
        assert "read_file" in agent_docs_only._registered_tool_names
        assert "search_files" in agent_docs_only._registered_tool_names

    def test_execute_python_not_registered_when_disabled(self, agent: ConcreteAgent) -> None:
        agent._sync_tools()
        assert "execute_python" not in agent._registered_tool_names

    def test_execute_python_registered_when_enabled(self, mock_emit: MagicMock) -> None:
        code_agent = ConcreteAgent(
            emit_message_callable=mock_emit,
            allow_code_execution=True,
            code_execution_globals={},
        )
        code_agent._sync_tools()
        assert "execute_python" in code_agent._registered_tool_names

    def test_execute_python_registered_without_workspace_when_enabled(self, mock_emit: MagicMock) -> None:
        code_agent = ConcreteAgent(
            emit_message_callable=mock_emit,
            workspace=None,
            allow_code_execution=True,
            code_execution_globals={},
        )
        code_agent._sync_tools()
        assert "execute_python" in code_agent._registered_tool_names

    def test_sync_clears_and_re_registers(self, agent: ConcreteAgent) -> None:
        """Calling _sync_tools multiple times doesn't duplicate registrations."""
        agent._sync_tools()
        count1 = len(agent._registered_tool_names)
        agent._sync_tools()
        count2 = len(agent._registered_tool_names)
        assert count1 == count2


# =============================================================================
# _execute_tool
# =============================================================================


class TestExecuteTool:
    """Tests for _execute_tool."""

    # --- Success cases ---

    def test_execute_with_required_param(self, agent: ConcreteAgent) -> None:
        result = agent._execute_tool("echo", {"message": "hi"})
        assert result == {"echoed": "hi"}

    def test_execute_with_multiple_params(self, agent: ConcreteAgent) -> None:
        result = agent._execute_tool("add_numbers", {"a": 3, "b": 7})
        assert result == {"sum": 10}

    def test_execute_no_params(self, agent: ConcreteAgent) -> None:
        result = agent._execute_tool("no_params", {})
        assert result == {"status": "ok"}

    def test_execute_with_optional_param_omitted(self, agent: ConcreteAgent) -> None:
        result = agent._execute_tool("optional_params", {"required": "hello"})
        assert result == {"required": "hello", "opt": 5}

    def test_execute_with_optional_param_provided(self, agent: ConcreteAgent) -> None:
        result = agent._execute_tool("optional_params", {"required": "hello", "opt": 99})
        assert result == {"required": "hello", "opt": 99}

    # --- Error cases ---

    def test_unknown_tool(self, agent: ConcreteAgent) -> None:
        result = agent._execute_tool("nonexistent", {})
        assert "error" in result
        assert "Unknown tool" in result["error"]

    def test_unavailable_tool(self, agent: ConcreteAgent) -> None:
        result = agent._execute_tool("disabled_tool", {})
        assert "error" in result
        assert "not currently available" in result["error"]

    def test_execute_python_unavailable_when_disabled(self, agent: ConcreteAgent) -> None:
        result = agent._execute_tool("execute_python", {"code": "print('hello')"})
        assert "error" in result
        assert "not currently available" in result["error"]

    def test_missing_required_param(self, agent: ConcreteAgent) -> None:
        result = agent._execute_tool("echo", {})
        assert "error" in result
        assert "Missing required parameter" in result["error"]
        assert "message" in result["error"]

    def test_extra_param(self, agent: ConcreteAgent) -> None:
        result = agent._execute_tool("no_params", {"bogus": 42})
        assert "error" in result
        assert "Unknown parameter" in result["error"]

    def test_wrong_type(self, agent: ConcreteAgent) -> None:
        result = agent._execute_tool("add_numbers", {"a": "not_int", "b": 2})
        assert "error" in result
        assert "Invalid argument type" in result["error"]

    # --- Pydantic model coercion ---

    def test_pydantic_model_coerced_from_dict(self, agent: ConcreteAgent) -> None:
        """A dict matching a Pydantic model's schema should be coerced into the model instance."""
        result = agent._execute_tool("search", {
            "params": {"query": "bluebox", "max_results": 5, "tags": ["api"]},
        })
        assert result["is_model"] is True
        assert result["query"] == "bluebox"
        assert result["max_results"] == 5
        assert result["tags"] == ["api"]

    def test_pydantic_model_defaults_applied(self, agent: ConcreteAgent) -> None:
        """Pydantic defaults should fill in when fields are omitted from the dict."""
        result = agent._execute_tool("search", {
            "params": {"query": "test"},  # max_results and tags use defaults
        })
        assert result["is_model"] is True
        assert result["max_results"] == 10
        assert result["tags"] == []

    def test_pydantic_model_validation_error(self, agent: ConcreteAgent) -> None:
        """Invalid data for the Pydantic model should return a validation error."""
        result = agent._execute_tool("search", {
            "params": {"query": 12345},  # query should be str
        })
        assert "error" in result
        assert "Invalid argument type" in result["error"]

    def test_pydantic_model_missing_required_field(self, agent: ConcreteAgent) -> None:
        """Missing required fields in the Pydantic model dict should fail validation."""
        result = agent._execute_tool("search", {
            "params": {},  # query is required
        })
        assert "error" in result

    def test_primitives_still_pass_through(self, agent: ConcreteAgent) -> None:
        """Primitive types should still work after the coercion change."""
        result = agent._execute_tool("echo", {"message": "hello"})
        assert result == {"echoed": "hello"}

        result = agent._execute_tool("add_numbers", {"a": 3, "b": 7})
        assert result == {"sum": 10}

    def test_execute_python_uses_configured_globals(
        self,
        mock_emit: MagicMock,
        tmp_path: Path,
    ) -> None:
        code_agent = ConcreteAgent(
            emit_message_callable=mock_emit,
            allow_code_execution=True,
            code_execution_globals={"value": 7},
        )
        code_agent._workspace = LocalAgentWorkspace.from_directory_path(tmp_path / "workspace")
        result = code_agent._execute_tool("execute_python", {"code": "print(value + 5)"})
        assert "error" not in result
        assert "12" in result.get("output", "")

    def test_execute_python_marks_raw_and_meta_read_only(
        self,
        mock_emit: MagicMock,
        tmp_path: Path,
    ) -> None:
        code_agent = ConcreteAgent(
            emit_message_callable=mock_emit,
            allow_code_execution=True,
            code_execution_globals={},
        )
        code_agent._workspace = LocalAgentWorkspace.from_directory_path(tmp_path / "workspace")

        with patch(
            "bluebox.agents.abstract_agent.execute_python_sandboxed",
            return_value={"output": "(no output)"},
        ) as mock_exec:
            result = code_agent._execute_tool("execute_python", {"code": "print('ok')"})

        assert "error" not in result
        kwargs = mock_exec.call_args.kwargs
        read_only_paths = kwargs["read_only_paths"]
        assert str((code_agent._workspace.root_path / "raw").resolve()) in read_only_paths
        assert str((code_agent._workspace.root_path / "meta").resolve()) in read_only_paths

    def test_execute_python_without_workspace_uses_compute_only_sandbox(self, mock_emit: MagicMock) -> None:
        code_agent = ConcreteAgent(
            emit_message_callable=mock_emit,
            workspace=None,
            allow_code_execution=True,
            code_execution_globals={"value": 7},
        )
        with patch(
            "bluebox.agents.abstract_agent.execute_python_sandboxed",
            return_value={"output": "7"},
        ) as mock_exec:
            result = code_agent._execute_tool("execute_python", {"code": "print(value)"})

        assert "error" not in result
        kwargs = mock_exec.call_args.kwargs
        assert kwargs["code"] == "print(value)"
        assert kwargs["extra_globals"] == {"value": 7}
        assert "work_dir" not in kwargs
        assert "read_only_paths" not in kwargs

    def test_persist_always_wraps_result_and_saves_artifact(self, agent: ConcreteAgent) -> None:
        result = agent._execute_tool("persist_always", {"text": "small"})
        assert result["persist_mode"] == "always"
        assert result["artifact_id"].startswith("a_")
        assert "artifact_path" in result
        assert "preview" in result
        artifacts = agent._workspace.list_artifacts(source="raw")
        assert len(artifacts) == 1
        assert artifacts[0].artifact_id == result["artifact_id"]

    def test_persist_overflow_under_limit_returns_raw_result(self, agent: ConcreteAgent) -> None:
        result = agent._execute_tool("persist_overflow", {"text": "short"})
        assert result == {"text": "short"}
        artifacts = agent._workspace.list_artifacts(source="raw")
        assert artifacts == []

    def test_persist_overflow_over_limit_wraps_and_saves_artifact(self, agent: ConcreteAgent) -> None:
        result = agent._execute_tool("persist_overflow", {"text": "x" * 200})
        assert result["persist_mode"] == "overflow"
        assert result["artifact_id"].startswith("a_")
        assert result["truncated"] is True
        assert "truncated" in result["preview"]
        artifacts = agent._workspace.list_artifacts(source="raw")
        assert len(artifacts) == 1
        assert artifacts[0].artifact_id == result["artifact_id"]

    def test_persist_never_returns_raw_and_does_not_save(self, agent: ConcreteAgent) -> None:
        result = agent._execute_tool("persist_never", {"text": "abc"})
        assert result == {"text": "abc"}
        artifacts = agent._workspace.list_artifacts(source="raw")
        assert artifacts == []

    def test_persist_always_returns_raw_without_workspace(self, mock_emit: MagicMock) -> None:
        no_workspace_agent = ConcreteAgent(emit_message_callable=mock_emit, workspace=None)
        result = no_workspace_agent._execute_tool("persist_always", {"text": "small"})
        assert result == {"text": "small"}

    def test_workspace_scope_returns_error_without_workspace_when_docs_tools_enabled(
        self,
        agent_docs_only: ConcreteAgent,
    ) -> None:
        result = agent_docs_only._execute_tool(
            "read_file",
            {"scope": "workspace", "path": "output/x.txt"},
        )
        assert isinstance(result, str)
        assert "workspace scope unavailable" in result

    def test_docs_scope_works_without_workspace(self, agent_docs_only: ConcreteAgent) -> None:
        result = agent_docs_only._execute_tool(
            "read_file",
            {"scope": "docs", "path": "guide.md"},
        )
        assert isinstance(result, str)
        assert "Docs Only" in result

    def test_token_optimized_persist_saves_raw_not_encoded(
        self,
        agent: ConcreteAgent,
    ) -> None:
        result = agent._execute_tool("persist_always_token_optimized", {})
        assert isinstance(result, str)
        assert "_token_optimized_note" in result

        artifacts = agent._workspace.list_artifacts(source="raw")
        assert len(artifacts) == 1
        artifact_id = artifacts[0].artifact_id
        stored = agent._workspace.read_artifact(artifact_id)
        assert "error" not in stored
        assert '"status": "ok"' in stored["content"]

    def test_token_optimized_without_persist_has_no_note(self, agent: ConcreteAgent) -> None:
        result = agent._execute_tool("token_optimized_no_persist", {})
        assert isinstance(result, str)
        assert "_token_optimized_note" not in result


# =============================================================================
# _execute_tool — LLMExclude stripping
# =============================================================================


class TestExecuteToolLLMExclude:
    """Tests that _execute_tool strips LLMExclude-annotated fields from tool results."""

    def test_model_in_dict_strips_excluded_fields(self, agent: ConcreteAgent) -> None:
        """Model returned inside a dict has LLMExclude fields removed."""
        result = agent._execute_tool("get_transaction", {})
        tx = result["transaction"]
        assert tx["url"] == "https://api.example.com/data"
        assert tx["method"] == "GET"
        assert tx["status_code"] == 200
        assert "response_body" not in tx
        assert "raw_headers" not in tx
        # non-model values in the dict are preserved
        assert result["count"] == 1

    def test_model_returned_directly_strips_excluded(self, agent: ConcreteAgent) -> None:
        """Model returned as the top-level result is converted to dict with exclusions."""
        result = agent._execute_tool("get_transaction_direct", {})
        assert isinstance(result, dict)
        assert result["url"] == "https://example.com"
        assert result["method"] == "POST"
        assert result["status_code"] == 201
        assert "response_body" not in result
        assert "raw_headers" not in result

    def test_computed_field_preserved(self, agent: ConcreteAgent) -> None:
        """@computed_field values are included in stripped output."""
        result = agent._execute_tool("get_computed", {})
        assert result["first"] == "Jane"
        assert result["last"] == "Doe"
        assert result["full_name"] == "Jane Doe"
        assert "internal" not in result

    def test_list_of_models_strips_all(self, agent: ConcreteAgent) -> None:
        """List of models inside a dict — each model has exclusions stripped."""
        result = agent._execute_tool("get_transaction_list", {})
        txs = result["transactions"]
        assert len(txs) == 3
        for i, tx in enumerate(txs):
            assert tx["url"] == f"https://api.example.com/{i}"
            assert "response_body" not in tx
            assert "raw_headers" not in tx

    def test_clean_model_unchanged(self, agent: ConcreteAgent) -> None:
        """Model with no LLMExclude annotations returns all fields."""
        result = agent._execute_tool("get_clean_model", {})
        assert result == {"x": "visible", "y": 42}

    def test_auto_execute_tool_strips_excluded(
        self, agent: ConcreteAgent, mock_emit: MagicMock,
    ) -> None:
        """LLMExclude stripping also works through _auto_execute_tool path."""
        result_json = agent._auto_execute_tool("get_transaction_direct", {})
        result = json.loads(result_json)
        assert result["url"] == "https://example.com"
        assert "response_body" not in result
        assert "raw_headers" not in result


# =============================================================================
# _auto_execute_tool
# =============================================================================


class TestAutoExecuteTool:
    """Tests for _auto_execute_tool."""

    def test_success_returns_json(self, agent: ConcreteAgent) -> None:
        result_str = agent._auto_execute_tool("echo", {"message": "hi"})
        result = json.loads(result_str)
        assert result == {"echoed": "hi"}

    def test_success_emits_message(self, agent: ConcreteAgent, mock_emit: MagicMock) -> None:
        agent._auto_execute_tool("echo", {"message": "hi"})
        # Find the ToolInvocationResultEmittedMessage among emitted messages
        tool_msgs = [
            c for c in mock_emit.call_args_list
            if isinstance(c[0][0], ToolInvocationResultEmittedMessage)
        ]
        assert len(tool_msgs) == 1
        assert tool_msgs[0][0][0].tool_result == {"echoed": "hi"}

    def test_exception_returns_error_json(self, agent: ConcreteAgent) -> None:
        result_str = agent._auto_execute_tool("raises_error", {})
        result = json.loads(result_str)
        assert "error" in result
        assert "intentional test error" in result["error"]

    def test_exception_emits_failed_message(self, agent: ConcreteAgent, mock_emit: MagicMock) -> None:
        agent._auto_execute_tool("raises_error", {})
        tool_msgs = [
            c for c in mock_emit.call_args_list
            if isinstance(c[0][0], ToolInvocationResultEmittedMessage)
        ]
        assert len(tool_msgs) == 1
        assert "error" in tool_msgs[0][0][0].tool_result


# =============================================================================
# _process_tool_calls
# =============================================================================


class TestProcessToolCalls:
    """Tests for _process_tool_calls."""

    def test_single_tool_call(self, agent: ConcreteAgent) -> None:
        """Single tool call is executed and added to chat history."""
        tool_call = LLMToolCall(tool_name="echo", tool_arguments={"message": "hi"}, call_id="c1")
        agent._process_tool_calls([tool_call])

        chats = agent.get_chats()
        assert len(chats) == 1
        assert chats[0].role == ChatRole.TOOL
        assert chats[0].tool_call_id == "c1"
        assert '"echoed"' in chats[0].content

    def test_multiple_tool_calls_parallel(self, agent: ConcreteAgent) -> None:
        """Multiple tool calls execute in parallel and results are in original order."""
        calls = [
            LLMToolCall(tool_name="echo", tool_arguments={"message": "first"}, call_id="c1"),
            LLMToolCall(tool_name="echo", tool_arguments={"message": "second"}, call_id="c2"),
            LLMToolCall(tool_name="add_numbers", tool_arguments={"a": 1, "b": 2}, call_id="c3"),
        ]
        agent._process_tool_calls(calls)

        chats = agent.get_chats()
        assert len(chats) == 3
        # Results are added in original call order
        assert chats[0].tool_call_id == "c1"
        assert chats[1].tool_call_id == "c2"
        assert chats[2].tool_call_id == "c3"


# =============================================================================
# reset()
# =============================================================================


class TestReset:
    """Tests for reset()."""

    def test_clears_chats(self, agent: ConcreteAgent) -> None:
        agent._add_chat(ChatRole.USER, "hello")
        assert len(agent.get_chats()) == 1
        agent.reset()
        assert len(agent.get_chats()) == 0

    def test_creates_new_thread(self, agent: ConcreteAgent) -> None:
        old_id = agent.chat_thread_id
        agent.reset()
        assert agent.chat_thread_id != old_id

    def test_clears_response_id(self, agent: ConcreteAgent) -> None:
        agent._previous_response_id = "resp_old"
        agent.reset()
        assert agent._previous_response_id is None

    def test_clears_response_id_index(self, agent: ConcreteAgent) -> None:
        agent._response_id_to_chat_index["resp_old"] = 0
        agent.reset()
        assert agent._response_id_to_chat_index == {}

    def test_syncs_tools_after_reset(self, agent: ConcreteAgent) -> None:
        """Tools are re-synced after reset."""
        agent.reset()
        assert "echo" in agent._registered_tool_names

    def test_persist_thread_deferred_after_reset(self, mock_emit: MagicMock) -> None:
        """After reset, thread persist is deferred until the next _add_chat."""
        mock_persist = MagicMock(side_effect=lambda t: t)
        agent = ConcreteAgent(
            emit_message_callable=mock_emit,
            persist_chat_thread_callable=mock_persist,
        )
        mock_persist.reset_mock()
        agent.reset()
        assert mock_persist.call_count == 0
        agent._add_chat(role=ChatRole.USER, content="hello")
        # Called twice: lazy thread create + thread update after chat
        assert mock_persist.call_count == 2


# =============================================================================
# Unified file tools (functional tests with real DocumentationDataLoader)
# =============================================================================


class TestFileTools:
    """Tests for the unified file tools on AbstractAgent."""

    def test_list_files_workspace(self, agent: ConcreteAgent) -> None:
        result = agent._execute_tool("list_files", {"scope": "workspace"})
        assert isinstance(result, str)
        assert "scope: workspace" in result

    def test_list_files_docs(self, agent_with_docs: ConcreteAgent) -> None:
        result = agent_with_docs._execute_tool("list_files", {"scope": "docs"})
        assert isinstance(result, str)
        assert "guide.md" in result
        assert "main.py" in result

    def test_read_file_workspace(self, agent: ConcreteAgent) -> None:
        (agent._workspace.root_path / "output").mkdir(parents=True, exist_ok=True)
        (agent._workspace.root_path / "output" / "x.txt").write_text("hello\nworld")
        result = agent._execute_tool(
            "read_file",
            {"scope": "workspace", "path": "output/x.txt", "start_line": 1, "end_line": 1},
        )
        assert isinstance(result, str)
        assert "hello" in result

    def test_read_file_docs(self, agent_with_docs: ConcreteAgent) -> None:
        result = agent_with_docs._execute_tool("read_file", {"scope": "docs", "path": "guide.md"})
        assert isinstance(result, str)
        assert "User Guide" in result

    def test_search_files_docs_exact(self, agent_with_docs: ConcreteAgent) -> None:
        result = agent_with_docs._execute_tool(
            "search_files",
            {"scope": "docs", "query": "Installation", "mode": "exact"},
        )
        assert isinstance(result, str)
        assert "files_with_matches" in result

    def test_search_files_docs_terms(self, agent_with_docs: ConcreteAgent) -> None:
        result = agent_with_docs._execute_tool(
            "search_files",
            {"scope": "docs", "query": "installation API", "mode": "terms"},
        )
        assert isinstance(result, str)
        assert "results_count" in result

    def test_search_files_docs_terms_splits_on_whitespace(self, agent_with_docs: ConcreteAgent) -> None:
        """Terms mode should split query tokens on spaces and commas."""
        result = agent_with_docs._search_files(
            scope="docs",
            query="installation API,Configuration",
            mode="terms",
        )
        assert isinstance(result, dict)
        assert result.get("mode") == "terms"
        assert result.get("terms") == ["installation", "API", "Configuration"]

    def test_search_files_docs_regex(self, agent_with_docs: ConcreteAgent) -> None:
        result = agent_with_docs._execute_tool(
            "search_files",
            {"scope": "docs", "query": r"def \w+\(", "mode": "regex"},
        )
        assert isinstance(result, str)
        assert "regex" in result.lower()

    def test_search_files_workspace_exact(self, agent: ConcreteAgent) -> None:
        (agent._workspace.root_path / "output").mkdir(parents=True, exist_ok=True)
        (agent._workspace.root_path / "output" / "data.txt").write_text("alpha\nbeta\nalpha")
        result = agent._execute_tool(
            "search_files",
            {"scope": "workspace", "query": "alpha", "mode": "exact"},
        )
        assert isinstance(result, str)
        assert "files_with_matches" in result

    def test_docs_scope_without_loader_returns_error(self, agent: ConcreteAgent) -> None:
        result = agent._execute_tool("search_files", {"scope": "docs", "query": "x"})
        assert isinstance(result, str)
        assert "docs scope unavailable" in result


# =============================================================================
# _get_documentation_prompt_section
# =============================================================================


class TestDocumentationPromptSection:
    """Tests for _get_documentation_prompt_section."""

    def test_empty_without_loader(self, agent: ConcreteAgent) -> None:
        assert agent._get_documentation_prompt_section() == ""

    def test_contains_stats(self, agent_with_docs: ConcreteAgent) -> None:
        section = agent_with_docs._get_documentation_prompt_section()
        assert "## Documentation" in section
        assert "indexed files" in section

    def test_contains_doc_file_names(self, agent_with_docs: ConcreteAgent) -> None:
        section = agent_with_docs._get_documentation_prompt_section()
        assert "guide.md" in section
        assert "api.md" in section

    def test_contains_code_file_names(self, agent_with_docs: ConcreteAgent) -> None:
        section = agent_with_docs._get_documentation_prompt_section()
        assert "main.py" in section

    def test_contains_doc_titles(self, agent_with_docs: ConcreteAgent) -> None:
        section = agent_with_docs._get_documentation_prompt_section()
        assert "User Guide" in section
        assert "API Reference" in section

    def test_truncates_long_titles(self, mock_emit: MagicMock, tmp_path: Path) -> None:
        """Long titles are truncated to 80 chars + '...'."""
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        long_title = "A" * 100
        (docs_dir / "long.md").write_text(f"# {long_title}\n\nSome content.\n")

        loader = DocumentationDataLoader(documentation_paths=[str(docs_dir)])
        agent = ConcreteAgent(
            emit_message_callable=mock_emit,
            documentation_data_loader=loader,
        )
        section = agent._get_documentation_prompt_section()
        # The title should be truncated at 80 chars + "..."
        assert "A" * 80 + "..." in section
        assert "A" * 100 not in section


# =============================================================================
# _call_llm - system prompt injection
# =============================================================================


class TestCallLLM:
    """Tests for _call_llm system prompt injection."""

    def test_no_docs_section_without_loader(self, agent: ConcreteAgent) -> None:
        """System prompt has tools + workspace sections, but no docs section without loader."""
        mock_response = LLMChatResponse(content="hello", response_id="r1")
        agent.llm_client.call_sync = MagicMock(return_value=mock_response)

        agent._call_llm([], "base prompt")

        call_args = agent.llm_client.call_sync.call_args
        system_prompt = call_args.kwargs["system_prompt"]
        assert system_prompt.startswith("base prompt")
        assert "## Tools" in system_prompt  # tool availability section always injected
        assert "## Workspace" in system_prompt
        assert "raw/" in system_prompt
        assert "output/" in system_prompt
        assert "## Documentation" not in system_prompt  # no docs without loader

    def test_docs_section_appended_with_loader(self, agent_with_docs: ConcreteAgent) -> None:
        """System prompt has workspace and documentation sections when loader is present."""
        mock_response = LLMChatResponse(content="hello", response_id="r1")
        agent_with_docs.llm_client.call_sync = MagicMock(return_value=mock_response)

        agent_with_docs._call_llm([], "base prompt")

        call_args = agent_with_docs.llm_client.call_sync.call_args
        system_prompt = call_args.kwargs["system_prompt"]
        assert system_prompt.startswith("base prompt")
        assert "## Workspace" in system_prompt
        assert "## Documentation" in system_prompt
        assert "guide.md" in system_prompt

    def test_streaming_also_gets_docs_section(self, agent_with_docs: ConcreteAgent) -> None:
        """Streaming path also appends workspace and documentation sections."""
        agent_with_docs._stream_chunk_callable = MagicMock()
        mock_response = LLMChatResponse(content="hello", response_id="r1")

        # Mock call_stream_sync to yield chunks then response
        def fake_stream(*args, **kwargs):
            yield "chunk1"
            yield mock_response
        agent_with_docs.llm_client.call_stream_sync = MagicMock(side_effect=fake_stream)

        agent_with_docs._call_llm([], "base prompt")

        call_args = agent_with_docs.llm_client.call_stream_sync.call_args
        system_prompt = call_args.kwargs["system_prompt"]
        assert "## Workspace" in system_prompt
        assert "## Documentation" in system_prompt

    def test_workspace_section_omitted_without_workspace(self, mock_emit: MagicMock) -> None:
        agent = ConcreteAgent(emit_message_callable=mock_emit, workspace=None)
        mock_response = LLMChatResponse(content="hello", response_id="r1")
        agent.llm_client.call_sync = MagicMock(return_value=mock_response)

        agent._call_llm([], "base prompt")

        call_args = agent.llm_client.call_sync.call_args
        system_prompt = call_args.kwargs["system_prompt"]
        assert "## Workspace" not in system_prompt

    def test_docs_section_present_without_workspace_when_loader_exists(
        self,
        agent_docs_only: ConcreteAgent,
    ) -> None:
        mock_response = LLMChatResponse(content="hello", response_id="r1")
        agent_docs_only.llm_client.call_sync = MagicMock(return_value=mock_response)

        agent_docs_only._call_llm([], "base prompt")

        call_args = agent_docs_only.llm_client.call_sync.call_args
        system_prompt = call_args.kwargs["system_prompt"]
        assert "## Workspace" not in system_prompt
        assert "## Documentation" in system_prompt


# =============================================================================
# _process_streaming_response
# =============================================================================


class TestStreamingResponse:
    """Tests for _process_streaming_response."""

    def test_streams_chunks_to_callback(self, agent: ConcreteAgent) -> None:
        chunk_callback = MagicMock()
        agent._stream_chunk_callable = chunk_callback

        mock_response = LLMChatResponse(content="final", response_id="r1")

        def fake_stream(*args, **kwargs):
            yield "chunk1"
            yield "chunk2"
            yield mock_response
        agent.llm_client.call_stream_sync = MagicMock(side_effect=fake_stream)

        response = agent._process_streaming_response([], "prompt")

        assert chunk_callback.call_count == 2
        chunk_callback.assert_any_call("chunk1")
        chunk_callback.assert_any_call("chunk2")
        assert response.content == "final"

    def test_raises_on_no_response(self, agent: ConcreteAgent) -> None:
        agent._stream_chunk_callable = MagicMock()

        def fake_stream(*args, **kwargs):
            yield "chunk_only"
        agent.llm_client.call_stream_sync = MagicMock(side_effect=fake_stream)

        with pytest.raises(ValueError, match="No final response"):
            agent._process_streaming_response([], "prompt")


# =============================================================================
# process_new_message
# =============================================================================


class TestProcessNewMessage:
    """Tests for process_new_message."""

    def test_adds_user_chat(self, agent: ConcreteAgent) -> None:
        """process_new_message adds the user chat to history."""
        mock_response = LLMChatResponse(content="reply")
        agent.llm_client.call_sync = MagicMock(return_value=mock_response)

        agent.process_new_message("hello")

        chats = agent.get_chats()
        assert chats[0].role == ChatRole.USER
        assert chats[0].content == "hello"

    def test_default_role_is_user(self, agent: ConcreteAgent) -> None:
        mock_response = LLMChatResponse(content="reply")
        agent.llm_client.call_sync = MagicMock(return_value=mock_response)

        agent.process_new_message("hello")
        assert agent.get_chats()[0].role == ChatRole.USER

    def test_custom_role(self, agent: ConcreteAgent) -> None:
        mock_response = LLMChatResponse(content="reply")
        agent.llm_client.call_sync = MagicMock(return_value=mock_response)

        agent.process_new_message("system info", role=ChatRole.SYSTEM)
        assert agent.get_chats()[0].role == ChatRole.SYSTEM

    def test_emits_response(self, agent: ConcreteAgent, mock_emit: MagicMock) -> None:
        mock_response = LLMChatResponse(content="reply", response_id="r1")
        agent.llm_client.call_sync = MagicMock(return_value=mock_response)

        agent.process_new_message("hello")

        # Find emitted ChatResponseEmittedMessage
        chat_msgs = [
            c for c in mock_emit.call_args_list
            if isinstance(c[0][0], ChatResponseEmittedMessage)
        ]
        assert len(chat_msgs) == 1
        assert chat_msgs[0][0][0].content == "reply"


# =============================================================================
# _run_agent_loop
# =============================================================================


class TestRunAgentLoop:
    """Tests for _run_agent_loop."""

    def test_stops_when_no_tool_calls(self, agent: ConcreteAgent) -> None:
        """Loop stops after LLM returns content without tool calls."""
        mock_response = LLMChatResponse(content="done", response_id="r1")
        agent.llm_client.call_sync = MagicMock(return_value=mock_response)

        agent._add_chat(ChatRole.USER, "hello")
        agent._run_agent_loop()

        # Only one LLM call should be made
        assert agent.llm_client.call_sync.call_count == 1

    def test_executes_tool_calls_and_loops(self, agent: ConcreteAgent) -> None:
        """Loop executes tool calls and re-calls LLM."""
        # First call: LLM requests a tool call
        tool_call = LLMToolCall(tool_name="echo", tool_arguments={"message": "hi"}, call_id="c1")
        response_with_tool = LLMChatResponse(content="let me call echo", tool_calls=[tool_call], response_id="r1")
        # Second call: LLM is done
        response_final = LLMChatResponse(content="all done", response_id="r2")

        agent.llm_client.call_sync = MagicMock(side_effect=[response_with_tool, response_final])

        agent._add_chat(ChatRole.USER, "hello")
        agent._run_agent_loop()

        assert agent.llm_client.call_sync.call_count == 2
        chats = agent.get_chats()
        # Should have: USER, ASSISTANT (tool call), TOOL (result), ASSISTANT (final)
        roles = [c.role for c in chats]
        assert ChatRole.TOOL in roles

    def test_emits_error_on_exception(self, agent: ConcreteAgent, mock_emit: MagicMock) -> None:
        """Loop emits ErrorEmittedMessage on LLM exception."""
        agent.llm_client.call_sync = MagicMock(side_effect=RuntimeError("LLM down"))

        agent._add_chat(ChatRole.USER, "hello")
        agent._run_agent_loop()

        error_msgs = [
            c for c in mock_emit.call_args_list
            if isinstance(c[0][0], ErrorEmittedMessage)
        ]
        assert len(error_msgs) == 1
        assert "LLM down" in error_msgs[0][0][0].error

    def test_max_iterations_cap(self, agent: ConcreteAgent) -> None:
        """Loop stops after 10 iterations even if tool calls continue."""
        # Every call returns a tool call (infinite loop scenario)
        tool_call = LLMToolCall(tool_name="no_params", tool_arguments={}, call_id="c1")
        response = LLMChatResponse(content="", tool_calls=[tool_call], response_id="r1")
        agent.llm_client.call_sync = MagicMock(return_value=response)

        agent._add_chat(ChatRole.USER, "hello")
        agent._run_agent_loop()

        assert agent.llm_client.call_sync.call_count == 10

    def test_response_id_tracked(self, agent: ConcreteAgent) -> None:
        """_previous_response_id is updated after each LLM call."""
        mock_response = LLMChatResponse(content="reply", response_id="resp_abc")
        agent.llm_client.call_sync = MagicMock(return_value=mock_response)

        agent._add_chat(ChatRole.USER, "hello")
        agent._run_agent_loop()

        assert agent._previous_response_id == "resp_abc"


# =============================================================================
# _emit_message
# =============================================================================


class TestEmitMessage:
    """Tests for _emit_message."""

    def test_calls_callback(self, agent: ConcreteAgent, mock_emit: MagicMock) -> None:
        msg = ErrorEmittedMessage(error="test")
        agent._emit_message(msg)
        mock_emit.assert_called_once_with(msg)


# =============================================================================
# @agent_tool decorator (tested via AbstractAgent directly)
# =============================================================================


class TestAgentToolDecorator:
    """Comprehensive tests for the @agent_tool decorator.

    Verifies that docstrings are extracted fully and correctly, parameter schemas
    are auto-generated with the right types/descriptions/required fields, explicit
    overrides work, and availability is wired through to _ToolMeta.
    """

    # ---- helpers ----

    @staticmethod
    def _get_tool(name: str) -> _ToolMeta:
        """Look up a tool by name from ConcreteAgent._collect_tools()."""
        for meta, _ in ConcreteAgent._collect_tools():
            if meta.name == name:
                return meta
        raise KeyError(f"Tool {name!r} not found in ConcreteAgent")

    # ---- tool name derivation ----

    def test_single_leading_underscore_stripped(self) -> None:
        assert self._get_tool("echo").name == "echo"  # _echo -> echo

    def test_multiple_leading_underscores_stripped(self) -> None:
        @agent_tool
        def __double_prefix(self, x: str) -> dict[str, Any]:
            """Double underscore tool."""
            return {}
        assert __double_prefix._tool_meta.name == "double_prefix"

    def test_no_underscore_prefix_unchanged(self) -> None:
        @agent_tool
        def plain_name(self, x: str) -> dict[str, Any]:
            """No underscore prefix."""
            return {}
        assert plain_name._tool_meta.name == "plain_name"

    # ---- description extraction from docstrings ----

    def test_single_line_docstring_extracted_exactly(self) -> None:
        meta = self._get_tool("no_params")
        assert meta.description == "A tool with no parameters."

    def test_multiline_description_before_args_extracted_in_full(self) -> None:
        """The full text before Args: should be in the description, nothing more."""
        meta = self._get_tool("echo")
        assert meta.description == "Echo the message back."

    def test_multi_paragraph_description_joined(self) -> None:
        @agent_tool
        def _multi_para(self) -> dict[str, Any]:
            """First paragraph here.

            Second paragraph with more detail.

            Args:
                (none)
            """
            return {}
        assert _multi_para._tool_meta.description == (
            "First paragraph here. Second paragraph with more detail."
        )

    def test_multiple_blank_lines_between_paragraphs(self) -> None:
        """Multiple consecutive blank lines should collapse, not produce extra spaces."""
        @agent_tool
        def _gappy(self) -> dict[str, Any]:
            """Line one.



            Line two after many blanks.


            Args:
                (none)
            """
            return {}
        desc = _gappy._tool_meta.description
        assert desc == "Line one. Line two after many blanks."
        assert "  " not in desc  # no double-spaces

    def test_blank_lines_before_args_section(self) -> None:
        """Blank lines right before Args: should not leak empty content."""
        @agent_tool
        def _blanks_before_args(self) -> dict[str, Any]:
            """Do the thing.



            Args:
                x: Something.
            """
            return {}
        assert _blanks_before_args._tool_meta.description == "Do the thing."

    def test_three_paragraphs_with_mixed_blank_lines(self) -> None:
        @agent_tool
        def _three_para(self) -> dict[str, Any]:
            """First paragraph.

            Second paragraph.


            Third paragraph.

            Args:
                (none)
            """
            return {}
        assert _three_para._tool_meta.description == (
            "First paragraph. Second paragraph. Third paragraph."
        )

    def test_args_section_excluded_from_description(self) -> None:
        """Args block must not leak into the description."""
        meta = self._get_tool("add_numbers")
        assert "Args:" not in meta.description
        assert "First number" not in meta.description
        assert meta.description == "Add two numbers."

    def test_returns_section_excluded_from_description(self) -> None:
        @agent_tool
        def _with_returns(self, x: str) -> dict[str, Any]:
            """Do something useful.

            Returns:
                A dict of results.
            """
            return {}
        assert _with_returns._tool_meta.description == "Do something useful."
        assert "Returns" not in _with_returns._tool_meta.description

    def test_raises_section_excluded_from_description(self) -> None:
        @agent_tool
        def _with_raises(self) -> dict[str, Any]:
            """Do something risky.

            Raises:
                ValueError: If things go wrong.
            """
            return {}
        assert _with_raises._tool_meta.description == "Do something risky."

    def test_example_section_excluded_from_description(self) -> None:
        @agent_tool
        def _with_example(self) -> dict[str, Any]:
            """Compute a value.

            Example:
                _with_example()
            """
            return {}
        assert _with_example._tool_meta.description == "Compute a value."

    def test_description_whitespace_collapsed(self) -> None:
        """Extra whitespace from indentation should be collapsed to single spaces."""
        @agent_tool
        def _spaced(self) -> dict[str, Any]:
            """
            Has    extra   whitespace   here.
            """
            return {}
        assert "  " not in _spaced._tool_meta.description
        assert _spaced._tool_meta.description == "Has extra whitespace here."

    # ---- explicit description override ----

    def test_explicit_description_overrides_docstring(self) -> None:
        @agent_tool(description="Custom override description.")
        def _overridden(self) -> dict[str, Any]:
            """This docstring should be ignored."""
            return {}
        assert _overridden._tool_meta.description == "Custom override description."
        assert "ignored" not in _overridden._tool_meta.description

    def test_explicit_description_works_without_docstring(self) -> None:
        @agent_tool(description="No docstring needed.")
        def _no_doc(self) -> dict[str, Any]:
            return {}
        assert _no_doc._tool_meta.description == "No docstring needed."

    # ---- error: no description at all ----

    def test_raises_without_description_or_docstring(self) -> None:
        with pytest.raises(ValueError, match="no description and no docstring"):
            @agent_tool
            def _bare(self) -> dict[str, Any]:
                pass

    def test_raises_with_empty_docstring(self) -> None:
        with pytest.raises(ValueError, match="no description and no docstring"):
            @agent_tool
            def _empty_doc(self) -> dict[str, Any]:
                ""
                pass

    # ---- parameter schema: structure ----

    def test_schema_top_level_structure(self) -> None:
        schema = self._get_tool("echo").parameters
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "required" in schema

    def test_self_excluded_from_schema(self) -> None:
        schema = self._get_tool("echo").parameters
        assert "self" not in schema["properties"]

    def test_no_params_tool_has_empty_schema(self) -> None:
        schema = self._get_tool("no_params").parameters
        assert schema["properties"] == {}
        assert schema["required"] == []

    # ---- parameter schema: required vs optional ----

    def test_all_params_required_when_no_defaults(self) -> None:
        schema = self._get_tool("add_numbers").parameters
        assert sorted(schema["required"]) == ["a", "b"]

    def test_param_with_default_not_required(self) -> None:
        schema = self._get_tool("optional_params").parameters
        assert "required" in schema["required"]
        assert "opt" not in schema["required"]

    def test_single_required_param(self) -> None:
        schema = self._get_tool("echo").parameters
        assert schema["required"] == ["message"]

    # ---- parameter schema: types ----

    def test_string_param_type(self) -> None:
        schema = self._get_tool("echo").parameters
        assert schema["properties"]["message"]["type"] == "string"

    def test_int_param_type(self) -> None:
        schema = self._get_tool("add_numbers").parameters
        assert schema["properties"]["a"]["type"] == "integer"
        assert schema["properties"]["b"]["type"] == "integer"

    def test_bool_param_type(self) -> None:
        @agent_tool
        def _bool_tool(self, flag: bool) -> dict[str, Any]:
            """Tool with bool param.

            Args:
                flag: A boolean flag.
            """
            return {}
        schema = _bool_tool._tool_meta.parameters
        assert schema["properties"]["flag"]["type"] == "boolean"

    def test_float_param_type(self) -> None:
        @agent_tool
        def _float_tool(self, value: float) -> dict[str, Any]:
            """Tool with float param.

            Args:
                value: A float value.
            """
            return {}
        schema = _float_tool._tool_meta.parameters
        assert schema["properties"]["value"]["type"] == "number"

    def test_list_param_type(self) -> None:
        @agent_tool
        def _list_tool(self, items: list[str]) -> dict[str, Any]:
            """Tool with list param.

            Args:
                items: List of strings.
            """
            return {}
        schema = _list_tool._tool_meta.parameters
        assert schema["properties"]["items"]["type"] == "array"
        assert schema["properties"]["items"]["items"]["type"] == "string"

    def test_dict_param_type(self) -> None:
        @agent_tool
        def _dict_tool(self, data: dict[str, int]) -> dict[str, Any]:
            """Tool with dict param.

            Args:
                data: A mapping of names to counts.
            """
            return {}
        schema = _dict_tool._tool_meta.parameters
        assert schema["properties"]["data"]["type"] == "object"
        assert schema["properties"]["data"]["additionalProperties"]["type"] == "integer"

    def test_nullable_param_type(self) -> None:
        @agent_tool
        def _nullable_tool(self, value: str | None) -> dict[str, Any]:
            """Tool with nullable param.

            Args:
                value: A nullable string.
            """
            return {}
        schema = _nullable_tool._tool_meta.parameters
        # pydantic represents str | None as anyOf
        assert "anyOf" in schema["properties"]["value"]

    # ---- parameter schema: descriptions from docstring Args ----

    def test_param_description_extracted_from_docstring(self) -> None:
        """Each param's description from the Args section should appear in the schema."""
        schema = self._get_tool("echo").parameters
        assert schema["properties"]["message"].get("description") == "The message to echo."

    def test_multiple_param_descriptions_extracted(self) -> None:
        schema = self._get_tool("add_numbers").parameters
        assert schema["properties"]["a"].get("description") == "First number."
        assert schema["properties"]["b"].get("description") == "Second number."

    def test_optional_param_description_extracted(self) -> None:
        schema = self._get_tool("optional_params").parameters
        assert schema["properties"]["required"].get("description") == "A required string."
        assert schema["properties"]["opt"].get("description") == "An optional integer."

    def test_no_param_description_when_args_section_missing(self) -> None:
        """Tool with no Args section should still have schema, just no descriptions."""
        schema = self._get_tool("no_params").parameters
        # No properties, so nothing to check descriptions on — just verify no crash
        assert schema["properties"] == {}

    def test_param_description_with_parenthetical_type(self) -> None:
        """Docstring format 'param (type): desc' should extract cleanly."""
        @agent_tool
        def _paren_type(self, query: str) -> dict[str, Any]:
            """Search tool.

            Args:
                query (str): The search query to use.
            """
            return {}
        schema = _paren_type._tool_meta.parameters
        assert schema["properties"]["query"].get("description") == "The search query to use."

    # ---- explicit parameters override ----

    def test_explicit_parameters_overrides_auto_generation(self) -> None:
        custom_schema = {
            "type": "object",
            "properties": {"custom_field": {"type": "string"}},
            "required": ["custom_field"],
        }

        @agent_tool(parameters=custom_schema)
        def _custom_schema(self, ignored: int) -> dict[str, Any]:
            """Tool with custom schema.

            Args:
                ignored: This type hint should be ignored.
            """
            return {}
        assert _custom_schema._tool_meta.parameters == custom_schema
        # The actual type hint (int) is NOT used
        assert "ignored" not in _custom_schema._tool_meta.parameters["properties"]

    # ---- availability ----

    def test_availability_default_true(self) -> None:
        assert self._get_tool("echo").availability is True

    def test_availability_false(self) -> None:
        assert self._get_tool("disabled_tool").availability is False

    def test_availability_callable_stored(self) -> None:
        assert callable(self._get_tool("gated_tool").availability)

    def test_availability_callable_evaluates_dynamically(self) -> None:
        """The callable should evaluate against the instance, not be fixed at decoration time."""
        avail_fn = self._get_tool("gated_tool").availability

        class FakeAgent:
            _feature_flag = False

        fake = FakeAgent()
        assert avail_fn(fake) is False
        fake._feature_flag = True
        assert avail_fn(fake) is True

    def test_token_optimized_flag_encodes_output(self) -> None:
        @agent_tool(token_optimized=True)
        def _encoded_tool(self) -> dict[str, Any]:
            """Returns a dict that should be toon-encoded."""
            return {"status": "ok", "count": 2}

        meta = _encoded_tool._tool_meta
        assert meta.token_optimized is True

    # ---- _tool_meta is attached to the method ----

    def test_tool_meta_attached_to_method(self) -> None:
        assert hasattr(ConcreteAgent._echo, "_tool_meta")
        assert isinstance(ConcreteAgent._echo._tool_meta, _ToolMeta)

    def test_tool_meta_is_frozen(self) -> None:
        meta = ConcreteAgent._echo._tool_meta
        with pytest.raises(AttributeError):
            meta.name = "tampered"  # type: ignore[misc]

    # ---- mixed types in a single tool ----

    def test_complex_mixed_signature(self) -> None:
        """Tool with diverse param types: all types and required/optional are correct."""
        @agent_tool
        def _complex(
            self,
            name: str,
            count: int,
            tags: list[str],
            verbose: bool = False,
            limit: int | None = None,
        ) -> dict[str, Any]:
            """Complex tool.

            Args:
                name: The resource name.
                count: How many to fetch.
                tags: Tags to filter by.
                verbose: Enable verbose output.
                limit: Max results (optional).
            """
            return {}

        meta = _complex._tool_meta
        schema = meta.parameters

        # required: name, count, tags (no defaults)
        assert sorted(schema["required"]) == ["count", "name", "tags"]

        # types
        assert schema["properties"]["name"]["type"] == "string"
        assert schema["properties"]["count"]["type"] == "integer"
        assert schema["properties"]["tags"]["type"] == "array"
        assert schema["properties"]["verbose"]["type"] == "boolean"
        assert "anyOf" in schema["properties"]["limit"]  # int | None

        # descriptions
        assert schema["properties"]["name"]["description"] == "The resource name."
        assert schema["properties"]["count"]["description"] == "How many to fetch."
        assert schema["properties"]["tags"]["description"] == "Tags to filter by."
        assert schema["properties"]["verbose"]["description"] == "Enable verbose output."
        assert schema["properties"]["limit"]["description"] == "Max results (optional)."

    # ---- Pydantic model as parameter ----

    def test_pydantic_model_param_generates_object_schema(self) -> None:
        """A Pydantic model param should produce a nested object schema with its fields."""
        meta = self._get_tool("search")
        schema = meta.parameters
        param_schema = schema["properties"]["params"]

        # Pydantic model becomes an object with its own properties
        assert param_schema["type"] == "object"
        assert "query" in param_schema["properties"]
        assert "max_results" in param_schema["properties"]
        assert "tags" in param_schema["properties"]

    def test_pydantic_model_field_types_in_schema(self) -> None:
        """The nested Pydantic model's field types should be correct in the schema."""
        meta = self._get_tool("search")
        param_schema = meta.parameters["properties"]["params"]

        assert param_schema["properties"]["query"]["type"] == "string"
        assert param_schema["properties"]["max_results"]["type"] == "integer"
        assert param_schema["properties"]["tags"]["type"] == "array"

    def test_pydantic_model_field_descriptions_in_schema(self) -> None:
        """Pydantic Field(description=...) should appear in the nested schema."""
        meta = self._get_tool("search")
        param_schema = meta.parameters["properties"]["params"]

        assert param_schema["properties"]["query"]["description"] == "The search query."
        assert param_schema["properties"]["max_results"]["description"] == "Maximum number of results."
        assert param_schema["properties"]["tags"]["description"] == "Tags to filter by."

    def test_pydantic_model_required_fields_in_schema(self) -> None:
        """Only fields without defaults should be required in the nested schema."""
        meta = self._get_tool("search")
        param_schema = meta.parameters["properties"]["params"]

        assert "query" in param_schema["required"]
        # max_results and tags have defaults, so not required
        assert "max_results" not in param_schema.get("required", [])
        assert "tags" not in param_schema.get("required", [])

    def test_pydantic_model_param_docstring_description(self) -> None:
        """The Args docstring description should appear on the param, not the model fields."""
        meta = self._get_tool("search")
        param_schema = meta.parameters["properties"]["params"]
        assert param_schema.get("description") == "The search parameters."

    # ---- real-world Routine model: full _ToolMeta snapshot ----

    def test_routine_param_tool_meta_matches_expected(self) -> None:
        """End-to-end: @agent_tool on a fn with Routine param produces the exact expected _ToolMeta.

        This test defines a standalone function taking a Routine, verifies that the
        decorator wires docstring, type hints, and Args descriptions into _ToolMeta
        correctly — and that the Routine schema matches what Pydantic generates.
        """
        from pydantic import TypeAdapter
        from bluebox.data_models.routine.routine import Routine

        @agent_tool
        def _update_routine(self, routine: Routine, dry_run: bool = False) -> dict[str, Any]:
            """Update an existing routine definition.

            Validates the routine and optionally persists it.

            Args:
                routine: The full routine object to update.
                dry_run: If True, validate only without saving.
            """
            return {}

        meta = _update_routine._tool_meta

        # --- name: leading underscore stripped ---
        assert meta.name == "update_routine"

        # --- description: full text before Args, joined ---
        assert meta.description == (
            "Update an existing routine definition. "
            "Validates the routine and optionally persists it."
        )

        # --- availability: default True ---
        assert meta.availability is True

        # --- parameters schema: build expected and compare ---
        # The routine param schema should match what Pydantic generates,
        # with $defs hoisted to the root level (not nested inside the property).
        routine_schema = TypeAdapter(Routine).json_schema()
        routine_schema.pop("title", None)
        routine_schema["description"] = "The full routine object to update."
        hoisted_defs = routine_schema.pop("$defs", {})

        expected_parameters: dict[str, Any] = {
            "type": "object",
            "properties": {
                "routine": routine_schema,
                "dry_run": {
                    "type": "boolean",
                    "description": "If True, validate only without saving.",
                },
            },
            "required": ["routine"],
        }
        if hoisted_defs:
            expected_parameters["$defs"] = hoisted_defs

        assert meta.parameters == expected_parameters


# =============================================================================
# _ToolMeta
# =============================================================================


class TestToolMeta:
    """Tests for _ToolMeta dataclass."""

    def test_frozen(self) -> None:
        meta = _ToolMeta(
            name="t", description="d",
            parameters={"type": "object", "properties": {}, "required": []},
            availability=True,
        )
        with pytest.raises(AttributeError):
            meta.name = "changed"  # type: ignore[misc]

    def test_stores_fields(self) -> None:
        params = {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]}
        meta = _ToolMeta(name="my_tool", description="desc", parameters=params, availability=True)
        assert meta.name == "my_tool"
        assert meta.description == "desc"
        assert meta.parameters == params
        assert meta.availability is True
        assert meta.persist == ToolResultPersistMode.NEVER
        assert meta.max_characters == 10_000
        assert meta.token_optimized is False


# =============================================================================
# Integration: full round-trip with docs
# =============================================================================


class TestIntegration:
    """Integration tests exercising multiple components together."""

    def test_process_message_with_tool_call_round_trip(
        self, agent: ConcreteAgent, mock_emit: MagicMock,
    ) -> None:
        """Full round-trip: user message -> LLM tool call -> tool execution -> LLM final."""
        tool_call = LLMToolCall(tool_name="echo", tool_arguments={"message": "round trip"}, call_id="c1")
        response1 = LLMChatResponse(content="", tool_calls=[tool_call], response_id="r1")
        response2 = LLMChatResponse(content="Echo result: round trip", response_id="r2")

        agent.llm_client.call_sync = MagicMock(side_effect=[response1, response2])
        agent.process_new_message("please echo round trip")

        chats = agent.get_chats()
        roles = [c.role for c in chats]
        assert roles == [ChatRole.USER, ChatRole.ASSISTANT, ChatRole.TOOL, ChatRole.ASSISTANT]
        assert chats[-1].content == "Echo result: round trip"

        # Verify the response was emitted
        chat_msgs = [
            c for c in mock_emit.call_args_list
            if isinstance(c[0][0], ChatResponseEmittedMessage)
        ]
        assert any("Echo result" in m[0][0].content for m in chat_msgs)

    def test_file_tools_functional_with_search_then_read(
        self, agent_with_docs: ConcreteAgent,
    ) -> None:
        """Search for content, then read the file — mimics typical docs workflow."""
        search_result = agent_with_docs._execute_tool(
            "search_files",
            {"scope": "docs", "query": "pip install", "mode": "exact"},
        )
        assert isinstance(search_result, str)
        assert "files_with_matches" in search_result

        read_result = agent_with_docs._execute_tool(
            "read_file", {"scope": "docs", "path": "guide.md"},
        )
        assert isinstance(read_result, str)
        assert "pip install" in read_result

    def test_reset_preserves_documentation_data_loader(
        self, agent_with_docs: ConcreteAgent,
    ) -> None:
        """Reset clears chat state but preserves the documentation data loader."""
        assert agent_with_docs._documentation_data_loader is not None
        agent_with_docs._add_chat(ChatRole.USER, "hello")

        agent_with_docs.reset()

        assert agent_with_docs._documentation_data_loader is not None
        assert len(agent_with_docs.get_chats()) == 0
        # Unified file tools stay registered
        assert "search_files" in agent_with_docs._registered_tool_names
