"""
bluebox/agents/dumb_agent.py

Minimal test agent for compute-only autonomous experimentation.
"""

from __future__ import annotations

from textwrap import dedent
from typing import Callable

from bluebox.agents.abstract_agent import AbstractAgent, AgentCard
from bluebox.agents.workspace import AgentWorkspace
from bluebox.data_models.llms.interaction import Chat, ChatThread, EmittedMessage, LLMChatResponse
from bluebox.data_models.llms.vendors import LLMModel, OpenAIModel


class DumbAgent(AbstractAgent):
    """
    Intentionally minimal agent.

    Defaults to no workspace and no docs tooling. When code execution is enabled
    (default), only compute-style `execute_python` is available.
    """

    AGENT_CARD = AgentCard(
        description="Minimal test agent with optional compute-only Python execution.",
    )

    SYSTEM_PROMPT: str = dedent("""\
        You are DumbAgent, a minimal test agent.
        Keep responses concise and rely on `execute_python` for calculations.
        If no workspace is attached, do not attempt file I/O.
    """).strip()

    def __init__(
        self,
        emit_message_callable: Callable[[EmittedMessage], None],
        workspace: AgentWorkspace | None = None,
        persist_chat_callable: Callable[[Chat], Chat] | None = None,
        persist_chat_thread_callable: Callable[[ChatThread], ChatThread] | None = None,
        stream_chunk_callable: Callable[[str], None] | None = None,
        llm_model: LLMModel = OpenAIModel.GPT_5_1,
        chat_thread: ChatThread | None = None,
        existing_chats: list[Chat] | None = None,
        on_llm_response: Callable[[LLMChatResponse], None] | None = None,
        allow_code_execution: bool = True,
    ) -> None:
        super().__init__(
            emit_message_callable=emit_message_callable,
            workspace=workspace,
            persist_chat_callable=persist_chat_callable,
            persist_chat_thread_callable=persist_chat_thread_callable,
            stream_chunk_callable=stream_chunk_callable,
            llm_model=llm_model,
            chat_thread=chat_thread,
            existing_chats=existing_chats,
            documentation_data_loader=None,
            on_llm_response=on_llm_response,
            allow_code_execution=allow_code_execution,
            code_execution_globals={},
        )

    def _get_system_prompt(self) -> str:
        return self.SYSTEM_PROMPT + self._generate_code_execution_prompt()
