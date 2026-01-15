"""
web_hacker/agents/guide_agent/guide_agent.py

Guide agent that guides the user through the process of creating or editing a routine.
"""

import uuid
from typing import Any, Callable

from data_models.guide_agent.conversation import (
    ConversationMessage,
    ConversationRole,
    GuideAgentConversationState,
    PendingToolInvocation,
    ToolInvocationStatus,
)
from data_models.guide_agent.message import (
    GuideAgentMessage,
    GuideAgentMessageType,
)
from data_models.llms import LLMModel, OpenAIModel
from llms.llm_client import LLMClient
from llms.tools.guide_agent_tools import (
    START_ROUTINE_DISCOVERY_TOOL_DESCRIPTION,
    START_ROUTINE_DISCOVERY_TOOL_NAME,
    StartRoutineDiscoveryJobCreationParams,
)
from utils.exceptions import UnknownToolError
from utils.logger import get_logger


logger = get_logger(name=__name__)


class GuideAgent:
    """
    Guide agent that guides the user through the process of creating or editing a routine.

    The agent maintains conversation state and uses LLM tool-calling to determine
    when to initiate routine discovery. Tool invocations require user confirmation
    via callback before execution.

    Usage:
        def handle_message(message: GuideAgentMessage) -> None:
            print(f"[{message.type}] {message.content}")

        agent = GuideAgent(emit_message_callable=handle_message)
        agent.process_user_message("I want to search for flights")
    """

    # Class constants ______________________________________________________________________________________________________

    SYSTEM_PROMPT: str = """You are a helpful assistant that guides users through \
creating automation routines. Your job is to:
1. Understand what task the user wants to automate
2. Clarify what data/output they expect from the routine
3. Identify required input parameters
4. Understand any filters or constraints

When you have enough information, use the start_routine_discovery_job_creation tool.
Ask clarifying questions if needed. Be conversational and helpful."""

    # Magic methods ________________________________________________________________________________________________________

    def __init__(
        self,
        emit_message_callable: Callable[[GuideAgentMessage], None],
        llm_model: LLMModel = OpenAIModel.GPT_5_MINI,
        guide_chat_id: str | None = None,
    ) -> None:
        """
        Initialize the guide agent.

        Args:
            emit_message_callable: Callback function to emit messages to the host.
            llm_model: The LLM model to use for conversation.
            guide_chat_id: Optional session ID. If None, generates a new UUIDv4.
        """
        self._emit_message_callable = emit_message_callable
        self.llm_model = llm_model
        self.llm_client = LLMClient(llm_model)

        # Register tools
        self._register_tools()

        # Initialize conversation state
        self._state = GuideAgentConversationState(
            guide_chat_id=guide_chat_id or str(uuid.uuid4())
        )

        logger.info(
            "Instantiated GuideAgent with model: %s, guide_chat_id: %s",
            llm_model,
            self._state.guide_chat_id,
        )

    # Properties ___________________________________________________________________________________________________________

    @property
    def guide_chat_id(self) -> str:
        """Return the current session ID."""
        return self._state.guide_chat_id

    @property
    def has_pending_tool_invocation(self) -> bool:
        """Check if there's a pending tool invocation awaiting confirmation."""
        return self._state.pending_tool_invocation is not None

    # Private methods ______________________________________________________________________________________________________

    def _register_tools(self) -> None:
        """Register all tools with the LLM client."""
        self.llm_client.register_tool(
            name=START_ROUTINE_DISCOVERY_TOOL_NAME,
            description=START_ROUTINE_DISCOVERY_TOOL_DESCRIPTION,
            parameters=StartRoutineDiscoveryJobCreationParams.model_json_schema(),
        )

    def _emit_message(self, message: GuideAgentMessage) -> None:
        """Emit a message via the callback."""
        self._emit_message_callable(message)

    def _add_message_to_history(
        self,
        role: ConversationRole,
        content: str,
    ) -> None:
        """Add a message to the conversation history."""
        self._state.messages.append(
            ConversationMessage(role=role, content=content)
        )

    def _build_conversation_prompt(self) -> str:
        """
        Build the conversation history as a prompt string for the LLM.

        Returns:
            Formatted prompt string containing conversation history.

        Raises:
            NotImplementedError: Business logic to be implemented in subsequent PR.
        """
        raise NotImplementedError("Business logic to be implemented")

    def _parse_llm_response(
        self,
        response: str,
    ) -> tuple[str | None, dict[str, Any] | None]:
        """
        Parse LLM response for text content and tool calls.

        Args:
            response: Raw LLM response

        Returns:
            Tuple of (text_content, tool_call_dict or None)

        Raises:
            NotImplementedError: Business logic to be implemented in subsequent PR.
        """
        raise NotImplementedError("Business logic to be implemented")

    def _create_tool_invocation_request(
        self,
        tool_name: str,
        tool_arguments: dict[str, Any],
    ) -> PendingToolInvocation:
        """
        Create a tool invocation request for user confirmation.

        Args:
            tool_name: Name of the tool to invoke
            tool_arguments: Arguments for the tool

        Returns:
            PendingToolInvocation stored in state and ready to emit
        """
        invocation_id = str(uuid.uuid4())

        pending = PendingToolInvocation(
            invocation_id=invocation_id,
            tool_name=tool_name,
            tool_arguments=tool_arguments,
        )

        # Store in state
        self._state.pending_tool_invocation = pending

        logger.info(
            "Created tool invocation request: %s (tool: %s)",
            invocation_id,
            tool_name,
        )

        return pending

    def _execute_tool(
        self,
        tool_name: str,
        tool_arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Execute a confirmed tool invocation.

        Args:
            tool_name: Name of the tool to execute
            tool_arguments: Arguments for the tool

        Returns:
            Tool execution result with guide_chat_id and params

        Raises:
            UnknownToolError: If tool_name is unknown
        """
        if tool_name == START_ROUTINE_DISCOVERY_TOOL_NAME:
            logger.info(
                "Executing tool %s with args: %s",
                tool_name,
                tool_arguments,
            )
            # Return data for handoff to routine discovery
            return {
                "guide_chat_id": self._state.guide_chat_id,
                **tool_arguments,
            }

        raise UnknownToolError(f"Unknown tool: {tool_name}")

    # Public methods _______________________________________________________________________________________________________

    def process_user_message(self, content: str) -> None:
        """
        Process a user message and emit responses via callback.

        This method handles the conversation loop:
        1. Adds user message to history
        2. Calls LLM to generate response
        3. Emits chat response or tool invocation request

        Args:
            content: The user's message content

        Raises:
            NotImplementedError: Business logic to be implemented in subsequent PR.
        """
        # Block new messages if there's a pending tool invocation
        if self._state.pending_tool_invocation:
            self._emit_message(
                GuideAgentMessage(
                    type=GuideAgentMessageType.ERROR,
                    error="Please confirm or deny the pending tool invocation before sending new messages",
                )
            )
            return

        # Add user message to history
        self._add_message_to_history(ConversationRole.USER, content)

        # Future: Call LLM, parse response, handle tool calls
        raise NotImplementedError("LLM conversation logic to be implemented")

    def confirm_tool_invocation(self, invocation_id: str) -> None:
        """
        Confirm a pending tool invocation and execute it.

        Args:
            invocation_id: ID of the tool invocation to confirm

        Emits:
            - TOOL_INVOCATION_RESULT with status "executed" and result on success
            - ERROR if no pending invocation or ID mismatch
        """
        pending = self._state.pending_tool_invocation

        if not pending:
            self._emit_message(
                GuideAgentMessage(
                    type=GuideAgentMessageType.ERROR,
                    error="No pending tool invocation to confirm",
                )
            )
            return

        if pending.invocation_id != invocation_id:
            self._emit_message(
                GuideAgentMessage(
                    type=GuideAgentMessageType.ERROR,
                    error=f"Invocation ID mismatch: expected {pending.invocation_id}",
                )
            )
            return

        # Update status
        pending.status = ToolInvocationStatus.CONFIRMED

        try:
            result = self._execute_tool(pending.tool_name, pending.tool_arguments)
            pending.status = ToolInvocationStatus.EXECUTED
            self._state.pending_tool_invocation = None

            self._emit_message(
                GuideAgentMessage(
                    type=GuideAgentMessageType.TOOL_INVOCATION_RESULT,
                    tool_invocation=pending,
                    tool_result=result,
                )
            )

            logger.info(
                "Tool invocation %s executed successfully",
                invocation_id,
            )

        except Exception as e:
            pending.status = ToolInvocationStatus.FAILED

            self._emit_message(
                GuideAgentMessage(
                    type=GuideAgentMessageType.TOOL_INVOCATION_RESULT,
                    tool_invocation=pending,
                    error=str(e),
                )
            )

            logger.exception(
                "Tool invocation %s failed: %s",
                invocation_id,
                e,
            )

    def deny_tool_invocation(
        self,
        invocation_id: str,
        reason: str | None = None,
    ) -> None:
        """
        Deny a pending tool invocation.

        Args:
            invocation_id: ID of the tool invocation to deny
            reason: Optional reason for denial

        Emits:
            - TOOL_INVOCATION_RESULT with status "denied"
            - ERROR if no pending invocation or ID mismatch
        """
        pending = self._state.pending_tool_invocation

        if not pending:
            self._emit_message(
                GuideAgentMessage(
                    type=GuideAgentMessageType.ERROR,
                    error="No pending tool invocation to deny",
                )
            )
            return

        if pending.invocation_id != invocation_id:
            self._emit_message(
                GuideAgentMessage(
                    type=GuideAgentMessageType.ERROR,
                    error=f"Invocation ID mismatch: expected {pending.invocation_id}",
                )
            )
            return

        # Update status and clear pending
        pending.status = ToolInvocationStatus.DENIED
        self._state.pending_tool_invocation = None

        # Add denial to conversation history
        denial_message = "Tool invocation denied"
        if reason:
            denial_message += f": {reason}"
        self._add_message_to_history(ConversationRole.SYSTEM, denial_message)

        self._emit_message(
            GuideAgentMessage(
                type=GuideAgentMessageType.TOOL_INVOCATION_RESULT,
                tool_invocation=pending,
                content=denial_message,
            )
        )

        logger.info(
            "Tool invocation %s denied: %s",
            invocation_id,
            reason or "no reason provided",
        )

    def get_state(self) -> GuideAgentConversationState:
        """
        Get the current conversation state.

        Returns:
            Current GuideAgentConversationState
        """
        return self._state

    def reset(self) -> None:
        """
        Reset the conversation to a fresh state.

        Generates a new guide_chat_id and clears all messages.
        """
        old_guide_chat_id = self._state.guide_chat_id
        self._state = GuideAgentConversationState(
            guide_chat_id=str(uuid.uuid4())
        )

        logger.info(
            "Reset conversation from %s to %s",
            old_guide_chat_id,
            self._state.guide_chat_id,
        )
