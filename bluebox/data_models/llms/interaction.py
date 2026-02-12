"""
bluebox/data_models/llms/interaction.py

Data models for LLM interactions and agent communication.
"""

from datetime import datetime, timezone
from enum import StrEnum
from typing import Annotated, Any, Literal, Union
from uuid import uuid4

from pydantic import BaseModel, Field

from bluebox.data_models.resource_base import ResourceBase
from bluebox.data_models.routine import Routine


class ChatRole(StrEnum):
    """
    Role in a chat message.
    """
    USER = "user"
    USER_ACTION = "user_action"
    ASSISTANT = "assistant"  # AI
    SYSTEM = "system"
    TOOL = "tool"
    SYSTEM_STATUS_UPDATE = "system_status_update"


class ToolInvocationStatus(StrEnum):
    """
    Status of a tool invocation.
    """
    PENDING_CONFIRMATION = "pending_confirmation"
    CONFIRMED = "confirmed"
    DENIED = "denied"
    EXECUTED = "executed"
    FAILED = "failed"


class SuggestedEditType(StrEnum):
    """
    Types of suggested edits.
    """
    ROUTINE = "routine"


class SuggestedEditStatus(StrEnum):
    """
    Status of a suggested edit.
    """
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    
    
class SuggestedEdit(ResourceBase):
    """
    Base model for suggested edits that require user approval.
    """
    type: SuggestedEditType = Field(
        ...,
        description="Type of suggested edit",
    )
    status: SuggestedEditStatus = Field(
        default=SuggestedEditStatus.PENDING,
        description="Current status of the suggested edit",
    )
    chat_thread_id: str = Field(
        ...,
        description="ID of the ChatThread this edit belongs to",
    )


class SuggestedEditRoutine(SuggestedEdit):
    """
    Suggested edit for a routine.
    """
    type: Literal[SuggestedEditType.ROUTINE] = SuggestedEditType.ROUTINE
    routine: Routine = Field(..., description="The new/modified routine object")


# Union of all suggested edit types - discriminated by 'type' field
SuggestedEditUnion = Annotated[
    Union[
        SuggestedEditRoutine,
        # Add new types here
    ],
    Field(discriminator="type"),
]


class PendingToolInvocation(BaseModel):
    """
    A tool invocation awaiting user confirmation.
    """
    invocation_id: str = Field(
        ...,
        description="Unique ID for this invocation (UUIDv4)",
    )
    tool_name: str = Field(
        ...,
        description="Name of the tool to invoke",
    )
    tool_arguments: dict[str, Any] = Field(
        ...,
        description="Arguments to pass to the tool",
    )
    call_id: str | None = Field(
        default=None,
        description="LLM's call ID for this tool invocation (for Responses API)",
    )
    status: ToolInvocationStatus = Field(
        default=ToolInvocationStatus.PENDING_CONFIRMATION,
        description="Current status of the invocation",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(tz=timezone.utc),
        description="When the invocation was created",
    )


class EmittedMessageType(StrEnum):
    """
    Types of messages the chat can emit via callback.
    """

    CHAT_RESPONSE = "chat_response"
    TOOL_INVOCATION_REQUEST = "tool_invocation_request"
    TOOL_INVOCATION_RESULT = "tool_invocation_result"
    SUGGESTED_EDIT = "suggested_edit"
    BROWSER_RECORDING_REQUEST = "browser_recording_request"
    ROUTINE_DISCOVERY_REQUEST = "routine_discovery_request"
    ROUTINE_CREATION_REQUEST = "routine_creation_request"
    STATUS_UPDATE = "status_update"
    ERROR = "error"


# Emitted message classes - discriminated by 'type' field _______________________________________

class BaseEmittedMessage(BaseModel):
    """
    Base class for messages emitted by the guide agent via callback.

    This is the internal message format used by GuideAgent to communicate
    with its host (e.g., CLI, WebSocket handler in servers repo).
    """
    type: EmittedMessageType = Field(
        ...,
        description="The type of message being emitted",
    )
    timestamp: float = Field(
        default_factory=lambda: datetime.now(tz=timezone.utc).timestamp(),
        description="When the message was created",
    )
    chat_thread_id: str | None = Field(
        default=None,
        description="ID of the ChatThread this message belongs to",
    )


class ChatResponseEmittedMessage(BaseEmittedMessage):
    """Chat response message with text content."""
    type: Literal[EmittedMessageType.CHAT_RESPONSE] = EmittedMessageType.CHAT_RESPONSE
    content: str = Field(..., description="Text content of the chat response")
    chat_id: str | None = Field(
        default=None,
        description="ID of the Chat message",
    )


class StatusUpdateEmittedMessage(BaseEmittedMessage):
    """Ephemeral status/progress update (not persisted as a chat message)."""
    type: Literal[EmittedMessageType.STATUS_UPDATE] = EmittedMessageType.STATUS_UPDATE
    content: str = Field(..., description="Status update text content")


class ToolInvocationRequestEmittedMessage(BaseEmittedMessage):
    """Tool invocation request awaiting confirmation."""
    type: Literal[EmittedMessageType.TOOL_INVOCATION_REQUEST] = EmittedMessageType.TOOL_INVOCATION_REQUEST
    tool_invocation: PendingToolInvocation = Field(
        ...,
        description="Tool invocation details",
    )


class ToolInvocationResultEmittedMessage(BaseEmittedMessage):
    """Result of a tool invocation."""
    type: Literal[EmittedMessageType.TOOL_INVOCATION_RESULT] = EmittedMessageType.TOOL_INVOCATION_RESULT
    tool_invocation: PendingToolInvocation = Field(
        ...,
        description="Tool invocation that was executed",
    )
    tool_result: str | dict[str, Any] = Field(
        ...,
        description="Result data from tool execution",
    )


class SuggestedEditEmittedMessage(BaseEmittedMessage):
    """Suggested edit for user approval."""
    type: Literal[EmittedMessageType.SUGGESTED_EDIT] = EmittedMessageType.SUGGESTED_EDIT
    suggested_edit: SuggestedEditUnion = Field(
        ...,
        description="Suggested edit details",
    )


class BrowserRecordingRequestEmittedMessage(BaseEmittedMessage):
    """Request to start browser recording."""
    type: Literal[EmittedMessageType.BROWSER_RECORDING_REQUEST] = EmittedMessageType.BROWSER_RECORDING_REQUEST
    browser_recording_task: str = Field(
        ...,
        description="Task description for the browser recording",
    )


class RoutineDiscoveryRequestEmittedMessage(BaseEmittedMessage):
    """Request to start routine discovery."""
    type: Literal[EmittedMessageType.ROUTINE_DISCOVERY_REQUEST] = EmittedMessageType.ROUTINE_DISCOVERY_REQUEST
    routine_discovery_task: str = Field(
        ...,
        description="Task description for routine discovery",
    )


class RoutineCreationRequestEmittedMessage(BaseEmittedMessage):
    """Request to create and save a new routine."""
    type: Literal[EmittedMessageType.ROUTINE_CREATION_REQUEST] = EmittedMessageType.ROUTINE_CREATION_REQUEST
    created_routine: Routine = Field(
        ...,
        description="The routine object to create",
    )


class ErrorEmittedMessage(BaseEmittedMessage):
    """Error message."""
    type: Literal[EmittedMessageType.ERROR] = EmittedMessageType.ERROR
    error: str = Field(..., description="Error message")
    content: str | None = Field(
        default=None,
        description="Additional error context",
    )


# Union of all emitted message types - discriminated by 'type' field
EmittedMessage = Annotated[
    Union[
        ChatResponseEmittedMessage,
        StatusUpdateEmittedMessage,
        ToolInvocationRequestEmittedMessage,
        ToolInvocationResultEmittedMessage,
        SuggestedEditEmittedMessage,
        BrowserRecordingRequestEmittedMessage,
        RoutineDiscoveryRequestEmittedMessage,
        RoutineCreationRequestEmittedMessage,
        ErrorEmittedMessage,
    ],
    Field(discriminator="type"),
]


class LLMToolCall(BaseModel):
    """
    A tool call requested by the LLM.
    """
    tool_name: str = Field(
        ...,
        description="Name of the tool to invoke",
    )
    tool_arguments: dict[str, Any] = Field(
        ...,
        description="Arguments to pass to the tool",
    )
    call_id: str | None = Field(
        default=None,
        description="Unique ID for this tool call (required for Responses API)",
    )


class LLMChatResponse(BaseModel):
    """
    Response from an LLM chat completion with tool support.
    """
    # TODO: capture more metadata here returned from the LLM provider; probably standardize in this class
    content: str | None = Field(
        default=None,
        description="Text content of the response",
    )
    tool_calls: list[LLMToolCall] = Field(
        default_factory=list,
        description="Tool calls requested by the LLM",
    )
    response_id: str | None = Field(
        default=None,
        description="Response ID for chaining (OpenAI Responses API)",
    )
    reasoning_content: str | None = Field(
        default=None,
        description="Extended reasoning content",
    )
    parsed: Any | None = Field(
        default=None,
        description="Parsed structured output when response_model is provided",
    )


class Chat(BaseModel):
    """
    A single message in a conversation.
    """
    id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique message ID (UUIDv4)",
    )
    chat_thread_id: str = Field(
        ...,
        description="ID of the parent thread this message belongs to",
    )
    role: ChatRole = Field(
        ...,
        description="The role of the message sender (user, assistant, system, tool)",
    )
    content: str = Field(
        ...,
        description="The content of the message",
    )
    tool_call_id: str | None = Field(
        default=None,
        description="For TOOL role messages, the call_id this is a response to",
    )
    tool_calls: list[LLMToolCall] = Field(
        default_factory=list,
        description="For ASSISTANT role messages, any tool calls made",
    )
    chat_cache_key: str | None = Field(
        default=None,
        description="Cache key for the message",
    )
    llm_provider_response_id: str | None = Field(
        default=None,
        description="Response ID for the message from the LLM provider",
    )


class ChatThread(BaseModel):
    """
    Container for a conversation thread.
    """
    id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique thread ID (UUIDv4)",
    )
    chat_ids: list[str] = Field(
        default_factory=list,
        description="Ordered list of message IDs in this thread",
    )
    suggested_edit_ids : list[str] = Field(
        default_factory=list,
        description="List of suggested edit IDs in this thread",
    )
    pending_tool_invocation: PendingToolInvocation | None = Field(
        default=None,
        description="Tool invocation awaiting user confirmation, if any",
    )
    updated_at: int = Field(
        default=0,
        description="Unix timestamp (seconds) when thread was last updated",
    )
