"""
web_hacker/data_models/llms/interaction.py

Data models for LLM interactions and agent communication.
"""

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class ChatRole(StrEnum):
    """
    Role in a chat message.
    """
    USER = "user"
    ASSISTANT = "assistant"  # AI
    SYSTEM = "system"
    TOOL = "tool"


class ToolInvocationStatus(StrEnum):
    """
    Status of a tool invocation.
    """
    PENDING_CONFIRMATION = "pending_confirmation"
    CONFIRMED = "confirmed"
    DENIED = "denied"
    EXECUTED = "executed"
    FAILED = "failed"


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
    status: ToolInvocationStatus = Field(
        default=ToolInvocationStatus.PENDING_CONFIRMATION,
        description="Current status of the invocation",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(tz=timezone.utc),
        description="When the invocation was created",
    )


class ChatMessageType(StrEnum):
    """
    Types of messages the chat can emit via callback.
    """

    CHAT_RESPONSE = "chat_response"
    TOOL_INVOCATION_REQUEST = "tool_invocation_request"
    TOOL_INVOCATION_RESULT = "tool_invocation_result"
    ERROR = "error"


class EmittedChatMessage(BaseModel):
    """
    Message emitted by the guide agent via callback.

    This is the internal message format used by GuideAgent to communicate
    with its host (e.g., CLI, WebSocket handler in servers repo).
    """

    type: ChatMessageType = Field(
        ...,
        description="The type of message being emitted",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(tz=timezone.utc),
        description="When the message was created",
    )
    content: str | None = Field(
        default=None,
        description="Text content for chat responses or error messages",
    )
    tool_invocation: PendingToolInvocation | None = Field(
        default=None,
        description="Tool invocation details for request/result messages",
    )
    tool_result: dict[str, Any] | None = Field(
        default=None,
        description="Result data from tool execution",
    )
    error: str | None = Field(
        default=None,
        description="Error message if type is ERROR",
    )


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


class LLMChatResponse(BaseModel):
    """
    Response from an LLM chat completion with tool support.
    """
    content: str | None = Field(
        default=None,
        description="Text content of the response",
    )
    tool_call: LLMToolCall | None = Field(
        default=None,
        description="Tool call requested by the LLM, if any",
    )


class Chat(BaseModel):
    """
    A single message in a conversation.
    """
    id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique message ID (UUIDv4)",
    )
    thread_id: str = Field(
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


class ChatThread(BaseModel):
    """
    Container for a conversation thread.
    """
    id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique thread ID (UUIDv4)",
    )
    message_ids: list[str] = Field(
        default_factory=list,
        description="Ordered list of message IDs in this thread",
    )
    pending_tool_invocation: PendingToolInvocation | None = Field(
        default=None,
        description="Tool invocation awaiting user confirmation, if any",
    )
    updated_at: int = Field(
        default=0,
        description="Unix timestamp (seconds) when thread was last updated",
    )
