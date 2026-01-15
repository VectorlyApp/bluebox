"""
web_hacker/data_models/guide_agent/conversation.py

Conversation state data models for the guide agent.
"""

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class ConversationRole(StrEnum):
    """Role in a conversation message."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class ConversationMessage(BaseModel):
    """A single message in the conversation history."""

    role: ConversationRole = Field(
        ...,
        description="The role of the message sender",
    )
    content: str = Field(
        ...,
        description="The content of the message",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the message was created",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata for the message",
    )


class ToolInvocationStatus(StrEnum):
    """Status of a tool invocation."""

    PENDING_CONFIRMATION = "pending_confirmation"
    CONFIRMED = "confirmed"
    DENIED = "denied"
    EXECUTED = "executed"
    FAILED = "failed"


class PendingToolInvocation(BaseModel):
    """A tool invocation awaiting user confirmation."""

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
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the invocation was created",
    )


class GuideAgentConversationState(BaseModel):
    """
    Full state of a guide agent conversation session.
    """

    guide_chat_id: str = Field(
        ...,
        description="Unique session identifier (UUIDv4)",
    )
    messages: list[ConversationMessage] = Field(
        default_factory=list,
        description="Conversation history",
    )
    pending_tool_invocation: PendingToolInvocation | None = Field(
        default=None,
        description="Tool invocation awaiting confirmation, if any",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the session was created",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the session was last updated",
    )
