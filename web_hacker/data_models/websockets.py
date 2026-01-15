"""
web_hacker/data_models/websockets.py

Data models for WebSocket communication across all agents.
Consolidates base, browser, and guide WebSocket types.
"""

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Literal, Self

from pydantic import BaseModel, Field


# Base response type enums ________________________________________________________________________

class WebSocketBaseCommandResponseType(StrEnum):
    """
    Common response types shared across all WebSocket endpoints.
    Subapps can extend with their own specific response types.
    """

    PONG = "pong"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"


class WebSocketBaseStreamResponseType(StrEnum):
    """
    Common stream response types shared across all WebSocket endpoints.
    Subapps can extend with their own specific stream types.
    """

    SESSION_ENDED = "session_ended"
    TOOL_INVOCATION_REQUEST = "tool_invocation_request"


# Base response classes ___________________________________________________________________________

class WebSocketResponseBase(BaseModel):
    """
    Base shape for all websocket response messages.
    All agent-specific responses should inherit from this.
    """

    type: str  # generic string type for base, subclasses narrow to specific literals
    timestamp: float = Field(
        default_factory=lambda: datetime.now(tz=timezone.utc).timestamp(),
        description="UTC unix timestamp",
    )


# Common command responses ________________________________________________________________________

class WebSocketPongResponse(WebSocketResponseBase):
    """Response to the ping command."""

    type: Literal[WebSocketBaseCommandResponseType.PONG] = (
        WebSocketBaseCommandResponseType.PONG
    )


class WebSocketSuccessResponse(WebSocketResponseBase):
    """Generic success response for commands that complete successfully."""

    type: Literal[WebSocketBaseCommandResponseType.SUCCESS] = (
        WebSocketBaseCommandResponseType.SUCCESS
    )
    message: str = Field(
        description="Human-readable success message",
    )


class WebSocketWarningResponse(WebSocketResponseBase):
    """Warning response used for non-fatal cases where the command could not be completed."""

    type: Literal[WebSocketBaseCommandResponseType.WARNING] = (
        WebSocketBaseCommandResponseType.WARNING
    )
    message: str = Field(
        description="Human-readable warning message",
    )


class WebSocketErrorResponse(WebSocketResponseBase):
    """Error envelope used when a command is invalid or fails."""

    type: Literal[WebSocketBaseCommandResponseType.ERROR] = (
        WebSocketBaseCommandResponseType.ERROR
    )
    message: str = Field(
        description="Human-readable error message",
    )


# Common stream responses _________________________________________________________________________

class WebSocketSessionEndedResponse(WebSocketResponseBase):
    """
    Notification sent to clients when the session is ending.
    Sent before the WebSocket connection is closed.
    """

    type: Literal[WebSocketBaseStreamResponseType.SESSION_ENDED] = (
        WebSocketBaseStreamResponseType.SESSION_ENDED
    )
    reason: str = Field(
        description="Reason for session end (e.g., 'Session ended manually', 'Timeout reached', 'Error occurred')",
    )
    error_message: str | None = Field(
        default=None,
        description="Optional error message if the session ended due to an error",
    )


class WebSocketToolInvocationRequestResponse(WebSocketResponseBase):
    """
    Generic tool invocation request requiring user confirmation.

    Sent when an agent wants to execute a tool but needs user approval first.
    The client should respond with a confirm or deny command.
    """

    type: Literal[WebSocketBaseStreamResponseType.TOOL_INVOCATION_REQUEST] = (
        WebSocketBaseStreamResponseType.TOOL_INVOCATION_REQUEST
    )
    invocation_id: str = Field(
        description="Unique ID for this invocation (UUIDv4)",
    )
    tool_name: str = Field(
        description="Name of the tool to invoke",
    )
    tool_arguments: dict[str, Any] = Field(
        description="Arguments to pass to the tool",
    )
    explanation: str = Field(
        description="Human-readable explanation of what the tool will do",
    )


# Browser WebSocket types _________________________________________________________________________

## Browser commands from client

class WebSocketClientCommandType(StrEnum):
    """
    Commands that can be sent from the client to /stream-cdp.
    """
    # simple commands
    PING = "ping"
    GET_STATS = "get_stats"
    BACK = "back"
    GET_CURRENT_URL = "get_current_url"

    # commands that expect arguments in the message payload
    SUBSCRIBE = "subscribe"  # client sends "subscribe:<comma-separated categories>" OR "subscribe:all" OR "subscribe" (all)
    NAVIGATE = "navigate"  # client sends "navigate:<url>"


class ParsedWebSocketClientCommand(BaseModel):
    """
    Parsed websocket command with normalized command type and extracted arguments.
    """
    command: WebSocketClientCommandType | None = Field(
        description="The parsed command type, or None if the command was invalid",
    )
    args: str | None = Field(
        default=None,
        description="Raw argument string extracted from the command (e.g., URL for navigate, categories for subscribe)",
    )
    error: str | None = Field(
        default=None,
        description="Error message if the command could not be parsed, or None if parsing succeeded",
    )

    @property
    def is_valid(self) -> bool:
        """Whether the command was successfully parsed."""
        return self.command is not None and self.error is None

    @property
    def subscribe_categories(self) -> set[str] | None:
        """
        Parse subscription categories from args.
        Returns:
            - None if not a subscribe command or args is "all"
            - set() if subscribing to all categories
            - set of category names if subscribing to specific categories
        """
        if self.command != WebSocketClientCommandType.SUBSCRIBE:
            return None
        if self.args == "all":
            return set()
        return {c.strip() for c in self.args.split(",") if c.strip()}

    @property
    def navigate_url(self) -> str | None:
        """
        Get the URL from a navigate command.
        Returns:
            - URL string if this is a navigate command
            - None if not a navigate command
        """
        if self.command != WebSocketClientCommandType.NAVIGATE:
            return None
        return self.args

    @classmethod
    def parse_raw_websocket_command(cls, raw_message: str) -> Self:
        """
        Parse a raw websocket message into a structured command with arguments.

        Args:
            raw_message: The raw message string from the client

        Returns:
            ParsedWebSocketClientCommand with normalized command type and extracted arguments

        Examples:
            >>> ParsedWebSocketClientCommand.parse_raw_websocket_command("ping")
            ParsedWebSocketClientCommand(command="ping", args=None, error=None)

            >>> ParsedWebSocketClientCommand.parse_raw_websocket_command("subscribe")
            ParsedWebSocketClientCommand(command="subscribe", args="all", error=None)

            >>> ParsedWebSocketClientCommand.parse_raw_websocket_command("subscribe:AsyncNetworkMonitor,AsyncStorageMonitor")
            ParsedWebSocketClientCommand(command="subscribe", args="AsyncNetworkMonitor,AsyncStorageMonitor", error=None)

            >>> ParsedWebSocketClientCommand.parse_raw_websocket_command("navigate:https://example.com")
            ParsedWebSocketClientCommand(command="navigate", args="https://example.com", error=None)
        """
        message = raw_message.strip().lower()

        # simple commands (no arguments)
        if message == WebSocketClientCommandType.PING:
            return cls(command=WebSocketClientCommandType.PING)

        if message == WebSocketClientCommandType.GET_STATS:
            return cls(command=WebSocketClientCommandType.GET_STATS)

        if message == WebSocketClientCommandType.BACK:
            return cls(command=WebSocketClientCommandType.BACK)

        if message == WebSocketClientCommandType.GET_CURRENT_URL:
            return cls(command=WebSocketClientCommandType.GET_CURRENT_URL)

        # subscribe command (with optional arguments)
        if message.startswith(WebSocketClientCommandType.SUBSCRIBE):
            remainder = message[len(WebSocketClientCommandType.SUBSCRIBE):]

            # normalize: "subscribe", "subscribe:", "subscribe:all" -> "all"
            if remainder in ("", ":", ":all"):
                return cls(command=WebSocketClientCommandType.SUBSCRIBE, args="all")

            # "subscribe:<categories>"
            if remainder.startswith(":"):
                categories_str = remainder[1:].strip()
                if not categories_str:
                    return cls(
                        command=None,
                        error=f'Invalid subscribe command: "{raw_message}" (missing categories after colon)',
                    )
                return cls(command=WebSocketClientCommandType.SUBSCRIBE, args=categories_str)

            # invalid format (e.g., "subscribeXYZ")
            return cls(
                command=None,
                error=f'Invalid subscribe command: "{raw_message}" (must be of the form "subscribe:<categories>")',
            )

        # navigate command (requires URL argument)
        if message.startswith(WebSocketClientCommandType.NAVIGATE):
            remainder = message[len(WebSocketClientCommandType.NAVIGATE):]
            if not remainder.startswith(":"):
                return cls(
                    command=None,
                    error=f'Invalid navigate command: "{raw_message}" (must be of the form "navigate:<url>")',
                )

            url = remainder[1:].strip()
            if not url:
                return cls(
                    command=None,
                    error=f'Invalid navigate command: "{raw_message}" (missing URL after colon)',
                )
            return cls(
                command=WebSocketClientCommandType.NAVIGATE,
                args=url,
            )

        # unknown command
        return cls(
            command=None,
            error=f'Unknown command: "{raw_message}"',
        )


## Browser response type enums

class WebSocketCommandResponseType(StrEnum):
    """
    Response envelope types the server sends back to the client for direct commands.
    Extends base types with browser-specific response types.
    """
    # inherited from base
    PONG = WebSocketBaseCommandResponseType.PONG
    SUCCESS = WebSocketBaseCommandResponseType.SUCCESS
    WARNING = WebSocketBaseCommandResponseType.WARNING
    ERROR = WebSocketBaseCommandResponseType.ERROR
    # browser-specific
    STATS = "stats"
    SUBSCRIBED = "subscribed"
    CURRENT_URL = "current_url"


class WebSocketStreamResponseType(StrEnum):
    """
    Message types used when the server streams CDP monitor data.
    Extends base types with browser-specific stream types.
    """
    # inherited from base
    SESSION_ENDED = WebSocketBaseStreamResponseType.SESSION_ENDED
    # browser-specific
    SNAPSHOT = "snapshot"
    UPDATE = "update"


## Browser-specific command responses

class WebSocketStatsResponse(WebSocketResponseBase):
    """
    Response to the get_stats command.
    """
    type: Literal[WebSocketCommandResponseType.STATS] = WebSocketCommandResponseType.STATS
    stats: dict[str, Any] = Field(
        description="Current accumulator stats from the event broadcaster",
    )


class WebSocketSubscribedResponse(WebSocketResponseBase):
    """
    Response to subscribe commands, echoing the applied categories.
    """
    type: Literal[WebSocketCommandResponseType.SUBSCRIBED] = WebSocketCommandResponseType.SUBSCRIBED
    categories: str | list[str] = Field(
        description="The categories that were subscribed to",
        default="all",
        examples=[
            "all",
            "AsyncNetworkMonitor",
            "AsyncNetworkMonitor,AsyncStorageMonitor,AsyncWindowPropertyMonitor",
        ],
    )


class WebSocketCurrentUrlResponse(WebSocketResponseBase):
    """
    Response containing the current page URL.
    """
    type: Literal[WebSocketCommandResponseType.CURRENT_URL] = WebSocketCommandResponseType.CURRENT_URL
    url: str = Field(
        description="Current page URL",
    )


## Browser-specific streamed responses

class WebSocketUpdateEvent(BaseModel):
    """
    Lightweight event summary emitted by monitors to connected websocket clients.
    """
    category: str = Field(
        description="Monitor category (class name) that emitted this event",
    )
    summary: dict[str, Any] = Field(
        description="Lightweight summary of the event for real-time streaming",
    )
    timestamp: float = Field(
        description="UTC unix timestamp when the event occurred",
    )


class WebSocketSnapshotResponse(WebSocketResponseBase):
    """
    Initial snapshot message sent after client registration containing current stats.
    """
    type: Literal[WebSocketStreamResponseType.SNAPSHOT] = WebSocketStreamResponseType.SNAPSHOT
    cdp_captures_id: str = Field(
        description="The CDP captures ID for this session",
    )
    stats: dict[str, Any] = Field(
        description="Current accumulator stats",
    )


class WebSocketUpdateResponse(WebSocketResponseBase):
    """
    Throttled update message containing stats plus recent monitor events.
    Sent periodically (throttled by broadcast_interval, default 1.0s).
    """
    type: Literal[WebSocketStreamResponseType.UPDATE] = WebSocketStreamResponseType.UPDATE
    stats: dict[str, Any] = Field(
        description="Current accumulator stats",
    )
    events: list[WebSocketUpdateEvent] = Field(
        description="Buffered events since last broadcast (max 50)",
    )


## Browser server response union

WebSocketServerResponse = (
    # command responses (from base)
    WebSocketPongResponse
    | WebSocketSuccessResponse
    | WebSocketWarningResponse
    | WebSocketErrorResponse
    # command responses (browser-specific)
    | WebSocketStatsResponse
    | WebSocketSubscribedResponse
    | WebSocketCurrentUrlResponse
    # streamed responses (from base)
    | WebSocketSessionEndedResponse
    # streamed responses (browser-specific)
    | WebSocketSnapshotResponse
    | WebSocketUpdateResponse
)


# Guide WebSocket types ___________________________________________________________________________

## Guide commands from client

class GuideWebSocketClientCommandType(StrEnum):
    """Commands that can be sent from the client to the guide agent."""

    # simple commands
    PING = "ping"
    GET_STATE = "get_state"
    RESET = "reset"

    # commands that expect arguments in the message payload
    SEND_MESSAGE = "send_message"  # client sends "send_message:<content>"
    CONFIRM_TOOL = "confirm_tool"  # client sends "confirm_tool:<invocation_id>"
    DENY_TOOL = "deny_tool"  # client sends "deny_tool:<invocation_id>" or "deny_tool:<invocation_id>:<reason>"


class ParsedGuideWebSocketClientCommand(BaseModel):
    """Parsed websocket command with normalized command type and extracted arguments."""

    command: GuideWebSocketClientCommandType | None = Field(
        description="The parsed command type, or None if the command was invalid",
    )
    args: str | None = Field(
        default=None,
        description="Raw argument string extracted from the command",
    )
    error: str | None = Field(
        default=None,
        description="Error message if the command could not be parsed, or None if parsing succeeded",
    )

    @property
    def is_valid(self) -> bool:
        """Whether the command was successfully parsed."""
        return self.command is not None and self.error is None

    @property
    def message_content(self) -> str | None:
        """
        Extract message content from send_message command.

        Returns:
            - None if not a send_message command
            - Empty string if send_message with no content
            - Content string if send_message with content
        """
        if self.command != GuideWebSocketClientCommandType.SEND_MESSAGE:
            return None
        return self.args or ""

    @property
    def invocation_id(self) -> str | None:
        """
        Extract invocation ID from confirm_tool or deny_tool command.

        Returns:
            - None if not a confirm/deny command
            - Invocation ID string if confirm/deny command
        """
        if self.command not in (
            GuideWebSocketClientCommandType.CONFIRM_TOOL,
            GuideWebSocketClientCommandType.DENY_TOOL,
        ):
            return None
        if not self.args:
            return None
        # For deny_tool, args could be "invocation_id:reason"
        return self.args.split(":")[0]

    @property
    def deny_reason(self) -> str | None:
        """
        Extract denial reason from deny_tool command.

        Returns:
            - None if not a deny_tool command or no reason provided
            - Reason string if deny_tool with reason
        """
        if self.command != GuideWebSocketClientCommandType.DENY_TOOL:
            return None
        if not self.args or ":" not in self.args:
            return None
        parts = self.args.split(":", 1)
        return parts[1] if len(parts) > 1 else None

    @classmethod
    def parse_raw_websocket_command(cls, raw_message: str) -> Self:
        """
        Parse a raw websocket message into a structured command with arguments.

        Args:
            raw_message: The raw message string from the client

        Returns:
            ParsedGuideWebSocketClientCommand with normalized command type and extracted arguments

        Examples:
            >>> ParsedGuideWebSocketClientCommand.parse_raw_websocket_command("ping")
            ParsedGuideWebSocketClientCommand(command="ping", args=None, error=None)

            >>> ParsedGuideWebSocketClientCommand.parse_raw_websocket_command("send_message:Hello")
            ParsedGuideWebSocketClientCommand(command="send_message", args="Hello", error=None)

            >>> ParsedGuideWebSocketClientCommand.parse_raw_websocket_command("confirm_tool:abc-123")
            ParsedGuideWebSocketClientCommand(command="confirm_tool", args="abc-123", error=None)
        """
        message = raw_message.strip().lower()

        # simple commands (no arguments)
        if message == GuideWebSocketClientCommandType.PING:
            return cls(command=GuideWebSocketClientCommandType.PING)

        if message == GuideWebSocketClientCommandType.GET_STATE:
            return cls(command=GuideWebSocketClientCommandType.GET_STATE)

        if message == GuideWebSocketClientCommandType.RESET:
            return cls(command=GuideWebSocketClientCommandType.RESET)

        # send_message command (preserve original case for content)
        if message.startswith(GuideWebSocketClientCommandType.SEND_MESSAGE):
            remainder = message[len(GuideWebSocketClientCommandType.SEND_MESSAGE):]

            if remainder == "":
                return cls(command=GuideWebSocketClientCommandType.SEND_MESSAGE, args=None)

            if remainder.startswith(":"):
                # preserve original case for message content
                original_remainder = raw_message.strip()[
                    len(GuideWebSocketClientCommandType.SEND_MESSAGE) + 1:
                ]
                return cls(
                    command=GuideWebSocketClientCommandType.SEND_MESSAGE,
                    args=original_remainder.strip() if original_remainder.strip() else None,
                )

            return cls(
                command=None,
                error=f'Invalid send_message command: "{raw_message}" (must be "send_message" or "send_message:<content>")',
            )

        # confirm_tool command
        if message.startswith(GuideWebSocketClientCommandType.CONFIRM_TOOL):
            remainder = message[len(GuideWebSocketClientCommandType.CONFIRM_TOOL):]

            if remainder == "":
                return cls(
                    command=None,
                    error='confirm_tool requires an invocation_id (e.g., "confirm_tool:<invocation_id>")',
                )

            if remainder.startswith(":"):
                original_remainder = raw_message.strip()[
                    len(GuideWebSocketClientCommandType.CONFIRM_TOOL) + 1:
                ]
                invocation_id = original_remainder.strip()
                if not invocation_id:
                    return cls(
                        command=None,
                        error='confirm_tool requires an invocation_id',
                    )
                return cls(
                    command=GuideWebSocketClientCommandType.CONFIRM_TOOL,
                    args=invocation_id,
                )

            return cls(
                command=None,
                error=f'Invalid confirm_tool command: "{raw_message}"',
            )

        # deny_tool command
        if message.startswith(GuideWebSocketClientCommandType.DENY_TOOL):
            remainder = message[len(GuideWebSocketClientCommandType.DENY_TOOL):]

            if remainder == "":
                return cls(
                    command=None,
                    error='deny_tool requires an invocation_id (e.g., "deny_tool:<invocation_id>" or "deny_tool:<invocation_id>:<reason>")',
                )

            if remainder.startswith(":"):
                original_remainder = raw_message.strip()[
                    len(GuideWebSocketClientCommandType.DENY_TOOL) + 1:
                ]
                args = original_remainder.strip()
                if not args:
                    return cls(
                        command=None,
                        error='deny_tool requires an invocation_id',
                    )
                return cls(
                    command=GuideWebSocketClientCommandType.DENY_TOOL,
                    args=args,
                )

            return cls(
                command=None,
                error=f'Invalid deny_tool command: "{raw_message}"',
            )

        # unknown command
        return cls(
            command=None,
            error=f'Unknown command: "{raw_message}"',
        )


## Guide response type enums

class GuideWebSocketCommandResponseType(StrEnum):
    """
    Response envelope types the server sends back to the client for direct commands.
    Extends base types with guide-specific response types.
    """

    # inherited from base
    PONG = WebSocketBaseCommandResponseType.PONG
    SUCCESS = WebSocketBaseCommandResponseType.SUCCESS
    WARNING = WebSocketBaseCommandResponseType.WARNING
    ERROR = WebSocketBaseCommandResponseType.ERROR
    # guide-specific
    MESSAGE = "message"
    STATE = "state"


class GuideWebSocketStreamResponseType(StrEnum):
    """
    Message types used when the server streams guide agent updates.
    Extends base types with guide-specific stream types.
    """

    # inherited from base
    SESSION_ENDED = WebSocketBaseStreamResponseType.SESSION_ENDED
    TOOL_INVOCATION_REQUEST = WebSocketBaseStreamResponseType.TOOL_INVOCATION_REQUEST
    # guide-specific
    TOOL_INVOCATION_RESULT = "tool_invocation_result"


## Guide-specific command responses

class GuideWebSocketMessageResponse(WebSocketResponseBase):
    """Agent chat message response."""

    type: Literal[GuideWebSocketCommandResponseType.MESSAGE] = (
        GuideWebSocketCommandResponseType.MESSAGE
    )
    content: str = Field(
        description="The agent's message content",
    )


class GuideWebSocketStateResponse(WebSocketResponseBase):
    """Response to the get_state command with current conversation state."""

    type: Literal[GuideWebSocketCommandResponseType.STATE] = (
        GuideWebSocketCommandResponseType.STATE
    )
    guide_chat_id: str = Field(
        description="Current session ID",
    )
    message_count: int = Field(
        description="Number of messages in conversation history",
    )
    has_pending_tool_invocation: bool = Field(
        description="Whether there's a pending tool invocation awaiting confirmation",
    )
    pending_tool_invocation_id: str | None = Field(
        default=None,
        description="ID of the pending tool invocation, if any",
    )


## Guide-specific streamed responses

class GuideWebSocketToolInvocationResultResponse(WebSocketResponseBase):
    """Result after tool confirmation/denial/execution."""

    type: Literal[GuideWebSocketStreamResponseType.TOOL_INVOCATION_RESULT] = (
        GuideWebSocketStreamResponseType.TOOL_INVOCATION_RESULT
    )
    invocation_id: str = Field(
        description="ID of the tool invocation",
    )
    status: str = Field(
        description="Status of the invocation (confirmed, denied, executed, failed)",
    )
    result: dict[str, Any] | None = Field(
        default=None,
        description="Result data from tool execution, if successful",
    )
    error: str | None = Field(
        default=None,
        description="Error message if execution failed",
    )


## Guide server response union

GuideWebSocketServerResponse = (
    # command responses (from base)
    WebSocketPongResponse
    | WebSocketSuccessResponse
    | WebSocketWarningResponse
    | WebSocketErrorResponse
    # command responses (guide-specific)
    | GuideWebSocketMessageResponse
    | GuideWebSocketStateResponse
    # streamed responses (from base)
    | WebSocketSessionEndedResponse
    | WebSocketToolInvocationRequestResponse
    # streamed responses (guide-specific)
    | GuideWebSocketToolInvocationResultResponse
)
