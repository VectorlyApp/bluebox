"""
bluebox/utils/tui_base.py

Base class for multi-pane Textual TUIs that wrap an agent.

Provides the shared layout (chat + input | tool log + status), streaming,
message routing, context estimation, slash-command dispatch, and chat history.
Subclasses override a handful of abstract / hook methods to customise the agent,
welcome screen, status panel, and any extra slash commands.
"""

from __future__ import annotations

import json
from abc import abstractmethod
from datetime import datetime
from textwrap import dedent
from typing import TYPE_CHECKING, ClassVar

from rich.markdown import Markdown as RichMarkdown
from rich.markup import escape
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.suggester import SuggestFromList
from textual.widgets import Input, RichLog, Static

from bluebox.data_models.llms.interaction import (
    ChatRole,
    BaseEmittedMessage,
    ChatResponseEmittedMessage,
    ToolInvocationResultEmittedMessage,
    ErrorEmittedMessage,
    ToolInvocationStatus,
)
from bluebox.data_models.llms.vendors import LLMModel

if TYPE_CHECKING:
    from bluebox.agents.abstract_agent import AbstractAgent


# ─── Shared constants ────────────────────────────────────────────────────────

MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    "gpt-4": 128_000,
    "gpt-5": 128_000,
    "o3": 200_000,
    "o4": 200_000,
}
DEFAULT_CONTEXT_WINDOW = 128_000

BASE_SLASH_COMMANDS: list[str] = [
    "/reset", "/status", "/chats", "/clear", "/help", "/quit",
]

BASE_HELP_TEXT = dedent("""\
    [bold]Commands:[/bold]
    [cyan]/status[/cyan]           Show current state
    [cyan]/chats[/cyan]            Show message history
    [cyan]/clear[/cyan]            Clear the chat display
    [cyan]/reset[/cyan]            Start new conversation
    [cyan]/help[/cyan]             Show this help
    [cyan]/quit[/cyan]             Exit
""")

APP_CSS = dedent("""\
    Screen {
        layout: horizontal;
    }

    #left-pane {
        width: 3fr;
        layout: vertical;
    }

    #right-pane {
        width: 1fr;
        layout: vertical;
    }

    #chat-log {
        height: 1fr;
        border: solid $accent;
        border-title-color: $accent;
    }

    #user-input {
        height: 3;
        margin: 0 0;
    }

    #tool-log {
        height: 1fr;
        border: solid $secondary;
        border-title-color: $secondary;
    }

    #status-panel {
        height: auto;
        min-height: 8;
        max-height: 16;
        border: solid $primary;
        border-title-color: $primary;
        padding: 0 1;
    }
""")


# ─── Helpers ─────────────────────────────────────────────────────────────────

def get_context_window_size(model_value: str) -> int:
    """Get approximate context window size for a model."""
    for prefix, size in MODEL_CONTEXT_WINDOWS.items():
        if model_value.startswith(prefix):
            return size
    return DEFAULT_CONTEXT_WINDOW


# ─── Slash-command suggester ─────────────────────────────────────────────────

class SlashCommandSuggester(SuggestFromList):
    """Only suggest when input starts with '/'."""

    async def get_suggestion(self, value: str) -> str | None:
        if not value.startswith("/"):
            return None
        return await super().get_suggestion(value)


# ─── Base TUI ────────────────────────────────────────────────────────────────

class AbstractAgentTUI(App):
    """Base Textual app for agent TUIs with the standard two-column layout.

    Subclasses must implement:
        _create_agent     — instantiate and return the agent
        _print_welcome    — write welcome content to the chat log
        _build_status_text — return Rich markup for the right-pane status panel

    Subclasses may override:
        _handle_custom_command   — handle agent-specific slash commands
        _handle_additional_message — handle extra emitted-message types
        _on_reset          — extra cleanup when /reset is invoked
        _pre_process_input — intercept input before slash-command dispatch
        _format_user_label — customise the "[bold green]You>[/bold green]" prefix
    """

    CSS = APP_CSS
    TITLE = "Agent"
    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", priority=True),
    ]

    # Override in subclasses to extend the command palette.
    SLASH_COMMANDS: ClassVar[list[str]] = BASE_SLASH_COMMANDS
    HELP_TEXT: ClassVar[str] = BASE_HELP_TEXT

    # ── Init ─────────────────────────────────────────────────────────────

    def __init__(self, llm_model: LLMModel) -> None:
        super().__init__()
        self._llm_model = llm_model
        self._context_window_size = get_context_window_size(llm_model.value)

        # Agent — set in on_mount via _create_agent()
        self._agent: AbstractAgent | None = None

        # Streaming state
        self._streaming_started: bool = False
        self._stream_buffer: list[str] = []

        # Counters
        self._tool_call_count: int = 0
        self._last_seen_chat_count: int = 0

    # ── Abstract / hook methods ──────────────────────────────────────────

    @abstractmethod
    def _create_agent(self) -> AbstractAgent:
        """Instantiate and return the agent (must extend AbstractAgent).

        Pass ``self._handle_message`` as the emit callback and
        ``self._handle_stream_chunk`` as the stream callback.
        """

    @abstractmethod
    def _print_welcome(self) -> None:
        """Write welcome / intro content to ``#chat-log``."""

    @abstractmethod
    def _build_status_text(self) -> str:
        """Return Rich markup string for the right-pane status panel."""

    def _handle_custom_command(self, cmd: str, raw_input: str) -> bool:
        """Handle an agent-specific slash command.

        Args:
            cmd: The lowercased input.
            raw_input: The original (untrimmed-case) input.

        Returns:
            True if the command was handled, False to fall through.
        """
        return False

    def _handle_additional_message(self, message: BaseEmittedMessage) -> bool:
        """Handle an emitted message type not covered by the base class.

        Returns True if handled, False to ignore.
        """
        return False

    def _on_reset(self) -> None:
        """Extra cleanup when the user runs /reset."""

    def _pre_process_input(self, user_input: str) -> bool:
        """Intercept user input before slash-command dispatch.

        Return True to consume the input (skip normal processing).
        """
        return False

    def _format_user_label(self) -> str:
        """Return Rich markup for the user-message prefix."""
        return "[bold green]You>[/bold green]"

    # ── Compose ──────────────────────────────────────────────────────────

    def compose(self) -> ComposeResult:
        with Horizontal():
            with Vertical(id="left-pane"):
                chat_log = RichLog(id="chat-log", wrap=True, highlight=True, markup=True)
                chat_log.border_title = "Chat"
                yield chat_log
                yield Input(
                    placeholder="Type a message or /help ...",
                    id="user-input",
                    suggester=SlashCommandSuggester(
                        self.SLASH_COMMANDS, case_sensitive=False,
                    ),
                )
            with Vertical(id="right-pane"):
                tool_log = RichLog(id="tool-log", wrap=True, highlight=True, markup=True)
                tool_log.border_title = "Tools"
                yield tool_log
                status = Static(id="status-panel")
                status.border_title = "Info"
                yield status

    # ── Lifecycle ────────────────────────────────────────────────────────

    def on_mount(self) -> None:
        chat = self.query_one("#chat-log", RichLog)
        try:
            self._agent = self._create_agent()
        except Exception as e:
            chat.write(Text.from_markup(
                f"[bold red]Failed to initialize agent:[/bold red] {escape(str(e))}"
            ))
            return

        self._print_welcome()
        self._update_status()
        self.set_interval(10, self._update_status)
        self.query_one("#user-input", Input).focus()

    # ── Status panel ─────────────────────────────────────────────────────

    def _estimate_context_usage(self) -> tuple[int, float]:
        """Estimate tokens used and percentage of context window filled."""
        if not self._agent:
            return 0, 0.0
        total_chars = 0
        for c in self._agent.get_chats():
            total_chars += len(c.content or "")
            if c.tool_calls:
                for tc in c.tool_calls:
                    if hasattr(tc, "tool_arguments"):
                        total_chars += len(json.dumps(tc.tool_arguments))
        tokens = total_chars // 4
        pct = (tokens / self._context_window_size) * 100 if self._context_window_size > 0 else 0.0
        return tokens, min(pct, 100.0)

    def _context_bar(self, pct: float, width: int = 15) -> str:
        """Render a text-based context-fill bar."""
        filled = int(pct / 100 * width)
        empty = width - filled
        if pct < 50:
            color = "green"
        elif pct < 80:
            color = "yellow"
        else:
            color = "red"
        bar = "\u2588" * filled + "\u2591" * empty
        return f"[{color}]{bar}[/{color}] {pct:.0f}%"

    def _update_status(self) -> None:
        """Refresh the right-pane status panel."""
        panel = self.query_one("#status-panel", Static)
        panel.update(Text.from_markup(self._build_status_text()))

    # ── Agent callbacks ──────────────────────────────────────────────────

    def _handle_stream_chunk(self, chunk: str) -> None:
        """Buffer streaming chunks for markdown rendering when complete."""
        chat = self.query_one("#chat-log", RichLog)
        if not self._streaming_started:
            chat.write(Text.from_markup("\n[bold cyan]Assistant[/bold cyan]"))
            self._streaming_started = True
        self._stream_buffer.append(chunk)

    def _flush_auto_executed_tools(self, tool_log: RichLog, chat: RichLog) -> None:
        """Detect and log auto-executed tools by scanning new chat entries."""
        if not self._agent:
            return
        chats = self._agent.get_chats()
        current_count = len(chats)
        if current_count <= self._last_seen_chat_count:
            self._last_seen_chat_count = current_count
            return

        ts = datetime.now().strftime("%H:%M:%S")
        for c in chats[self._last_seen_chat_count:]:
            if c.role.value == "tool":
                self._tool_call_count += 1
                tool_name = "tool"
                if c.content and c.content.startswith("Tool '"):
                    end = c.content.find("'", 6)
                    if end > 6:
                        tool_name = c.content[6:end]

                chat.write(Text.from_markup(
                    f"[green]\u2713[/green] [dim]{tool_name} (auto)[/dim]"
                ))
                tool_log.write(Text.from_markup(
                    f"[dim]{ts}[/dim] [green]AUTO[/green] [bold]{tool_name}[/bold]"
                ))
                result_preview = (
                    c.content[c.content.find("result: ") + 8:]
                    if "result: " in (c.content or "")
                    else (c.content or "")
                )
                if len(result_preview) > 300:
                    result_preview = result_preview[:300] + "..."
                tool_log.write(result_preview)
                tool_log.write("")

            elif c.role.value == "assistant" and c.tool_calls:
                for tc in c.tool_calls:
                    tool_log.write(Text.from_markup(
                        f"[dim]{ts}[/dim] [yellow]CALL[/yellow] [bold]{tc.tool_name}[/bold]"
                    ))
                    if hasattr(tc, "tool_arguments") and tc.tool_arguments:
                        args_str = json.dumps(tc.tool_arguments, indent=2)
                        lines = args_str.split("\n")
                        if len(lines) > 15:
                            args_str = "\n".join(lines[:15]) + f"\n... ({len(lines) - 15} more)"
                        tool_log.write(args_str)
                    tool_log.write("")

        self._last_seen_chat_count = current_count

    def _handle_message(self, message: BaseEmittedMessage) -> None:
        """Route emitted messages to the appropriate pane."""
        chat = self.query_one("#chat-log", RichLog)
        tool_log = self.query_one("#tool-log", RichLog)

        # Let subclass handle first — if it returns True, we're done.
        if self._handle_additional_message(message):
            self._update_status()
            return

        if isinstance(message, ChatResponseEmittedMessage):
            self._flush_auto_executed_tools(tool_log, chat)

            if not self._streaming_started:
                chat.write(Text.from_markup("\n[bold cyan]Assistant[/bold cyan]"))

            # Prefer the message content; fall back to stream buffer
            content = message.content or "".join(self._stream_buffer)
            self._stream_buffer.clear()
            self._streaming_started = False

            if content:
                chat.write(RichMarkdown(content))
            chat.write("")

        elif isinstance(message, ToolInvocationResultEmittedMessage):
            inv = message.tool_invocation
            ts = datetime.now().strftime("%H:%M:%S")
            self._tool_call_count += 1

            if inv.status == ToolInvocationStatus.EXECUTED:
                chat.write(Text.from_markup(
                    f"[green]\u2713[/green] [dim]{inv.tool_name} executed[/dim]"
                ))
                tool_log.write(Text.from_markup(
                    f"[dim]{ts}[/dim] [green]RESULT[/green] [bold]{inv.tool_name}[/bold]"
                ))
                if isinstance(message.tool_result, dict):
                    result_str = json.dumps(message.tool_result, indent=2)
                    lines = result_str.split("\n")
                    if len(lines) > 30:
                        result_str = "\n".join(lines[:30]) + f"\n... ({len(lines) - 30} more)"
                    tool_log.write(result_str)
                elif message.tool_result:
                    tool_log.write(str(message.tool_result)[:500])
                tool_log.write("")

            elif inv.status == ToolInvocationStatus.FAILED:
                error = (
                    message.tool_result.get("error")
                    if isinstance(message.tool_result, dict)
                    else str(message.tool_result)
                )
                chat.write(Text.from_markup(f"[red]\u2717 {inv.tool_name} failed[/red]"))
                tool_log.write(Text.from_markup(
                    f"[dim]{ts}[/dim] [red]FAILED[/red] [bold]{inv.tool_name}[/bold]: {error}"
                ))
                tool_log.write("")

            elif inv.status == ToolInvocationStatus.DENIED:
                chat.write(Text.from_markup(
                    f"[yellow]\u2717 {inv.tool_name} denied[/yellow]"
                ))

        elif isinstance(message, ErrorEmittedMessage):
            chat.write(Text.from_markup(
                f"[bold red]Error:[/bold red] {escape(message.error)}"
            ))

        self._update_status()

    # ── Input handling ───────────────────────────────────────────────────

    def on_input_submitted(self, event: Input.Submitted) -> None:
        user_input = event.value.strip()
        if not user_input:
            return

        inp = self.query_one("#user-input", Input)
        inp.value = ""

        # Let subclass intercept (e.g. tool-confirmation mode)
        if self._pre_process_input(user_input):
            return

        chat = self.query_one("#chat-log", RichLog)

        # Show user message
        chat.write(Text.from_markup(
            f"{self._format_user_label()} {escape(user_input)}"
        ))

        cmd = user_input.lower()

        # ── Common commands ──
        if cmd in ("/quit", "/exit", "/q"):
            self.exit()
            return
        if cmd in ("/help", "/h", "/?"):
            chat.write(Text.from_markup(self.HELP_TEXT))
            return
        if cmd == "/reset":
            if self._agent:
                self._agent.reset()
            self._last_seen_chat_count = 0
            self._on_reset()
            chat.write(Text.from_markup("[yellow]\u21ba Conversation reset[/yellow]"))
            self._update_status()
            return
        if cmd == "/status":
            self._show_status_in_chat()
            return
        if cmd == "/chats":
            self._show_chats()
            return
        if cmd == "/clear":
            self.query_one("#chat-log", RichLog).clear()
            return

        # ── Agent-specific commands ──
        if self._handle_custom_command(cmd, user_input):
            return

        # ── Regular message → agent ──
        if not self._agent:
            chat.write(Text.from_markup("[red]Agent not initialized.[/red]"))
            return
        self._send_to_agent(user_input)

    @work(thread=True)
    def _send_to_agent(self, user_input: str) -> None:
        """Send message to agent in a background thread."""
        self._agent.process_new_message(user_input, ChatRole.USER)
        self.call_from_thread(self._update_status)

    # ── Slash-command handlers ───────────────────────────────────────────

    def _show_status_in_chat(self) -> None:
        """Write status info into the chat pane (default: reuse panel text)."""
        chat = self.query_one("#chat-log", RichLog)
        chat.write(Text.from_markup(self._build_status_text()))

    def _show_chats(self) -> None:
        chat = self.query_one("#chat-log", RichLog)
        if not self._agent:
            chat.write(Text.from_markup("[yellow]Agent not initialized.[/yellow]"))
            return
        chats = self._agent.get_chats()
        if not chats:
            chat.write(Text.from_markup("[yellow]No messages yet.[/yellow]"))
            return

        chat.write(Text.from_markup(
            f"[bold cyan]Chat History ({len(chats)} messages)[/bold cyan]"
        ))
        role_colors = {
            "user": "green", "assistant": "cyan",
            "system": "yellow", "tool": "magenta",
        }
        for i, c in enumerate(chats, 1):
            color = role_colors.get(c.role.value, "white")
            content = (c.content or "").replace("\n", " ")[:60]
            suffix = "..." if len(c.content or "") > 60 else ""
            if c.role.value == "tool":
                tid = (c.tool_call_id[:8] + "...") if c.tool_call_id else "?"
                chat.write(Text.from_markup(
                    f"[dim]{i}.[/dim] [{color}]TOOL[/{color}] "
                    f"[dim]({tid})[/dim] {escape(content)}{suffix}"
                ))
            elif c.role.value == "assistant" and c.tool_calls:
                tool_names = ", ".join(tc.tool_name for tc in c.tool_calls)
                chat.write(Text.from_markup(
                    f"[dim]{i}.[/dim] [{color}]ASSISTANT[/{color}] "
                    f"[yellow]\u2192 {tool_names}[/yellow]"
                ))
            else:
                chat.write(Text.from_markup(
                    f"[dim]{i}.[/dim] [{color}]{c.role.value.upper()}[/{color}] "
                    f"{escape(content)}{suffix}"
                ))
