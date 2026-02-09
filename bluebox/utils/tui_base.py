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
from rich.syntax import Syntax
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.suggester import SuggestFromList
from textual.widgets import Input, RichLog, Static, Tree

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
        layout: vertical;
    }

    #main-row {
        height: 1fr;
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

    #input-row {
        height: 3;
    }

    #input-prompt {
        width: 2;
        height: 3;
        padding: 1 0 0 0;
        color: green;
        text-style: bold;
    }

    #user-input {
        width: 1fr;
    }

    #status-bar {
        height: 1;
        padding: 0 1;
    }

    #tool-log {
        height: 1fr;
        border: solid $secondary;
        border-title-color: $secondary;
        overflow-y: auto;
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

    def __init__(self, llm_model: LLMModel, working_dir: str | None = None) -> None:
        """
        Initialize the base agent TUI.

        Args:
            llm_model: The LLM model to use for the agent.
            working_dir: Optional path shown in a "Working directory" pane on the right.
                If None, the pane is not rendered.
        """
        super().__init__()
        self._llm_model = llm_model
        self._working_dir = working_dir
        self._context_window_size = get_context_window_size(llm_model.value)

        # Agent — set in on_mount via _create_agent()
        self._agent: AbstractAgent | None = None

        # Streaming state
        self._streaming_started: bool = False
        self._stream_buffer: list[str] = []
        self._in_code_block: bool = False
        self._code_block_lang: str = ""
        self._code_block_buffer: list[str] = []

        # Processing guard
        self._processing: bool = False

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
        with Horizontal(id="main-row"):
            with Vertical(id="left-pane"):
                chat_log = RichLog(id="chat-log", wrap=True, highlight=True, markup=True)
                chat_log.border_title = "Chat"
                yield chat_log
                with Horizontal(id="input-row"):
                    yield Static(">", id="input-prompt")
                    yield Input(
                        placeholder="Type a message or /help ...",
                        id="user-input",
                        suggester=SlashCommandSuggester(
                            self.SLASH_COMMANDS, case_sensitive=False,
                        ),
                    )
            with Vertical(id="right-pane"):
                tool_tree = Tree("Tools", id="tool-log")
                tool_tree.show_root = False
                tool_tree.border_title = "Tools invoked"
                yield tool_tree
        yield Static(id="status-bar")

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
        self.call_after_refresh(self._update_status)  # defer until layout is done
        self.set_interval(10, self._update_status)
        self.query_one("#user-input", Input).focus()

    def on_resize(self) -> None:
        """Re-render status bar on terminal resize."""
        self._update_status()

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
        """Refresh the bottom status bar."""
        bar = self.query_one("#status-bar", Static)
        left = Text.from_markup(self._build_status_bar_text())
        hint = Text.from_markup("[dim]Shift+drag to select[/dim]")
        # Pad the left text so the hint is right-aligned to the bar's width
        bar_width = bar.size.width or self.size.width
        pad = max(0, bar_width - left.cell_len - hint.cell_len)
        left.append(" " * pad)
        left.append_text(hint)
        bar.update(left)

    def _build_status_bar_text(self) -> str:
        """Return Rich markup for the bottom status bar."""
        now = datetime.now().astimezone().strftime("%I:%M %p %Z").lstrip("0")
        tokens_used, ctx_pct = self._estimate_context_usage()
        ctx_bar = self._context_bar(ctx_pct, width=10)
        parts = [
            f"  [dim]{now}[/dim] |  [bold purple]Vectorly[/bold purple]  |  "
            f"{self.TITLE}  |  [dim]{self._llm_model.value}[/dim]  |  {ctx_bar}",
        ]
        if self._working_dir:
            parts.append(f"  |  📁 [dim]Output dir:[/dim] {self._working_dir}")
        return "".join(parts)

    # ── Agent callbacks ──────────────────────────────────────────────────

    def _handle_stream_chunk(self, chunk: str) -> None:
        """Buffer streaming chunks, rendering markdown line-by-line."""
        chat = self.query_one("#chat-log", RichLog)
        if not self._streaming_started:
            chat.write(Text.from_markup("\n[bold cyan]Assistant[/bold cyan]"))
            self._streaming_started = True

        self._stream_buffer.append(chunk)

        combined = "".join(self._stream_buffer)
        if "\n" in combined:
            lines = combined.split("\n")
            for line in lines[:-1]:
                self._write_md_line(chat, line)
            self._stream_buffer.clear()
            if lines[-1]:
                self._stream_buffer.append(lines[-1])

    # ── Markdown helpers ──────────────────────────────────────────────

    def _write_md_line(self, chat: RichLog, line: str) -> None:
        """Write a single line to chat with markdown formatting."""
        stripped = line.strip()

        # Code block delimiters
        if stripped.startswith("```"):
            if self._in_code_block:
                # Closing fence — render buffered code with syntax highlighting
                code = "\n".join(self._code_block_buffer)
                lang = self._code_block_lang or "text"
                chat.write(Syntax(
                    code, lang, theme="monokai",
                    line_numbers=False, word_wrap=True,
                ))
                self._code_block_buffer.clear()
                self._in_code_block = False
            else:
                # Opening fence — start buffering
                self._in_code_block = True
                self._code_block_lang = stripped[3:].strip()
                self._code_block_buffer.clear()
            return

        # Inside code block — buffer, don't render yet
        if self._in_code_block:
            self._code_block_buffer.append(line)
            return

        if not stripped:
            chat.write("")
            return

        chat.write(self._render_md_line(line))

    def _render_md_line(self, line: str) -> Text:
        """Convert a single markdown line to a Rich Text renderable."""
        stripped = line.strip()

        # Headers
        if stripped.startswith("### "):
            return Text(stripped[4:], style="bold")
        if stripped.startswith("## "):
            return Text(stripped[3:], style="bold underline")
        if stripped.startswith("# "):
            return Text(stripped[2:], style="bold underline")

        # Unordered list items
        if stripped.startswith(("- ", "* ")):
            result = Text("  \u2022 ")
            result.append_text(self._apply_inline_md(stripped[2:]))
            return result

        return self._apply_inline_md(line)

    def _apply_inline_md(self, text: str) -> Text:
        """Apply **bold**, `code`, and *italic* inline formatting."""
        result = Text()
        i = 0
        while i < len(text):
            # Bold: **text**
            if text[i : i + 2] == "**":
                end = text.find("**", i + 2)
                if end != -1:
                    result.append(text[i + 2 : end], style="bold")
                    i = end + 2
                    continue
            # Inline code: `text`
            if text[i] == "`":
                end = text.find("`", i + 1)
                if end != -1:
                    result.append(text[i + 1 : end], style="bold cyan")
                    i = end + 1
                    continue
            # Italic: *text* (but not **)
            if text[i] == "*" and text[i : i + 2] != "**":
                end = text.find("*", i + 1)
                if end != -1 and text[end : end + 2] != "**":
                    result.append(text[i + 1 : end], style="italic")
                    i = end + 1
                    continue
            result.append(text[i])
            i += 1
        return result

    def _add_tool_node(
        self,
        label: Text,
        details: list[str],
        *,
        expand: bool = False,
        max_lines: int = 15,
        max_line_len: int = 200,
    ) -> None:
        """Add a collapsed node to the tool tree with detail lines as leaves.

        Lines beyond *max_lines* are replaced by a clickable "show more"
        leaf.  Clicking it inserts the remaining lines at the same level.
        """
        tool_tree = self.query_one("#tool-log", Tree)
        node = tool_tree.root.add(label, expand=expand)

        visible = details[:max_lines]
        overflow = details[max_lines:]

        for line in visible:
            node.add_leaf(line[:max_line_len])

        if overflow:
            node.add_leaf(
                Text(f"... ({len(overflow)} more lines)", style="dim italic"),
                data={"overflow": overflow, "max_line_len": max_line_len},
            )

        tool_tree.scroll_end(animate=False)

    def on_tree_node_selected(self, event: Tree.NodeSelected) -> None:
        """Expand a 'show more' placeholder into sibling leaves."""
        node = event.node
        if not isinstance(node.data, dict) or "overflow" not in node.data:
            return
        overflow: list[str] = node.data["overflow"]
        max_line_len: int = node.data.get("max_line_len", 200)
        parent = node.parent

        # Add overflow lines at the same level, then remove the placeholder
        for line in overflow:
            parent.add_leaf(line[:max_line_len])
        node.remove()

    def _flush_auto_executed_tools(self, chat: RichLog) -> None:
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
                result_text = (
                    c.content[c.content.find("result: ") + 8:]
                    if "result: " in (c.content or "")
                    else (c.content or "")
                )
                # Try to pretty-print JSON results for line-by-line display
                try:
                    result_text = json.dumps(json.loads(result_text), indent=2)
                except (json.JSONDecodeError, TypeError):
                    pass
                self._add_tool_node(
                    Text.assemble(
                        (ts, "dim"), " ", ("AUTO", "green"), " ", (tool_name, "bold"),
                    ),
                    result_text.split("\n") if result_text.strip() else [],
                )

            elif c.role.value == "assistant" and c.tool_calls:
                for tc in c.tool_calls:
                    details: list[str] = []
                    if hasattr(tc, "tool_arguments") and tc.tool_arguments:
                        details = json.dumps(tc.tool_arguments, indent=2).split("\n")
                    self._add_tool_node(
                        Text.assemble(
                            (ts, "dim"), " ", ("CALL", "yellow"), " ", (tc.tool_name, "bold"),
                        ),
                        details,
                    )

        self._last_seen_chat_count = current_count

    def _handle_message(self, message: BaseEmittedMessage) -> None:
        """Route emitted messages to the appropriate pane."""
        chat = self.query_one("#chat-log", RichLog)

        # Eagerly flush so CALL/AUTO nodes appear before any RESULT
        self._flush_auto_executed_tools(chat)

        # Let subclass handle first — if it returns True, we're done.
        if self._handle_additional_message(message):
            self._update_status()
            return

        if isinstance(message, ChatResponseEmittedMessage):

            if self._streaming_started:
                # Flush remaining partial line with formatting
                remainder = "".join(self._stream_buffer)
                if remainder.strip():
                    self._write_md_line(chat, remainder)
                # Flush any unclosed code block
                if self._in_code_block and self._code_block_buffer:
                    code = "\n".join(self._code_block_buffer)
                    lang = self._code_block_lang or "text"
                    chat.write(Syntax(
                        code, lang, theme="monokai",
                        line_numbers=False, word_wrap=True,
                    ))
                    self._code_block_buffer.clear()
                self._stream_buffer.clear()
                self._streaming_started = False
                self._in_code_block = False
            else:
                # Non-streaming: render full response as markdown
                chat.write(Text.from_markup("\n[bold cyan]Assistant[/bold cyan]"))
                if message.content:
                    chat.write(RichMarkdown(message.content))
            chat.write("")

        elif isinstance(message, ToolInvocationResultEmittedMessage):
            inv = message.tool_invocation
            ts = datetime.now().strftime("%H:%M:%S")
            self._tool_call_count += 1

            if inv.status == ToolInvocationStatus.EXECUTED:
                chat.write(Text.from_markup(
                    f"[green]\u2713[/green] [dim]{inv.tool_name} executed[/dim]"
                ))
                details: list[str] = []
                if isinstance(message.tool_result, dict):
                    details = json.dumps(message.tool_result, indent=2).split("\n")
                elif message.tool_result:
                    details = str(message.tool_result).split("\n")
                self._add_tool_node(
                    Text.assemble(
                        (ts, "dim"), " ", ("RESULT", "green"), " ", (inv.tool_name, "bold"),
                    ),
                    details,
                )

            elif inv.status == ToolInvocationStatus.FAILED:
                error = (
                    message.tool_result.get("error")
                    if isinstance(message.tool_result, dict)
                    else str(message.tool_result)
                )
                chat.write(Text.from_markup(f"[red]\u2717 {inv.tool_name} failed[/red]"))
                self._add_tool_node(
                    Text.assemble(
                        (ts, "dim"), " ", ("FAILED", "red"), " ", (inv.tool_name, "bold"),
                    ),
                    [str(error)[:500]] if error else [],
                )

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
        if self._processing:
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
        self._processing = True
        self._send_to_agent(user_input)

    @work(thread=True)
    def _send_to_agent(self, user_input: str) -> None:
        """Send message to agent in a background thread."""
        try:
            self._agent.process_new_message(user_input, ChatRole.USER)
        finally:
            self.call_from_thread(self._finish_processing)

    def _finish_processing(self) -> None:
        """Re-enable input submission after agent finishes."""
        self._processing = False
        self._update_status()

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
