"""
bluebox/scripts/run_guide_agent_tui.py

Multi-pane terminal UI for the Guide Agent using Textual.

Layout:
  ┌─────────────────────────────┬──────────────────────┐
  │                             │  Tool Calls History   │
  │       Chat (scrolling)      │                       │
  │                             ├──────────────────────┤
  │  ┌────────────────────────┐ │  Status / Branding    │
  │  │ Input                  │ │                       │
  │  └────────────────────────┘ │                       │
  └─────────────────────────────┴──────────────────────┘

Usage:
    bluebox-guide-tui
    bluebox-guide-tui --model gpt-5.1
    bluebox-guide-tui --cdp-captures-dir ./cdp_captures -q
"""

import argparse
import difflib
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from openai import OpenAI
from rich import box
from rich.console import Console
from rich.markup import escape
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.suggester import SuggestFromList
from textual.widgets import Input, RichLog, Static

from bluebox.agents.guide_agent import GuideAgent
from bluebox.config import Config
from bluebox.data_models.llms.vendors import OpenAIModel
from bluebox.data_models.llms.interaction import (
    ChatRole,
    BaseEmittedMessage,
    ChatResponseEmittedMessage,
    ToolInvocationRequestEmittedMessage,
    ToolInvocationResultEmittedMessage,
    SuggestedEditEmittedMessage,
    BrowserRecordingRequestEmittedMessage,
    RoutineDiscoveryRequestEmittedMessage,
    RoutineCreationRequestEmittedMessage,
    ErrorEmittedMessage,
    PendingToolInvocation,
    SuggestedEditRoutine,
    ToolInvocationStatus,
)
from bluebox.data_models.routine.routine import Routine
from bluebox.llms.tools.guide_agent_tools import validate_routine
from bluebox.llms.infra.data_store import LocalDiscoveryDataStore
from bluebox.utils.chrome_utils import ensure_chrome_running

# Package root for code_paths
BLUEBOX_PACKAGE_ROOT = Path(__file__).resolve().parent.parent

# Browser monitoring constants
PORT = 9222
DEFAULT_CDP_CAPTURES_DIR = Path("./cdp_captures")

# Rough context window sizes (tokens) per model family
MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    "gpt-4": 128_000,
    "gpt-5": 128_000,
    "o3": 200_000,
    "o4": 200_000,
}
DEFAULT_CONTEXT_WINDOW = 128_000

# ─── Slash commands ──────────────────────────────────────────────────────────

SLASH_COMMANDS = [
    "/load", "/unload", "/show", "/validate", "/execute",
    "/monitor", "/diff", "/accept", "/reject",
    "/status", "/chats", "/clear", "/reset", "/help", "/quit",
]

HELP_TEXT = """\
[bold]Commands:[/bold]
  [cyan]/load <file.json>[/cyan]     Load a routine file
  [cyan]/unload[/cyan]               Unload the current routine
  [cyan]/show[/cyan]                 Show current routine details
  [cyan]/validate[/cyan]             Validate the current routine
  [cyan]/execute[/cyan]              Execute the loaded routine
  [cyan]/monitor[/cyan]              Start browser monitoring
  [cyan]/diff[/cyan]                 Show pending edit diff
  [cyan]/accept[/cyan]               Accept pending edit
  [cyan]/reject[/cyan]               Reject pending edit
  [cyan]/status[/cyan]               Show current state
  [cyan]/chats[/cyan]                Show message history
  [cyan]/clear[/cyan]                Clear the chat display
  [cyan]/reset[/cyan]                Start new conversation
  [cyan]/help[/cyan]                 Show this help
  [cyan]/quit[/cyan]                 Exit
"""


# ─── Textual CSS ─────────────────────────────────────────────────────────────

APP_CSS = """
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
    max-height: 12;
    border: solid $primary;
    border-title-color: $primary;
    padding: 0 1;
}
"""


# ─── Helpers ─────────────────────────────────────────────────────────────────

def safe_parse_routine(routine_str: str | None) -> tuple[dict[str, Any] | None, str | None]:
    """Parse a routine string to dict. Returns (dict, None) or (None, error)."""
    if routine_str is None:
        return None, None
    try:
        return json.loads(routine_str), None
    except json.JSONDecodeError as e:
        return None, f"Invalid JSON: {e}"


def get_context_window_size(model_value: str) -> int:
    """Get approximate context window size for a model."""
    for prefix, size in MODEL_CONTEXT_WINDOWS.items():
        if model_value.startswith(prefix):
            return size
    return DEFAULT_CONTEXT_WINDOW


def configure_logging(quiet: bool = False, log_file: str | None = None) -> None:
    """Configure logging — in TUI mode, always redirect to file or suppress.

    Without this, StreamHandler output bleeds into the Textual display.
    """
    # Redirect ALL logging to file — any StreamHandler output corrupts the TUI
    resolved_log_file = log_file or ".bluebox_tui.log"

    # Clear root logger handlers (prevents any stderr StreamHandlers)
    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    # Clear bluebox logger
    wh_logger = logging.getLogger("bluebox")
    wh_logger.handlers.clear()
    wh_logger.propagate = False

    if quiet:
        wh_logger.setLevel(logging.CRITICAL + 1)
        root_logger.setLevel(logging.CRITICAL + 1)
        return

    file_handler = logging.FileHandler(resolved_log_file)
    file_handler.setFormatter(logging.Formatter(
        fmt="[%(asctime)s] %(levelname)s:%(name)s:%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    wh_logger.addHandler(file_handler)
    # Also redirect root logger to the same file
    root_logger.addHandler(file_handler)


# ─── Slash command suggester ─────────────────────────────────────────────────

class SlashCommandSuggester(SuggestFromList):
    """Only suggest when input starts with '/'."""

    async def get_suggestion(self, value: str) -> str | None:
        if not value.startswith("/"):
            return None
        return await super().get_suggestion(value)


# ─── Textual App ─────────────────────────────────────────────────────────────

class GuideAgentTUI(App):
    """Multi-pane TUI for the Guide Agent."""

    CSS = APP_CSS
    TITLE = "Guide Agent"
    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", priority=True),
    ]

    def __init__(
        self,
        llm_model: OpenAIModel,
        data_store: LocalDiscoveryDataStore | None = None,
        cdp_captures_dir: Path = DEFAULT_CDP_CAPTURES_DIR,
    ) -> None:
        super().__init__()
        self._llm_model = llm_model
        self._data_store = data_store
        self._cdp_captures_dir = cdp_captures_dir
        self._context_window_size = get_context_window_size(llm_model.value)

        # Agent state
        self._agent: GuideAgent | None = None
        self._pending_invocation: PendingToolInvocation | None = None
        self._pending_suggested_edit: SuggestedEditRoutine | None = None
        self._loaded_routine_path: Path | None = None
        self._streaming_started = False
        self._stream_buffer: list[str] = []  # buffer chunks until full message arrives
        self._tool_call_count = 0
        self._estimated_tokens_used = 0
        self._last_seen_chat_count = 0  # track to detect auto-executed tools

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
                    suggester=SlashCommandSuggester(SLASH_COMMANDS, case_sensitive=False),
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
        """Initialize agent and show welcome."""
        self._agent = GuideAgent(
            emit_message_callable=self._handle_message,
            stream_chunk_callable=self._handle_stream_chunk,
            llm_model=self._llm_model,
            data_store=self._data_store,
        )
        self._print_welcome()
        self._update_status()
        # Refresh clock every 10 seconds
        self.set_interval(10, self._update_status)
        self.query_one("#user-input", Input).focus()

    # ── Welcome ──────────────────────────────────────────────────────────

    def _print_welcome(self) -> None:
        chat = self.query_one("#chat-log", RichLog)
        chat.write(Text.from_markup(
            "[bold magenta]Guide Agent[/bold magenta]  "
            "[dim]powered by vectorly[/dim]"
        ))
        chat.write("")
        chat.write(Text.from_markup(
            "Welcome! I'll help you create web automation routines.\n"
            "Type [cyan]/help[/cyan] for commands, or just describe what you need."
        ))
        chat.write("")

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
        tokens = total_chars // 4  # ~4 chars per token
        pct = (tokens / self._context_window_size) * 100 if self._context_window_size > 0 else 0.0
        return tokens, min(pct, 100.0)

    def _context_bar(self, pct: float, width: int = 15) -> str:
        """Render a text-based context fill bar like [████░░░░░░] 34%."""
        filled = int(pct / 100 * width)
        empty = width - filled
        if pct < 50:
            color = "green"
        elif pct < 80:
            color = "yellow"
        else:
            color = "red"
        bar = "█" * filled + "░" * empty
        return f"[{color}]{bar}[/{color}] {pct:.0f}%"

    def _update_status(self) -> None:
        """Refresh the status panel content."""
        panel = self.query_one("#status-panel", Static)

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        msg_count = len(self._agent.get_chats()) if self._agent else 0
        tokens_used, ctx_pct = self._estimate_context_usage()
        ctx_bar = self._context_bar(ctx_pct)

        status_text = (
            f"[bold magenta]VECTORLY[/bold magenta]\n"
            f"[dim]──────────────────[/dim]\n"
            f"[dim]Model:[/dim]    {self._llm_model.value}\n"
            f"[dim]Messages:[/dim] {msg_count}\n"
            f"[dim]Tools:[/dim]    {self._tool_call_count}\n"
            f"[dim]Context:[/dim]  {ctx_bar}\n"
            f"[dim](est.)     ~{tokens_used:,} / {self._context_window_size:,}[/dim]\n"
            f"[dim]Time:[/dim]     {now}\n"
        )
        panel.update(Text.from_markup(status_text))

    # ── Agent callbacks ──────────────────────────────────────────────────

    def _handle_stream_chunk(self, chunk: str) -> None:
        """Buffer streaming chunks line-by-line, flushing complete lines immediately."""
        chat = self.query_one("#chat-log", RichLog)
        if not self._streaming_started:
            chat.write(Text.from_markup("\n[bold cyan]Assistant[/bold cyan]"))
            self._streaming_started = True

        self._stream_buffer.append(chunk)

        # Flush complete lines as they come in for a real streaming feel
        combined = "".join(self._stream_buffer)
        if "\n" in combined:
            # Split on newlines — write all complete lines, keep the remainder
            lines = combined.split("\n")
            for line in lines[:-1]:  # all complete lines
                chat.write(line)
            # Keep the last (incomplete) part in the buffer
            self._stream_buffer.clear()
            if lines[-1]:  # leftover partial line
                self._stream_buffer.append(lines[-1])

    def _flush_auto_executed_tools(self, tool_log: RichLog, chat: RichLog) -> None:
        """Detect and log auto-executed tools by scanning new chat entries."""
        if not self._agent:
            return
        chats = self._agent.get_chats()
        current_count = len(chats)
        if current_count <= self._last_seen_chat_count:
            self._last_seen_chat_count = current_count
            return

        # Scan new messages for TOOL-role entries (auto-executed results)
        ts = datetime.now().strftime("%H:%M:%S")
        for c in chats[self._last_seen_chat_count:]:
            if c.role.value == "tool":
                self._tool_call_count += 1
                # Extract tool name from content like "Tool 'name' result: ..."
                tool_name = "tool"
                if c.content and c.content.startswith("Tool '"):
                    end = c.content.find("'", 6)
                    if end > 6:
                        tool_name = c.content[6:end]

                # Compact line in chat pane
                chat.write(Text.from_markup(
                    f"[green]✓[/green] [dim]{tool_name} (auto)[/dim]"
                ))

                # Details in tool pane
                tool_log.write(Text.from_markup(
                    f"[dim]{ts}[/dim] [green]AUTO[/green] [bold]{tool_name}[/bold]"
                ))
                # Show truncated result
                result_preview = c.content[c.content.find("result: ") + 8:] if "result: " in c.content else c.content
                if len(result_preview) > 300:
                    result_preview = result_preview[:300] + "..."
                tool_log.write(result_preview)
                tool_log.write("")

            elif c.role.value == "assistant" and c.tool_calls:
                # Assistant message with tool_calls — log the calls themselves
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

        if isinstance(message, ChatResponseEmittedMessage):
            # Flush any auto-executed tools that happened before this response
            self._flush_auto_executed_tools(tool_log, chat)

            if self._streaming_started:
                # Flush any remaining buffered text
                remainder = "".join(self._stream_buffer)
                if remainder:
                    chat.write(remainder)
                self._stream_buffer.clear()
                self._streaming_started = False
            else:
                chat.write(Text.from_markup("\n[bold cyan]Assistant[/bold cyan]"))
                chat.write(message.content)
            chat.write("")

        elif isinstance(message, ToolInvocationRequestEmittedMessage):
            inv = message.tool_invocation
            self._pending_invocation = inv
            self._tool_call_count += 1

            # Compact in chat pane
            chat.write(Text.from_markup(
                f"[yellow]▶ Tool:[/yellow] [bold]{inv.tool_name}[/bold] "
                f"[dim]— y/n to approve[/dim]"
            ))

            # Full details in tool pane
            ts = datetime.now().strftime("%H:%M:%S")
            tool_log.write(Text.from_markup(
                f"[dim]{ts}[/dim] [yellow]REQUEST[/yellow] [bold]{inv.tool_name}[/bold]"
            ))
            args_str = json.dumps(inv.tool_arguments, indent=2)
            lines = args_str.split("\n")
            if len(lines) > 20:
                args_str = "\n".join(lines[:20]) + f"\n... ({len(lines) - 20} more lines)"
            tool_log.write(args_str)
            tool_log.write("")

            # Switch input placeholder
            inp = self.query_one("#user-input", Input)
            inp.placeholder = "Approve tool call? (y/n)"

        elif isinstance(message, ToolInvocationResultEmittedMessage):
            inv = message.tool_invocation
            ts = datetime.now().strftime("%H:%M:%S")

            if inv.status == ToolInvocationStatus.EXECUTED:
                chat.write(Text.from_markup(
                    f"[green]✓[/green] [dim]{inv.tool_name} executed[/dim]"
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
                error = message.tool_result.get("error") if isinstance(message.tool_result, dict) else str(message.tool_result)
                chat.write(Text.from_markup(f"[red]✗ {inv.tool_name} failed[/red]"))
                tool_log.write(Text.from_markup(
                    f"[dim]{ts}[/dim] [red]FAILED[/red] [bold]{inv.tool_name}[/bold]: {error}"
                ))
                tool_log.write("")

            elif inv.status == ToolInvocationStatus.DENIED:
                chat.write(Text.from_markup(f"[yellow]✗ {inv.tool_name} denied[/yellow]"))

        elif isinstance(message, SuggestedEditEmittedMessage):
            self._pending_suggested_edit = message.suggested_edit
            chat.write(Text.from_markup(
                "[bold yellow]Agent suggested a routine edit[/bold yellow]\n"
                "[dim]Use /diff, /accept, or /reject[/dim]"
            ))

        elif isinstance(message, BrowserRecordingRequestEmittedMessage):
            chat.write(Text.from_markup(
                "[yellow]Agent requests browser recording. Use /monitor to start.[/yellow]"
            ))

        elif isinstance(message, RoutineDiscoveryRequestEmittedMessage):
            chat.write(Text.from_markup(
                f"[yellow]Agent requests routine discovery:[/yellow] {message.routine_discovery_task}\n"
                "[dim]Use bluebox-guide for discovery features.[/dim]"
            ))

        elif isinstance(message, RoutineCreationRequestEmittedMessage):
            self._handle_routine_creation(message.created_routine)

        elif isinstance(message, ErrorEmittedMessage):
            chat.write(Text.from_markup(f"[bold red]Error:[/bold red] {escape(message.error)}"))

        self._update_status()

    # ── Input handling ───────────────────────────────────────────────────

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle user input submission."""
        user_input = event.value.strip()
        if not user_input:
            return

        inp = self.query_one("#user-input", Input)
        inp.value = ""

        # Tool confirmation mode
        if self._pending_invocation:
            self._handle_tool_confirmation(user_input)
            return

        chat = self.query_one("#chat-log", RichLog)

        # Show user message with routine label
        routine_label = ""
        if self._agent and self._agent.routine_state.current_routine_str:
            rd, _ = safe_parse_routine(self._agent.routine_state.current_routine_str)
            if rd:
                name = rd.get("name", "")[:20]
                routine_label = f" [dim]({name})[/dim]"
        chat.write(Text.from_markup(f"[bold green]You{routine_label}>[/bold green] {escape(user_input)}"))

        # Route slash commands
        cmd = user_input.lower()

        if cmd in ("/quit", "/exit", "/q"):
            self.exit()
            return
        if cmd in ("/help", "/h", "/?"):
            chat.write(Text.from_markup(HELP_TEXT))
            return
        if cmd == "/reset":
            self._agent.reset()
            self._pending_invocation = None
            self._pending_suggested_edit = None
            self._loaded_routine_path = None
            chat.write(Text.from_markup("[yellow]↺ Conversation reset[/yellow]"))
            self._update_status()
            return
        if cmd == "/status":
            self._show_status_in_chat()
            return
        if cmd == "/chats":
            self._show_chats()
            return
        if cmd == "/show":
            self._show_routine()
            return
        if cmd == "/validate":
            self._validate_routine()
            return
        if cmd == "/diff":
            self._show_diff()
            return
        if cmd == "/accept":
            self._accept_edit()
            return
        if cmd == "/reject":
            self._reject_edit()
            return
        if cmd == "/clear":
            self.query_one("#chat-log", RichLog).clear()
            return
        if cmd == "/unload":
            self._unload_routine()
            return
        if cmd == "/monitor":
            chat.write(Text.from_markup(
                "[yellow]Browser monitoring requires the scrolling terminal.[/yellow]\n"
                "[dim]Use bluebox-guide for monitor/discovery features.[/dim]"
            ))
            return
        if user_input.startswith("/load "):
            self._load_routine(user_input[6:].strip())
            return
        if user_input.startswith("/execute"):
            params_path = user_input[8:].strip() or None
            self._execute_routine(params_path)
            return

        # Regular message → send to agent in background
        self._reload_routine_from_file()
        self._send_to_agent(user_input)

    @work(thread=True)
    def _send_to_agent(self, user_input: str) -> None:
        """Send message to agent in a background thread."""
        self._agent.process_new_message(user_input, ChatRole.USER)
        self.call_from_thread(self._update_status)

    # ── Tool confirmation ────────────────────────────────────────────────

    def _handle_tool_confirmation(self, user_input: str) -> None:
        normalized = user_input.strip().lower()
        chat = self.query_one("#chat-log", RichLog)
        inp = self.query_one("#user-input", Input)

        if normalized in ("y", "yes"):
            invocation_id = self._pending_invocation.invocation_id
            self._pending_invocation = None
            inp.placeholder = "Type a message or /help ..."
            self._confirm_tool(invocation_id)
        elif normalized in ("n", "no"):
            invocation_id = self._pending_invocation.invocation_id
            self._pending_invocation = None
            inp.placeholder = "Type a message or /help ..."
            self._deny_tool(invocation_id)
        else:
            chat.write(Text.from_markup("[yellow]Please enter 'y' or 'n'[/yellow]"))

    @work(thread=True)
    def _confirm_tool(self, invocation_id: str) -> None:
        self._agent.confirm_tool_invocation(invocation_id)
        self.call_from_thread(self._update_status)

    @work(thread=True)
    def _deny_tool(self, invocation_id: str) -> None:
        self._agent.deny_tool_invocation(invocation_id, reason="User declined")
        self.call_from_thread(self._update_status)

    # ── Slash command handlers ───────────────────────────────────────────

    def _load_routine(self, file_path: str) -> None:
        chat = self.query_one("#chat-log", RichLog)
        try:
            path = Path(file_path).resolve()
            if not path.exists():
                chat.write(Text.from_markup(f"[red]✗ File not found: {file_path}[/red]"))
                return

            with open(path, encoding="utf-8") as f:
                routine_str = f.read()

            self._loaded_routine_path = path
            self._agent.routine_state.update_current_routine(routine_str)

            routine_dict, parse_error = safe_parse_routine(routine_str)
            if routine_dict is not None:
                name = routine_dict.get("name", "N/A")
                ops = len(routine_dict.get("operations", []))
                params = len(routine_dict.get("parameters", []))
                validation = validate_routine(routine_dict)
                valid_str = "[green]✓ valid[/green]" if validation.get("valid") else f"[red]✗ {validation.get('error', '')}[/red]"
                chat.write(Text.from_markup(
                    f"[green]✓ Loaded:[/green] {name}\n"
                    f"  Operations: {ops} | Parameters: {params}\n"
                    f"  {valid_str}\n"
                    f"  [dim]File: {path}[/dim]"
                ))
            else:
                chat.write(Text.from_markup(
                    f"[yellow]⚠ Loaded file with invalid JSON: {parse_error}[/yellow]\n"
                    "[dim]Ask the agent for help fixing the JSON.[/dim]"
                ))
            self._update_status()
        except Exception as e:
            chat.write(Text.from_markup(f"[red]✗ Error loading routine: {e}[/red]"))

    def _unload_routine(self) -> None:
        chat = self.query_one("#chat-log", RichLog)
        if self._agent.routine_state.current_routine_str is None:
            chat.write(Text.from_markup("[yellow]No routine loaded.[/yellow]"))
            return
        self._loaded_routine_path = None
        self._agent.routine_state.update_current_routine(None)
        chat.write(Text.from_markup("[yellow]✓ Routine unloaded[/yellow]"))
        self._update_status()

    def _show_routine(self) -> None:
        chat = self.query_one("#chat-log", RichLog)
        routine_str = self._agent.routine_state.current_routine_str
        if routine_str is None:
            chat.write(Text.from_markup("[yellow]No routine loaded.[/yellow]"))
            return

        routine_dict, parse_error = safe_parse_routine(routine_str)
        if routine_dict is None:
            chat.write(Text.from_markup(f"[red]✗ {parse_error}[/red]"))
            return

        name = routine_dict.get("name", "N/A")
        desc = routine_dict.get("description", "N/A")
        ops = routine_dict.get("operations", [])
        params = routine_dict.get("parameters", [])

        lines = [f"[bold cyan]Routine: {name}[/bold cyan]", f"  {desc}", ""]
        if params:
            lines.append("[dim]Parameters:[/dim]")
            for p in params:
                req = "[red]*[/red]" if p.get("required") else " "
                lines.append(f"  {req}{p.get('name', '?')} ({p.get('type', '?')}): {p.get('description', '')}")
            lines.append("")
        if ops:
            lines.append(f"[dim]Operations ({len(ops)}):[/dim]")
            for i, op in enumerate(ops[:10], 1):
                lines.append(f"  {i}. {op.get('type', '?')}")
            if len(ops) > 10:
                lines.append(f"  ... and {len(ops) - 10} more")
        if self._loaded_routine_path:
            lines.append(f"\n[dim]File: {self._loaded_routine_path}[/dim]")
        chat.write(Text.from_markup("\n".join(lines)))

    def _validate_routine(self) -> None:
        chat = self.query_one("#chat-log", RichLog)
        routine_str = self._agent.routine_state.current_routine_str
        if routine_str is None:
            chat.write(Text.from_markup("[yellow]No routine loaded.[/yellow]"))
            return
        routine_dict, parse_error = safe_parse_routine(routine_str)
        if routine_dict is None:
            chat.write(Text.from_markup(f"[red]✗ {parse_error}[/red]"))
            return
        result = validate_routine(routine_dict)
        if result.get("valid"):
            chat.write(Text.from_markup(f"[green]✓ Valid:[/green] {result.get('message', 'OK')}"))
        else:
            chat.write(Text.from_markup(f"[red]✗ Invalid:[/red] {result.get('error', 'Unknown')}"))

    def _show_chats(self) -> None:
        chat = self.query_one("#chat-log", RichLog)
        chats = self._agent.get_chats()
        if not chats:
            chat.write(Text.from_markup("[yellow]No messages yet.[/yellow]"))
            return

        chat.write(Text.from_markup(f"[bold cyan]Chat History ({len(chats)} messages)[/bold cyan]"))
        role_colors = {"user": "green", "assistant": "cyan", "system": "yellow", "tool": "magenta"}
        for i, c in enumerate(chats, 1):
            color = role_colors.get(c.role.value, "white")
            content = c.content.replace("\n", " ")[:60]
            suffix = "..." if len(c.content) > 60 else ""
            if c.role.value == "tool":
                tid = (c.tool_call_id[:8] + "...") if c.tool_call_id else "?"
                chat.write(Text.from_markup(f"[dim]{i}.[/dim] [{color}]TOOL[/{color}] [dim]({tid})[/dim] {escape(content)}{suffix}"))
            elif c.role.value == "assistant" and c.tool_calls:
                tool_names = ", ".join(tc.tool_name for tc in c.tool_calls)
                chat.write(Text.from_markup(f"[dim]{i}.[/dim] [{color}]ASSISTANT[/{color}] [yellow]→ {tool_names}[/yellow]"))
            else:
                chat.write(Text.from_markup(f"[dim]{i}.[/dim] [{color}]{c.role.value.upper()}[/{color}] {escape(content)}{suffix}"))

    def _show_status_in_chat(self) -> None:
        chat = self.query_one("#chat-log", RichLog)
        routine_str = self._agent.routine_state.current_routine_str
        routine_name = "None"
        if routine_str:
            rd, _ = safe_parse_routine(routine_str)
            if rd:
                routine_name = rd.get("name", "Unnamed")
        msg_count = len(self._agent.get_chats())
        thread_id = self._agent.chat_thread_id[:8] + "..."
        tokens_used, ctx_pct = self._estimate_context_usage()
        chat.write(Text.from_markup(
            f"[bold cyan]Status[/bold cyan]\n"
            f"  Routine: {routine_name}\n"
            f"  Messages: {msg_count}\n"
            f"  Thread: {thread_id}\n"
            f"  Context: ~{tokens_used:,}t ({ctx_pct:.0f}%)\n"
            f"  File: {self._loaded_routine_path or 'N/A'}"
        ))

    def _show_diff(self) -> None:
        chat = self.query_one("#chat-log", RichLog)
        if not self._pending_suggested_edit:
            chat.write(Text.from_markup("[yellow]No pending suggested edit.[/yellow]"))
            return
        current_str = self._agent.routine_state.current_routine_str or "{}"
        new_str = json.dumps(self._pending_suggested_edit.routine.model_dump(), indent=2)
        diff = difflib.unified_diff(
            current_str.splitlines(keepends=True),
            new_str.splitlines(keepends=True),
            fromfile="current", tofile="suggested", lineterm="",
        )
        diff_lines = list(diff)
        if diff_lines:
            output_lines = []
            for line in diff_lines:
                line = line.rstrip("\n")
                if line.startswith("+"):
                    output_lines.append(f"[green]{escape(line)}[/green]")
                elif line.startswith("-"):
                    output_lines.append(f"[red]{escape(line)}[/red]")
                elif line.startswith("@@"):
                    output_lines.append(f"[cyan]{escape(line)}[/cyan]")
                else:
                    output_lines.append(escape(line))
            chat.write(Text.from_markup("[bold cyan]Suggested Edit Diff:[/bold cyan]\n" + "\n".join(output_lines)))
        else:
            chat.write(Text.from_markup("[yellow]No differences found.[/yellow]"))

    def _accept_edit(self) -> None:
        chat = self.query_one("#chat-log", RichLog)
        if not self._pending_suggested_edit:
            chat.write(Text.from_markup("[yellow]No pending suggested edit.[/yellow]"))
            return
        routine = self._pending_suggested_edit.routine
        routine_dict = routine.model_dump()
        routine_str = json.dumps(routine_dict)
        self._agent.routine_state.update_current_routine(routine_str)
        self._persist_routine(routine_dict)
        chat.write(Text.from_markup("[green]✓ Edit accepted and applied[/green]"))
        self._pending_suggested_edit = None
        self._update_status()

    def _reject_edit(self) -> None:
        chat = self.query_one("#chat-log", RichLog)
        if not self._pending_suggested_edit:
            chat.write(Text.from_markup("[yellow]No pending suggested edit.[/yellow]"))
            return
        chat.write(Text.from_markup("[yellow]✗ Edit rejected[/yellow]"))
        self._pending_suggested_edit = None

    def _persist_routine(self, routine_dict: dict[str, Any]) -> None:
        """Save routine to loaded file path."""
        chat = self.query_one("#chat-log", RichLog)
        if self._loaded_routine_path is None:
            chat.write(Text.from_markup("[yellow]⚠ No file loaded — routine updated in memory only[/yellow]"))
            return
        try:
            new_content = json.dumps(routine_dict, indent=2)
            with open(self._loaded_routine_path, "w", encoding="utf-8") as f:
                f.write(new_content)
            chat.write(Text.from_markup(f"[green]✓ Saved to {self._loaded_routine_path}[/green]"))
        except Exception as e:
            chat.write(Text.from_markup(f"[red]✗ Failed to save: {e}[/red]"))

    def _execute_routine(self, params_path: str | None) -> None:
        """Execute the loaded routine."""
        chat = self.query_one("#chat-log", RichLog)
        routine_str = self._agent.routine_state.current_routine_str
        if routine_str is None:
            chat.write(Text.from_markup("[red]✗ No routine loaded.[/red]"))
            return
        routine_dict, parse_error = safe_parse_routine(routine_str)
        if routine_dict is None:
            chat.write(Text.from_markup(f"[red]✗ Cannot execute: {parse_error}[/red]"))
            return

        params: dict[str, Any] = {}
        if params_path:
            try:
                path = Path(params_path)
                if not path.exists():
                    chat.write(Text.from_markup(f"[red]✗ Params file not found: {params_path}[/red]"))
                    return
                with open(path, encoding="utf-8") as f:
                    params = json.load(f)
            except json.JSONDecodeError as e:
                chat.write(Text.from_markup(f"[red]✗ Invalid params JSON: {e}[/red]"))
                return

        if not ensure_chrome_running(PORT):
            chat.write(Text.from_markup(f"[red]✗ Chrome not running on port {PORT}[/red]"))
            return
        self._run_execution(routine_dict, params)

    @work(thread=True)
    def _run_execution(self, routine_dict: dict[str, Any], params: dict[str, Any]) -> None:
        """Execute routine in background thread."""
        chat = self.query_one("#chat-log", RichLog)
        try:
            routine = Routine(**routine_dict)
            result = routine.execute(params)
            result_dict = result.model_dump()
            self._agent.routine_state.update_last_execution(
                routine=routine_dict, parameters=params, result=result_dict,
            )
            if result.ok:
                self.call_from_thread(
                    lambda: chat.write(Text.from_markup("[green]✓ Execution succeeded[/green]"))
                )
            else:
                self.call_from_thread(
                    lambda: chat.write(Text.from_markup(
                        f"[red]✗ Execution failed: {result.error or 'Unknown'}[/red]"
                    ))
                )
        except Exception as e:
            self.call_from_thread(
                lambda: chat.write(Text.from_markup(f"[red]✗ Execution error: {e}[/red]"))
            )
        self.call_from_thread(self._update_status)

    def _handle_routine_creation(self, routine: Routine | None) -> None:
        """Handle routine creation from agent."""
        chat = self.query_one("#chat-log", RichLog)
        if not routine:
            return
        try:
            safe_name = routine.name.lower().replace(" ", "_").replace("-", "_")
            safe_name = "".join(c for c in safe_name if c.isalnum() or c == "_")
            routines_dir = Path("./example_data/example_routines")
            routines_dir.mkdir(parents=True, exist_ok=True)
            routine_path = routines_dir / f"{safe_name}.json"
            routine_path.write_text(json.dumps(routine.model_dump(), indent=2))

            routine_str = routine_path.read_text()
            self._loaded_routine_path = routine_path
            self._agent.routine_state.update_current_routine(routine_str)

            chat.write(Text.from_markup(
                f"[green]✓ Routine created:[/green] {routine.name}\n"
                f"  Ops: {len(routine.operations)} | Params: {len(routine.parameters)}\n"
                f"  [dim]Saved: {routine_path}[/dim]"
            ))

            system_message = (
                f"[ACTION REQUIRED] Routine '{routine.name}' has been created and saved to {routine_path}. "
                f"It has {len(routine.operations)} operations and {len(routine.parameters)} parameters. "
                "The routine is now loaded into context. Review it using get_current_routine and explain "
                "to the user what it does, what parameters it needs, and how to use it."
            )
            self._agent.process_new_message(system_message, ChatRole.SYSTEM)
            self._update_status()
        except Exception as e:
            chat.write(Text.from_markup(f"[red]✗ Failed to create routine: {e}[/red]"))

    def _reload_routine_from_file(self) -> None:
        """Re-read routine from file (picks up external edits)."""
        if self._loaded_routine_path is None:
            return
        try:
            with open(self._loaded_routine_path, encoding="utf-8") as f:
                routine_str = f.read()
            self._agent.routine_state.update_current_routine(routine_str)
        except Exception:
            pass


# ─── Entry point ─────────────────────────────────────────────────────────────

def parse_model(model_str: str) -> OpenAIModel:
    """Parse a model string into an OpenAIModel enum value."""
    for model in OpenAIModel:
        if model.value == model_str or model.name == model_str:
            return model
    raise ValueError(f"Unknown model: {model_str}")


def main() -> None:
    """Entry point for the guide agent TUI."""
    parser = argparse.ArgumentParser(description="Guide Agent — Multi-pane TUI")
    parser.add_argument(
        "--model", type=str, default=OpenAIModel.GPT_5_1.value,
        help=f"LLM model to use (default: {OpenAIModel.GPT_5_1.value})",
    )
    parser.add_argument(
        "--cdp-captures-dir", type=str, default=None,
        help="Path to CDP captures directory",
    )
    parser.add_argument(
        "--output-dir", type=str, default="./guide_agent_output",
        help="Output directory for temporary files",
    )
    parser.add_argument(
        "--docs-dir", type=str,
        default=str(BLUEBOX_PACKAGE_ROOT / "agent_docs"),
        help="Documentation directory",
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress logs")
    parser.add_argument("--log-file", type=str, default=None, help="Log to file")
    args = parser.parse_args()

    configure_logging(quiet=args.quiet, log_file=args.log_file)

    console = Console()

    if Config.OPENAI_API_KEY is None:
        console.print("[bold red]Error: OPENAI_API_KEY environment variable is not set[/bold red]")
        sys.exit(1)

    try:
        llm_model = parse_model(args.model)
        openai_client = OpenAI(api_key=Config.OPENAI_API_KEY)

        data_store_kwargs: dict[str, Any] = {
            "client": openai_client,
            "documentation_paths": [args.docs_dir],
            "code_paths": [
                str(BLUEBOX_PACKAGE_ROOT / "data_models" / "routine"),
                str(BLUEBOX_PACKAGE_ROOT / "data_models" / "ui_elements.py"),
                str(BLUEBOX_PACKAGE_ROOT / "agents" / "routine_discovery_agent.py"),
                str(BLUEBOX_PACKAGE_ROOT / "llms" / "infra" / "data_store.py"),
                str(BLUEBOX_PACKAGE_ROOT / "utils" / "js_utils.py"),
                str(BLUEBOX_PACKAGE_ROOT / "utils" / "data_utils.py"),
                "!" + str(BLUEBOX_PACKAGE_ROOT / "**" / "__init__.py"),
            ],
        }

        if args.cdp_captures_dir:
            data_store_kwargs.update({
                "cdp_captures_dir": args.cdp_captures_dir,
                "tmp_dir": str(Path(args.output_dir) / "tmp"),
            })

        with console.status("[bold blue]Initializing...[/bold blue]") as status:
            status.update("[bold blue]Creating data store...[/bold blue]")
            data_store = LocalDiscoveryDataStore(**data_store_kwargs)

            if args.cdp_captures_dir:
                status.update("[bold blue]Creating CDP captures vectorstore...[/bold blue]")
                data_store.make_cdp_captures_vectorstore()

            status.update("[bold blue]Creating documentation vectorstore...[/bold blue]")
            data_store.make_documentation_vectorstore()

        console.print("[green]✓ Vectorstores ready![/green]")
        console.print()

        cdp_captures_dir = Path(args.cdp_captures_dir) if args.cdp_captures_dir else DEFAULT_CDP_CAPTURES_DIR

        app = GuideAgentTUI(
            llm_model=llm_model,
            data_store=data_store,
            cdp_captures_dir=cdp_captures_dir,
        )
        app.run()

    except ValueError as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[bold red]Fatal error: {e}[/bold red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
