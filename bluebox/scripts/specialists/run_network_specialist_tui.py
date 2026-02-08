"""
bluebox/scripts/specialists/run_network_specialist_tui.py

Multi-pane terminal UI for the NetworkSpecialist using Textual.

Layout:
  ┌─────────────────────────────┬──────────────────────┐
  │                             │  Tool Calls History   │
  │       Chat (scrolling)      │                       │
  │                             ├──────────────────────┤
  │  ┌────────────────────────┐ │  Status / Stats       │
  │  │ Input                  │ │                       │
  │  └────────────────────────┘ │                       │
  └─────────────────────────────┴──────────────────────┘

Usage:
    bluebox-network-specialist-tui --jsonl-path ./cdp_captures/network/events.jsonl
    bluebox-network-specialist-tui --jsonl-path ./cdp_captures/network/events.jsonl --model gpt-5.1
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.markup import escape
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.suggester import SuggestFromList
from textual.widgets import Input, RichLog, Static

from bluebox.agents.specialists.network_specialist import NetworkSpecialist
from bluebox.data_models.llms.interaction import (
    ChatRole,
    BaseEmittedMessage,
    ChatResponseEmittedMessage,
    ToolInvocationResultEmittedMessage,
    ErrorEmittedMessage,
    ToolInvocationStatus,
)
from bluebox.data_models.llms.vendors import LLMModel, OpenAIModel
from bluebox.data_models.orchestration.result import SpecialistResultWrapper
from bluebox.llms.data_loaders.network_data_loader import NetworkDataLoader
from bluebox.utils.cli_utils import add_model_argument, resolve_model
from bluebox.utils.logger import get_logger

logger = get_logger(name=__name__)

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
    "/discover", "/reset", "/status", "/chats", "/clear", "/help", "/quit",
]

HELP_TEXT = """\
[bold]Commands:[/bold]
  [cyan]/discover <task>[/cyan]  Discover API endpoints for a task
  [cyan]/status[/cyan]           Show current state
  [cyan]/chats[/cyan]            Show message history
  [cyan]/clear[/cyan]            Clear the chat display
  [cyan]/reset[/cyan]            Start new conversation
  [cyan]/help[/cyan]             Show this help
  [cyan]/quit[/cyan]             Exit
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
    min-height: 10;
    max-height: 16;
    border: solid $primary;
    border-title-color: $primary;
    padding: 0 1;
}
"""


# ─── Helpers ─────────────────────────────────────────────────────────────────

def get_context_window_size(model_value: str) -> int:
    """Get approximate context window size for a model."""
    for prefix, size in MODEL_CONTEXT_WINDOWS.items():
        if model_value.startswith(prefix):
            return size
    return DEFAULT_CONTEXT_WINDOW


def configure_logging(quiet: bool = False, log_file: str | None = None) -> None:
    """Configure logging — in TUI mode, always redirect to file or suppress."""
    import logging

    resolved_log_file = log_file or ".bluebox_network_tui.log"

    root_logger = logging.getLogger()
    root_logger.handlers.clear()

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
    root_logger.addHandler(file_handler)


# ─── Slash command suggester ─────────────────────────────────────────────────

class SlashCommandSuggester(SuggestFromList):
    """Only suggest when input starts with '/'."""

    async def get_suggestion(self, value: str) -> str | None:
        if not value.startswith("/"):
            return None
        return await super().get_suggestion(value)


# ─── Textual App ─────────────────────────────────────────────────────────────

class NetworkSpecialistTUI(App):
    """Multi-pane TUI for the Network Specialist."""

    CSS = APP_CSS
    TITLE = "Network Spy"
    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", priority=True),
    ]

    def __init__(
        self,
        llm_model: LLMModel,
        network_store: NetworkDataLoader,
        data_path: str = "",
    ) -> None:
        super().__init__()
        self._llm_model = llm_model
        self._network_store = network_store
        self._data_path = data_path
        self._context_window_size = get_context_window_size(llm_model.value)

        # Agent state
        self._agent: NetworkSpecialist | None = None
        self._streaming_started = False
        self._stream_buffer: list[str] = []
        self._tool_call_count = 0
        self._last_seen_chat_count = 0

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
        self._agent = NetworkSpecialist(
            emit_message_callable=self._handle_message,
            stream_chunk_callable=self._handle_stream_chunk,
            network_data_store=self._network_store,
            llm_model=self._llm_model,
        )
        self._print_welcome()
        self._update_status()
        self.set_interval(10, self._update_status)
        self.query_one("#user-input", Input).focus()

    # ── Welcome ──────────────────────────────────────────────────────────

    def _print_welcome(self) -> None:
        chat = self.query_one("#chat-log", RichLog)
        chat.write(Text.from_markup(
            "[bold cyan]Network Spy[/bold cyan]  "
            "[dim]powered by vectorly[/dim]"
        ))
        chat.write("")

        stats = self._network_store.stats

        # Stats summary
        lines = [
            f"[dim]Total Requests:[/dim] {stats.total_requests}",
            f"[dim]Unique URLs:[/dim]    {stats.unique_urls}",
            f"[dim]Unique Hosts:[/dim]   {stats.unique_hosts}",
        ]

        # Methods
        if stats.methods:
            methods_str = ", ".join(f"{m}: {c}" for m, c in sorted(stats.methods.items(), key=lambda x: -x[1]))
            lines.append(f"[dim]Methods:[/dim]        {methods_str}")

        # Features
        features = []
        if stats.has_cookies:
            features.append("Cookies")
        if stats.has_auth_headers:
            features.append("Auth Headers")
        if stats.has_json_requests:
            features.append("JSON")
        if stats.has_form_data:
            features.append("Form Data")
        if features:
            lines.append(f"[dim]Features:[/dim]       {', '.join(features)}")

        if self._data_path:
            lines.append(f"[dim]File:[/dim]           {self._data_path}")

        chat.write(Text.from_markup("\n".join(lines)))
        chat.write("")

        # Show top hosts
        host_stats = self._network_store.get_host_stats()
        if host_stats:
            host_lines = ["[dim]Top Hosts:[/dim]"]
            for hs in host_stats[:8]:
                methods_str = ", ".join(f"{m}:{c}" for m, c in sorted(hs["methods"].items()))
                host = hs["host"][:45] + "..." if len(hs["host"]) > 45 else hs["host"]
                host_lines.append(f"  {host} ({hs['request_count']} reqs, {methods_str})")
            if len(host_stats) > 8:
                host_lines.append(f"  [dim]... and {len(host_stats) - 8} more hosts[/dim]")
            chat.write(Text.from_markup("\n".join(host_lines)))
            chat.write("")

        # Show likely API endpoints
        likely_urls = self._network_store.api_urls
        if likely_urls:
            url_lines = [f"[dim]Likely API Endpoints ({len(likely_urls)}):[/dim]"]
            for url in likely_urls[:15]:
                url_lines.append(f"  [yellow]{escape(url)}[/yellow]")
            if len(likely_urls) > 15:
                url_lines.append(f"  [dim]... and {len(likely_urls) - 15} more[/dim]")
            chat.write(Text.from_markup("\n".join(url_lines)))
            chat.write("")

        chat.write(Text.from_markup(
            "Type [cyan]/help[/cyan] for commands, or ask questions about the network traffic."
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
        tokens = total_chars // 4
        pct = (tokens / self._context_window_size) * 100 if self._context_window_size > 0 else 0.0
        return tokens, min(pct, 100.0)

    def _context_bar(self, pct: float, width: int = 15) -> str:
        """Render a text-based context fill bar."""
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
        stats = self._network_store.stats

        status_text = (
            f"[bold cyan]NETWORK SPY[/bold cyan]\n"
            f"[dim]──────────────────[/dim]\n"
            f"[dim]Model:[/dim]     {self._llm_model.value}\n"
            f"[dim]Messages:[/dim]  {msg_count}\n"
            f"[dim]Tools:[/dim]     {self._tool_call_count}\n"
            f"[dim]Context:[/dim]   {ctx_bar}\n"
            f"[dim](est.)      ~{tokens_used:,} / {self._context_window_size:,}[/dim]\n"
            f"[dim]──────────────────[/dim]\n"
            f"[dim]Requests:[/dim]  {stats.total_requests}\n"
            f"[dim]URLs:[/dim]      {stats.unique_urls}\n"
            f"[dim]Hosts:[/dim]     {stats.unique_hosts}\n"
            f"[dim]Time:[/dim]      {now}\n"
        )
        panel.update(Text.from_markup(status_text))

    # ── Agent callbacks ──────────────────────────────────────────────────

    def _handle_stream_chunk(self, chunk: str) -> None:
        """Buffer streaming chunks, flushing complete lines immediately."""
        chat = self.query_one("#chat-log", RichLog)
        if not self._streaming_started:
            chat.write(Text.from_markup("\n[bold cyan]Assistant[/bold cyan]"))
            self._streaming_started = True

        self._stream_buffer.append(chunk)

        combined = "".join(self._stream_buffer)
        if "\n" in combined:
            lines = combined.split("\n")
            for line in lines[:-1]:
                chat.write(line)
            self._stream_buffer.clear()
            if lines[-1]:
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
                    f"[green]✓[/green] [dim]{tool_name} (auto)[/dim]"
                ))

                tool_log.write(Text.from_markup(
                    f"[dim]{ts}[/dim] [green]AUTO[/green] [bold]{tool_name}[/bold]"
                ))
                result_preview = c.content[c.content.find("result: ") + 8:] if "result: " in c.content else c.content
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

        if isinstance(message, ChatResponseEmittedMessage):
            self._flush_auto_executed_tools(tool_log, chat)

            if self._streaming_started:
                remainder = "".join(self._stream_buffer)
                if remainder:
                    chat.write(remainder)
                self._stream_buffer.clear()
                self._streaming_started = False
            else:
                chat.write(Text.from_markup("\n[bold cyan]Assistant[/bold cyan]"))
                chat.write(message.content)
            chat.write("")

        elif isinstance(message, ToolInvocationResultEmittedMessage):
            inv = message.tool_invocation
            ts = datetime.now().strftime("%H:%M:%S")
            self._tool_call_count += 1

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

        chat = self.query_one("#chat-log", RichLog)

        # Show user message
        chat.write(Text.from_markup(f"[bold green]You>[/bold green] {escape(user_input)}"))

        cmd = user_input.lower()

        if cmd in ("/quit", "/exit", "/q"):
            self.exit()
            return
        if cmd in ("/help", "/h", "/?"):
            chat.write(Text.from_markup(HELP_TEXT))
            return
        if cmd == "/reset":
            self._agent.reset()
            self._last_seen_chat_count = 0
            chat.write(Text.from_markup("[yellow]↺ Conversation reset[/yellow]"))
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

        # Handle /discover <task>
        if user_input.lower().startswith("/discover"):
            task = user_input[9:].strip()
            if not task:
                chat.write(Text.from_markup("[yellow]Usage: /discover <task>[/yellow]"))
                return
            self._run_discovery(task)
            return

        # Regular message → send to agent
        self._send_to_agent(user_input)

    @work(thread=True)
    def _send_to_agent(self, user_input: str) -> None:
        """Send message to agent in a background thread."""
        self._agent.process_new_message(user_input, ChatRole.USER)
        self.call_from_thread(self._update_status)

    # ── Autonomous discovery ─────────────────────────────────────────────

    @work(thread=True)
    def _run_discovery(self, task: str) -> None:
        """Run autonomous endpoint discovery in a background thread."""
        chat = self.query_one("#chat-log", RichLog)
        tool_log = self.query_one("#tool-log", RichLog)

        self.call_from_thread(
            lambda: chat.write(Text.from_markup(
                f"\n[bold magenta]Starting Autonomous Discovery[/bold magenta]\n"
                f"[dim]Task:[/dim] {escape(task)}"
            ))
        )

        self._agent.reset()
        self._last_seen_chat_count = 0

        start_time = time.perf_counter()
        result = self._agent.run_autonomous(task)
        elapsed = time.perf_counter() - start_time
        iterations = self._agent.autonomous_iteration

        def _show_result() -> None:
            chat.write("")

            if isinstance(result, SpecialistResultWrapper) and result.success and result.output:
                # Success — show the output
                output = result.output
                chat.write(Text.from_markup(
                    f"[bold green]✓ Discovery Complete[/bold green] "
                    f"[dim]({iterations} iterations, {elapsed:.1f}s)[/dim]"
                ))
                output_str = json.dumps(output, indent=2)
                output_lines = output_str.split("\n")
                if len(output_lines) > 40:
                    output_str = "\n".join(output_lines[:40]) + f"\n... ({len(output_lines) - 40} more lines)"
                chat.write(output_str)

                # Also log to tool pane
                tool_log.write(Text.from_markup(
                    f"[green]DISCOVERY RESULT[/green] [dim]({iterations} iter, {elapsed:.1f}s)[/dim]"
                ))
                tool_log.write(output_str[:500])
                tool_log.write("")

            elif isinstance(result, SpecialistResultWrapper) and not result.success:
                # Explicit failure
                reason = result.failure_reason or "Unknown"
                chat.write(Text.from_markup(
                    f"[bold red]✗ Endpoint Not Found[/bold red] "
                    f"[dim]({iterations} iterations, {elapsed:.1f}s)[/dim]\n"
                    f"[red]Reason:[/red] {escape(reason)}"
                ))

                # Notes if any
                if result.notes:
                    notes_str = "\n".join(f"  - {n}" for n in result.notes[:10])
                    chat.write(Text.from_markup(f"[dim]Notes:[/dim]\n{notes_str}"))

            else:
                # None or unexpected — max iterations without finalization
                chat.write(Text.from_markup(
                    f"[bold yellow]⚠ Discovery Incomplete[/bold yellow] "
                    f"[dim]({iterations} iterations, {elapsed:.1f}s)[/dim]\n"
                    "[yellow]Agent reached max iterations without finalizing.[/yellow]"
                ))

            chat.write("")
            self._update_status()

        self.call_from_thread(_show_result)

    # ── Slash command handlers ───────────────────────────────────────────

    def _show_status_in_chat(self) -> None:
        chat = self.query_one("#chat-log", RichLog)
        stats = self._network_store.stats
        msg_count = len(self._agent.get_chats()) if self._agent else 0
        tokens_used, ctx_pct = self._estimate_context_usage()

        chat.write(Text.from_markup(
            f"[bold cyan]Status[/bold cyan]\n"
            f"  Model: {self._llm_model.value}\n"
            f"  Messages: {msg_count}\n"
            f"  Context: ~{tokens_used:,}t ({ctx_pct:.0f}%)\n"
            f"  Requests: {stats.total_requests}\n"
            f"  URLs: {stats.unique_urls}\n"
            f"  Hosts: {stats.unique_hosts}\n"
            f"  File: {self._data_path or 'N/A'}"
        ))

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
            content = (c.content or "").replace("\n", " ")[:60]
            suffix = "..." if len(c.content or "") > 60 else ""
            if c.role.value == "tool":
                tid = (c.tool_call_id[:8] + "...") if c.tool_call_id else "?"
                chat.write(Text.from_markup(f"[dim]{i}.[/dim] [{color}]TOOL[/{color}] [dim]({tid})[/dim] {escape(content)}{suffix}"))
            elif c.role.value == "assistant" and c.tool_calls:
                tool_names = ", ".join(tc.tool_name for tc in c.tool_calls)
                chat.write(Text.from_markup(f"[dim]{i}.[/dim] [{color}]ASSISTANT[/{color}] [yellow]→ {tool_names}[/yellow]"))
            else:
                chat.write(Text.from_markup(f"[dim]{i}.[/dim] [{color}]{c.role.value.upper()}[/{color}] {escape(content)}{suffix}"))


# ─── Entry point ─────────────────────────────────────────────────────────────

def main() -> None:
    """Entry point for the network specialist TUI."""
    parser = argparse.ArgumentParser(description="Network Spy — Multi-pane TUI")
    parser.add_argument(
        "--jsonl-path",
        type=str,
        required=True,
        help="Path to the JSONL file containing NetworkTransactionEvent entries",
    )
    add_model_argument(parser)
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress logs")
    parser.add_argument("--log-file", type=str, default=None, help="Log to file")
    args = parser.parse_args()

    configure_logging(quiet=args.quiet, log_file=args.log_file)

    console = Console()

    # Load JSONL file
    jsonl_path = Path(args.jsonl_path)
    if not jsonl_path.exists():
        console.print(f"[bold red]Error: JSONL file not found: {jsonl_path}[/bold red]")
        sys.exit(1)

    console.print(f"[dim]Loading JSONL file: {jsonl_path}[/dim]")

    try:
        network_store = NetworkDataLoader(jsonl_path)
    except ValueError as e:
        console.print(f"[bold red]Error parsing JSONL file: {e}[/bold red]")
        sys.exit(1)

    llm_model = resolve_model(args.model, console)

    console.print(f"[green]✓ Loaded {network_store.stats.total_requests} requests[/green]")
    console.print()

    app = NetworkSpecialistTUI(
        llm_model=llm_model,
        network_store=network_store,
        data_path=str(jsonl_path),
    )
    app.run()


if __name__ == "__main__":
    main()
