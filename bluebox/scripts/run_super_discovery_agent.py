"""
bluebox/scripts/run_super_discovery_agent.py

Interactive terminal interface for the SuperDiscoveryAgent.
Guides users through discovering web automation routines from CDP captures.

Usage:
    # Basic usage (starts interactive session)
    bluebox-super-discovery --cdp-captures-dir ./cdp_captures
    
    # Run discovery immediately
    bluebox-super-discovery --cdp-captures-dir ./cdp_captures --task "Endpoint/routine for obtaining all Premier League matches for a given week"

    # Suppress logs for cleaner output
    bluebox-super-discovery --cdp-captures-dir ./cdp_captures -q
    
    # Run with routine validation (will validate the routine after discovery)
    bluebox-super-discovery --cdp-captures-dir ./cdp_captures --remote-debugging-address http://127.0.0.1:9222

Commands:
    /discover <task>         Start routine discovery for the given task
    /execute                 Execute the discovered routine with test parameters
    /status                  Show current discovery state
    /chats                   Show all messages in the thread
    /routine                 Show the current/discovered routine
    /save <path>             Save routine to file
    /reset                   Start a new conversation
    /help                    Show help
    /quit                    Exit
"""

import argparse
import hashlib
import json
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from prompt_toolkit import prompt as pt_prompt
from prompt_toolkit.formatted_text import HTML
from rich import box
from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from bluebox.agents.super_discovery_agent import SuperDiscoveryAgent
from bluebox.config import Config
from bluebox.data_models.llms.interaction import (
    BaseEmittedMessage,
    ChatResponseEmittedMessage,
    ChatRole,
    ErrorEmittedMessage,
    ToolInvocationResultEmittedMessage,
)
from bluebox.data_models.llms.vendors import OpenAIModel
from bluebox.data_models.routine.routine import Routine
from bluebox.llms.data_loaders.documentation_data_loader import DocumentationDataLoader
from bluebox.llms.data_loaders.interactions_data_loader import InteractionsDataLoader
from bluebox.llms.data_loaders.js_data_loader import JSDataLoader
from bluebox.llms.data_loaders.network_data_loader import NetworkDataLoader
from bluebox.llms.data_loaders.storage_data_loader import StorageDataLoader
from bluebox.llms.data_loaders.window_property_data_loader import WindowPropertyDataLoader
from bluebox.llms.tools.execute_routine_tool import execute_routine
from bluebox.utils.logger import get_logger
from bluebox.utils.terminal_utils import SlashCommandCompleter, SlashCommandLexer


# Package root for code_paths
BLUEBOX_PACKAGE_ROOT = Path(__file__).resolve().parent.parent

logger = get_logger(__name__)
console = Console()

SLASH_COMMANDS = [
    ("/discover", "Start routine discovery — /discover <task description>"),
    ("/execute", "Execute the discovered routine with test parameters"),
    ("/status", "Show current discovery state"),
    ("/chats", "Show all messages in the thread"),
    ("/routine", "Show the current/discovered routine"),
    ("/save", "Save routine to file — /save <path.json>"),
    ("/reset", "Start a new conversation"),
    ("/help", "Show help"),
    ("/quit", "Exit"),
]

BANNER = """
[bold cyan]╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║   ███████╗██╗   ██╗██████╗ ███████╗██████╗     ██████╗ ██╗███████╗ ██████╗ ██████╗ ██╗   ██╗███████╗██████╗ ██╗   ██╗   ║
║   ██╔════╝██║   ██║██╔══██╗██╔════╝██╔══██╗    ██╔══██╗██║██╔════╝██╔════╝██╔═══██╗██║   ██║██╔════╝██╔══██╗╚██╗ ██╔╝   ║
║   ███████╗██║   ██║██████╔╝█████╗  ██████╔╝    ██║  ██║██║███████╗██║     ██║   ██║██║   ██║█████╗  ██████╔╝ ╚████╔╝    ║
║   ╚════██║██║   ██║██╔═══╝ ██╔══╝  ██╔══██╗    ██║  ██║██║╚════██║██║     ██║   ██║╚██╗ ██╔╝██╔══╝  ██╔══██╗  ╚██╔╝     ║
║   ███████║╚██████╔╝██║     ███████╗██║  ██║    ██████╔╝██║███████║╚██████╗╚██████╔╝ ╚████╔╝ ███████╗██║  ██║   ██║      ║
║   ╚══════╝ ╚═════╝ ╚═╝     ╚══════╝╚═╝  ╚═╝    ╚═════╝ ╚═╝╚══════╝ ╚═════╝ ╚═════╝   ╚═══╝  ╚══════╝╚═╝  ╚═╝   ╚═╝      ║
║                                                                                                                        ║
║[/bold cyan][dim]                                          powered by Vectorly                                                         [/dim][bold cyan]║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝[/bold cyan]
"""


def configure_logging(quiet: bool = False, log_file: str | None = None) -> None:
    """
    Configure logging for all bluebox modules.

    Args:
        quiet: If True, suppress all logs to console.
        log_file: If provided, write logs to this file instead of console.
    """
    wh_logger = logging.getLogger("bluebox")

    if quiet:
        wh_logger.setLevel(logging.CRITICAL + 1)
        return

    if log_file:
        wh_logger.handlers.clear()
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(
            fmt="[%(asctime)s] %(levelname)s:%(name)s:%(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        ))
        wh_logger.addHandler(file_handler)
        wh_logger.propagate = False


def print_welcome(model: str, data_summary: dict[str, Any]) -> None:
    """Print welcome message and help."""
    console.print(BANNER)
    console.print()

    # Data summary
    data_info = []
    if data_summary.get("network"):
        data_info.append(f"[green]✓[/green] Network: {data_summary['network']} transactions")
    if data_summary.get("storage"):
        data_info.append(f"[green]✓[/green] Storage: {data_summary['storage']} events")
    if data_summary.get("window_props"):
        data_info.append(f"[green]✓[/green] Window Props: {data_summary['window_props']} events")
    if data_summary.get("js"):
        data_info.append(f"[green]✓[/green] JS Files: {data_summary['js']} files")
    if data_summary.get("interactions"):
        data_info.append(f"[green]✓[/green] Interactions: {data_summary['interactions']} events")

    data_section = "\n".join(data_info) if data_info else "[yellow]No CDP data loaded[/yellow]"

    console.print(Panel(
        f"""[bold]Welcome![/bold] I'll help you discover web automation routines from your
CDP (Chrome DevTools Protocol) captures.

I coordinate specialist agents (network, JS, value-resolver) to analyze
your captured traffic and build reusable routines.

[bold]Loaded Data:[/bold]
{data_section}

[bold]Commands:[/bold]
  [cyan]/discover <task>[/cyan]         Start routine discovery for the given task
  [cyan]/execute[/cyan]                 Execute routine with test parameters
  [cyan]/status[/cyan]                  Show current discovery state
  [cyan]/chats[/cyan]                   Show all messages in the thread
  [cyan]/routine[/cyan]                 Show the current/discovered routine
  [cyan]/save <path>[/cyan]             Save routine to file
  [cyan]/reset[/cyan]                   Start a new conversation
  [cyan]/help[/cyan]                    Show all commands
  [cyan]/quit[/cyan]                    Exit

[bold]Quick Start:[/bold]
  Type [cyan]/discover Search for trains from NYC to Boston[/cyan] to begin!

[bold]Links:[/bold]
  [link=https://vectorly.app/docs]https://vectorly.app/docs[/link]
  [link=https://console.vectorly.app]https://console.vectorly.app[/link]""",
        title="[bold cyan]Super Discovery Agent[/bold cyan]",
        subtitle=f"[dim]Model: {model}[/dim]",
        border_style="cyan",
        box=box.ROUNDED,
    ))
    console.print()


def print_assistant_message(content: str) -> None:
    """Print an assistant response using markdown rendering."""
    console.print()
    console.print("[bold cyan]Assistant[/bold cyan]")
    console.print()
    console.print(Markdown(content))
    console.print()


def print_error(error: str) -> None:
    """Print an error message."""
    console.print()
    console.print(f"[bold red]⚠ Error:[/bold red] [red]{escape(error)}[/red]")
    console.print()


def print_tool_result(tool_name: str, status: str, result: Any) -> None:
    """Print a tool invocation result."""
    # Format result preview (truncate if too long)
    if isinstance(result, str):
        result_preview = result[:200] + "..." if len(result) > 200 else result
    elif isinstance(result, dict):
        result_str = json.dumps(result, indent=2)
        lines = result_str.split("\n")
        if len(lines) > 10:
            result_preview = "\n".join(lines[:10]) + f"\n... ({len(lines) - 10} more lines)"
        else:
            result_preview = result_str
    else:
        result_preview = str(result)[:200]

    status_color = "green" if status == "executed" else "yellow" if status == "denied" else "red"
    console.print(f"  [{status_color}]🔧 {tool_name}[/{status_color}] [dim]({status})[/dim]")


class TerminalSuperDiscoveryChat:
    """Interactive terminal chat interface for the SuperDiscoveryAgent."""

    def __init__(
        self,
        network_data_loader: NetworkDataLoader,
        storage_data_loader: StorageDataLoader | None = None,
        window_property_data_loader: WindowPropertyDataLoader | None = None,
        js_data_loader: JSDataLoader | None = None,
        interaction_data_loader: InteractionsDataLoader | None = None,
        documentation_data_loader: DocumentationDataLoader | None = None,
        llm_model: OpenAIModel = OpenAIModel.GPT_5_1,
        subagent_llm_model: OpenAIModel | None = None,
        max_iterations: int = 50,
        remote_debugging_address: str | None = None,
        output_dir: Path | None = None,
    ) -> None:
        """Initialize the terminal chat interface."""
        self._network_data_loader = network_data_loader
        self._storage_data_loader = storage_data_loader
        self._window_property_data_loader = window_property_data_loader
        self._js_data_loader = js_data_loader
        self._interaction_data_loader = interaction_data_loader
        self._documentation_data_loader = documentation_data_loader
        self._llm_model = llm_model
        self._subagent_llm_model = subagent_llm_model
        self._max_iterations = max_iterations
        self._remote_debugging_address = remote_debugging_address
        self._output_dir = output_dir or Path("./routine_discovery_output")

        # State
        self._streaming_started: bool = False
        self._current_agent: SuperDiscoveryAgent | None = None
        self._chat_agent: SuperDiscoveryAgent | None = None  # Lazy-initialized for chat mode
        self._discovered_routine: Routine | None = None
        self._is_discovering: bool = False
        self._message_history: list[dict] = []
        self._last_state_hash: str | None = None  # For tracking state changes

    def _get_prompt(self) -> HTML:
        """Get the input prompt (HTML format for prompt_toolkit)."""
        if self._is_discovering:
            return HTML("<b><ansigreen>You</ansigreen></b> <ansiyellow>(discovering)</ansiyellow><b><ansigreen>&gt;</ansigreen></b> ")
        elif self._discovered_routine:
            name = self._discovered_routine.name[:15] + "..." if len(self._discovered_routine.name) > 15 else self._discovered_routine.name
            return HTML(f"<b><ansigreen>You</ansigreen></b> <ansigray>({name})</ansigray><b><ansigreen>&gt;</ansigreen></b> ")
        return HTML("<b><ansigreen>You&gt;</ansigreen></b> ")

    def _handle_stream_chunk(self, chunk: str) -> None:
        """Handle a streaming text chunk from the LLM."""
        if not self._streaming_started:
            console.print()
            console.print("[bold cyan]Assistant[/bold cyan]")
            console.print()
            self._streaming_started = True

        print(chunk, end="", flush=True)

    def _handle_message(self, message: BaseEmittedMessage) -> None:
        """Handle messages emitted by the SuperDiscoveryAgent."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        message_dict: dict[str, Any] = {
            "type": message.__class__.__name__,
            "timestamp": timestamp,
        }

        if isinstance(message, ChatResponseEmittedMessage):
            if self._streaming_started:
                print()  # End the streamed line
                print()  # Add spacing
                self._streaming_started = False
            else:
                # Show shortened preview in discovery mode
                if self._is_discovering:
                    preview = message.content[:150] + "..." if len(message.content) > 150 else message.content
                    console.print(f"[dim]💬 {preview}[/dim]")
                else:
                    print_assistant_message(message.content)
            message_dict["content"] = message.content

        elif isinstance(message, ErrorEmittedMessage):
            print_error(message.error)
            message_dict["error"] = message.error

        elif isinstance(message, ToolInvocationResultEmittedMessage):
            tool_name = message.tool_invocation.tool_name
            status = message.tool_invocation.status.value
            tool_result = message.tool_result

            if self._is_discovering:
                print_tool_result(tool_name, status, tool_result)

            message_dict["tool_name"] = tool_name
            message_dict["status"] = status
            message_dict["tool_arguments"] = message.tool_invocation.tool_arguments
            message_dict["result"] = tool_result

        self._message_history.append(message_dict)

        # Dump full chat thread after each message
        self._dump_chat_thread()

        # Dump state if changed
        self._dump_state_if_changed()

    def _dump_chat_thread(self) -> None:
        """Dump chat threads for super agent and all subagents to chat_threads/ directory."""
        if not self._current_agent:
            return

        # Create chat_threads directory
        chat_threads_dir = self._output_dir / "chat_threads"
        chat_threads_dir.mkdir(parents=True, exist_ok=True)

        # Dump super agent's chat thread
        self._dump_agent_thread(
            self._current_agent._thread,
            self._current_agent._chats,
            chat_threads_dir / "super_agent.json",
            agent_type="super_agent",
        )

        # Dump each subagent's chat thread
        for agent_id, agent_instance in self._current_agent._agent_instances.items():
            # Get the subagent type from orchestration state
            subagent_info = self._current_agent._orchestration_state.subagents.get(agent_id)
            agent_type = subagent_info.type.value if subagent_info else "unknown"

            self._dump_agent_thread(
                agent_instance._thread,
                agent_instance._chats,
                chat_threads_dir / f"{agent_type}_{agent_id}.json",
                agent_type=agent_type,
                agent_id=agent_id,
            )

    def _dump_agent_thread(
        self,
        thread: Any,
        chats: dict[str, Any],
        output_path: Path,
        agent_type: str,
        agent_id: str | None = None,
    ) -> None:
        """Dump a single agent's chat thread to a JSON file."""
        # Build ordered list of messages
        messages = []
        for chat_id in thread.chat_ids:
            chat = chats.get(chat_id)
            if chat:
                messages.append({
                    "id": chat.id,
                    "role": chat.role.value,
                    "content": chat.content,
                    "tool_calls": [tc.model_dump() for tc in chat.tool_calls] if chat.tool_calls else None,
                    "tool_call_id": chat.tool_call_id,
                })

        # Write to file
        thread_data: dict[str, Any] = {
            "agent_type": agent_type,
            "thread_id": thread.id,
            "updated_at": thread.updated_at,
            "messages": messages,
        }
        if agent_id:
            thread_data["agent_id"] = agent_id

        output_path.write_text(json.dumps(thread_data, indent=2, default=str))

    def _dump_state_if_changed(self) -> None:
        """Dump discovery and orchestration state to state/ directory if changed."""
        if not self._current_agent:
            return

        # Build current state snapshot
        discovery_state = self._current_agent._discovery_state
        orchestration_state = self._current_agent._orchestration_state

        state_snapshot: dict[str, Any] = {
            "discovery_state": {
                "phase": discovery_state.phase.value,
                "root_transaction": discovery_state.root_transaction.model_dump() if discovery_state.root_transaction else None,
                "transaction_queue": list(discovery_state.transaction_queue),
                "processed_transactions": list(discovery_state.processed_transactions),
                "transaction_data": {
                    tx_id: {
                        "request": tx_data.get("request"),
                        "extracted_variables": tx_data["extracted_variables"].model_dump() if tx_data.get("extracted_variables") else None,
                        "resolved_variables": [rv.model_dump() for rv in tx_data.get("resolved_variables", [])],
                    }
                    for tx_id, tx_data in discovery_state.transaction_data.items()
                },
                "production_routine": discovery_state.production_routine.model_dump() if discovery_state.production_routine else None,
                "test_parameters": discovery_state.test_parameters,
                "construction_attempts": discovery_state.construction_attempts,
            },
            "orchestration_state": {
                "tasks": {
                    task_id: {
                        "id": task.id,
                        "agent_type": task.agent_type.value,
                        "status": task.status.value,
                        "prompt": task.prompt,
                        "result": task.result,
                        "error": task.error,
                        "loops_used": task.loops_used,
                        "max_loops": task.max_loops,
                    }
                    for task_id, task in orchestration_state.tasks.items()
                },
                "subagents": {
                    agent_id: {
                        "id": subagent.id,
                        "type": subagent.type.value,
                        "task_ids": subagent.task_ids,
                    }
                    for agent_id, subagent in orchestration_state.subagents.items()
                },
            },
        }

        # Compute hash of current state
        state_json = json.dumps(state_snapshot, sort_keys=True, default=str)
        state_hash = hashlib.sha256(state_json.encode()).hexdigest()[:16]

        # Only save if state has changed
        if state_hash == self._last_state_hash:
            return

        self._last_state_hash = state_hash

        # Save to state directory with timestamp
        state_dir = self._output_dir / "state"
        state_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        state_path = state_dir / f"{timestamp}.json"

        state_snapshot["_meta"] = {
            "timestamp": timestamp,
            "hash": state_hash,
        }

        state_path.write_text(json.dumps(state_snapshot, indent=2, default=str))

    def _create_agent(self, task: str) -> SuperDiscoveryAgent:
        """Create a new SuperDiscoveryAgent for the given task."""
        return SuperDiscoveryAgent(
            emit_message_callable=self._handle_message,
            stream_chunk_callable=self._handle_stream_chunk,
            network_data_loader=self._network_data_loader,
            task=task,
            storage_data_loader=self._storage_data_loader,
            window_property_data_loader=self._window_property_data_loader,
            js_data_loader=self._js_data_loader,
            interaction_data_loader=self._interaction_data_loader,
            documentation_data_loader=self._documentation_data_loader,
            llm_model=self._llm_model,
            subagent_llm_model=self._subagent_llm_model,
            max_iterations=self._max_iterations,
            remote_debugging_address=self._remote_debugging_address,
        )

    def _init_chat_agent(self) -> None:
        """Initialize the chat agent for conversational mode."""
        # Create an agent for answering questions about the captured data
        # The task is a placeholder - we use process_new_message() for chat, not run()
        self._chat_agent = SuperDiscoveryAgent(
            emit_message_callable=self._handle_message,
            stream_chunk_callable=self._handle_stream_chunk,
            network_data_loader=self._network_data_loader,
            task="Help the user understand their CDP captures and answer questions about routine discovery.",
            storage_data_loader=self._storage_data_loader,
            window_property_data_loader=self._window_property_data_loader,
            js_data_loader=self._js_data_loader,
            interaction_data_loader=self._interaction_data_loader,
            documentation_data_loader=self._documentation_data_loader,
            llm_model=self._llm_model,
            subagent_llm_model=self._subagent_llm_model,
            max_iterations=self._max_iterations,
            remote_debugging_address=self._remote_debugging_address,
        )

    def _handle_chat_message(self, user_input: str) -> None:
        """Handle a normal chat message (non-command)."""
        if not self._chat_agent:
            self._init_chat_agent()

        try:
            self._chat_agent.process_new_message(user_input, ChatRole.USER)
        except Exception as e:
            print_error(f"Chat error: {e}")

    def _handle_discover_command(self, task: str) -> None:
        """Handle /discover command to start routine discovery."""
        if not task.strip():
            console.print()
            console.print("[yellow]Usage: /discover <task description>[/yellow]")
            console.print("[dim]Example: /discover Search for trains from NYC to Boston[/dim]")
            console.print()
            return

        # Clear existing output directory and reset state tracking
        if self._output_dir.exists():
            shutil.rmtree(self._output_dir)
            console.print(f"[dim]Cleared existing output directory: {self._output_dir}[/dim]")
        self._last_state_hash = None  # Reset to capture initial state

        console.print()
        console.print(Panel(
            f"[bold]Task:[/bold] {task}",
            title="[bold cyan]Starting Discovery[/bold cyan]",
            border_style="cyan",
            box=box.ROUNDED,
        ))
        console.print()

        self._is_discovering = True
        self._message_history = []

        try:
            # Create agent and run discovery
            self._current_agent = self._create_agent(task)

            routine = self._current_agent.run()

            if routine:
                self._discovered_routine = routine
                console.print()
                console.print("[bold green]✓ Discovery completed successfully![/bold green]")
                self._print_routine_summary(routine)

                # Auto-save routine
                self._output_dir.mkdir(parents=True, exist_ok=True)
                routine_path = self._output_dir / "routine.json"
                routine_path.write_text(json.dumps(routine.model_dump(), indent=2))
                console.print(f"[dim]Saved to: {routine_path}[/dim]")

                # Save test parameters from discovery state (provided by agent)
                test_params = self._current_agent._discovery_state.test_parameters
                test_params_path = self._output_dir / "test_parameters.json"
                test_params_path.write_text(json.dumps(test_params, indent=2))
                console.print(f"[dim]Test params: {test_params_path}[/dim]")
            else:
                console.print()
                console.print("[bold red]✗ Discovery failed - no routine produced[/bold red]")

        except KeyboardInterrupt:
            console.print()
            console.print("[yellow]Discovery interrupted by user[/yellow]")
        except Exception as e:
            console.print()
            console.print(f"[bold red]✗ Discovery error: {e}[/bold red]")
        finally:
            self._is_discovering = False

        console.print()

    def _print_routine_summary(self, routine: Routine) -> None:
        """Print a summary of the discovered routine."""
        table = Table(box=box.ROUNDED, show_header=False)
        table.add_column("Field", style="dim")
        table.add_column("Value", style="white")

        table.add_row("Name", routine.name)
        table.add_row("Description", routine.description[:100] + "..." if len(routine.description) > 100 else routine.description)
        table.add_row("Operations", str(len(routine.operations)))
        table.add_row("Parameters", str(len(routine.parameters)))

        if routine.parameters:
            param_names = ", ".join(p.name for p in routine.parameters)
            table.add_row("Param Names", param_names)

        console.print()
        console.print(Panel(table, title="[bold green]Discovered Routine[/bold green]", border_style="green", box=box.ROUNDED))

    def _handle_status_command(self) -> None:
        """Handle /status command to show current state."""
        table = Table(box=box.ROUNDED, show_header=False)
        table.add_column("Field", style="dim")
        table.add_column("Value", style="white")

        # Discovery status
        if self._is_discovering:
            table.add_row("Status", "[yellow]Discovering...[/yellow]")
        elif self._discovered_routine:
            table.add_row("Status", "[green]Routine discovered[/green]")
        else:
            table.add_row("Status", "[dim]Ready[/dim]")

        # Routine info
        if self._discovered_routine:
            table.add_row("Routine", self._discovered_routine.name)
            table.add_row("Operations", str(len(self._discovered_routine.operations)))
            table.add_row("Parameters", str(len(self._discovered_routine.parameters)))
        else:
            table.add_row("Routine", "[dim]None[/dim]")

        # Agent state
        if self._current_agent:
            phase = self._current_agent._discovery_state.phase.value
            table.add_row("Phase", phase)

        # Data sources
        data_info = []
        if self._network_data_loader:
            data_info.append(f"Network: {self._network_data_loader.stats.total_requests}")
        if self._storage_data_loader:
            data_info.append(f"Storage: {self._storage_data_loader.stats.total_events}")
        table.add_row("Data Sources", ", ".join(data_info) if data_info else "[dim]None[/dim]")

        # Browser
        if self._remote_debugging_address:
            table.add_row("Browser", f"[green]Connected[/green] ({self._remote_debugging_address})")
        else:
            table.add_row("Browser", "[yellow]Not connected[/yellow]")

        console.print()
        console.print(Panel(table, title="[bold cyan]Status[/bold cyan]", border_style="cyan", box=box.ROUNDED))
        console.print()

    def _handle_chats_command(self) -> None:
        """Handle /chats command to show message history."""
        if not self._message_history:
            console.print()
            console.print("[yellow]No messages in history yet.[/yellow]")
            console.print()
            return

        console.print()
        console.print(f"[bold cyan]Message History ({len(self._message_history)} messages)[/bold cyan]")
        console.print()

        for i, msg in enumerate(self._message_history[-20:], 1):  # Show last 20
            msg_type = msg.get("type", "Unknown")
            if msg_type == "ChatResponseEmittedMessage":
                content = msg.get("content", "")[:60]
                console.print(f"[dim]{i}.[/dim] [cyan]ASSISTANT[/cyan] {escape(content)}...")
            elif msg_type == "ErrorEmittedMessage":
                error = msg.get("error", "")[:60]
                console.print(f"[dim]{i}.[/dim] [red]ERROR[/red] {escape(error)}...")
            elif msg_type == "ToolInvocationResultEmittedMessage":
                tool_name = msg.get("tool_name", "?")
                status = msg.get("status", "?")
                console.print(f"[dim]{i}.[/dim] [yellow]TOOL[/yellow] {tool_name} ({status})")

        console.print()

    def _handle_routine_command(self) -> None:
        """Handle /routine command to show the discovered routine."""
        if not self._discovered_routine:
            console.print()
            console.print("[yellow]No routine discovered yet. Use /discover <task> to start.[/yellow]")
            console.print()
            return

        routine = self._discovered_routine

        # Build detailed view
        table = Table(box=box.ROUNDED, show_header=False, expand=True)
        table.add_column("Field", style="dim", width=15)
        table.add_column("Value", style="white")

        table.add_row("Name", routine.name)
        table.add_row("Description", routine.description)

        # Parameters
        if routine.parameters:
            param_lines = []
            for p in routine.parameters:
                obs = f" = {p.observed_value}" if p.observed_value else ""
                param_lines.append(f"  {p.name} ({p.type}){obs}")
            table.add_row("Parameters", "\n".join(param_lines))
        else:
            table.add_row("Parameters", "[dim]None[/dim]")

        # Operations summary
        if routine.operations:
            op_lines = []
            for i, op in enumerate(routine.operations, 1):
                op_type = op.type
                op_lines.append(f"  {i}. {op_type}")
            table.add_row("Operations", "\n".join(op_lines[:10]))
            if len(routine.operations) > 10:
                table.add_row("", f"[dim]... and {len(routine.operations) - 10} more[/dim]")

        console.print()
        console.print(Panel(table, title="[bold cyan]Discovered Routine[/bold cyan]", border_style="cyan", box=box.ROUNDED))
        console.print()

    def _handle_save_command(self, path: str) -> None:
        """Handle /save command to save routine to file."""
        if not self._discovered_routine:
            console.print()
            console.print("[yellow]No routine to save. Use /discover <task> first.[/yellow]")
            console.print()
            return

        if not path.strip():
            console.print()
            console.print("[yellow]Usage: /save <path.json>[/yellow]")
            console.print()
            return

        try:
            save_path = Path(path.strip())
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_path.write_text(json.dumps(self._discovered_routine.model_dump(), indent=2))
            console.print()
            console.print(f"[green]✓ Routine saved to: {save_path}[/green]")
            console.print()
        except Exception as e:
            console.print()
            console.print(f"[red]✗ Failed to save: {e}[/red]")
            console.print()

    def _handle_execute_command(self) -> None:
        """Handle /execute command to run the discovered routine with test parameters."""
        if not self._discovered_routine:
            console.print()
            console.print("[yellow]No routine discovered yet. Use /discover <task> first.[/yellow]")
            console.print()
            return

        # Get test parameters from the agent's discovery state (can be empty if routine has no params)
        test_params: dict[str, str] = {}
        if self._current_agent and self._current_agent._discovery_state.test_parameters is not None:
            test_params = self._current_agent._discovery_state.test_parameters

        # Warn if routine has parameters but test_params is empty
        if self._discovered_routine.parameters and not test_params:
            console.print()
            console.print("[yellow]Warning: Routine has parameters but no test values available.[/yellow]")
            console.print()

        if not self._remote_debugging_address:
            console.print()
            console.print("[yellow]No browser connected. Use --remote-debugging-address to connect.[/yellow]")
            console.print("[dim]Example: bluebox-super-discovery --cdp-captures-dir ./cdp_captures --remote-debugging-address http://127.0.0.1:9222[/dim]")
            console.print()
            return

        console.print()
        console.print(Panel(
            f"[bold]Routine:[/bold] {self._discovered_routine.name}\n"
            f"[bold]Parameters:[/bold] {json.dumps(test_params, indent=2)}",
            title="[bold cyan]Executing Routine[/bold cyan]",
            border_style="cyan",
            box=box.ROUNDED,
        ))
        console.print()

        try:
            with console.status("[bold blue]Executing routine...[/bold blue]"):
                result = execute_routine(
                    routine=self._discovered_routine.model_dump(),
                    parameters=test_params,
                    remote_debugging_address=self._remote_debugging_address,
                    timeout=120,
                    close_tab_when_done=True,
                )

            if result.get("success"):
                exec_result = result.get("result")
                if exec_result and exec_result.ok and exec_result.data is not None:
                    # Save result to file
                    result_path = self._output_dir / "execution_result.json"
                    result_path.write_text(json.dumps({
                        "success": True,
                        "data": exec_result.data,
                    }, indent=2, default=str))

                    console.print("[bold green]✓ Execution succeeded![/bold green]")
                    console.print()

                    # Show data preview
                    data_str = json.dumps(exec_result.data, indent=2, default=str)
                    if len(data_str) > 1000:
                        console.print(f"[dim]Data preview (truncated):[/dim]\n{data_str[:1000]}...")
                    else:
                        console.print(f"[dim]Data:[/dim]\n{data_str}")

                    console.print()
                    console.print(f"[dim]Full result saved to: {result_path}[/dim]")
                else:
                    console.print("[yellow]⚠ Execution completed but no data returned.[/yellow]")
                    if exec_result:
                        console.print(f"[dim]Result: {exec_result.model_dump()}[/dim]")
            else:
                error = result.get("error", "Unknown error")
                console.print(f"[bold red]✗ Execution failed: {error}[/bold red]")

        except Exception as e:
            console.print(f"[bold red]✗ Execution error: {e}[/bold red]")

        console.print()

    def _handle_reset_command(self) -> None:
        """Handle /reset command to clear state."""
        self._current_agent = None
        self._discovered_routine = None
        self._message_history = []
        self._is_discovering = False
        console.print()
        console.print("[yellow]↺ State reset[/yellow]")
        console.print()

    def get_data_summary(self) -> dict[str, Any]:
        """Get a summary of loaded data for the welcome message."""
        summary: dict[str, Any] = {}
        if self._network_data_loader:
            summary["network"] = self._network_data_loader.stats.total_requests
        if self._storage_data_loader:
            summary["storage"] = self._storage_data_loader.stats.total_events
        if self._window_property_data_loader:
            summary["window_props"] = self._window_property_data_loader.stats.total_events
        if self._js_data_loader:
            summary["js"] = self._js_data_loader.stats.total_files
        if self._interaction_data_loader:
            summary["interactions"] = self._interaction_data_loader.stats.total_events
        return summary

    def run(self, initial_task: str | None = None) -> None:
        """Run the interactive chat loop."""
        print_welcome(str(self._llm_model.value), self.get_data_summary())

        # Auto-run discovery if task provided
        if initial_task:
            self._handle_discover_command(initial_task)

        while True:
            try:
                user_input = pt_prompt(
                    self._get_prompt(),
                    completer=SlashCommandCompleter(SLASH_COMMANDS),
                    lexer=SlashCommandLexer(),
                    complete_while_typing=True,
                )

                # Skip empty input
                if not user_input.strip():
                    continue

                # Check for commands
                cmd = user_input.strip().lower()

                if cmd in ("/quit", "/exit", "/q"):
                    console.print()
                    console.print("[bold cyan]Goodbye![/bold cyan]")
                    console.print()
                    break

                if cmd in ("/help", "/h", "/?"):
                    console.print()
                    console.print(Panel(
                        """[bold]Commands:[/bold]
  [cyan]/discover <task>[/cyan]         Start routine discovery for the given task
  [cyan]/execute[/cyan]                 Execute routine with test parameters
  [cyan]/status[/cyan]                  Show current discovery state
  [cyan]/chats[/cyan]                   Show all messages in the thread
  [cyan]/routine[/cyan]                 Show the current/discovered routine
  [cyan]/save <path>[/cyan]             Save routine to file
  [cyan]/reset[/cyan]                   Start a new conversation
  [cyan]/help[/cyan]                    Show this help message
  [cyan]/quit[/cyan]                    Exit

[bold]Tips:[/bold]
  - Use /discover with a clear task description
  - The agent will analyze your CDP captures automatically
  - Use /routine to see the discovered routine details
  - Use /execute to run the routine (requires --remote-debugging-address)
  - Use /save to export the routine to a JSON file""",
                        title="[bold cyan]Help[/bold cyan]",
                        border_style="cyan",
                        box=box.ROUNDED,
                    ))
                    console.print()
                    continue

                if cmd == "/status":
                    self._handle_status_command()
                    continue

                if cmd == "/chats":
                    self._handle_chats_command()
                    continue

                if cmd == "/routine":
                    self._handle_routine_command()
                    continue

                if cmd == "/execute":
                    self._handle_execute_command()
                    continue

                if cmd == "/reset":
                    self._handle_reset_command()
                    continue

                if user_input.strip().startswith("/discover "):
                    task = user_input.strip()[10:].strip()
                    self._handle_discover_command(task)
                    continue

                if user_input.strip().startswith("/save "):
                    path = user_input.strip()[6:].strip()
                    self._handle_save_command(path)
                    continue

                # Handle /discover without argument
                if cmd == "/discover":
                    console.print()
                    console.print("[yellow]Usage: /discover <task description>[/yellow]")
                    console.print("[dim]Example: /discover Search for trains from NYC to Boston[/dim]")
                    console.print()
                    continue

                # Non-command input - chat with the agent
                self._handle_chat_message(user_input)

            except KeyboardInterrupt:
                console.print()
                console.print("[cyan]Interrupted. Goodbye![/cyan]")
                console.print()
                break

            except EOFError:
                console.print()
                console.print("[cyan]Goodbye![/cyan]")
                console.print()
                break


def parse_model(model_str: str) -> OpenAIModel:
    """Parse a model string into an OpenAIModel enum value."""
    for model in OpenAIModel:
        if model.value == model_str or model.name == model_str:
            return model
    raise ValueError(f"Unknown model: {model_str}")


def main() -> None:
    """Entry point for the super discovery agent terminal."""
    parser = argparse.ArgumentParser(description="Interactive Super Discovery Agent terminal")

    # CDP captures directory (convenience option - auto-discovers JSONL files)
    parser.add_argument(
        "--cdp-captures-dir",
        type=str,
        default=None,
        help="Directory with CDP captures. Auto-discovers JSONL files within.",
    )

    # Individual JSONL file paths (explicit option)
    parser.add_argument("--network-jsonl", type=str, default=None, help="Path to network events JSONL file.")
    parser.add_argument("--storage-jsonl", type=str, default=None, help="Path to storage events JSONL file.")
    parser.add_argument("--window-props-jsonl", type=str, default=None, help="Path to window properties JSONL file.")
    parser.add_argument("--js-jsonl", type=str, default=None, help="Path to JavaScript events JSONL file.")
    parser.add_argument("--interaction-jsonl", type=str, default=None, help="Path to interaction events JSONL file.")

    # Task (optional - can run discovery immediately)
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Task description for immediate discovery (optional).",
    )

    # Output and model options
    parser.add_argument("--output-dir", type=str, default="./routine_discovery_output", help="Output directory.")
    parser.add_argument(
        "--llm-model",
        type=str,
        default=OpenAIModel.GPT_5_1.value,
        help=f"LLM model for orchestrator (default: {OpenAIModel.GPT_5_1.value}).",
    )
    parser.add_argument(
        "--subagent-llm-model",
        type=str,
        default=None,
        help="LLM model for subagents (defaults to --llm-model).",
    )
    parser.add_argument(
        "--remote-debugging-address",
        type=str,
        default=None,
        help="Chrome remote debugging address (e.g., http://127.0.0.1:9222).",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=50,
        help="Max iterations for discovery loop.",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress all log output.",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Write logs to file instead of console.",
    )

    args = parser.parse_args()

    # Configure logging
    configure_logging(quiet=args.quiet, log_file=args.log_file)

    # Validate API key
    if Config.OPENAI_API_KEY is None:
        console.print("[bold red]Error: OPENAI_API_KEY environment variable is not set[/bold red]")
        sys.exit(1)

    # Resolve JSONL paths - explicit paths take precedence over cdp-captures-dir
    network_jsonl = args.network_jsonl
    storage_jsonl = args.storage_jsonl
    window_props_jsonl = args.window_props_jsonl
    js_jsonl = args.js_jsonl
    interaction_jsonl = args.interaction_jsonl

    if args.cdp_captures_dir:
        cdp_dir = Path(args.cdp_captures_dir)
        if not network_jsonl:
            candidate = cdp_dir / "network" / "events.jsonl"
            if candidate.exists():
                network_jsonl = str(candidate)
        if not storage_jsonl:
            candidate = cdp_dir / "storage" / "events.jsonl"
            if candidate.exists():
                storage_jsonl = str(candidate)
        if not window_props_jsonl:
            candidate = cdp_dir / "window_properties" / "events.jsonl"
            if candidate.exists():
                window_props_jsonl = str(candidate)
        if not js_jsonl:
            candidate = cdp_dir / "network" / "javascript_events.jsonl"
            if candidate.exists():
                js_jsonl = str(candidate)
        if not interaction_jsonl:
            candidate = cdp_dir / "interaction" / "events.jsonl"
            if candidate.exists():
                interaction_jsonl = str(candidate)

    # Validate that we have at least network data
    if not network_jsonl:
        console.print("[bold red]Error: No network data source provided. Use --network-jsonl or --cdp-captures-dir[/bold red]")
        sys.exit(1)

    try:
        llm_model = parse_model(args.llm_model)
        subagent_model = parse_model(args.subagent_llm_model) if args.subagent_llm_model else None

        # Load data loaders with status
        with console.status("[bold blue]Loading data...[/bold blue]") as status:
            status.update("[bold blue]Loading network data...[/bold blue]")
            network_data_loader = NetworkDataLoader(network_jsonl)
            logger.info("Network data loaded: %d transactions", network_data_loader.stats.total_requests)

            storage_data_loader: StorageDataLoader | None = None
            if storage_jsonl and Path(storage_jsonl).exists():
                status.update("[bold blue]Loading storage data...[/bold blue]")
                storage_data_loader = StorageDataLoader(storage_jsonl)
                logger.info("Storage data loaded: %d events", storage_data_loader.stats.total_events)

            window_property_data_loader: WindowPropertyDataLoader | None = None
            if window_props_jsonl and Path(window_props_jsonl).exists():
                status.update("[bold blue]Loading window property data...[/bold blue]")
                window_property_data_loader = WindowPropertyDataLoader(window_props_jsonl)
                logger.info("Window property data loaded: %d events", window_property_data_loader.stats.total_events)

            js_data_loader: JSDataLoader | None = None
            if js_jsonl and Path(js_jsonl).exists():
                status.update("[bold blue]Loading JS data...[/bold blue]")
                js_data_loader = JSDataLoader(js_jsonl)
                logger.info("JS data loaded: %d files", js_data_loader.stats.total_files)

            interaction_data_loader: InteractionsDataLoader | None = None
            if interaction_jsonl and Path(interaction_jsonl).exists():
                status.update("[bold blue]Loading interaction data...[/bold blue]")
                interaction_data_loader = InteractionsDataLoader.from_jsonl(interaction_jsonl)
                logger.info("Interaction data loaded: %d events", interaction_data_loader.stats.total_events)

            # Initialize documentation data loader
            status.update("[bold blue]Loading documentation...[/bold blue]")
            DEFAULT_DOCS_DIR = str(BLUEBOX_PACKAGE_ROOT / "agent_docs")
            DEFAULT_CODE_PATHS = [
                str(BLUEBOX_PACKAGE_ROOT / "data_models" / "routine"),
                str(BLUEBOX_PACKAGE_ROOT / "data_models" / "ui_elements.py"),
                str(BLUEBOX_PACKAGE_ROOT / "agents" / "routine_discovery_agent.py"),
                str(BLUEBOX_PACKAGE_ROOT / "llms" / "infra" / "data_store.py"),
                str(BLUEBOX_PACKAGE_ROOT / "utils" / "js_utils.py"),
                str(BLUEBOX_PACKAGE_ROOT / "utils" / "data_utils.py"),
                "!" + str(BLUEBOX_PACKAGE_ROOT / "**" / "__init__.py"),
            ]

            documentation_data_loader = DocumentationDataLoader(
                documentation_paths=[DEFAULT_DOCS_DIR],
                code_paths=DEFAULT_CODE_PATHS,
            )
            logger.info("Documentation data loaded: %d docs, %d code files",
                        documentation_data_loader.stats.total_docs,
                        documentation_data_loader.stats.total_code)

        console.print("[green]✓ Data loaded![/green]")
        console.print()

        # Create and run terminal
        chat = TerminalSuperDiscoveryChat(
            network_data_loader=network_data_loader,
            storage_data_loader=storage_data_loader,
            window_property_data_loader=window_property_data_loader,
            js_data_loader=js_data_loader,
            interaction_data_loader=interaction_data_loader,
            documentation_data_loader=documentation_data_loader,
            llm_model=llm_model,
            subagent_llm_model=subagent_model,
            max_iterations=args.max_iterations,
            remote_debugging_address=args.remote_debugging_address,
            output_dir=Path(args.output_dir),
        )
        chat.run(initial_task=args.task)

    except ValueError as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[bold red]Fatal error: {e}[/bold red]")
        raise


if __name__ == "__main__":
    main()
