"""
bluebox/scripts/run_routine_discovery_agent_beta.py

Multi-pane terminal UI for the RoutineDiscoveryAgentBeta using Textual.

Layout:
  +-----------------------------+----------------------+
  |                             |  Tool Calls History   |
  |       Chat (scrolling)      |                       |
  |                             +-----------------------+
  |  +------------------------+ |  Saved Files          |
  |  | Input                  | |                       |
  |  +------------------------+ |                       |
  +-----------------------------+-----------------------+

Usage:
    bluebox-routine-discovery-agent-beta --cdp-captures-dir ./cdp_captures
    bluebox-routine-discovery-agent-beta --cdp-captures-dir ./cdp_captures --task "Search for trains from NYC to Boston"
    bluebox-routine-discovery-agent-beta --cdp-captures-dir ./cdp_captures --model gpt-5.1
    bluebox-routine-discovery-agent-beta --cdp-captures-dir ./cdp_captures --remote-debugging-address http://127.0.0.1:9222
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rich.console import Console
from rich.markup import escape
from rich.text import Text
from textual import work
from textual.widgets import RichLog

from bluebox.agents.routine_discovery_agent_beta import RoutineDiscoveryAgentBeta
from bluebox.config import Config
from bluebox.data_models.llms.interaction import BaseEmittedMessage
from bluebox.data_models.llms.vendors import LLMModel
from bluebox.data_models.routine.routine import Routine
from bluebox.llms.data_loaders.documentation_data_loader import DocumentationDataLoader
from bluebox.llms.data_loaders.interactions_data_loader import InteractionsDataLoader
from bluebox.llms.data_loaders.js_data_loader import JSDataLoader
from bluebox.llms.data_loaders.network_data_loader import NetworkDataLoader
from bluebox.llms.data_loaders.storage_data_loader import StorageDataLoader
from bluebox.llms.data_loaders.window_property_data_loader import WindowPropertyDataLoader
from bluebox.llms.tools.execute_routine_tool import execute_routine
from bluebox.utils.cli_utils import add_model_argument, resolve_model
from bluebox.utils.logger import enable_tui_logging, get_logger
from bluebox.utils.tui_base import AbstractAgentTUI, BASE_SLASH_COMMANDS

if TYPE_CHECKING:
    from bluebox.agents.abstract_agent import AbstractAgent


# Package root for documentation code_paths
BLUEBOX_PACKAGE_ROOT = Path(__file__).resolve().parent.parent

logger = get_logger(__name__)


# ─── Slash commands ──────────────────────────────────────────────────────────

SLASH_COMMANDS: dict[str, str] = {
    "/discover": "Start routine discovery — /discover <task>",
    "/execute": "Execute the discovered routine",
    "/routine": "Show the discovered routine",
    "/save": "Save routine — /save <path.json>",
    **BASE_SLASH_COMMANDS,
}

HELP_TEXT = """\
[bold]Commands:[/bold]
  [cyan]/discover <task>[/cyan]  Start routine discovery for the given task
  [cyan]/execute[/cyan]          Execute the discovered routine
  [cyan]/routine[/cyan]          Show the discovered routine
  [cyan]/save <path>[/cyan]      Save routine to file
  [cyan]/status[/cyan]           Show current state
  [cyan]/chats[/cyan]            Show message history
  [cyan]/clear[/cyan]            Clear the chat display
  [cyan]/reset[/cyan]            Start new conversation
  [cyan]/help[/cyan]             Show this help
  [cyan]/quit[/cyan]             Exit
"""


# ─── Textual App ─────────────────────────────────────────────────────────────

class RotutineDiscoveryBetaTUI(AbstractAgentTUI):
    """Multi-pane TUI for the Routine Discovery Beta Agent."""

    TITLE = "Routine Discovery Beta"
    SLASH_COMMANDS = SLASH_COMMANDS
    HELP_TEXT = HELP_TEXT
    SHOW_SAVED_FILES_PANE = True

    def __init__(
        self,
        llm_model: LLMModel,
        network_data_loader: NetworkDataLoader,
        storage_data_loader: StorageDataLoader | None = None,
        window_property_data_loader: WindowPropertyDataLoader | None = None,
        js_data_loader: JSDataLoader | None = None,
        interaction_data_loader: InteractionsDataLoader | None = None,
        documentation_data_loader: DocumentationDataLoader | None = None,
        subagent_llm_model: LLMModel | None = None,
        max_iterations: int = 50,
        remote_debugging_address: str | None = None,
        output_dir: Path | None = None,
        initial_task: str | None = None,
    ) -> None:
        output = output_dir or Path("./routine_discovery_output")
        super().__init__(llm_model, working_dir=str(output))
        self._network_data_loader = network_data_loader
        self._storage_data_loader = storage_data_loader
        self._window_property_data_loader = window_property_data_loader
        self._js_data_loader = js_data_loader
        self._interaction_data_loader = interaction_data_loader
        self._documentation_data_loader = documentation_data_loader
        self._subagent_llm_model = subagent_llm_model
        self._max_iterations = max_iterations
        self._remote_debugging_address = remote_debugging_address
        self._output_dir = output
        self._initial_task = initial_task

        # Discovery state
        self._discovery_agent: RoutineDiscoveryAgentBeta | None = None
        self._discovered_routine: Routine | None = None
        self._is_discovering: bool = False
        self._last_state_hash: str | None = None

    # ── Abstract implementations ─────────────────────────────────────────

    def _create_agent(self) -> AbstractAgent:
        """Create the chat-mode agent (placeholder task for conversational use)."""
        return RoutineDiscoveryAgentBeta(
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

    def _print_welcome(self) -> None:
        chat = self.query_one("#chat-log", RichLog)
        chat.write(Text.from_markup(
            "[bold cyan]Routine Discovery Beta Agent[/bold cyan]  "
            "[dim]powered by Vectorly[/dim]"
        ))
        chat.write("")

        # Data summary
        lines: list[str] = []
        stats = self._network_data_loader.stats
        lines.append(f"[dim]Network:[/dim]     {stats.total_requests} transactions, {stats.unique_hosts} hosts")
        if self._storage_data_loader:
            lines.append(f"[dim]Storage:[/dim]     {self._storage_data_loader.stats.total_events} events")
        if self._window_property_data_loader:
            lines.append(f"[dim]Window:[/dim]      {self._window_property_data_loader.stats.total_events} events")
        if self._js_data_loader:
            lines.append(f"[dim]JS Files:[/dim]    {self._js_data_loader.stats.total_files} files")
        if self._interaction_data_loader:
            lines.append(f"[dim]Interactions:[/dim] {self._interaction_data_loader.stats.total_events} events")
        if self._remote_debugging_address:
            lines.append(f"[dim]Browser:[/dim]     [green]{self._remote_debugging_address}[/green]")

        chat.write(Text.from_markup("\n".join(lines)))
        chat.write("")
        chat.write(Text.from_markup(
            "Type [cyan]/discover <task>[/cyan] to start discovery, "
            "or ask questions about the captured traffic."
        ))
        chat.write("")

    def _build_status_text(self) -> str:
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        msg_count = len(self._agent.get_chats()) if self._agent else 0
        tokens_used, ctx_pct = self._estimate_context_usage()
        ctx_bar = self._context_bar(ctx_pct)
        stats = self._network_data_loader.stats

        phase = "N/A"
        if self._discovery_agent:
            phase = self._discovery_agent._discovery_state.phase.value

        if self._is_discovering:
            discovery_status = "[yellow]Discovering...[/yellow]"
        elif self._discovered_routine:
            name = self._discovered_routine.name[:25]
            discovery_status = f"[green]{name}[/green]"
        else:
            discovery_status = "[dim]Ready[/dim]"

        return (
            f"[bold cyan]Routine Discovery Beta[/bold cyan]\n"
            f"[dim]\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500[/dim]\n"
            f"[dim]Model:[/dim]     {self._llm_model.value}\n"
            f"[dim]Messages:[/dim]  {msg_count}\n"
            f"[dim]Tools:[/dim]     {self._tool_call_count}\n"
            f"[dim]Context:[/dim]   {ctx_bar}\n"
            f"[dim](est.)      ~{tokens_used:,} / {self._context_window_size:,}[/dim]\n"
            f"[dim]\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500[/dim]\n"
            f"[dim]Status:[/dim]    {discovery_status}\n"
            f"[dim]Phase:[/dim]     {phase}\n"
            f"[dim]Requests:[/dim]  {stats.total_requests}\n"
            f"[dim]Hosts:[/dim]     {stats.unique_hosts}\n"
            f"[dim]Browser:[/dim]   {self._remote_debugging_address or '[yellow]Not connected[/yellow]'}\n"
            f"[dim]Time:[/dim]      {now}\n"
        )

    # ── Lifecycle override ────────────────────────────────────────────────

    def on_mount(self) -> None:
        super().on_mount()
        if self._initial_task:
            self._run_discovery(self._initial_task)

    # ── Message handler override (adds state dumping) ─────────────────────

    def _handle_message(self, message: BaseEmittedMessage) -> None:
        super()._handle_message(message)
        if self._discovery_agent:
            self._dump_chat_thread()
            self._dump_state_if_changed()

    # ── Custom commands ──────────────────────────────────────────────────

    def _handle_custom_command(self, cmd: str, raw_input: str) -> bool:
        if raw_input.lower().startswith("/discover"):
            task = raw_input[9:].strip()
            chat = self.query_one("#chat-log", RichLog)
            if not task:
                chat.write(Text.from_markup(
                    "[yellow]Usage: /discover <task>[/yellow]\n"
                    "[dim]Example: /discover Search for trains from NYC to Boston[/dim]"
                ))
            else:
                self._run_discovery(task)
            return True

        if cmd == "/execute":
            self._handle_execute_command()
            return True

        if cmd == "/routine":
            self._handle_routine_command()
            return True

        if raw_input.lower().startswith("/save"):
            path = raw_input[5:].strip()
            self._handle_save_command(path)
            return True

        return False

    # ── Discovery ────────────────────────────────────────────────────────

    @work(thread=True)
    def _run_discovery(self, task: str) -> None:
        """Run autonomous routine discovery in a background thread."""
        chat = self.query_one("#chat-log", RichLog)

        # Clear existing output directory
        if self._output_dir.exists():
            shutil.rmtree(self._output_dir)
        self._last_state_hash = None

        self.call_from_thread(
            lambda: chat.write(Text.from_markup(
                f"\n[bold magenta]Starting Discovery[/bold magenta]\n"
                f"[dim]Task:[/dim] {escape(task)}"
            ))
        )

        self._is_discovering = True
        self._processing = True

        try:
            agent = RoutineDiscoveryAgentBeta(
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
            self._discovery_agent = agent

            routine = agent.run()

            def _show_result() -> None:
                chat.write("")
                if routine:
                    self._discovered_routine = routine
                    chat.write(Text.from_markup(
                        "[bold green]\u2713 Discovery completed successfully![/bold green]"
                    ))

                    lines = [
                        f"[dim]Name:[/dim]       {routine.name}",
                        f"[dim]Operations:[/dim] {len(routine.operations)}",
                        f"[dim]Parameters:[/dim] {len(routine.parameters)}",
                    ]
                    if routine.parameters:
                        param_names = ", ".join(p.name for p in routine.parameters)
                        lines.append(f"[dim]Params:[/dim]     {param_names}")
                    chat.write(Text.from_markup("\n".join(lines)))

                    # Auto-save routine and test params
                    self._output_dir.mkdir(parents=True, exist_ok=True)
                    routine_path = self._output_dir / "routine.json"
                    routine_path.write_text(json.dumps(routine.model_dump(), indent=2))
                    self._add_saved_file(str(routine_path))

                    test_params = agent._discovery_state.test_parameters
                    if test_params:
                        params_path = self._output_dir / "test_parameters.json"
                        params_path.write_text(json.dumps(test_params, indent=2))
                        self._add_saved_file(str(params_path))

                    chat.write(Text.from_markup(
                        f"[dim]Saved to: {self._output_dir}[/dim]"
                    ))
                else:
                    chat.write(Text.from_markup(
                        "[bold red]\u2717 Discovery failed - no routine produced[/bold red]"
                    ))
                chat.write("")
                self._update_status()

            self.call_from_thread(_show_result)

        except Exception as e:
            self.call_from_thread(
                lambda: chat.write(Text.from_markup(
                    f"[bold red]\u2717 Discovery error: {escape(str(e))}[/bold red]"
                ))
            )
        finally:
            self._is_discovering = False
            self.call_from_thread(self._finish_processing)

    # ── /execute ─────────────────────────────────────────────────────────

    def _handle_execute_command(self) -> None:
        """Validate preconditions then run execution in background."""
        chat = self.query_one("#chat-log", RichLog)

        if not self._discovered_routine:
            chat.write(Text.from_markup(
                "[yellow]No routine discovered yet. Use /discover <task> first.[/yellow]"
            ))
            return

        if not self._remote_debugging_address:
            chat.write(Text.from_markup(
                "[yellow]No browser connected. Use --remote-debugging-address to connect.[/yellow]"
            ))
            return

        self._run_execute()

    @work(thread=True)
    def _run_execute(self) -> None:
        """Execute the discovered routine in a background thread."""
        chat = self.query_one("#chat-log", RichLog)
        self._processing = True

        test_params: dict[str, str] = {}
        if self._discovery_agent and self._discovery_agent._discovery_state.test_parameters:
            test_params = self._discovery_agent._discovery_state.test_parameters

        self.call_from_thread(
            lambda: chat.write(Text.from_markup(
                f"\n[bold magenta]Executing Routine[/bold magenta]\n"
                f"[dim]Routine:[/dim] {self._discovered_routine.name}\n"
                f"[dim]Params:[/dim]  {json.dumps(test_params)}"
            ))
        )

        try:
            result = execute_routine(
                routine=self._discovered_routine.model_dump(),
                parameters=test_params,
                remote_debugging_address=self._remote_debugging_address,
                timeout=120,
                close_tab_when_done=True,
            )

            def _show_result() -> None:
                if result.get("success"):
                    exec_result = result.get("result")
                    if exec_result and exec_result.ok and exec_result.data is not None:
                        result_path = self._output_dir / "execution_result.json"
                        result_path.write_text(json.dumps({
                            "success": True,
                            "data": exec_result.data,
                        }, indent=2, default=str))

                        chat.write(Text.from_markup(
                            "[bold green]\u2713 Execution succeeded![/bold green]"
                        ))
                        data_str = json.dumps(exec_result.data, indent=2, default=str)
                        preview = data_str[:500] + "..." if len(data_str) > 500 else data_str
                        chat.write(preview)
                        self._add_saved_file(str(result_path))
                    else:
                        chat.write(Text.from_markup(
                            "[yellow]Execution completed but no data returned.[/yellow]"
                        ))
                else:
                    error = result.get("error", "Unknown error")
                    chat.write(Text.from_markup(
                        f"[bold red]\u2717 Execution failed: {escape(str(error))}[/bold red]"
                    ))
                chat.write("")
                self._update_status()

            self.call_from_thread(_show_result)

        except Exception as e:
            self.call_from_thread(
                lambda: chat.write(Text.from_markup(
                    f"[bold red]\u2717 Execution error: {escape(str(e))}[/bold red]"
                ))
            )
        finally:
            self.call_from_thread(self._finish_processing)

    # ── /routine ─────────────────────────────────────────────────────────

    def _handle_routine_command(self) -> None:
        """Show the discovered routine details in chat."""
        chat = self.query_one("#chat-log", RichLog)

        if not self._discovered_routine:
            chat.write(Text.from_markup(
                "[yellow]No routine discovered yet. Use /discover <task> first.[/yellow]"
            ))
            return

        routine = self._discovered_routine
        desc = routine.description[:100] + "..." if len(routine.description) > 100 else routine.description
        lines = [
            f"\n[bold cyan]Discovered Routine[/bold cyan]",
            f"[dim]Name:[/dim]        {routine.name}",
            f"[dim]Description:[/dim] {desc}",
        ]

        if routine.parameters:
            lines.append("[dim]Parameters:[/dim]")
            for p in routine.parameters:
                obs = f" = {p.observed_value}" if p.observed_value else ""
                lines.append(f"  {p.name} ({p.type}){obs}")
        else:
            lines.append("[dim]Parameters:[/dim]  None")

        if routine.operations:
            lines.append(f"[dim]Operations ({len(routine.operations)}):[/dim]")
            for i, op in enumerate(routine.operations[:10], 1):
                lines.append(f"  {i}. {op.type}")
            if len(routine.operations) > 10:
                lines.append(f"  [dim]... and {len(routine.operations) - 10} more[/dim]")

        chat.write(Text.from_markup("\n".join(lines)))
        chat.write("")

    # ── /save ────────────────────────────────────────────────────────────

    def _handle_save_command(self, path: str) -> None:
        """Save the discovered routine to a JSON file."""
        chat = self.query_one("#chat-log", RichLog)

        if not self._discovered_routine:
            chat.write(Text.from_markup(
                "[yellow]No routine to save. Use /discover <task> first.[/yellow]"
            ))
            return

        if not path.strip():
            chat.write(Text.from_markup("[yellow]Usage: /save <path.json>[/yellow]"))
            return

        try:
            save_path = Path(path.strip())
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_path.write_text(json.dumps(self._discovered_routine.model_dump(), indent=2))
            chat.write(Text.from_markup(f"[green]\u2713 Routine saved to: {save_path}[/green]"))
            self._add_saved_file(str(save_path))
        except Exception as e:
            chat.write(Text.from_markup(
                f"[red]\u2717 Failed to save: {escape(str(e))}[/red]"
            ))

    # ── Reset override ───────────────────────────────────────────────────

    def _on_reset(self) -> None:
        self._discovery_agent = None
        self._discovered_routine = None
        self._is_discovering = False
        self._last_state_hash = None

    # ── Status in chat ───────────────────────────────────────────────────

    def _show_status_in_chat(self) -> None:
        """Show a compact status summary in the chat pane."""
        chat = self.query_one("#chat-log", RichLog)
        msg_count = len(self._agent.get_chats()) if self._agent else 0
        tokens_used, ctx_pct = self._estimate_context_usage()
        stats = self._network_data_loader.stats

        phase = "N/A"
        if self._discovery_agent:
            phase = self._discovery_agent._discovery_state.phase.value

        if self._is_discovering:
            discovery_status = "Discovering..."
        elif self._discovered_routine:
            discovery_status = self._discovered_routine.name
        else:
            discovery_status = "Ready"

        chat.write(Text.from_markup(
            f"[bold cyan]Status[/bold cyan]\n"
            f"  Model: {self._llm_model.value}\n"
            f"  Messages: {msg_count}\n"
            f"  Context: ~{tokens_used:,}t ({ctx_pct:.0f}%)\n"
            f"  Discovery: {discovery_status}\n"
            f"  Phase: {phase}\n"
            f"  Requests: {stats.total_requests}\n"
            f"  Hosts: {stats.unique_hosts}\n"
            f"  Browser: {self._remote_debugging_address or 'Not connected'}"
        ))

    # ── State dumping ────────────────────────────────────────────────────

    def _dump_chat_thread(self) -> None:
        """Dump chat threads for the discovery agent and subagents."""
        if not self._discovery_agent:
            return

        chat_threads_dir = self._output_dir / "chat_threads"
        chat_threads_dir.mkdir(parents=True, exist_ok=True)

        self._dump_agent_thread(
            self._discovery_agent._thread,
            self._discovery_agent._chats,
            chat_threads_dir / "orchestration_agent.json",
            agent_type="orchestration_agent",
        )

        for agent_id, agent_instance in self._discovery_agent._agent_instances.items():
            subagent_info = self._discovery_agent._orchestration_state.subagents.get(agent_id)
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
        messages = []
        for chat_id in thread.chat_ids:
            chat_msg = chats.get(chat_id)
            if chat_msg:
                messages.append({
                    "id": chat_msg.id,
                    "role": chat_msg.role.value,
                    "content": chat_msg.content,
                    "tool_calls": [tc.model_dump() for tc in chat_msg.tool_calls] if chat_msg.tool_calls else None,
                    "tool_call_id": chat_msg.tool_call_id,
                })

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
        """Dump discovery and orchestration state if changed."""
        if not self._discovery_agent:
            return

        discovery_state = self._discovery_agent._discovery_state
        orchestration_state = self._discovery_agent._orchestration_state

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
                    sa_id: {
                        "id": subagent.id,
                        "type": subagent.type.value,
                        "task_ids": subagent.task_ids,
                    }
                    for sa_id, subagent in orchestration_state.subagents.items()
                },
            },
        }

        state_json = json.dumps(state_snapshot, sort_keys=True, default=str)
        state_hash = hashlib.sha256(state_json.encode()).hexdigest()[:16]

        if state_hash == self._last_state_hash:
            return

        self._last_state_hash = state_hash

        state_dir = self._output_dir / "state"
        state_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        state_path = state_dir / f"{timestamp}.json"

        state_snapshot["_meta"] = {
            "timestamp": timestamp,
            "hash": state_hash,
        }

        state_path.write_text(json.dumps(state_snapshot, indent=2, default=str))


# ─── Entry point ─────────────────────────────────────────────────────────────

def main() -> None:
    """Entry point for the Routine Discovery Beta agent TUI."""
    parser = argparse.ArgumentParser(description="Routine Discovery Beta Agent \u2014 Multi-pane TUI")

    # CDP captures directory
    parser.add_argument(
        "--cdp-captures-dir",
        type=str,
        default=None,
        help="Directory with CDP captures. Auto-discovers JSONL files within.",
    )

    # Individual JSONL file paths
    parser.add_argument("--network-jsonl", type=str, default=None, help="Path to network events JSONL file.")
    parser.add_argument("--storage-jsonl", type=str, default=None, help="Path to storage events JSONL file.")
    parser.add_argument("--window-props-jsonl", type=str, default=None, help="Path to window properties JSONL file.")
    parser.add_argument("--js-jsonl", type=str, default=None, help="Path to JavaScript events JSONL file.")
    parser.add_argument("--interaction-jsonl", type=str, default=None, help="Path to interaction events JSONL file.")

    # Task
    parser.add_argument("--task", type=str, default=None, help="Task description for immediate discovery (optional).")

    # Output and model options
    parser.add_argument("--output-dir", type=str, default="./routine_discovery_output", help="Output directory.")
    add_model_argument(parser)
    parser.add_argument(
        "--subagent-model",
        type=str,
        default=None,
        help="LLM model for subagents (defaults to --model).",
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
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress logs")
    parser.add_argument("--log-file", type=str, default=None, help="Log to file")

    args = parser.parse_args()

    console = Console()

    # Validate API key
    if Config.OPENAI_API_KEY is None:
        console.print("[bold red]Error: OPENAI_API_KEY environment variable is not set[/bold red]")
        sys.exit(1)

    # Resolve JSONL paths — explicit paths take precedence over cdp-captures-dir
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

    if not network_jsonl:
        console.print("[bold red]Error: No network data source provided. Use --network-jsonl or --cdp-captures-dir[/bold red]")
        sys.exit(1)

    try:
        llm_model = resolve_model(args.model, console)
        subagent_model = resolve_model(args.subagent_model, console) if args.subagent_model else None

        # Load data
        with console.status("[bold blue]Loading data...[/bold blue]") as status:
            status.update("[bold blue]Loading network data...[/bold blue]")
            network_data_loader = NetworkDataLoader(network_jsonl)

            storage_data_loader: StorageDataLoader | None = None
            if storage_jsonl and Path(storage_jsonl).exists():
                status.update("[bold blue]Loading storage data...[/bold blue]")
                storage_data_loader = StorageDataLoader(storage_jsonl)

            window_property_data_loader: WindowPropertyDataLoader | None = None
            if window_props_jsonl and Path(window_props_jsonl).exists():
                status.update("[bold blue]Loading window property data...[/bold blue]")
                window_property_data_loader = WindowPropertyDataLoader(window_props_jsonl)

            js_data_loader: JSDataLoader | None = None
            if js_jsonl and Path(js_jsonl).exists():
                status.update("[bold blue]Loading JS data...[/bold blue]")
                js_data_loader = JSDataLoader(js_jsonl)

            interaction_data_loader: InteractionsDataLoader | None = None
            if interaction_jsonl and Path(interaction_jsonl).exists():
                status.update("[bold blue]Loading interaction data...[/bold blue]")
                interaction_data_loader = InteractionsDataLoader.from_jsonl(interaction_jsonl)

            status.update("[bold blue]Loading documentation...[/bold blue]")
            docs_dir = str(BLUEBOX_PACKAGE_ROOT / "agent_docs")
            code_paths = [
                str(BLUEBOX_PACKAGE_ROOT / "data_models" / "routine"),
                str(BLUEBOX_PACKAGE_ROOT / "data_models" / "ui_elements.py"),
                str(BLUEBOX_PACKAGE_ROOT / "agents" / "routine_discovery_agent.py"),
                str(BLUEBOX_PACKAGE_ROOT / "llms" / "infra" / "data_store.py"),
                str(BLUEBOX_PACKAGE_ROOT / "utils" / "js_utils.py"),
                str(BLUEBOX_PACKAGE_ROOT / "utils" / "data_utils.py"),
                "!" + str(BLUEBOX_PACKAGE_ROOT / "**" / "__init__.py"),
            ]
            documentation_data_loader = DocumentationDataLoader(
                documentation_paths=[docs_dir],
                code_paths=code_paths,
            )

        console.print(f"[green]\u2713 Loaded {network_data_loader.stats.total_requests} network transactions[/green]")
        console.print()

        # Redirect logging before TUI takes over
        enable_tui_logging(log_file=args.log_file or ".bluebox_routine_discovery_agent_beta_tui.log", quiet=args.quiet)

        app = RotutineDiscoveryBetaTUI(
            llm_model=llm_model,
            network_data_loader=network_data_loader,
            storage_data_loader=storage_data_loader,
            window_property_data_loader=window_property_data_loader,
            js_data_loader=js_data_loader,
            interaction_data_loader=interaction_data_loader,
            documentation_data_loader=documentation_data_loader,
            subagent_llm_model=subagent_model,
            max_iterations=args.max_iterations,
            remote_debugging_address=args.remote_debugging_address,
            output_dir=Path(args.output_dir),
            initial_task=args.task,
        )
        app.run()

    except ValueError as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[bold red]Fatal error: {e}[/bold red]")
        raise


if __name__ == "__main__":
    main()
