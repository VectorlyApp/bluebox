"""
bluebox/scripts/specialists/run_experiment_worker_agent.py

Interactive TUI for ExperimentWorker.

Features:
- Chat directly with the worker agent.
- Run `/experiement <text prompt>` to execute an autonomous worker run
  against the mounted capture data and live browser tools.

Autonomous mode example:
    python -m bluebox.scripts.specialists.run_experiment_worker_agent \
      --cdp-captures-dir ./cdp_captures \
      --workspace-dir ./agent_workspaces/experiment_worker_agent \
      --remote-debugging-address http://127.0.0.1:9222 \
      --model gpt-5.2

Inside TUI:
    /experiement replay the exact captured request and return a minimal reproducible recipe
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from rich.console import Console
from rich.markup import escape
from rich.text import Text
from textual import work
from textual.widgets import RichLog

from bluebox.agents.abstract_agent import AutonomousRunConfig
from bluebox.workspace import LocalAgentWorkspace
from bluebox.agents.workers.experiment_worker import ExperimentWorker
from bluebox.data_models.llms.vendors import LLMModel
from bluebox.data_models.orchestration.result import SpecialistResultWrapper
from bluebox.llms.data_loaders.documentation_data_loader import DocumentationDataLoader
from bluebox.llms.data_loaders.dom_data_loader import DOMDataLoader
from bluebox.llms.data_loaders.network_data_loader import NetworkDataLoader
from bluebox.llms.data_loaders.storage_data_loader import StorageDataLoader
from bluebox.llms.data_loaders.window_property_data_loader import WindowPropertyDataLoader
from bluebox.utils.cli_utils import add_model_argument, resolve_model
from bluebox.utils.logger import enable_tui_logging, get_logger
from bluebox.utils.tui_base import AbstractAgentTUI, BASE_HELP_TEXT, BASE_SLASH_COMMANDS

logger = get_logger(name=__name__)
BLUEBOX_PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent

DEFAULT_WORKER_TIMEOUT_SECONDS = 300
DEFAULT_WORKER_MAX_ITERATIONS = 30


def _build_documentation_loader() -> DocumentationDataLoader:
    """Build the same docs/code loader family used by agent TUIs."""
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
    return DocumentationDataLoader(
        documentation_paths=[docs_dir],
        code_paths=code_paths,
    )


def _collect_capture_input_files(cdp_captures_dir: Path) -> list[tuple[str, Path]]:
    """Collect capture JSONL files and map them to mounted input names."""
    candidates: list[tuple[str, Path]] = [
        ("network_events", cdp_captures_dir / "network" / "events.jsonl"),
        ("storage_events", cdp_captures_dir / "storage" / "events.jsonl"),
        ("dom_events", cdp_captures_dir / "dom" / "events.jsonl"),
        ("window_properties_events", cdp_captures_dir / "window_properties" / "events.jsonl"),
        ("js_events", cdp_captures_dir / "js" / "events.jsonl"),
        ("interaction_events", cdp_captures_dir / "interaction" / "events.jsonl"),
    ]
    return [(name, path) for name, path in candidates if path.exists()]


def _mount_capture_inputs(workspace: LocalAgentWorkspace, capture_inputs: list[tuple[str, Path]]) -> None:
    """Attach capture inputs into workspace raw/."""
    for name, source_path in capture_inputs:
        workspace.attach_input_file(name=name, source_path=source_path)


def _load_if_exists(loader_cls: type, jsonl_path: Path) -> Any | None:
    """Load a data loader only when the JSONL path exists."""
    if not jsonl_path.exists():
        return None
    try:
        return loader_cls(jsonl_path=str(jsonl_path))
    except Exception as e:
        logger.warning("Failed to load %s from %s: %s", loader_cls.__name__, jsonl_path, e)
        return None


class ExperimentWorkerAgentTUI(AbstractAgentTUI):
    """TUI for chatting with ExperimentWorker and running autonomous experiments."""

    TITLE = "Experiment Worker Agent"
    SHOW_SAVED_FILES_PANE = True
    SLASH_COMMANDS = {
        **BASE_SLASH_COMMANDS,
        "/experiement": "Run autonomous experiment from plain text prompt",
    }
    HELP_TEXT = BASE_HELP_TEXT + (
        "\n    [cyan]/experiement[/cyan]    Run autonomous worker experiment"
        "\n                       [dim]Usage: /experiement <text prompt>[/dim]"
        "\n"
    )

    def __init__(
        self,
        llm_model: LLMModel,
        workspace_dir: str,
        remote_debugging_address: str | None,
        cdp_captures_dir: Path | None,
        worker_timeout_seconds: int,
        max_worker_iterations: int,
    ) -> None:
        super().__init__(llm_model=llm_model, working_dir=workspace_dir)
        self._workspace_dir = workspace_dir
        self._remote_debugging_address = remote_debugging_address
        self._worker_timeout_seconds = worker_timeout_seconds
        self._max_worker_iterations = max_worker_iterations

        self._workspace = LocalAgentWorkspace.from_directory_path(self._workspace_dir)
        self._documentation_loader = _build_documentation_loader()

        self._network_loader: NetworkDataLoader | None = None
        self._storage_loader: StorageDataLoader | None = None
        self._dom_loader: DOMDataLoader | None = None
        self._window_loader: WindowPropertyDataLoader | None = None

        if cdp_captures_dir is not None:
            capture_inputs = _collect_capture_input_files(cdp_captures_dir)
            _mount_capture_inputs(self._workspace, capture_inputs)
            self._network_loader = _load_if_exists(
                NetworkDataLoader, cdp_captures_dir / "network" / "events.jsonl",
            )
            self._storage_loader = _load_if_exists(
                StorageDataLoader, cdp_captures_dir / "storage" / "events.jsonl",
            )
            self._dom_loader = _load_if_exists(
                DOMDataLoader, cdp_captures_dir / "dom" / "events.jsonl",
            )
            self._window_loader = _load_if_exists(
                WindowPropertyDataLoader, cdp_captures_dir / "window_properties" / "events.jsonl",
            )

        self._last_experiment: dict[str, Any] | None = None

    def _create_agent(self) -> ExperimentWorker:
        return ExperimentWorker(
            emit_message_callable=self._handle_message,
            stream_chunk_callable=self._handle_stream_chunk,
            llm_model=self._llm_model,
            workspace=self._workspace,
            documentation_data_loader=self._documentation_loader,
            remote_debugging_address=self._remote_debugging_address,
            network_data_loader=self._network_loader,
            storage_data_loader=self._storage_loader,
            dom_data_loader=self._dom_loader,
            window_property_data_loader=self._window_loader,
        )

    def _create_runtime_worker(self) -> ExperimentWorker:
        """Create a fresh worker instance for `/experiement` runs."""
        return ExperimentWorker(
            emit_message_callable=self._handle_message,
            stream_chunk_callable=self._handle_stream_chunk,
            llm_model=self._llm_model,
            workspace=self._workspace,
            documentation_data_loader=self._documentation_loader,
            remote_debugging_address=self._remote_debugging_address,
            network_data_loader=self._network_loader,
            storage_data_loader=self._storage_loader,
            dom_data_loader=self._dom_loader,
            window_property_data_loader=self._window_loader,
        )

    def _print_welcome(self) -> None:
        chat = self.query_one("#chat-log", RichLog)
        chat.write(Text.from_markup(
            "[bold green]Experiment Worker Agent[/bold green]  "
            "[dim]chat + autonomous experiment harness[/dim]"
        ))
        chat.write("")
        chat.write(Text.from_markup(
            f"[dim]Model:[/dim]       {self._llm_model.value}\n"
            f"[dim]Browser:[/dim]     {self._remote_debugging_address or '(disabled)'}\n"
            f"[dim]Workspace:[/dim]   {self._workspace_dir}\n"
            f"[dim]Data loaders:[/dim] network={bool(self._network_loader)} "
            f"storage={bool(self._storage_loader)} dom={bool(self._dom_loader)} "
            f"window={bool(self._window_loader)}\n"
            f"[dim]Tip:[/dim]         /experiement replay captured journey-solution-option POST and return exact minimal headers/body"
        ))
        chat.write("")

    def _build_status_text(self) -> str:
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        msg_count = len(self._agent.get_chats()) if self._agent else 0
        tokens_used, ctx_pct = self._estimate_context_usage()
        ctx_bar = self._context_bar(ctx_pct)

        last = self._last_experiment or {}
        last_status = last.get("status", "n/a")
        last_iters = last.get("iterations", "n/a")
        return (
            f"[bold green]EXPERIMENT WORKER[/bold green]\n"
            f"[dim]──────────────────[/dim]\n"
            f"[dim]Model:[/dim]     {self._llm_model.value}\n"
            f"[dim]Messages:[/dim]  {msg_count}\n"
            f"[dim]Tools:[/dim]     {self._tool_call_count}\n"
            f"[dim]Last run:[/dim]  {last_status} ({last_iters} iters)\n"
            f"[dim]Context:[/dim]   {ctx_bar}\n"
            f"[dim](est.)      ~{tokens_used:,} / {self._context_window_size:,}[/dim]\n"
            f"[dim]──────────────────[/dim]\n"
            f"[dim]Time:[/dim]      {now}\n"
        )

    def _handle_custom_command(self, cmd: str, raw_input: str) -> bool:
        lower = raw_input.lower()
        if not lower.startswith("/experiement"):
            return False

        chat = self.query_one("#chat-log", RichLog)
        parts = raw_input.split(maxsplit=1)
        if len(parts) < 2 or not parts[1].strip():
            chat.write(Text.from_markup(
                "[yellow]Usage:[/yellow] /experiement <text prompt>\n"
                "[dim]Example:[/dim] /experiement find the exact request needed for station alerts and return a minimal reproducible recipe"
            ))
            return True

        task = parts[1].strip()
        chat.write(Text.from_markup(
            f"[yellow]Running autonomous experiment[/yellow]\n"
            f"[dim]Task:[/dim] {escape(task)}"
        ))

        self._processing = True
        self._assistant_header_printed = False
        self._status_update_printed = False
        self._run_experiment_async(task=task)
        return True

    @work(thread=True)
    def _run_experiment_async(self, task: str) -> None:
        worker: ExperimentWorker | None = None
        start_time = time.perf_counter()
        try:
            worker = self._create_runtime_worker()
            config = AutonomousRunConfig(
                min_iterations=1,
                max_iterations=self._max_worker_iterations,
            )
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(worker.run_autonomous, task, config)
                result = future.result(timeout=self._worker_timeout_seconds)

            elapsed = time.perf_counter() - start_time
            iterations = worker.autonomous_iteration

            raw_result = result.model_dump() if isinstance(result, BaseModel) else result
            if raw_result is None:
                raw_result = {"result": None}
            if not isinstance(raw_result, dict):
                raw_result = {"result": raw_result}

            status = "incomplete"
            if isinstance(result, SpecialistResultWrapper):
                status = "success" if result.success else "failure"

            record = {
                "task": task,
                "status": status,
                "iterations": iterations,
                "elapsed_seconds": round(elapsed, 3),
                "result": raw_result,
            }
            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            ref = self._workspace.save_artifact(
                source="output",
                filename=f"experiment_worker_{ts}.json",
                content=json.dumps(record, indent=2, default=str),
                tool_name="run_experiment_worker_agent",
                content_type="json",
                metadata={"task": task[:200]},
            )

            self.call_from_thread(
                self._show_experiment_success,
                status,
                iterations,
                elapsed,
                ref.relative_path,
            )
        except FuturesTimeoutError:
            self.call_from_thread(
                self._show_experiment_error,
                f"Experiment timed out after {self._worker_timeout_seconds}s",
            )
        except Exception as e:
            self.call_from_thread(self._show_experiment_error, str(e))
        finally:
            if worker is not None:
                try:
                    worker.close()
                except Exception:
                    logger.exception("Failed to close worker browser tab")

    def _show_experiment_success(
        self,
        status: str,
        iterations: int,
        elapsed: float,
        saved_relative_path: str,
    ) -> None:
        self._last_experiment = {
            "status": status,
            "iterations": iterations,
            "elapsed_seconds": elapsed,
        }
        chat = self.query_one("#chat-log", RichLog)
        color = "green" if status == "success" else "red" if status == "failure" else "yellow"
        chat.write(Text.from_markup(
            f"[bold {color}]Experiment {status.upper()}[/bold {color}] "
            f"[dim]({iterations} iterations, {elapsed:.1f}s)[/dim]\n"
            f"[dim]saved:[/dim] {saved_relative_path}"
        ))
        self._add_saved_file(str(self._workspace.root_path / saved_relative_path))
        self._processing = False
        self._update_status()

    def _show_experiment_error(self, error: str) -> None:
        chat = self.query_one("#chat-log", RichLog)
        chat.write(Text.from_markup(f"[bold red]Experiment failed:[/bold red] {escape(error)}"))
        self._processing = False
        self._update_status()


def main() -> None:
    """Entry point for ExperimentWorker Agent TUI."""
    parser = argparse.ArgumentParser(description="Experiment Worker Agent — Multi-pane TUI")
    add_model_argument(parser)
    parser.add_argument(
        "--workspace-dir",
        type=str,
        default="./agent_workspaces/experiment_worker_agent",
        help="Workspace directory for worker artifacts (default: ./agent_workspaces/experiment_worker_agent)",
    )
    parser.add_argument(
        "--remote-debugging-address",
        type=str,
        default="http://127.0.0.1:9222",
        help="Chrome remote debugging address (set empty string to disable browser tools)",
    )
    parser.add_argument(
        "--cdp-captures-dir",
        type=Path,
        default=None,
        help="Optional captures dir; mounts raw/*.jsonl into workspace for tool/Python access",
    )
    parser.add_argument(
        "--worker-timeout-seconds",
        type=int,
        default=DEFAULT_WORKER_TIMEOUT_SECONDS,
        help=f"Timeout for autonomous worker run (default: {DEFAULT_WORKER_TIMEOUT_SECONDS})",
    )
    parser.add_argument(
        "--max-worker-iterations",
        type=int,
        default=DEFAULT_WORKER_MAX_ITERATIONS,
        help=f"Max autonomous iterations (default: {DEFAULT_WORKER_MAX_ITERATIONS})",
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress logs")
    parser.add_argument("--log-file", type=str, default=None, help="Log to file")
    args = parser.parse_args()

    console = Console()
    llm_model = resolve_model(args.model, console)
    remote_debugging_address = args.remote_debugging_address.strip() or None

    console.print(f"[dim]Model: {llm_model.value}[/dim]")
    console.print(f"[dim]Workspace: {args.workspace_dir}[/dim]")
    console.print(f"[dim]Browser: {remote_debugging_address or '(disabled)'}[/dim]")
    if args.cdp_captures_dir:
        console.print(f"[dim]Captures mounted from: {args.cdp_captures_dir}[/dim]")
    console.print()

    enable_tui_logging(log_file=args.log_file or ".bluebox_experiment_worker_tui.log", quiet=args.quiet)

    app = ExperimentWorkerAgentTUI(
        llm_model=llm_model,
        workspace_dir=args.workspace_dir,
        remote_debugging_address=remote_debugging_address,
        cdp_captures_dir=args.cdp_captures_dir,
        worker_timeout_seconds=args.worker_timeout_seconds,
        max_worker_iterations=args.max_worker_iterations,
    )
    app.run()


if __name__ == "__main__":
    main()
