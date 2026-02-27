"""
bluebox/scripts/run_inspector_agent.py

Interactive TUI for RoutineInspector.

Features:
- Chat directly with the inspector agent.
- Run `/inspect <routine.json> ['{...}'|@params.json]` to:
  1) load a routine JSON
  2) execute it with test params (defaults to test_parameters in file)
  3) run the inspector autonomously with the same inspection prompt pattern
     used by PrincipalInvestigator
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import shlex
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from rich.console import Console
from rich.text import Text
from textual import work
from textual.widgets import RichLog

from bluebox.agents.abstract_agent import AutonomousRunConfig
from bluebox.agents.routine_inspector import RoutineInspector
from bluebox.agents.workspace import LocalWorkspace
from bluebox.data_models.llms.vendors import LLMModel
from bluebox.data_models.orchestration.inspection import RoutineInspectionResult
from bluebox.data_models.routine.execution import RoutineExecutionResultWithMetadata
from bluebox.data_models.routine.routine import Routine
from bluebox.llms.data_loaders.documentation_data_loader import DocumentationDataLoader
from bluebox.utils.cli_utils import add_model_argument, resolve_model
from bluebox.utils.logger import enable_tui_logging, get_logger
from bluebox.utils.tui_base import AbstractAgentTUI, BASE_HELP_TEXT, BASE_SLASH_COMMANDS


logger = get_logger(name=__name__)
BLUEBOX_PACKAGE_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_INSPECTOR_TIMEOUT_SECONDS = 180
DEFAULT_INSPECTOR_MAX_ITERATIONS = 10
DEFAULT_ROUTINE_EXECUTION_TIMEOUT_SECONDS = 120.0
INSPECTOR_INLINE_EXECUTION_MAX_CHARS = 20_000
INSPECTION_PROMPT_MAX_CHARS = 120_000


def _build_documentation_loader() -> DocumentationDataLoader:
    """Build the same docs/code loader used by PI for inspector guidance."""
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


def _mount_capture_inputs(workspace: LocalWorkspace, capture_inputs: list[tuple[str, Path]]) -> None:
    """Attach capture inputs into workspace raw/."""
    for name, source_path in capture_inputs:
        workspace.attach_input_file(name=name, source_path=source_path)


def _load_exploration_summaries(exploration_dir: Path | None) -> dict[str, str]:
    """Load saved exploration summaries as JSON strings for prompt injection."""
    if exploration_dir is None:
        return {}

    base = exploration_dir
    if (base / "exploration").exists():
        base = base / "exploration"

    summaries: dict[str, str] = {}
    for domain in ("network", "storage", "dom", "ui"):
        path = base / f"{domain}.json"
        if not path.exists():
            continue
        try:
            raw = json.loads(path.read_text())
            summaries[domain] = json.dumps(raw, indent=2, default=str)
        except Exception as e:
            logger.warning("Failed to load exploration summary %s: %s", path, e)
    return summaries


def _resolve_routine_file(routine_path: Path) -> tuple[Routine, dict[str, Any]]:
    """
    Resolve a routine file into (Routine, default_test_parameters).

    Expects a raw routine JSON with name, description, operations.
    Optionally includes test_parameters at the top level.
    """
    raw = json.loads(routine_path.read_text())
    if not isinstance(raw, dict):
        raise ValueError("Routine file must contain a JSON object")

    if not all(k in raw for k in ("name", "description", "operations")):
        raise ValueError(
            "Routine file must contain 'name', 'description', and 'operations' keys.",
        )

    default_params: dict[str, Any] = {}
    if isinstance(raw.get("test_parameters"), dict):
        default_params = raw["test_parameters"]

    routine = Routine.model_validate(raw)
    return routine, default_params


def _parse_params_expression(params_expr: str) -> dict[str, Any]:
    """Parse params from JSON string, Python dict literal, or @file.json."""
    expr = params_expr.strip()
    if not expr:
        return {}

    if expr.startswith("@"):
        params_path = Path(expr[1:]).expanduser()
        payload = json.loads(params_path.read_text())
        if not isinstance(payload, dict):
            raise ValueError("Params file must contain a JSON object")
        return payload

    path_candidate = Path(expr).expanduser()
    if path_candidate.exists() and path_candidate.is_file():
        payload = json.loads(path_candidate.read_text())
        if not isinstance(payload, dict):
            raise ValueError("Params file must contain a JSON object")
        return payload

    # Try strict JSON first
    try:
        payload = json.loads(expr)
    except json.JSONDecodeError:
        # Fallback for Python dict literal input convenience
        payload = ast.literal_eval(expr)

    if not isinstance(payload, dict):
        raise ValueError("Params must be an object/dict")
    return payload


class InspectorAgentTUI(AbstractAgentTUI):
    """TUI for chatting with RoutineInspector and running local inspection mocks."""

    TITLE = "Inspector Agent"
    SHOW_SAVED_FILES_PANE = True
    SLASH_COMMANDS = {
        **BASE_SLASH_COMMANDS,
        "/inspect": "Execute routine + autonomous inspection: /inspect <routine.json> ['{...}'|@params.json]  (params default to test_parameters in file)",
    }
    HELP_TEXT = BASE_HELP_TEXT + (
        "\n    [cyan]/inspect[/cyan]         Execute routine + run autonomous inspector"
        "\n                       [dim]Usage: /inspect <routine.json> ['{...}'|@params.json][/dim]"
        "\n                       [dim]Params: inline JSON/dict, @file.json path, or omit to use test_parameters from file.[/dim]\n"
    )

    def __init__(
        self,
        llm_model: LLMModel,
        workspace_dir: str,
        remote_debugging_address: str,
        cdp_captures_dir: Path | None,
        exploration_dir: Path | None,
        inspector_timeout_seconds: int,
        routine_timeout_seconds: float,
        max_inspector_iterations: int,
    ) -> None:
        super().__init__(llm_model=llm_model, working_dir=workspace_dir)
        self._workspace_dir = workspace_dir
        self._remote_debugging_address = remote_debugging_address
        self._inspector_timeout_seconds = inspector_timeout_seconds
        self._routine_timeout_seconds = routine_timeout_seconds
        self._max_inspector_iterations = max_inspector_iterations

        self._workspace = LocalWorkspace.from_directory_path(self._workspace_dir)
        self._documentation_loader = _build_documentation_loader()
        self._exploration_summaries = _load_exploration_summaries(exploration_dir)

        if cdp_captures_dir is not None:
            capture_inputs = _collect_capture_input_files(cdp_captures_dir)
            _mount_capture_inputs(self._workspace, capture_inputs)

        self._last_inspection: dict[str, Any] | None = None

    def _create_agent(self) -> RoutineInspector:
        return RoutineInspector(
            emit_message_callable=self._handle_message,
            stream_chunk_callable=self._handle_stream_chunk,
            llm_model=self._llm_model,
            workspace=self._workspace,
            documentation_data_loader=self._documentation_loader,
        )

    def _create_runtime_inspector(self) -> RoutineInspector:
        """Create a fresh inspector instance for `/inspect` runs."""
        return RoutineInspector(
            emit_message_callable=self._handle_message,
            stream_chunk_callable=self._handle_stream_chunk,
            llm_model=self._llm_model,
            workspace=self._workspace,
            documentation_data_loader=self._documentation_loader,
        )

    def _print_welcome(self) -> None:
        chat = self.query_one("#chat-log", RichLog)
        chat.write(Text.from_markup(
            "[bold green]Inspector Agent[/bold green]  [dim]chat + autonomous inspection harness[/dim]"
        ))
        chat.write("")
        loaded_domains = ", ".join(sorted(self._exploration_summaries.keys())) or "(none)"
        chat.write(Text.from_markup(
            f"[dim]Model:[/dim]       {self._llm_model.value}\n"
            f"[dim]Browser:[/dim]     {self._remote_debugging_address}\n"
            f"[dim]Workspace:[/dim]   {self._workspace_dir}\n"
            f"[dim]Exploration:[/dim] {loaded_domains}\n"
            f"[dim]Tip:[/dim]         /inspect /abs/path/to/routine.json '{{\"param\": \"value\"}}'"
        ))
        chat.write("")

    def _build_status_text(self) -> str:
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        msg_count = len(self._agent.get_chats()) if self._agent else 0
        tokens_used, ctx_pct = self._estimate_context_usage()
        ctx_bar = self._context_bar(ctx_pct)
        last = self._last_inspection or {}
        last_pass = last.get("overall_pass")
        last_score = last.get("overall_score")
        if last_pass is None:
            last_summary = "n/a"
        else:
            verdict = "PASS" if last_pass else "FAIL"
            last_summary = f"{verdict} ({last_score}/100)"
        return (
            f"[bold green]INSPECTOR AGENT[/bold green]\n"
            f"[dim]──────────────────[/dim]\n"
            f"[dim]Model:[/dim]     {self._llm_model.value}\n"
            f"[dim]Messages:[/dim]  {msg_count}\n"
            f"[dim]Tools:[/dim]     {self._tool_call_count}\n"
            f"[dim]Last inspect:[/dim] {last_summary}\n"
            f"[dim]Context:[/dim]   {ctx_bar}\n"
            f"[dim](est.)      ~{tokens_used:,} / {self._context_window_size:,}[/dim]\n"
            f"[dim]──────────────────[/dim]\n"
            f"[dim]Time:[/dim]      {now}\n"
        )

    def _handle_custom_command(self, cmd: str, raw_input: str) -> bool:
        if not raw_input.lower().startswith("/inspect"):
            return False

        chat = self.query_one("#chat-log", RichLog)
        try:
            parts = shlex.split(raw_input)
        except ValueError as e:
            chat.write(Text.from_markup(f"[red]Invalid command syntax:[/red] {e}"))
            return True

        if len(parts) < 2:
            chat.write(Text.from_markup(
                "[yellow]Usage:[/yellow] /inspect <routine.json> ['{...}'|@params.json]\n"
                "[dim]Params default to test_parameters in the routine file if omitted.[/dim]"
            ))
            return True

        routine_path = Path(parts[1]).expanduser()
        if not routine_path.exists():
            chat.write(Text.from_markup(f"[red]Routine file not found:[/red] {routine_path}"))
            return True

        try:
            routine, default_params = _resolve_routine_file(routine_path)
        except Exception as e:
            chat.write(Text.from_markup(f"[red]Failed to load routine:[/red] {e}"))
            return True

        params_expr = " ".join(parts[2:]) if len(parts) > 2 else ""
        try:
            test_parameters = (
                _parse_params_expression(params_expr)
                if params_expr
                else (default_params or {})
            )
        except Exception as e:
            chat.write(Text.from_markup(f"[red]Failed to parse parameters:[/red] {e}"))
            return True

        chat.write(Text.from_markup(
            f"[yellow]Running inspection[/yellow] for [bold]{routine.name}[/bold]\n"
            f"[dim]Routine file:[/dim] {routine_path}\n"
            f"[dim]Test params:[/dim] {json.dumps(test_parameters, default=str)}"
        ))

        self._processing = True
        self._assistant_header_printed = False
        self._status_update_printed = False
        self._inspect_async(
            routine_path=str(routine_path),
            routine_data=routine.model_dump(mode="json"),
            test_parameters=test_parameters,
        )
        return True

    def _execute_routine_with_params(
        self,
        routine: Routine,
        test_parameters: dict[str, Any],
    ) -> RoutineExecutionResultWithMetadata | None:
        """Execute routine in live browser, mirroring PI settings."""
        if not self._remote_debugging_address:
            return None
        try:
            return routine.execute(
                parameters_dict=test_parameters,
                remote_debugging_address=self._remote_debugging_address,
                timeout=self._routine_timeout_seconds,
                close_tab_when_done=True,
                incognito=True,
            )
        except Exception as e:
            logger.error("Routine execution failed in run_inspector_agent: %s", e)
            return None

    def _build_inspection_prompt(
        self,
        routine: Routine,
        execution_result: RoutineExecutionResultWithMetadata | None,
    ) -> str:
        """Build inspection prompt using the same structure PI uses."""
        prompt_parts: list[str] = [
            f"## Routine Name\n{routine.name}\n",
            f"## Routine Description\n{routine.description}\n",
            f"## Routine JSON\n```json\n{json.dumps(routine.model_dump(), indent=2, default=str)}\n```\n",
        ]

        if execution_result is not None:
            exec_data = execution_result.model_dump(mode="json")
            exec_json = json.dumps(exec_data, indent=2, default=str)

            persisted_for_inspector = False
            if len(exec_json) > INSPECTOR_INLINE_EXECUTION_MAX_CHARS:
                try:
                    self._workspace.ensure_dirs()
                    safe_name = re.sub(r"[^a-zA-Z0-9_.-]+", "_", routine.name).strip("_")
                    if not safe_name:
                        safe_name = "routine"
                    artifact_ref = self._workspace.save_artifact(
                        source="raw",
                        filename=f"{safe_name}_execution_result.json",
                        content=exec_json,
                        tool_name="run_inspector_agent",
                        content_type="json",
                        metadata={
                            "routine_name": routine.name,
                            "char_count": len(exec_json),
                        },
                    )
                    prompt_parts.append(
                        "## Execution Result\n"
                        f"Execution payload is large ({len(exec_json)} chars) and was saved to:\n"
                        f"- workspace path: `{artifact_ref.relative_path}`\n"
                        f"- artifact_id: `{artifact_ref.artifact_id}`\n\n"
                        "Use `execute_python` or `read_file(scope=\"workspace\", path=\"...\")` to inspect "
                        "this file directly and base your judgment on the full payload.\n"
                        "Do not claim truncation — the full execution result is available in workspace raw/.\n"
                    )
                    persisted_for_inspector = True
                except Exception as e:
                    logger.warning(
                        "Failed to persist large execution payload for inspector run: %s",
                        e,
                    )

            if not persisted_for_inspector:
                prompt_parts.append(f"## Execution Result\n```json\n{exec_json}\n```\n")
        else:
            prompt_parts.append("## Execution Result\nNot available (no browser or execution failed).\n")

        base_prompt = "\n".join(prompt_parts)

        exploration_section = ""
        if self._exploration_summaries:
            exploration_parts: list[str] = ["## Exploration Summaries\n"]
            for domain, summary in self._exploration_summaries.items():
                exploration_parts.append(f"### {domain}\n{summary}\n")
            exploration_section = "\n".join(exploration_parts)

        inspection_prompt = (
            f"{base_prompt}\n\n{exploration_section}"
            if exploration_section
            else base_prompt
        )
        if len(inspection_prompt) > INSPECTION_PROMPT_MAX_CHARS and exploration_section:
            inspection_prompt = (
                f"{base_prompt}\n\n"
                "## Exploration Summaries\n"
                "[omitted due prompt size; consult persisted exploration artifacts if needed]\n"
            )
        return inspection_prompt

    @work(thread=True)
    def _inspect_async(
        self,
        routine_path: str,
        routine_data: dict[str, Any],
        test_parameters: dict[str, Any],
    ) -> None:
        """Execute routine and run autonomous inspector in a background thread."""
        try:
            routine = Routine.model_validate(routine_data)
            execution_result = self._execute_routine_with_params(routine, test_parameters)

            inspector = self._create_runtime_inspector()
            inspection_prompt = self._build_inspection_prompt(routine, execution_result)
            config = AutonomousRunConfig(
                min_iterations=1,
                max_iterations=self._max_inspector_iterations,
            )

            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    inspector.run_autonomous,
                    task=inspection_prompt,
                    config=config,
                    output_schema=RoutineInspectionResult.model_json_schema(),
                    output_description="RoutineInspectionResult with scores, blocking issues, and verdict",
                )
                result = future.result(timeout=self._inspector_timeout_seconds)

            raw_result = result.model_dump() if isinstance(result, BaseModel) else result
            if not isinstance(raw_result, dict):
                raw_result = {"result": raw_result}

            inspection_output = raw_result.get("output", raw_result)
            if not isinstance(inspection_output, dict):
                inspection_output = {"output": inspection_output}

            record = {
                "routine_file": routine_path,
                "routine_name": routine.name,
                "test_parameters": test_parameters,
                "execution_result": (
                    execution_result.model_dump(mode="json")
                    if execution_result is not None
                    else None
                ),
                "inspection_result": raw_result,
            }
            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            filename = f"inspection_{routine.name}_{ts}.json"
            ref = self._workspace.save_artifact(
                source="output",
                filename=filename,
                content=json.dumps(record, indent=2, default=str),
                tool_name="run_inspector_agent",
                content_type="json",
                metadata={"routine_name": routine.name},
            )

            self.call_from_thread(
                self._show_inspection_success,
                routine.name,
                inspection_output,
                ref.relative_path,
            )
        except FuturesTimeoutError:
            self.call_from_thread(
                self._show_inspection_error,
                f"Inspector timed out after {self._inspector_timeout_seconds}s",
            )
        except Exception as e:
            self.call_from_thread(self._show_inspection_error, str(e))

    def _show_inspection_success(
        self,
        routine_name: str,
        inspection_output: dict[str, Any],
        saved_relative_path: str,
    ) -> None:
        """Render inspection summary in chat and update status."""
        self._last_inspection = inspection_output
        overall_pass = inspection_output.get("overall_pass")
        overall_score = inspection_output.get("overall_score")
        blocking = inspection_output.get("blocking_issues", [])
        recommendations = inspection_output.get("recommendations", [])

        chat = self.query_one("#chat-log", RichLog)
        verdict = "PASS" if overall_pass else "FAIL"
        color = "green" if overall_pass else "red"
        chat.write(Text.from_markup(
            f"[bold {color}]Inspection {verdict}[/bold {color}] for [bold]{routine_name}[/bold] "
            f"(score: {overall_score})\n"
            f"[dim]blocking_issues:[/dim] {len(blocking)}   "
            f"[dim]recommendations:[/dim] {len(recommendations)}\n"
            f"[dim]saved:[/dim] {saved_relative_path}"
        ))

        self._add_saved_file(str(self._workspace.root_path / saved_relative_path))
        self._processing = False
        self._update_status()

    def _show_inspection_error(self, error: str) -> None:
        """Render inspection error in chat and update status."""
        chat = self.query_one("#chat-log", RichLog)
        chat.write(Text.from_markup(f"[bold red]Inspection failed:[/bold red] {error}"))
        self._processing = False
        self._update_status()


def main() -> None:
    """Entry point for InspectorAgent TUI."""
    parser = argparse.ArgumentParser(description="Routine Inspector Agent — Multi-pane TUI")
    add_model_argument(parser)
    parser.add_argument(
        "--workspace-dir",
        type=str,
        default="./agent_workspaces/inspector_agent",
        help="Workspace directory for inspector artifacts (default: ./agent_workspaces/inspector_agent)",
    )
    parser.add_argument(
        "--remote-debugging-address",
        type=str,
        default="http://127.0.0.1:9222",
        help="Chrome remote debugging address for routine execution",
    )
    parser.add_argument(
        "--cdp-captures-dir",
        type=Path,
        default=None,
        help="Optional captures dir; mounts raw/*.jsonl into workspace for execute_python/read_file access",
    )
    parser.add_argument(
        "--exploration-dir",
        type=Path,
        default=None,
        help="Optional exploration dir (or api_indexing output dir containing exploration/) to mirror PI inspection context",
    )
    parser.add_argument(
        "--inspector-timeout-seconds",
        type=int,
        default=DEFAULT_INSPECTOR_TIMEOUT_SECONDS,
        help=f"Timeout for autonomous inspector run (default: {DEFAULT_INSPECTOR_TIMEOUT_SECONDS})",
    )
    parser.add_argument(
        "--routine-timeout-seconds",
        type=float,
        default=DEFAULT_ROUTINE_EXECUTION_TIMEOUT_SECONDS,
        help=f"Timeout for routine.execute (default: {DEFAULT_ROUTINE_EXECUTION_TIMEOUT_SECONDS})",
    )
    parser.add_argument(
        "--max-inspector-iterations",
        type=int,
        default=DEFAULT_INSPECTOR_MAX_ITERATIONS,
        help=f"Max autonomous iterations for inspector (default: {DEFAULT_INSPECTOR_MAX_ITERATIONS})",
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress logs")
    parser.add_argument("--log-file", type=str, default=None, help="Log to file")
    args = parser.parse_args()

    console = Console()
    llm_model = resolve_model(args.model, console)
    console.print(f"[dim]Model: {llm_model.value}[/dim]")
    console.print(f"[dim]Workspace: {args.workspace_dir}[/dim]")
    console.print(f"[dim]Browser: {args.remote_debugging_address}[/dim]")
    if args.cdp_captures_dir:
        console.print(f"[dim]Captures mounted from: {args.cdp_captures_dir}[/dim]")
    if args.exploration_dir:
        console.print(f"[dim]Exploration context: {args.exploration_dir}[/dim]")
    console.print()

    enable_tui_logging(log_file=args.log_file or ".bluebox_inspector_tui.log", quiet=args.quiet)

    app = InspectorAgentTUI(
        llm_model=llm_model,
        workspace_dir=args.workspace_dir,
        remote_debugging_address=args.remote_debugging_address,
        cdp_captures_dir=args.cdp_captures_dir,
        exploration_dir=args.exploration_dir,
        inspector_timeout_seconds=args.inspector_timeout_seconds,
        routine_timeout_seconds=args.routine_timeout_seconds,
        max_inspector_iterations=args.max_inspector_iterations,
    )
    app.run()


if __name__ == "__main__":
    main()
