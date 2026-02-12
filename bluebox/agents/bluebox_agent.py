"""
bluebox/agents/bluebox_agent.py

Agent specialized in browsing the web using Vectorly routines.

Contains:
- BlueBoxAgent: Agent for searching and executing Vectorly routines
- Uses: AbstractAgent base class for all agent plumbing
"""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from textwrap import dedent
from typing import Any, Callable

import requests

from bluebox.agents.abstract_agent import AbstractAgent, AgentCard, agent_tool
from bluebox.config import Config
from bluebox.data_models.llms.interaction import (
    Chat,
    ChatResponseEmittedMessage,
    ChatThread,
    EmittedMessage,
)
from bluebox.data_models.llms.vendors import LLMModel, OpenAIModel
from bluebox.data_models.routine.routine import RoutineExecutionRequest, RoutineInfo
from bluebox.utils.code_execution_sandbox import execute_python_sandboxed
from bluebox.utils.infra_utils import read_file_lines
from bluebox.utils.llm_utils import token_optimized
from bluebox.utils.logger import get_logger

logger = get_logger(name=__name__)


class BlueBoxAgent(AbstractAgent):
    """
    BlueBoxAgent that searches and executes web automation routines.

    The agent uses AbstractAgent as its base and provides tools to search,
    inspect, and execute Vectorly routines.
    """

    AGENT_CARD = AgentCard(
        description="Searches and executes pre-built Vectorly routines to fulfill user requests.",
    )

    AGENT_LOOP_MAX_ITERATIONS: int = 100

    SYSTEM_PROMPT: str = dedent("""
        You are a web automation agent. Your job is to fulfill user requests by running pre-built Vectorly routines, or falling back to the browser agent for free-form tasks.

        ## Workflow
        1. **Search broadly**: When the user makes a request, use `search_routines` with a task description that describes what the user wants to do. This runs semantic search, so add some detail. You can run this multiple times if needed to get more results.
        2. **Execute all relevant routines**: Run ALL routines that could plausibly fulfill the user's request via `execute_routines_in_parallel`. When in doubt, include the routine — running an extra routine is cheap, missing a relevant one is costly.
        3. **Fallback to browser agent**: If NO routines match after thorough searching, use `execute_browser_task` to perform the task via an AI-driven browser agent. Write a clear, detailed natural language instruction for the task.
        4. **Post-process results**: Use `run_python_code` to transform routine results into clean output files (CSV, JSON, JSONL, etc.) for the user.
        5. **Verify output**: After writing files, use `list_workspace_files` and `read_workspace_file` to verify the output looks correct. If it doesn't, fix the code and rerun.
        6. **Report results**: Summarize what was executed and the output files to the user.

        ## Post-Processing with Python
        - After routines return results, ALWAYS use `run_python_code` to post-process data and generate clean output files.
        - The variable `routine_results` is pre-loaded: a list of dicts, one per JSON file in the raw/ directory.
        - You have full read/write file access to the workspace directory. Use open() to read/write files.
        - `json`, `csv`, and `Path` (from pathlib) are pre-loaded.
        - Output files are saved in the outputs/ subdirectory. Write there: `with open("outputs/results.csv", "w") as f: ...`
        - **ALWAYS add debug print() statements** in your code so you can see what's happening: print key counts, data shapes, sample values, etc. stdout is captured and returned to you.
        - **Be persistent**: If your code errors or produces unexpected results, read the error/output carefully, use `list_workspace_files` and `read_workspace_file` to inspect the data, fix the code, and try again. Keep iterating until you produce the correct output file. NEVER give up after one failed attempt — debug and retry.

        ## Inspecting the Workspace
        - Use `list_workspace_files` to see all files in the workspace (raw/, outputs/, etc.).
        - Use `read_workspace_file` to read any file by relative path (e.g. "raw/25-01-15-143052-routine_result_1.json" or "outputs/results.csv"). Use optional start_line/end_line to read specific line ranges for large files.

        ## Important Rules
        - **Always prefer routines over `execute_browser_task`**. Routines are faster, cheaper, and more reliable. Only use the browser agent as a fallback when no suitable routine exists.
        - When using `execute_browser_task`, write a specific, step-by-step task description so the browser agent knows exactly what to do.
        - If your first search returns no results, try rephrasing the task description before giving up.
        - Be concise in responses.
    """).strip()

    ## Magic methods

    def __init__(
        self,
        emit_message_callable: Callable[[EmittedMessage], None],
        persist_chat_callable: Callable[[Chat], Chat] | None = None,
        persist_chat_thread_callable: Callable[[ChatThread], ChatThread] | None = None,
        stream_chunk_callable: Callable[[str], None] | None = None,
        llm_model: LLMModel = OpenAIModel.GPT_5_1,
        chat_thread: ChatThread | None = None,
        existing_chats: list[Chat] | None = None,
        workspace_dir: str = "./bluebox_workspace",
    ) -> None:
        """
        Initialize the BlueBox Agent.

        Args:
            emit_message_callable: Callback function to emit messages to the host.
            persist_chat_callable: Optional callback to persist Chat objects.
            persist_chat_thread_callable: Optional callback to persist ChatThread.
            stream_chunk_callable: Optional callback for streaming text chunks.
            llm_model: The LLM model to use for conversation.
            chat_thread: Existing ChatThread to continue, or None for new conversation.
            existing_chats: Existing Chat messages if loading from persistence.
            workspace_dir: Root workspace directory. Raw routine results go in raw/,
                agent-generated output files go in outputs/.
        """
        # Validate required config
        if not Config.VECTORLY_API_KEY:
            raise ValueError("VECTORLY_API_KEY is not set")
        if not Config.VECTORLY_API_BASE:
            raise ValueError("VECTORLY_API_BASE is not set")

        self._workspace_dir = Path(workspace_dir)
        self._raw_dir = self._workspace_dir / "raw"
        self._outputs_dir = self._workspace_dir / "outputs"
        self._routine_cache: dict[str, RoutineInfo] = {}
        self._execution_counter: int = 0
        self._counter_lock = threading.Lock()

        super().__init__(
            emit_message_callable=emit_message_callable,
            persist_chat_callable=persist_chat_callable,
            persist_chat_thread_callable=persist_chat_thread_callable,
            stream_chunk_callable=stream_chunk_callable,
            llm_model=llm_model,
            chat_thread=chat_thread,
            existing_chats=existing_chats,
            documentation_data_loader=None,
        )

        logger.debug(
            "BlueBoxAgent initialized with model: %s, chat_thread_id: %s",
            llm_model,
            self._thread.id,
        )

    ## Abstract method implementations

    def _get_system_prompt(self) -> str:
        """Get system prompt with current time."""
        now = datetime.now()
        time_info = f"\n\n## Current Time\n{now.strftime('%Y-%m-%d %H:%M:%S %Z').strip()}"
        return self.SYSTEM_PROMPT + time_info

    ## Routine cache

    def _cache_routines_from_response(self, response: dict[str, Any] | list[Any]) -> None:
        """Parse search response and cache RoutineInfo objects for later validation."""
        if isinstance(response, list):
            items = response
        else:
            items = response.get("results", response.get("routines", []))
        if not isinstance(items, list):
            return
        for item in items:
            try:
                if not isinstance(item, dict) or "routine_id" not in item:
                    continue
                info = RoutineInfo.model_validate(item)
                self._routine_cache[info.routine_id] = info
                logger.debug("Cached routine: %s (%s)", info.name, info.routine_id)
            except Exception:
                logger.debug("Skipped caching item: %s", item.get("routine_id", "unknown"))

    def _validate_routine_params(self, routine_id: str, params: dict[str, Any]) -> str | None:
        """Validate params against cached routine info. Returns error string or None."""
        info = self._routine_cache.get(routine_id)
        if not info:
            return None  # Not cached, skip validation

        required = {p.name for p in info.parameters if p.required}
        provided = set(params.keys())
        missing = required - provided
        if missing:
            param_summary = [
                {"name": p.name, "type": p.type.value, "required": p.required, "description": p.description}
                for p in info.parameters
            ]
            return (
                f"Routine '{info.name}' ({routine_id}): missing required parameter(s) {sorted(missing)}. "
                f"Expected parameters: {param_summary}"
            )
        return None

    ## Tool handlers

    @agent_tool()
    @token_optimized
    def _search_routines(self, task: str) -> dict[str, Any]:
        """
        Search for routines by keywords. Matches against routine name and description.

        Args:
            task: Task description to search for.
        """
        url = f"{Config.VECTORLY_API_BASE}/routines/semantic-search"
        headers = {
            "Content-Type": "application/json",
            "X-Service-Token": Config.VECTORLY_API_KEY,
        }
        payload = {
            "query": task,
            "top_n": 5,
            "threshold": 0.0,
            "keywords": [],
        }

        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        self._cache_routines_from_response(data)
        return data

    @agent_tool()
    def _execute_routines_in_parallel(
        self,
        routine_executions: list[RoutineExecutionRequest],
    ) -> dict[str, Any]:
        """
        Execute one or more routines in parallel via the Vectorly API.

        Args:
            routine_executions: List of routines to execute. Each item needs routine_id and parameters.
        """
        if not routine_executions:
            return {"error": "No routine executions provided"}

        # Pre-flight validation against cached routine metadata
        validation_errors: list[str] = []
        for req in routine_executions:
            error = self._validate_routine_params(req.routine_id, req.parameters)
            if error:
                validation_errors.append(error)

        if validation_errors:
            return {"error": "Parameter validation failed. Fix and retry.\n" + "\n".join(validation_errors)}

        headers = {
            "Content-Type": "application/json",
            "X-Service-Token": Config.VECTORLY_API_KEY,
        }

        def save_result(result: dict[str, Any]) -> dict[str, Any]:
            """Save a single routine result to a JSON file in raw/."""
            try:
                self._raw_dir.mkdir(parents=True, exist_ok=True)
                with self._counter_lock:
                    self._execution_counter += 1
                    idx = self._execution_counter
                timestamp = datetime.now().strftime("%y-%m-%d-%H%M%S")
                output_path = self._raw_dir / f"{timestamp}-routine_result_{idx}.json"
                output_path.write_text(json.dumps(result, indent=2, default=str))
                result["output_file"] = str(output_path)
                logger.info("Routine result saved to %s", output_path)
            except Exception as e:
                logger.exception("Failed to save routine result to file: %s", e)
                result["output_file_error"] = str(e)
            return result

        def execute_one(req: RoutineExecutionRequest) -> dict[str, Any]:
            url = f"{Config.VECTORLY_API_BASE}/routines/{req.routine_id}/execute"
            try:
                response = requests.post(
                    url,
                    headers=headers,
                    json={"parameters": req.parameters},
                    timeout=300,
                )
                response.raise_for_status()
                return save_result({"success": True, "routine_id": req.routine_id, "data": response.json()})
            except requests.RequestException as e:
                logger.error("Routine execution failed for %s: %s", req.routine_id, e)
                return save_result({"success": False, "routine_id": req.routine_id, "error": str(e)})

        total = len(routine_executions)
        results: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(execute_one, req): req for req in routine_executions}
            for future in as_completed(futures):
                result = future.result()
                results.append(result)

                status = "succeeded" if result.get("success") else "FAILED"
                self._emit_message(ChatResponseEmittedMessage(
                    content=f"[{len(results)}/{total}] Routine '{result.get('routine_id')}' {status}.",
                ))

        succeeded = sum(1 for r in results if r.get("success"))
        return {
            "success": succeeded == total,
            "total_requested": total,
            "succeeded": succeeded,
            "failed": total - succeeded,
            "results": results,
        }

    @agent_tool()
    @token_optimized
    def _execute_browser_task(
        self,
        task: str,
        timeout_seconds: int = 300,
        use_vision: bool = True,
    ) -> dict[str, Any]:
        """
        Execute a free-form browser task using the AI browser agent.

        Use this as a FALLBACK when no pre-built routine matches the user's request.
        The browser agent receives a natural language task and autonomously navigates
        the web to complete it. This is slower and more expensive than routines.
        Progress is streamed in real time via SSE.

        Args:
            task: Detailed natural language instruction for the browser agent. Be specific and step-by-step.
            timeout_seconds: Maximum execution time in seconds (30-1800). Defaults to 300.
            use_vision: Whether the agent should use vision (screenshots). Defaults to True.
        """
        if not task or not task.strip():
            return {"error": "Task description cannot be empty"}

        timeout_seconds = max(30, min(timeout_seconds, 1800))

        headers = {
            "Content-Type": "application/json",
            "X-Service-Token": Config.VECTORLY_API_KEY,
        }

        payload = {
            "task": task,
            "timeout_seconds": timeout_seconds,
            "use_vision": use_vision,
        }

        self._emit_message(ChatResponseEmittedMessage(
            content="Starting browser agent task... This may take a few minutes.",
        ))

        try:
            with requests.post(
                f"{Config.VECTORLY_API_BASE}/buagent/execute/stream",
                headers=headers,
                json=payload,
                stream=True,
                timeout=timeout_seconds + 30,
            ) as response:
                response.raise_for_status()
                return self._consume_sse_stream(response)

        except requests.Timeout:
            return {"error": f"Browser agent timed out after {timeout_seconds}s"}
        except requests.RequestException as e:
            logger.error("Browser agent API call failed: %s", e)
            return {"error": f"Browser agent request failed: {e}"}

    def _consume_sse_stream(self, response: requests.Response) -> dict[str, Any]:
        """Parse an SSE stream from the BU agent and emit progress messages."""
        current_event = ""
        result: dict[str, Any] = {"error": "Stream ended without a terminal event"}

        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue

            if line.startswith("event: "):
                current_event = line[7:]
                continue

            if not line.startswith("data: "):
                continue

            try:
                data = json.loads(line[6:])
            except json.JSONDecodeError:
                logger.warning("Malformed SSE data line: %s", line)
                continue

            if current_event == "step":
                step_num = data.get("step_number", "?")
                max_steps = data.get("max_steps", "?")
                goal = data.get("next_goal", "")
                msg = f"[Step {step_num}/{max_steps}]"
                if goal:
                    msg += f" {goal}"
                self._emit_message(ChatResponseEmittedMessage(content=msg))

            elif current_event == "done":
                status = "succeeded" if data.get("is_successful") else "completed (not confirmed successful)"
                if not data.get("is_done"):
                    status = "did not finish"
                self._emit_message(ChatResponseEmittedMessage(
                    content=f"Browser agent task {status} in {data.get('duration_seconds', 0):.1f}s ({data.get('n_steps', 0)} steps).",
                ))
                result = {
                    "success": data.get("is_successful", False),
                    "is_done": data.get("is_done", False),
                    "final_result": data.get("final_result"),
                    "errors": data.get("errors", []),
                    "n_steps": data.get("n_steps", 0),
                    "duration_seconds": data.get("duration_seconds"),
                    "execution_id": data.get("execution_id"),
                }

            elif current_event == "error":
                error_msg = data.get("error", "Unknown error")
                self._emit_message(ChatResponseEmittedMessage(
                    content=f"Browser agent error: {error_msg}",
                ))
                result = {"error": error_msg, "execution_id": data.get("execution_id")}

        return result

    @agent_tool()
    def _run_python_code(self, code: str) -> dict[str, Any]:
        """
        Execute Python code to post-process routine results and generate output files.

        The code runs with full read/write access to the workspace directory.
        Pre-loaded variables: `routine_results` (list of dicts from all JSON files
        in the raw/ directory), `json`, `csv`, and `Path` (pathlib.Path).

        Write output files to the outputs/ subdirectory:
            with open("outputs/results.csv", "w") as f: ...

        IMPORTANT: Always include print() statements for debugging — print data shapes,
        key names, row counts, sample values, etc. If the code fails, use the output
        to diagnose and fix. Keep iterating until the output file is correct.

        Args:
            code: Python code to execute. Has full file access to the workspace.
                Pre-loaded: routine_results (list[dict]), json, csv, Path.
                Write output files to outputs/ subdirectory. Always add print()
                statements for debugging.
        """
        # Ensure directories exist
        self._raw_dir.mkdir(parents=True, exist_ok=True)
        self._outputs_dir.mkdir(parents=True, exist_ok=True)
        work_dir = str(self._workspace_dir.resolve())

        # Snapshot files in outputs/ before execution
        files_before: dict[str, float] = {}
        for p in self._outputs_dir.iterdir():
            if p.is_file():
                files_before[str(p)] = p.stat().st_mtime

        # Load all JSON files from raw/ as routine_results
        routine_results: list[dict[str, Any]] = []
        for json_file in sorted(self._raw_dir.glob("*.json")):
            try:
                data = json.loads(json_file.read_text())
                routine_results.append(data)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Failed to load %s: %s", json_file, e)

        # Execute in sandbox with work_dir for file access
        sandbox_result = execute_python_sandboxed(
            code,
            extra_globals={"routine_results": routine_results},
            work_dir=work_dir,
        )

        # Diff files in outputs/ to find new/modified ones
        files_created: list[str] = []
        for p in self._outputs_dir.iterdir():
            if not p.is_file():
                continue
            path_str = str(p)
            mtime = p.stat().st_mtime
            if path_str not in files_before or files_before[path_str] < mtime:
                files_created.append(path_str)

        # Build response
        result: dict[str, Any] = {}

        if "error" in sandbox_result:
            result["error"] = sandbox_result["error"]
            result["_hint"] = (
                "Code failed. Read the error and stdout above carefully. "
                "Use list_workspace_files and read_workspace_file to inspect the data, "
                "then fix the code and call run_python_code again."
            )

        output = sandbox_result.get("output", "")
        if output and output != "(no output)":
            result["output"] = output

        if files_created:
            result["files_created"] = files_created
            result["output_file"] = files_created[0]
            result["_hint"] = (
                "Files were created. Use read_workspace_file to verify the output "
                "is correct (check first few lines). If not, fix the code and rerun."
            )
        elif "error" not in sandbox_result:
            result["output"] = result.get("output", "") or "Code ran but produced no files."
            result["_hint"] = (
                "No files were created in outputs/. Make sure your code writes to "
                "outputs/ (e.g. open('outputs/results.csv', 'w')). Fix and rerun."
            )

        return result

    @agent_tool()
    @token_optimized
    def _list_workspace_files(self) -> dict[str, Any]:
        """
        List all files in the workspace directory as a tree.

        Shows the full directory structure including raw/ (routine results)
        and outputs/ (generated files).
        """
        self._workspace_dir.mkdir(parents=True, exist_ok=True)

        tree_lines: list[str] = []
        total_files = 0

        for dirpath, dirnames, filenames in sorted(self._workspace_dir.walk()):
            # Relative path from workspace root
            rel_dir = dirpath.relative_to(self._workspace_dir)
            depth = len(rel_dir.parts)
            indent = "  " * depth
            dir_name = rel_dir.name or str(self._workspace_dir.name)
            tree_lines.append(f"{indent}{dir_name}/")

            dirnames.sort()
            for filename in sorted(filenames):
                filepath = dirpath / filename
                size = filepath.stat().st_size
                if size < 1024:
                    size_str = f"{size}B"
                elif size < 1024 * 1024:
                    size_str = f"{size / 1024:.1f}KB"
                else:
                    size_str = f"{size / (1024 * 1024):.1f}MB"
                tree_lines.append(f"{indent}  {filename}  ({size_str})")
                total_files += 1

        return {
            "tree": "\n".join(tree_lines),
            "total_files": total_files,
        }

    @agent_tool()
    def _read_workspace_file(
        self,
        path: str,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> dict[str, Any]:
        """
        Read a file from the workspace by relative path.

        Use this to inspect raw routine results, verify generated output files,
        or debug data issues. Supports optional line ranges for large files.

        Args:
            path: Relative path within the workspace (e.g. "raw/routine_results_2024.json"
                or "outputs/results.csv").
            start_line: Optional 1-based start line number. Omit to read from the beginning.
            end_line: Optional 1-based end line number (inclusive). Omit to read to the end.
        """
        # Resolve and validate path stays within workspace
        resolved = (self._workspace_dir / path).resolve()
        workspace_resolved = self._workspace_dir.resolve()
        try:
            resolved.relative_to(workspace_resolved)
        except ValueError:
            return {"error": f"Access denied: '{path}' is outside the workspace directory"}

        result = read_file_lines(resolved, start_line=start_line, end_line=end_line)
        result["path"] = path
        return result
