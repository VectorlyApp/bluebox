"""
bluebox/agents/bluebox_agent.py

Agent specialized in browsing the web using Vectorly routines.

Contains:
- BlueBoxAgent: Agent for searching and executing Vectorly routines
- Uses: AbstractAgent base class for all agent plumbing
"""

from __future__ import annotations

import itertools
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from textwrap import dedent
from typing import Any, Callable

import requests

from bluebox.agents.abstract_agent import AbstractAgent, AgentCard, agent_tool
from bluebox.workspace import AgentWorkspace
from bluebox.config import Config
from bluebox.data_models.agents.context import BlueBoxAgentContext, UsedRoutine
from bluebox.data_models.browser_agent import (
    BrowserAgentDoneEvent,
    BrowserAgentErrorEvent,
    BrowserAgentStepEvent,
    sse_event_adapter,
)
from bluebox.data_models.llms.interaction import (
    BrowserAgentStepEmittedMessage,
    Chat,
    ChatResponseEmittedMessage,
    ChatThread,
    EmittedMessage,
    ErrorEmittedMessage,
    LLMChatResponse,
    StatusUpdateEmittedMessage,
)
from bluebox.data_models.llms.vendors import LLMModel, OpenAIModel
from bluebox.data_models.routine.routine import RoutineExecutionRequest, RoutineInfo
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

    AGENT_LOOP_MAX_ITERATIONS: int = 30

    SYSTEM_PROMPT: str = dedent("""
        You are a web automation agent. Your job is to fulfill user requests by running pre-built Vectorly routines, or falling back to the browser agent for free-form tasks.

        ## Workflow
        1. **Search broadly**: When the user makes a request, use `search_routines` with a task description that describes what the user wants to do. This runs semantic search, so add some detail. You can run this multiple times if needed to get more results.
        2. **Execute all relevant routines**: Run ALL routines that could plausibly fulfill the user's request via `execute_routines_in_parallel`. When in doubt, include the routine — running an extra routine is cheap, missing a relevant one is costly. Each routine execution requires a `routine_id` from the search results and a `parameters` dict keyed by parameter name with the corresponding value (e.g. {"origin": "New York", "date": "2025-03-01"}). Make sure to provide all required parameters as listed in the search results.
        3. **Fallback to browser agent**: If NO routines match after thorough searching, use `execute_browser_task` to perform the task via an AI-driven browser agent. Write a clear, detailed natural language instruction for the task.
        4. **Post-process results**: Use `execute_python` to transform routine results into clean output files (CSV, JSON, JSONL, etc.) for the user.
        5. **Verify output**: After writing files, use `list_files(scope="workspace")` and `read_file(scope="workspace", path=...)` to verify the output looks correct. If it doesn't, fix the code and rerun.
        6. **Report results**: Summarize what was executed and the output files to the user.

        ## Workspace
        Your workspace has the following structure:
        - `raw/` (read-only) — routine result JSON files and mounted inputs
        - `output/` — write all your generated output files here (CSV, JSON, JSONL, etc.)
        - `context/` — context files (JSON + Markdown) saved by `generate_context`, used for session replay
        - `meta/` (read-only) — system-managed manifests and metadata

        **Reading routine outputs in `execute_python`:**
        - Use `list_files(scope="workspace")` to see files in `raw/`
        - Read raw JSON files directly in Python:
          `records = [json.loads(p.read_text()) for p in Path("raw").glob("*.json")]`
        - Use `read_file(scope="workspace", path="...")` to inspect any file by relative path (e.g. "raw/25-01-15-143052-routine_result_1.json" or "output/results.csv"). Use optional start_line/end_line for large files.

        **Writing output files:**
        - Write to the output/ subdirectory: `with open("output/results.csv", "w") as f: ...`

        ## Routine Result Structure
        Each JSON file in `raw/` from `execute_routines_in_parallel` has this structure:

        ```
        {
          "execution_id": "...",
          "routine_id": "Routine_...",
          "routine_name": "RoutineName",
          "status": "completed",       # "completed" or "failed"
          "parameters": { ... },       # the input parameters used for this execution
          "result": {                  # execution result
            "ok": true,
            "error": null,
            "data": { ... }            # ← the actual payload lives HERE
          }
        }
        ```

        **Path to the payload:** `record["result"]["data"]`.
        **Input parameters:** `record["parameters"]`.

        **Important:** The payload shape varies per routine — different routines return different key names and structures. Always inspect a few raw records first before extracting fields.

        ## Post-Processing with Python
        - After routines return results, ALWAYS use `execute_python` to post-process data and generate clean output files.
        - **ALWAYS add debug print() statements** in your code so you can see what's happening: print key counts, data shapes, sample values, etc. stdout is captured and returned to you.
        - **On first pass, always explore the data**: before writing any output file, load records from `raw/*.json`, print routine names and top-level keys, then write extraction code.
        - **Be persistent**: If your code errors or produces unexpected results, read the error/output carefully, use `list_files(scope="workspace")` and `read_file(scope="workspace", path=...)` to inspect the data, fix the code, and try again. Keep iterating until you produce the correct output file. NEVER give up after one failed attempt — debug and retry.

        ## Important Rules
        - **Always prefer routines over `execute_browser_task`**. Routines are faster, cheaper, and more reliable. Only use the browser agent as a fallback when no suitable routine exists.
        - When using `execute_browser_task`, write a specific, step-by-step task description so the browser agent knows exactly what to do.
        - If your first search returns no results, try rephrasing the task description before giving up.
        - Be concise in responses.
        - Be thorough and persistent — keep iterating until the output is correct.
    """).strip()

    ## Magic methods

    def __init__(
        self,
        emit_message_callable: Callable[[EmittedMessage], None],
        workspace: AgentWorkspace,
        persist_chat_callable: Callable[[Chat], Chat] | None = None,
        persist_chat_thread_callable: Callable[[ChatThread], ChatThread] | None = None,
        stream_chunk_callable: Callable[[str], None] | None = None,
        llm_model: LLMModel = OpenAIModel.GPT_5_2,
        chat_thread: ChatThread | None = None,
        existing_chats: list[Chat] | None = None,
        auth_headers_provider: Callable[[], dict[str, str]] | None = None,
        on_llm_response: Callable[[LLMChatResponse], None] | None = None,
        context_file: str | None = None,
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
            workspace: Workspace for file I/O.
            auth_headers_provider: Optional callback that returns auth headers for
                downstream API calls. If not provided, falls back to Config.VECTORLY_SERVICE_TOKEN.
            on_llm_response: Optional callback invoked after each LLM call with the response (for token tracking).
            context_file: Optional path to a context file (.json or .md) from a previous
                session. If not provided, auto-discovers the most recent context file from
                the workspace's context/ directory.
        """
        # Validate required config
        self._auth_headers_provider = auth_headers_provider
        if not auth_headers_provider and not Config.VECTORLY_SERVICE_TOKEN:
            raise ValueError("Either auth_headers_provider or VECTORLY_SERVICE_TOKEN must be provided")

        self._workspace = workspace
        self._routine_cache: dict[str, RoutineInfo] = {}
        self._routine_execution_counter = itertools.count(
            self._get_next_routine_result_index()
        )

        # Load context from explicit path or auto-discover from workspace
        self._agent_context: BlueBoxAgentContext | None = self._load_context(context_file)

        super().__init__(
            emit_message_callable=emit_message_callable,
            workspace=self._workspace,
            persist_chat_callable=persist_chat_callable,
            persist_chat_thread_callable=persist_chat_thread_callable,
            stream_chunk_callable=stream_chunk_callable,
            llm_model=llm_model,
            chat_thread=chat_thread,
            existing_chats=existing_chats,
            documentation_data_loader=None,
            on_llm_response=on_llm_response,
            allow_code_execution=True,
        )

        logger.debug(
            "BlueBoxAgent initialized with model: %s, chat_thread_id: %s, sandbox_mode: %s, has_context: %s",
            llm_model,
            self._thread.id,
            self._sandbox_mode,
            self._agent_context is not None,
        )

    ## Properties

    @property
    def loaded_context(self) -> BlueBoxAgentContext | None:
        """The context loaded on init, if any."""
        return self._agent_context

    ## Auth

    def _get_auth_headers(self) -> dict[str, str]:
        """Build auth headers for downstream API calls.
        Uses auth_headers_provider if set, otherwise falls back to Config.VECTORLY_SERVICE_TOKEN."""
        if self._auth_headers_provider:
            return self._auth_headers_provider()
        return {"X-Service-Token": Config.VECTORLY_SERVICE_TOKEN}

    ## Abstract method implementations

    def _get_system_prompt(self) -> str:
        """Get system prompt with current time and sandbox-specific guidance."""
        now = datetime.now()
        time_info = f"\n\n## Current Time\n{now.strftime('%Y-%m-%d %H:%M:%S %Z').strip()}"
        prompt = self.SYSTEM_PROMPT + time_info
        prompt += self._generate_code_execution_prompt()
        if self._agent_context:
            prompt += self._get_context_prompt_section()
        return prompt

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

    ## Context loading

    _CONTEXT_PROMPT_MAX_CHARS: int = 20_000
    _ROUTINE_RESULT_PATTERN = re.compile(r"-routine_result_(\d+)\.json$")

    def _get_next_routine_result_index(self) -> int:
        """Scan raw/ for existing routine_result files and return max index + 1."""
        raw_dir = self._workspace.root_path / "raw"
        max_idx = 0
        if raw_dir.is_dir():
            for f in raw_dir.iterdir():
                m = self._ROUTINE_RESULT_PATTERN.search(f.name)
                if m:
                    max_idx = max(max_idx, int(m.group(1)))
        return max_idx + 1

    def _load_context(self, context_file: str | None) -> BlueBoxAgentContext | None:
        """Load context from an explicit path or auto-discover from workspace context/ dir.

        Resolution order for context_file:
        1. Absolute path
        2. Relative to workspace root

        If context_file is None, auto-discovers the most recent .json file in context/.
        """
        if context_file:
            return self._load_context_from_path(context_file)
        return self._auto_discover_context()

    def _load_context_from_path(self, context_file: str) -> BlueBoxAgentContext | None:
        """Load a context file from an explicit path (absolute or workspace-relative)."""
        path = Path(context_file)
        if not path.is_file() and not path.is_absolute():
            path = self._workspace.root_path / context_file
        if not path.is_file():
            logger.warning("Context file not found: %s", path)
            return None
        try:
            raw = path.read_text(encoding="utf-8")
            if path.suffix == ".md":
                ctx = BlueBoxAgentContext.from_markdown(raw)
            else:
                ctx = BlueBoxAgentContext.model_validate_json(raw)
            logger.info("Loaded agent context from %s", path)
            return ctx
        except Exception as e:
            logger.warning("Failed to load context file %s: %s", path, e)
            return None

    def _auto_discover_context(self) -> BlueBoxAgentContext | None:
        """Find and load the most recent context file from workspace context/ dir.

        Prefers .json files over .md when both exist. Falls back to .md if no
        JSON context files are present.
        """
        context_dir = self._workspace.root_path / "context"
        if not context_dir.is_dir():
            return None
        # Prefer JSON, fall back to Markdown
        for ext in ("*.json", "*.md"):
            files = sorted(context_dir.glob(ext), key=lambda p: p.stat().st_mtime, reverse=True)
            if files:
                return self._load_context_from_path(str(files[0]))
        return None

    def _get_context_prompt_section(self) -> str:
        """Build a system prompt section from a loaded BlueBoxAgentContext."""
        ctx = self._agent_context
        if not ctx:
            return ""

        section = (
            "\n\n## Prior Context\n"
            "A previous session already solved a similar task. Use this as a starting point.\n"
            "Replicate this path if the user's goal matches. "
            "Adjust parameters for the new request. Skip trial and error.\n\n"
            + ctx.to_markdown()
        )

        if len(section) > self._CONTEXT_PROMPT_MAX_CHARS:
            section = section[:self._CONTEXT_PROMPT_MAX_CHARS] + (
                "\n\n... (context truncated — use `read_file(scope=\"workspace\", path=\"...\")` to read "
                "the full context files in `context/` for more detail)"
            )

        return section

    def _extract_routines_from_raw(self) -> list[UsedRoutine]:
        """Extract routine info from raw/ execution result files.

        Each raw JSON file contains routine_id, routine_name, parameters,
        and status from a previous execution. Returns deduplicated list
        of successfully executed routines.
        """
        raw_results: list[dict[str, Any]] = []
        raw_refs = sorted(
            (ref for ref in self._workspace.list_artifacts("raw") if ref.relative_path.endswith(".json")),
            key=lambda ref: ref.index,
        )
        for ref in raw_refs:
            try:
                file_data = self._workspace.read_file(ref.relative_path)
                content = file_data.get("content")
                if isinstance(content, str):
                    raw_results.append(json.loads(content))
            except Exception as e:
                logger.warning("Failed to parse raw JSON artifact %s: %s", ref.relative_path, e)
        seen: set[str] = set()
        routines: list[UsedRoutine] = []
        for rr in raw_results:
            rid = rr.get("routine_id")
            if not rid or rid in seen:
                continue
            # Only include completed executions
            if rr.get("status") != "completed":
                continue
            seen.add(rid)
            routines.append(UsedRoutine.from_dict_params(
                routine_id=rid,
                routine_name=rr.get("routine_name", rid),
                parameters=rr.get("parameters", {}),
            ))
        return routines

    ## Tool handlers

    @agent_tool(token_optimized=True)
    def _search_routines(self, task: str) -> dict[str, Any]:
        """
        Search for routines by keywords. Matches against routine name and description.

        Args:
            task: Task description to search for.
        """
        url = f"{Config.VECTORLY_API_BASE}/routines/semantic-search"
        headers = self._get_auth_headers()
        payload = {
            "query": task,
            "top_n": 5,
            "threshold": 0.0,
            "keywords": [],
        }

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
            self._cache_routines_from_response(data)
            return data
        except requests.RequestException as e:
            logger.error("Routine search failed: %s", e)
            return {"error": f"Routine search failed: {e}", "results": []}

    @agent_tool()
    def _execute_routines_in_parallel(
        self,
        routine_executions: list[RoutineExecutionRequest],
    ) -> dict[str, Any]:
        """
        Execute one or more routines in parallel via the Vectorly API.

        Args:
            routine_executions: List of routines to execute. Each item needs routine_id and parameters. Parameters is a dict keyed by parameter name (as shown in search_routines results) with the corresponding value, e.g. {"origin": "New York", "date": "2025-03-01"}. All required parameters listed in the routine's parameter definitions must be provided.
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

        headers = self._get_auth_headers()

        def save_result(result: dict[str, Any]) -> dict[str, Any]:
            """Save a single routine result to a JSON file in raw/."""
            try:
                idx = next(self._routine_execution_counter)
                ts = datetime.now().strftime("%y-%m-%d-%H%M%S")
                ref = self._workspace.save_artifact(
                    "raw",
                    f"{ts}-routine_result_{idx}.json",
                    json.dumps(result, indent=2, default=str),
                )
                result.update(
                    {
                        "output_file": str(self._workspace.root_path / ref.relative_path),
                        "artifact_id": ref.artifact_id,
                    },
                )
            except Exception as e:
                logger.exception("Failed to save routine result to file: %s", e)
                result["output_file_error"] = str(e)
            return result

        def _summarize_result(full_result: dict[str, Any], req: RoutineExecutionRequest) -> dict[str, Any]:
            """Build a compact summary for the agent with a 4K char preview."""
            cached = self._routine_cache.get(req.routine_id)
            is_error = "error" in full_result and "execution_id" not in full_result
            routine_name = (
                full_result.get("routine_name")
                or (cached.name if cached else None)
                or req.routine_id
            )
            summary: dict[str, Any] = {
                "success": not is_error,
                "routine_name": routine_name,
                "routine_id": req.routine_id,
                "parameters": req.parameters,
                "output_file": full_result.get("output_file"),
            }
            if is_error:
                summary["error"] = full_result.get("error")
                return summary
            raw = json.dumps(full_result, indent=2, default=str)
            max_preview = 4000
            if len(raw) > max_preview:
                summary["response_preview"] = raw[:max_preview]
                summary["response_truncated"] = True
                summary["response_total_chars"] = len(raw)
                summary["_hint"] = (
                    f"Response truncated ({len(raw)} chars). "
                    f"Full result saved to {full_result.get('output_file')}. "
                    "Use read_file(scope='workspace', path='...') to inspect the full data, or execute_python to parse it."
                )
            else:
                summary["response_preview"] = raw
            return summary

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
                full_result = save_result(response.json())
                return _summarize_result(full_result, req)
            except requests.RequestException as e:
                logger.error("Routine execution failed for %s: %s", req.routine_id, e)
                full_result = save_result({"error": str(e), "routine_id": req.routine_id})
                return _summarize_result(full_result, req)

        total = len(routine_executions)
        results: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(execute_one, req): req for req in routine_executions}
            for future in as_completed(futures):
                result = future.result()
                results.append(result)

                status = "succeeded" if result.get("success") else "FAILED"
                routine_id = result.get("routine_id", "")
                cached = self._routine_cache.get(routine_id)
                label = cached.name if cached else routine_id
                self._emit_message(StatusUpdateEmittedMessage(
                    content=f"[{len(results)}/{total}] Routine '{label}' {status}.",
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
    def _execute_browser_task(
        self,
        task: str,
    ) -> dict[str, Any]:
        """
        Execute a free-form browser task using the AI browser agent.

        Use this as a FALLBACK when no pre-built routine matches the user's request.
        The browser agent receives a natural language task and autonomously navigates
        the web to complete it. This is slower and more expensive than routines.
        Progress is streamed in real time via SSE.

        Args:
            task: Detailed natural language instruction for the browser agent. Be specific and step-by-step.
                Example: "Go to google.com, search for 'best flight deals NYC to LA March 2026',
                click the first result, and extract the price and airline name."
        """
        if not task or not task.strip():
            return {"error": "Task description cannot be empty"}

        headers = self._get_auth_headers()

        payload = {
            "task": task,
            "timeout_seconds": 300,
            "use_vision": True,
        }

        self._emit_message(StatusUpdateEmittedMessage(
            content="Starting browser agent task. This may take a few minutes...",
        ))

        try:
            with requests.post(
                f"{Config.VECTORLY_API_BASE}/browser-agent/execute/stream",
                headers=headers,
                json=payload,
                stream=True,
                timeout=330,
            ) as response:
                if response.status_code == 402:
                    self._emit_message(ErrorEmittedMessage(
                        error="Insufficient credits. Please add credits or a payment method to continue. Billing: https://console.vectorly.app/billing",
                        code="INSUFFICIENT_CREDITS",
                    ))
                    return {
                        "error": "Insufficient credits. Please add credits or a payment method to continue. Billing: https://console.vectorly.app/billing",
                        "code": "INSUFFICIENT_CREDITS",
                    }
                response.raise_for_status()
                result = self._consume_sse_stream(response)

        except requests.Timeout:
            return {"error": "Browser agent timed out after 300s"}
        except requests.RequestException as e:
            logger.error("Browser agent API call failed: %s", e)
            return {"error": f"Browser agent request failed: {e}"}

        # Save final_result as a markdown file in output/
        final_result = result.get("final_result")
        if final_result:
            try:
                ts = datetime.now().strftime("%y-%m-%d-%H%M%S")
                ref = self._workspace.save_artifact(
                    "output",
                    f"{ts}-browser_agent.md",
                    final_result,
                )
                result.update(
                    {
                        "output_file": str(self._workspace.root_path / ref.relative_path),
                        "artifact_id": ref.artifact_id,
                    },
                )
            except Exception as e:
                logger.exception("Failed to save browser agent result: %s", e)
                result["output_file_error"] = str(e)

        return result

    def _consume_sse_stream(self, response: requests.Response) -> dict[str, Any]:
        """Parse an SSE stream from the browser agent and emit progress messages."""
        result: dict[str, Any] = {"error": "Stream ended without a terminal event"}
        step_counter = 0
        steps: list[dict[str, Any]] = []

        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue

            try:
                data = json.loads(line[6:])
            except json.JSONDecodeError:
                logger.warning("Malformed SSE data line: %s", line)
                continue

            try:
                event = sse_event_adapter.validate_python(data)
            except Exception as e:
                logger.warning("Unknown or invalid SSE event (type=%s): %s", data.get("type"), e)
                continue

            if isinstance(event, BrowserAgentStepEvent):
                step_counter += 1
                if step_counter > 1:
                    self._emit_message(StatusUpdateEmittedMessage(content=""))
                msg = f"[Step {step_counter}]"
                if event.next_goal:
                    msg += f" {event.next_goal}"
                self._emit_message(BrowserAgentStepEmittedMessage(
                    content=msg,
                    step_number=step_counter,
                    goal=event.next_goal,
                ))
                steps.append({"step": step_counter, "goal": event.next_goal, "is_done": event.is_done})

            elif isinstance(event, BrowserAgentDoneEvent):
                status = "succeeded" if event.is_successful else "completed (not confirmed successful)"
                if not event.is_done:
                    status = "did not finish"
                done_msg = f"Browser agent task {status} in {event.duration_seconds or 0:.1f}s ({event.n_steps} steps)."
                self._emit_message(StatusUpdateEmittedMessage(content=done_msg))
                result = {
                    "success": event.is_successful or False,
                    "is_done": event.is_done,
                    "final_result": event.final_result,
                    "errors": event.errors,
                    "n_steps": event.n_steps,
                    "duration_seconds": event.duration_seconds,
                    "execution_id": event.execution_id,
                    "steps": data.get("steps_detail", steps),  # prefer detailed steps from server if available
                    "prompt_tokens": event.prompt_tokens,
                    "completion_tokens": event.completion_tokens,
                    "total_tokens": event.total_tokens,
                }

            elif isinstance(event, BrowserAgentErrorEvent):
                error_msg = f"Browser agent error: {event.error}"
                self._emit_message(StatusUpdateEmittedMessage(content=error_msg))
                result = {"error": event.error, "execution_id": event.execution_id, "steps": steps}

        return result

    ## Context generation (structured output, called by TUI slash command)

    def generate_context(self, focus: str | None = None) -> BlueBoxAgentContext:
        """Generate a context file from the current session using structured output.

        Makes a direct LLM call with response_model=BlueBoxAgentContext to get
        a validated Pydantic model back. Saves both JSON and Markdown files to
        the workspace context/ directory.

        Args:
            focus: Optional user-provided focus prompt to guide context generation.

        Returns:
            The validated BlueBoxAgentContext.

        Raises:
            ValueError: If the LLM fails to produce a valid context.
        """
        raw_routines = self._extract_routines_from_raw()

        system_prompt = (
            "You are analyzing a BlueBox Agent conversation to extract a reusable context file. "
            "Fill in every field of the BlueBoxAgentContext schema based on the conversation.\n\n"
            "CRITICAL: routines_used must include every routine that was executed with exact "
            "routine_id, routine_name, and parameter values.\n"
            "Include the final working python_code snippet if post-processing was done.\n"
            "Include output_files with relative paths of files written to output/.\n"
        )
        if raw_routines:
            system_prompt += "\nRoutines found in execution results:\n"
            for r in raw_routines:
                system_prompt += f"- {r.routine_name} ({r.routine_id}): {json.dumps(r.parameters_as_dict(), default=str)}\n"
        if focus:
            system_prompt += f"\nUser focus: {focus}\n"

        # One-off structured output call that sees the full conversation via
        # OpenAI's response chaining (previous_response_id reconstructs the
        # thread server-side). We don't update self._previous_response_id
        # afterward so this call doesn't affect the agent loop.
        response = self.llm_client.call_sync(
            input="Generate a reusable context file from this conversation.",
            system_prompt=system_prompt,
            response_model=BlueBoxAgentContext,
            previous_response_id=self._previous_response_id,
        )
        context = response.parsed
        if context is None:
            raise ValueError("LLM failed to produce a valid BlueBoxAgentContext")

        # Safety net: merge raw routines if LLM left routines_used empty
        if not context.routines_used and raw_routines:
            context.routines_used = raw_routines
            logger.info(
                "Auto-populated %d routine(s) from raw/ execution results",
                len(raw_routines),
            )

        # Save canonical JSON
        json_ref = self._workspace.save_artifact(
            "context",
            "agent_context.json",
            context.model_dump_json(indent=2),
        )

        # Save companion Markdown
        md_ref = self._workspace.save_artifact(
            "context",
            "agent_context.md",
            context.to_markdown(),
        )

        logger.info(
            "Context files saved: %s, %s",
            self._workspace.root_path / json_ref.relative_path,
            self._workspace.root_path / md_ref.relative_path,
        )
        return context
