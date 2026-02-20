"""
bluebox/agents/principal_investigator.py

PrincipalInvestigator (PI) agent — the orchestrator for Phase 2: Experiment-Driven
Routine Construction.

The PI has NO browser and NO domain tools. It only:
- Reads exploration summaries (in its system prompt)
- Reads the Discovery Ledger (routines planned, experiments, proven artifacts)
- Plans what routines to build from the exploration data
- Creates experiment tasks with specific hypotheses
- Records findings and proven artifacts
- Assembles routines and submits them for inspection
- Ships a catalog of routines when done
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from textwrap import dedent
from typing import Any, Callable, TYPE_CHECKING

from pydantic import BaseModel

from bluebox.agents.abstract_agent import AbstractAgent, AgentCard, agent_tool
from bluebox.agents.routine_inspector import INSPECTION_OUTPUT_SCHEMA, RoutineInspector
from bluebox.agents.specialists.abstract_specialist import AbstractSpecialist, AutonomousConfig, RunMode
from bluebox.agents.workers.experiment_worker import ExperimentWorker
from bluebox.data_models.llms.interaction import (
    Chat,
    ChatRole,
    ChatThread,
    EmittedMessage,
)
from bluebox.data_models.llms.vendors import LLMModel, OpenAIModel
from bluebox.data_models.orchestration.experiment import (
    ArtifactType,
    ExperimentEntry,
    ExperimentStatus,
    ExperimentVerdict,
)
from bluebox.data_models.orchestration.ledger import (
    DiscoveryLedger,
    RoutineAttempt,
    RoutineAttemptStatus,
    RoutineCatalog,
    RoutineSpec,
    RoutineSpecStatus,
    ShippedRoutine,
)
from bluebox.data_models.orchestration.task import (
    SubAgent,
    Task,
    TaskStatus,
    SpecialistAgentType,
)
from bluebox.data_models.orchestration.state import AgentOrchestrationState
from bluebox.data_models.routine.execution import RoutineExecutionResultWithMetadata
from bluebox.data_models.routine.routine import Routine
from bluebox.llms.data_loaders.dom_data_loader import DOMDataLoader
from bluebox.llms.data_loaders.network_data_loader import NetworkDataLoader
from bluebox.llms.data_loaders.storage_data_loader import StorageDataLoader
from bluebox.llms.data_loaders.window_property_data_loader import WindowPropertyDataLoader
from bluebox.utils.logger import get_logger

if TYPE_CHECKING:
    from bluebox.llms.data_loaders.documentation_data_loader import DocumentationDataLoader
    from websocket import WebSocket

logger = get_logger(name=__name__)


# ---------------------------------------------------------------------------
# Worker tool descriptions — injected into PI system prompt so the PI knows
# what workers can do and references tools by name in experiment prompts.
# ---------------------------------------------------------------------------

WORKER_CAPABILITIES = dedent("""\
    ## Worker Capabilities

    Workers have access to the following tools. When writing experiment prompts,
    reference these tools by name so the worker knows exactly what to use.

    BROWSER TOOLS (act in the live browser):
      browser_navigate(url) — go to a URL and wait for page load.
        TIP: Navigating directly to an API URL (e.g. https://api.example.com/data)
        bypasses CORS restrictions since it's a top-level navigation, not a fetch.
        The worker can then read the page body to get the JSON response.
      browser_eval_js(expression) — run JavaScript in the page context.
        Use for fetch() calls, DOM reads, clicks, storage access.
        If fetch() fails with CORS, try: mode 'no-cors', or navigate to the URL first.
      browser_cdp_command(method, params) — raw Chrome DevTools Protocol command.
        POWERFUL: Can intercept/modify network requests below the browser security layer.
        Key CDP methods for bypassing CORS/auth issues:
          - Fetch.enable + Fetch.continueRequest: intercept and modify requests
          - Network.enable + Network.getResponseBody: capture responses at protocol level
          - Network.setExtraHTTPHeaders: add headers to all requests
          - Page.navigate: navigate and capture response at CDP level
        Workers should use this when browser_eval_js fetch() fails due to CORS.
      browser_get_dom(selector?, max_depth?, include_tags?) — filtered view of current DOM

    CAPTURE LOOKUP TOOLS (search RECORDED session data — old, potentially stale):
      capture_search_transactions(query) — find requests in the recorded capture
      capture_get_transaction(request_id) — get full recorded request/response details
        USE THIS FIRST when an API call fails — it shows the exact headers, cookies,
        and parameters that worked during the original recorded session.
      capture_search_storage(query) — find recorded storage events
      capture_trace_value(value) — find where a value appears across the recorded capture
      capture_get_page_structure(snapshot_index?) — get recorded DOM structure
      capture_get_element(element_type, snapshot_index?) — get recorded element details

    Workers also receive the exploration summaries as shared context.
""")


class PrincipalInvestigator(AbstractAgent):
    """
    Orchestrator agent for experiment-driven routine construction.

    The PI reads exploration summaries, plans a catalog of routines,
    dispatches experiments to ExperimentWorker agents, reviews results,
    and assembles proven artifacts into shipped routines.

    The PI is self-organizing — no strict phases. It decides what to work on,
    when to switch routines, and when to call it done.
    """

    AGENT_CARD = AgentCard(
        description=(
            "Orchestrates experiment-driven routine construction. Reads exploration "
            "summaries, plans a routine catalog, dispatches experiments to workers, "
            "reviews results, and assembles proven artifacts into shipped routines."
        ),
    )

    # -----------------------------------------------------------------------
    # System prompts
    # -----------------------------------------------------------------------

    SYSTEM_PROMPT_CORE: str = dedent("""\
        You are a Principal Investigator (PI) for automated routine construction.

        ## Your Role

        You are the strategist. You have NO browser and NO domain tools. You:
        1. Read exploration summaries to understand the captured session
        2. Plan what routines to build (a catalog — often multiple routines)
        3. Design experiments with specific, falsifiable hypotheses
        4. Dispatch experiments to workers who have browser + capture lookup tools
        5. Review results and record verdicts
        6. Accumulate proven artifacts (fetches, navigations, tokens, parameters)
        7. Assemble routines and submit them for validation + inspection
        8. Ship routines that pass, iterate on ones that fail
        9. Call mark_complete when all routines are addressed

        ## Catalog-First Thinking

        Your FIRST job after reading exploration summaries is to call plan_routines
        to declare what routines you intend to build. The exploration data reveals
        the API surface — each distinct capability can be a routine.

        Then work through routines by priority, starting with shared dependencies
        (auth, navigation) that apply across routines.

        ## How Experiments Work

        Each experiment has:
        - **hypothesis**: What you're testing (specific and falsifiable)
        - **rationale**: WHY — what evidence led here, what you expect to learn
        - **prompt**: Instructions for the worker (reference worker tools by name!)

        The worker executes the experiment and returns structured output.
        You review the output, record a verdict, and decide what to try next.

        ## Strategy

        1. Call plan_routines early to declare your catalog plan
        2. Start with shared experiments: auth, base navigation, tokens
        3. For each routine: experiment → prove → assemble → submit_routine → ship or fix
        4. Use follow_up to ask the SAME worker clarifying questions (preserves context)
        5. submit_routine REQUIRES test_parameters — provide realistic values for EVERY parameter!
           The routine will be executed in a live browser and reviewed by an independent inspector.
        6. When a routine passes inspection: mark_routine_shipped
        7. When a routine is hopeless: mark_routine_failed
        8. When all routines are addressed: mark_complete with a usage guide

        ## Parallel Experiments

        Use dispatch_experiments_batch to run multiple INDEPENDENT experiments at the
        same time on separate workers. This is much faster than sequential dispatch.

        Good candidates for parallel dispatch:
        - Testing multiple API endpoints that don't depend on each other
        - Probing different auth strategies simultaneously
        - Validating multiple routines' core fetches at once

        Each batch experiment gets its own worker and browser tab. Use dispatch_experiment
        (singular) only when you need to follow_up on a specific worker's context.

        ## CRITICAL RULES

        - NEVER guess at request details. Always dispatch experiments to verify.
        - Write experiment prompts that reference worker tools by name.
        - Record a verdict for EVERY completed experiment via record_finding.
        - If an experiment is ambiguous, use follow_up — don't dispatch a new one.
        - ALWAYS provide test_parameters when calling submit_routine — the routine
          WILL be executed and inspected. Use realistic values the experiments proved work.
        - Prioritize: auth → main API calls → parameters → assembly.

        ## Resilience — NEVER Give Up Early

        - NEVER call mark_failed after fewer than 5 experiments per routine.
          CORS failures, 400 errors, and network issues are NORMAL obstacles, not
          reasons to quit. They mean you need a different approach, not that the
          pipeline is hopeless.
        - When a fetch fails (CORS, 400, timeout), iterate with alternative approaches:
          1. Use capture_search_transactions / capture_get_transaction to see the EXACT
             request headers and patterns that worked in the recorded session, then
             replicate them in the worker's experiment.
          2. Use browser_cdp_command with Fetch.enable to intercept requests at the CDP
             level — this bypasses CORS entirely since it operates below the browser
             security layer.
          3. Try navigating directly to the API URL with browser_navigate — GET requests
             via top-level navigation don't have CORS restrictions.
          4. Try fetch with mode: 'no-cors' or from a different origin context.
          5. Check if the site's JS uses a proxy path (e.g. /api/* proxied to the API
             domain) — search the captured network data for path patterns.
        - If ALL alternative approaches fail for a routine, mark_routine_failed for THAT
          routine and move on to the next one. Do NOT call mark_failed (pipeline-level)
          unless every single routine has been individually addressed.
        - Use follow_up liberally — it's cheaper than dispatching a new experiment and
          preserves the worker's browser state and context.
    """)

    # -----------------------------------------------------------------------
    # Constructor
    # -----------------------------------------------------------------------

    def __init__(
        self,
        emit_message_callable: Callable[[EmittedMessage], None],
        task: str,
        # Exploration summaries — injected into system prompt
        exploration_summaries: dict[str, str] | None = None,
        # Data loaders — passed through to workers
        network_data_loader: NetworkDataLoader | None = None,
        storage_data_loader: StorageDataLoader | None = None,
        dom_data_loader: DOMDataLoader | None = None,
        window_property_data_loader: WindowPropertyDataLoader | None = None,
        documentation_data_loader: DocumentationDataLoader | None = None,
        # Browser context — passed through to workers
        remote_debugging_address: str | None = None,
        # Resume support — pass an existing ledger to pick up where a previous PI left off
        ledger: DiscoveryLedger | None = None,
        # LLM config
        llm_model: LLMModel = OpenAIModel.GPT_5_1,
        worker_llm_model: LLMModel | None = None,
        max_iterations: int = 200,
        worker_max_loops: int = 10,
        max_attempts_per_routine: int = 5,
        # Agent pool sizes
        num_workers: int = 3,
        num_inspectors: int = 1,
        # Persistence callbacks
        on_ledger_change: Callable[[DiscoveryLedger, str], None] | None = None,
        on_agent_thread: Callable[[str, str, list[dict[str, Any]]], None] | None = None,
        # Standard agent args
        persist_chat_callable: Callable[[Chat], Chat] | None = None,
        persist_chat_thread_callable: Callable[[ChatThread], ChatThread] | None = None,
        stream_chunk_callable: Callable[[str], None] | None = None,
        chat_thread: ChatThread | None = None,
        existing_chats: list[Chat] | None = None,
    ) -> None:
        # Task
        self._task = task
        self._max_iterations = max_iterations
        self._worker_max_loops = worker_max_loops
        self._max_attempts_per_routine = max_attempts_per_routine

        # Exploration context
        self._exploration_summaries = exploration_summaries or {}

        # Data loaders (passed through to workers)
        self._network_data_loader = network_data_loader
        self._storage_data_loader = storage_data_loader
        self._dom_data_loader = dom_data_loader
        self._window_property_data_loader = window_property_data_loader
        self._documentation_data_loader = documentation_data_loader

        # Browser context (passed through to workers)
        self._remote_debugging_address = remote_debugging_address

        # LLM
        self._worker_llm_model = worker_llm_model or llm_model

        # Agent pools
        self._num_workers = num_workers
        self._num_inspectors = num_inspectors
        self._worker_counter = 0  # Round-robin counter for workers
        self._inspector_counter = 0  # Round-robin counter for inspectors

        # Persistence callbacks
        self._on_ledger_change = on_ledger_change
        self._on_agent_thread = on_agent_thread

        # Internal state — the Discovery Ledger tracks everything
        # Accept an existing ledger for resume after context exhaustion
        self._ledger = ledger or DiscoveryLedger(user_task=task)
        self._orchestration_state = AgentOrchestrationState()
        self._agent_instances: dict[str, AbstractSpecialist] = {}
        self._is_done = False
        self._pipeline_result: RoutineCatalog | None = None
        self._recent_tool_calls: list[str] = []  # Track recent tool names for loop detection

        super().__init__(
            emit_message_callable=emit_message_callable,
            persist_chat_callable=persist_chat_callable,
            persist_chat_thread_callable=persist_chat_thread_callable,
            stream_chunk_callable=stream_chunk_callable,
            llm_model=llm_model,
            chat_thread=chat_thread,
            existing_chats=existing_chats,
            documentation_data_loader=documentation_data_loader,
        )

        logger.debug(
            "PrincipalInvestigator initialized: task=%s, explorations=%s",
            task[:80],
            list(self._exploration_summaries.keys()),
        )

    # -----------------------------------------------------------------------
    # System prompt
    # -----------------------------------------------------------------------

    def _get_system_prompt(self) -> str:
        parts: list[str] = [self.SYSTEM_PROMPT_CORE]

        # Worker capabilities
        parts.append(WORKER_CAPABILITIES)

        # Exploration summaries
        if self._exploration_summaries:
            parts.append("\n## Exploration Summaries\n")
            for domain, summary in self._exploration_summaries.items():
                parts.append(f"### {domain}\n{summary}\n")

        # Discovery Ledger
        ledger_summary = self._ledger.to_summary()
        if ledger_summary != "(no activity yet)":
            parts.append(f"\n## Discovery Ledger\n\n{ledger_summary}")

        # Task queue status
        queue = self._orchestration_state.get_queue_status()
        if any(v > 0 for v in queue.values()):
            parts.append(f"\n## Task Queue\n{json.dumps(queue)}")

        return "".join(parts)

    # -----------------------------------------------------------------------
    # Public entry point
    # -----------------------------------------------------------------------

    def run(self) -> RoutineCatalog | None:
        """
        Run the PI loop to completion.

        Returns:
            A RoutineCatalog of shipped routines, or None if construction failed.
        """
        # Seed the conversation — detect resume vs fresh start
        is_resume = bool(self._ledger.experiments or self._ledger.routine_specs)

        if is_resume:
            initial_message = (
                f"TASK: {self._task}\n\n"
                "You are RESUMING a previous session that ran out of context. "
                "Your Discovery Ledger has been preserved with all prior work.\n\n"
                "FIRST: Call get_ledger to see exactly where things stand — "
                "what routines are planned, what experiments have been run, "
                "what's shipped, and what still needs work.\n\n"
                "Then pick up where the previous session left off. Do NOT repeat "
                "experiments that already have verdicts."
            )
            logger.info(
                "PI resuming: %d specs, %d experiments, %d attempts",
                len(self._ledger.routine_specs),
                len(self._ledger.experiments),
                len(self._ledger.attempts),
            )
        else:
            initial_message = (
                f"TASK: {self._task}\n\n"
                "Start by analyzing the exploration summaries. Then:\n"
                "1. Call plan_routines to declare what routines to build\n"
                "2. Dispatch experiments for shared dependencies (auth, navigation)\n"
                "3. Work through routines by priority\n"
                "4. Call mark_complete when all routines are addressed\n\n"
                "Use dispatch_experiment to send experiments to workers, then "
                "record_finding to log what you learned."
            )
        self._add_chat(ChatRole.USER, initial_message)

        for iteration in range(self._max_iterations):
            if self._is_done:
                logger.info("PI completed after %d iterations", iteration)
                self._dump_agent_thread("principal_investigator", self)
                return self._pipeline_result

            messages = self._build_messages_for_llm()
            response = self._call_llm(
                messages,
                self._get_system_prompt(),
                tool_choice="required",
            )

            # Add assistant response to chat
            self._add_chat(
                ChatRole.ASSISTANT,
                content=response.content or "",
                tool_calls=response.tool_calls,
                llm_provider_response_id=response.response_id,
            )

            if response.tool_calls:
                self._process_tool_calls(response.tool_calls)

                # Loop detection: track recent tool calls
                for tc in response.tool_calls:
                    self._recent_tool_calls.append(tc.tool_name)
                # Keep only last 5
                self._recent_tool_calls = self._recent_tool_calls[-5:]

                # If last 3+ calls are the same non-productive tool, nudge
                if (
                    len(self._recent_tool_calls) >= 3
                    and len(set(self._recent_tool_calls[-3:])) == 1
                    and self._recent_tool_calls[-1] in ("set_active_routine", "get_ledger", "get_experiment_result")
                ):
                    stuck_tool = self._recent_tool_calls[-1]
                    self._recent_tool_calls.clear()
                    shipped = sum(1 for s in self._ledger.routine_specs if s.status == RoutineSpecStatus.SHIPPED)
                    total_attempts = len(self._ledger.attempts)
                    self._add_chat(
                        ChatRole.USER,
                        f"You appear stuck calling {stuck_tool} repeatedly. "
                        f"Current progress: {shipped} shipped, {total_attempts} attempts. "
                        "To make progress you MUST either:\n"
                        "1. dispatch_experiment or dispatch_experiments_batch to run more experiments\n"
                        "2. submit_routine with a COMPLETE routine_json dict AND test_parameters to create an attempt\n"
                        "3. mark_routine_failed for routines you've proven can't work (requires 2+ experiments)\n"
                        "Pick the next actionable step and execute it NOW.",
                    )
            else:
                # Nudge the PI to act
                self._add_chat(
                    ChatRole.USER,
                    "You must use a tool. Dispatch an experiment, record a finding, "
                    "submit a routine, or mark_complete if done.",
                )

        logger.warning("PI exhausted %d iterations without completing", self._max_iterations)
        # Dump PI thread before returning partial results
        self._dump_agent_thread("principal_investigator", self)
        # Return whatever we've shipped so far
        return self._build_partial_catalog()

    # ===================================================================
    # Persistence — notify external listener after ledger mutations
    # ===================================================================

    def _persist(self, reason: str) -> None:
        """Fire the on_ledger_change callback if registered."""
        if self._on_ledger_change is not None:
            try:
                self._on_ledger_change(self._ledger, reason)
            except Exception as e:
                logger.warning("on_ledger_change callback failed: %s", e)

    def _dump_agent_thread(self, agent_label: str, agent: AbstractAgent | AbstractSpecialist) -> None:
        """Dump an agent's full message history via the on_agent_thread callback."""
        if self._on_agent_thread is None:
            return
        try:
            chats = agent.get_chats()
            messages = [
                {
                    "id": c.id,
                    "role": c.role.value,
                    "content": c.content,
                    "tool_calls": [
                        {"tool_name": tc.tool_name, "arguments": tc.tool_arguments, "call_id": tc.call_id}
                        for tc in c.tool_calls
                    ] if c.tool_calls else [],
                    "tool_call_id": c.tool_call_id,
                    "created_at": str(c.created_at) if c.created_at else None,
                }
                for c in chats
            ]
            thread_id = agent.get_thread().id if agent.get_thread() else "unknown"
            self._on_agent_thread(agent_label, thread_id, messages)
        except Exception as e:
            logger.warning("on_agent_thread callback failed for %s: %s", agent_label, e)

    # ===================================================================
    # CATALOG PLANNING TOOLS
    # ===================================================================

    @agent_tool()
    def _plan_routines(
        self,
        specs: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Declare what routines to build from the exploration data.

        Call this early after analyzing exploration summaries. Each spec
        represents a distinct capability to extract from the site.
        Can be called again to add new specs discovered during experimentation.

        Args:
            specs: List of routine specs. Each dict has:
                - name: Short name (e.g. "get_league_standings")
                - description: What the routine does
                - priority: 1=must-have, 2=should-have, 3=nice-to-have (default 1)
        """
        created_ids: list[str] = []
        for spec_dict in specs:
            spec = RoutineSpec(
                name=spec_dict.get("name") or spec_dict.get("id", "unnamed"),
                description=spec_dict.get("description", ""),
                priority=spec_dict.get("priority", 1),
            )
            self._ledger.add_spec(spec)
            created_ids.append(spec.id)

        # Auto-set the first one as active if none is active
        if self._ledger.active_spec_id is None and self._ledger.routine_specs:
            self._ledger.active_spec_id = self._ledger.routine_specs[0].id

        self._persist("plan_routines")
        return {
            "created": len(created_ids),
            "spec_ids": created_ids,
            "total_specs": len(self._ledger.routine_specs),
        }

    @agent_tool()
    def _set_active_routine(self, spec_id: str) -> dict[str, Any]:
        """
        Switch focus to a different routine. The PI works on one routine at a
        time but can switch when blocked or when dependencies are shared.

        Args:
            spec_id: ID of the RoutineSpec to focus on.
        """
        spec = self._ledger.get_spec(spec_id)
        if spec is None:
            return {"error": f"No spec found with ID: {spec_id}"}

        self._ledger.active_spec_id = spec_id
        return {"active": spec.name, "status": spec.status.value}

    # ===================================================================
    # EXPERIMENT TOOLS
    # ===================================================================

    @agent_tool()
    def _dispatch_experiment(
        self,
        hypothesis: str,
        rationale: str,
        prompt: str,
        routine_spec_id: str | None = None,
        output_schema: dict[str, Any] | None = None,
        output_description: str | None = None,
        priority: int = 1,
    ) -> dict[str, Any]:
        """
        Create and dispatch an experiment to a worker.

        The worker has browser tools and capture lookup tools. Write the prompt
        so the worker knows exactly what to do — reference tools by name.

        Args:
            hypothesis: What we're testing. Specific and falsifiable.
            rationale: WHY we're testing this — evidence, reasoning, expectations.
            prompt: Instructions for the worker. Reference worker tools by name.
            routine_spec_id: Which routine this experiment is for (None = shared/auth).
            output_schema: Optional JSON schema for the worker's structured answer.
            output_description: Description of expected output.
            priority: 1=critical, 2=important, 3=nice-to-have.
        """
        # Create experiment entry
        experiment = ExperimentEntry(
            hypothesis=hypothesis,
            rationale=rationale,
            prompt=prompt,
            priority=priority,
            status=ExperimentStatus.RUNNING,
        )
        self._ledger.add_experiment(experiment)

        # Link to routine spec if provided
        spec_id = routine_spec_id or self._ledger.active_spec_id
        if spec_id:
            spec = self._ledger.get_spec(spec_id)
            if spec:
                spec.experiment_ids.append(experiment.id)
                if spec.status == RoutineSpecStatus.PLANNED:
                    spec.status = RoutineSpecStatus.EXPERIMENTING

        # Create and dispatch task
        task = Task(
            agent_type=SpecialistAgentType.EXPERIMENT_WORKER,
            prompt=prompt,
            max_loops=self._worker_max_loops,
            output_schema=output_schema,
            output_description=output_description,
        )
        self._orchestration_state.add_task(task)
        experiment.task_id = task.id

        # Execute immediately
        result = self._execute_task(task)

        # Update experiment status from task
        if task.status == TaskStatus.COMPLETED:
            experiment.status = ExperimentStatus.DONE
            experiment.output = task.result
        elif task.status == TaskStatus.FAILED:
            experiment.status = ExperimentStatus.FAILED
            experiment.output = {"error": task.error}
        elif task.status == TaskStatus.PAUSED:
            experiment.status = ExperimentStatus.RUNNING

        self._persist(f"experiment_{experiment.id}")
        return {
            "experiment_id": experiment.id,
            "task_id": task.id,
            "status": experiment.status.value,
            "result": result,
        }

    @agent_tool()
    def _dispatch_experiments_batch(
        self,
        experiments: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Dispatch multiple experiments IN PARALLEL to separate workers.

        Each experiment runs on its own worker with its own browser tab,
        all executing concurrently. Use this when you have independent
        experiments that don't depend on each other's results.

        Much faster than calling dispatch_experiment sequentially — N experiments
        run in roughly the time of 1.

        Args:
            experiments: List of experiment dicts, each with:
                - hypothesis: What we're testing (specific and falsifiable)
                - rationale: WHY we're testing this
                - prompt: Instructions for the worker (reference tools by name!)
                - routine_spec_id: (optional) Which routine this is for
                - output_schema: (optional) JSON schema for structured output
                - output_description: (optional) Description of expected output
                - priority: (optional) 1=critical, 2=important, 3=nice-to-have
        """
        if not experiments:
            return {"error": "No experiments provided"}

        # Cap at num_workers to avoid overwhelming the system
        max_parallel = self._num_workers
        if len(experiments) > max_parallel:
            logger.warning(
                "Batch of %d experiments exceeds worker pool (%d), running first %d",
                len(experiments), max_parallel, max_parallel,
            )
            experiments = experiments[:max_parallel]

        # Phase 1: Create all experiment entries and tasks (sequential — fast, no I/O)
        task_experiment_pairs: list[tuple[Task, ExperimentEntry]] = []
        for exp_dict in experiments:
            experiment = ExperimentEntry(
                hypothesis=exp_dict.get("hypothesis", ""),
                rationale=exp_dict.get("rationale", ""),
                prompt=exp_dict.get("prompt", ""),
                priority=exp_dict.get("priority", 1),
                status=ExperimentStatus.RUNNING,
            )
            self._ledger.add_experiment(experiment)

            # Link to routine spec
            spec_id = exp_dict.get("routine_spec_id") or self._ledger.active_spec_id
            if spec_id:
                spec = self._ledger.get_spec(spec_id)
                if spec:
                    spec.experiment_ids.append(experiment.id)
                    if spec.status == RoutineSpecStatus.PLANNED:
                        spec.status = RoutineSpecStatus.EXPERIMENTING

            task = Task(
                agent_type=SpecialistAgentType.EXPERIMENT_WORKER,
                prompt=exp_dict.get("prompt", ""),
                max_loops=self._worker_max_loops,
                output_schema=exp_dict.get("output_schema"),
                output_description=exp_dict.get("output_description"),
            )
            self._orchestration_state.add_task(task)
            experiment.task_id = task.id
            task_experiment_pairs.append((task, experiment))

        self._persist("batch_dispatched")

        # Phase 2: Execute all tasks in parallel using ThreadPoolExecutor
        results: list[dict[str, Any]] = []

        def _run_one(pair: tuple[Task, ExperimentEntry]) -> dict[str, Any]:
            task, experiment = pair
            # Create a dedicated worker for this parallel task
            worker = self._create_worker()
            subagent = SubAgent(
                type=task.agent_type,
                llm_model=self._worker_llm_model.value,
            )
            self._orchestration_state.subagents[subagent.id] = subagent
            self._agent_instances[subagent.id] = worker
            task.agent_id = subagent.id
            subagent.task_ids.append(task.id)

            try:
                task.status = TaskStatus.IN_PROGRESS
                task.started_at = datetime.now()

                config = AutonomousConfig(
                    min_iterations=1,
                    max_iterations=task.max_loops,
                )
                result = worker.run_autonomous(
                    task=task.prompt,
                    config=config,
                    output_schema=task.output_schema,
                    output_description=task.output_description,
                )
                task.loops_used += worker.autonomous_iteration
                self._dump_agent_thread(f"worker_{subagent.id}", worker)

                if result is not None:
                    task.status = TaskStatus.COMPLETED
                    task.completed_at = datetime.now()
                    task.result = result.model_dump() if isinstance(result, BaseModel) else result
                    experiment.status = ExperimentStatus.DONE
                    experiment.output = task.result
                else:
                    task.status = TaskStatus.FAILED
                    task.error = "Max loops reached without result"
                    experiment.status = ExperimentStatus.FAILED
                    experiment.output = {"error": task.error}
            except Exception as e:
                task.status = TaskStatus.FAILED
                task.error = str(e)
                task.completed_at = datetime.now()
                experiment.status = ExperimentStatus.FAILED
                experiment.output = {"error": str(e)}
                logger.error("Parallel task %s failed: %s", task.id, e)
            finally:
                worker.close()

            return {
                "experiment_id": experiment.id,
                "hypothesis": experiment.hypothesis[:100],
                "status": experiment.status.value,
                "result_preview": str(experiment.output)[:300] if experiment.output else None,
            }

        with ThreadPoolExecutor(max_workers=max_parallel) as pool:
            futures = {
                pool.submit(_run_one, pair): pair
                for pair in task_experiment_pairs
            }
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    pair = futures[future]
                    logger.error("Batch experiment failed: %s", e)
                    results.append({
                        "experiment_id": pair[1].id,
                        "status": "failed",
                        "error": str(e),
                    })

        self._persist("batch_completed")

        completed = sum(1 for r in results if r.get("status") == "done")
        failed = sum(1 for r in results if r.get("status") == "failed")

        return {
            "total": len(results),
            "completed": completed,
            "failed": failed,
            "experiments": results,
        }

    @agent_tool()
    def _get_experiment_result(self, experiment_id: str) -> dict[str, Any]:
        """
        Read the result of a completed experiment.

        Args:
            experiment_id: ID of the experiment.
        """
        experiment = self._ledger.get_experiment(experiment_id)
        if experiment is None:
            return {"error": f"No experiment found with ID: {experiment_id}"}

        return {
            "experiment_id": experiment.id,
            "hypothesis": experiment.hypothesis,
            "status": experiment.status.value,
            "verdict": experiment.verdict.value if experiment.verdict else None,
            "summary": experiment.summary,
            "output": experiment.output,
        }

    @agent_tool()
    def _follow_up(
        self,
        experiment_id: str,
        message: str,
    ) -> dict[str, Any]:
        """
        Send a follow-up message to the SAME worker that ran an experiment.
        The worker retains its full context — no cold start.

        Use this when:
        - The result is ambiguous and you need clarification
        - You want the worker to try a variation
        - You need more detail about the findings

        Args:
            experiment_id: ID of the experiment to follow up on.
            message: Follow-up instructions for the worker.
        """
        experiment = self._ledger.get_experiment(experiment_id)
        if experiment is None:
            return {"error": f"No experiment found with ID: {experiment_id}"}

        if experiment.task_id is None:
            return {"error": "Experiment has no associated task"}

        task = self._orchestration_state.tasks.get(experiment.task_id)
        if task is None:
            return {"error": f"Task {experiment.task_id} not found"}

        if task.agent_id is None:
            return {"error": "No agent instance associated with this task"}

        agent = self._agent_instances.get(task.agent_id)
        if agent is None:
            return {"error": f"Agent instance {task.agent_id} no longer exists"}

        # Send follow-up via the agent's conversational interface
        agent.process_new_message(message)

        # Collect the last assistant message as the response
        last_chat = None
        for chat_id in reversed(agent._thread.chat_ids):
            chat = agent._chats.get(chat_id)
            if chat and chat.role == ChatRole.ASSISTANT:
                last_chat = chat
                break

        response_text = last_chat.content if last_chat else "(no response)"
        return {
            "experiment_id": experiment.id,
            "follow_up_response": response_text,
        }

    # ===================================================================
    # RECORDING TOOLS
    # ===================================================================

    @agent_tool()
    def _record_finding(
        self,
        experiment_id: str,
        verdict: str,
        summary: str,
    ) -> dict[str, Any]:
        """
        Record a verdict after reviewing an experiment result.

        MUST be called for every completed experiment. This builds the
        experiment log that drives your next decisions.

        Args:
            experiment_id: ID of the experiment.
            verdict: One of 'confirmed', 'refuted', 'partial', 'needs_followup'.
            summary: What we learned, in one or two sentences.
        """
        experiment = self._ledger.get_experiment(experiment_id)
        if experiment is None:
            return {"error": f"No experiment found with ID: {experiment_id}"}

        try:
            experiment.verdict = ExperimentVerdict(verdict)
        except ValueError:
            return {
                "error": f"Invalid verdict: {verdict}. "
                f"Must be one of: {[v.value for v in ExperimentVerdict]}"
            }

        experiment.summary = summary

        self._persist(f"finding_{experiment.id}")
        return {
            "experiment_id": experiment.id,
            "verdict": experiment.verdict.value,
            "summary": summary,
        }

    @agent_tool()
    def _record_proven_artifact(
        self,
        artifact_type: str,
        details: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Add a proven artifact to the ledger. Call this when an experiment confirms
        a fetch, navigation, token, or parameter that will be part of a routine.

        Args:
            artifact_type: One of 'fetch', 'navigation', 'token', 'parameter'.
            details: Artifact-specific info. Examples:
                fetch: {url, method, headers, body, response_preview}
                navigation: {url, sets_up: [cookies, storage_keys]}
                token: {name, source, storage_type, key_name}
                parameter: {name, type, description, example_value}
        """
        try:
            atype = ArtifactType(artifact_type)
        except ValueError:
            return {
                "error": f"Invalid artifact_type: {artifact_type}. "
                f"Must be one of: {[t.value for t in ArtifactType]}"
            }

        proven = self._ledger.proven
        if atype == ArtifactType.FETCH:
            proven.fetches.append(details)
        elif atype == ArtifactType.NAVIGATION:
            proven.navigations.append(details)
        elif atype == ArtifactType.TOKEN:
            proven.tokens.append(details)
        elif atype == ArtifactType.PARAMETER:
            proven.parameters.append(details)

        self._persist(f"artifact_{artifact_type}")
        return {"ok": True, "artifact_type": artifact_type, "details": details}

    # ===================================================================
    # ROUTINE SUBMISSION TOOLS
    # ===================================================================

    @agent_tool()
    def _submit_routine(
        self,
        spec_id: str,
        routine_json: dict[str, Any],
        test_parameters: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Submit a routine attempt for validation, execution, and inspection.

        Pipeline: validate → execute with test_parameters → inspect → verdict.

        IMPORTANT: You MUST provide test_parameters with realistic values for
        every parameter defined in the routine. The routine will be executed
        in a live browser and the result sent to an independent inspector.

        Args:
            spec_id: Which RoutineSpec this routine fulfills.
            routine_json: The complete routine as a dict.
            test_parameters: Parameter values for test execution. REQUIRED —
                must include a value for every parameter in the routine.
        """
        spec = self._ledger.get_spec(spec_id)
        if spec is None:
            return {"error": f"No spec found with ID: {spec_id}"}

        if not test_parameters:
            return {
                "error": "test_parameters is required. Provide realistic values "
                "for every parameter so the routine can be executed and inspected."
            }

        # Check attempt limit
        existing_attempts = self._ledger.get_attempts_for_spec(spec_id)
        if len(existing_attempts) >= self._max_attempts_per_routine:
            return {
                "error": f"Max attempts ({self._max_attempts_per_routine}) reached for {spec.name}. "
                "Consider mark_routine_failed if this routine can't be built."
            }

        # Step 1: Validate routine JSON against the Routine model
        try:
            routine = Routine.model_validate(routine_json)
        except Exception as e:
            return {
                "success": False,
                "stage": "validation",
                "validation_errors": [str(e)],
                "routine_json": routine_json,
            }

        # Create attempt record
        parent_id = existing_attempts[-1].id if existing_attempts else None
        attempt = RoutineAttempt(
            routine_spec_id=spec_id,
            routine_json=json.loads(routine.model_dump_json()),
            status=RoutineAttemptStatus.VALIDATING,
            test_parameters=test_parameters,
            parent_attempt_id=parent_id,
        )
        self._ledger.add_attempt(attempt)
        spec.status = RoutineSpecStatus.VALIDATING
        self._persist(f"attempt_{attempt.id}_validated")

        # Step 2: Execute the routine with test parameters
        attempt.status = RoutineAttemptStatus.EXECUTING
        self._persist(f"attempt_{attempt.id}_executing")

        execution_result = self._execute_routine_with_params(routine, test_parameters)

        if execution_result is not None:
            attempt.execution_result = execution_result.model_dump()
            if not execution_result.ok:
                attempt.execution_error = execution_result.error
                logger.warning(
                    "Routine %s execution failed: %s", spec.name, execution_result.error,
                )
        else:
            attempt.execution_error = "Execution unavailable (no browser or execution crashed)"

        self._persist(f"attempt_{attempt.id}_executed")

        # Step 3: Send to inspector for quality review
        attempt.status = RoutineAttemptStatus.INSPECTING
        self._persist(f"attempt_{attempt.id}_inspecting")

        inspection_result = self._run_inspection(routine, execution_result, spec)

        if inspection_result is not None:
            attempt.inspection_result = inspection_result
            attempt.overall_pass = inspection_result.get("overall_pass", False)
            attempt.blocking_issues = inspection_result.get("blocking_issues", [])
            attempt.recommendations = inspection_result.get("recommendations", [])

            if attempt.overall_pass:
                attempt.status = RoutineAttemptStatus.PASSED
            else:
                attempt.status = RoutineAttemptStatus.FAILED
        else:
            # Inspector failed — treat as passed with warning (let PI decide)
            attempt.status = RoutineAttemptStatus.PASSED
            attempt.recommendations = ["Inspector was unavailable — manual review recommended"]

        self._persist(f"attempt_{attempt.id}_inspected")

        # Build response
        response: dict[str, Any] = {
            "success": True,
            "attempt_id": attempt.id,
            "spec_id": spec_id,
            "operations_count": len(routine.operations),
            "parameters_count": len(routine.parameters),
        }

        # Execution summary
        if execution_result is not None:
            response["execution"] = {
                "ok": execution_result.ok,
                "error": execution_result.error,
                "content_type": str(execution_result.content_type) if execution_result.content_type else None,
                "data_preview": str(execution_result.data)[:500] if execution_result.data else None,
                "warnings": execution_result.warnings,
            }
        else:
            response["execution"] = {"ok": False, "error": attempt.execution_error}

        # Inspection summary
        if inspection_result is not None:
            response["inspection"] = {
                "overall_pass": attempt.overall_pass,
                "overall_score": inspection_result.get("overall_score"),
                "blocking_issues": attempt.blocking_issues,
                "recommendations": attempt.recommendations,
                "summary": inspection_result.get("summary"),
            }
        else:
            response["inspection"] = {"overall_pass": None, "note": "Inspector unavailable"}

        response["verdict"] = attempt.status.value

        return response

    @agent_tool()
    def _mark_routine_shipped(
        self,
        spec_id: str,
        attempt_id: str,
        when_to_use: str,
        parameters_summary: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Mark a routine as shipped after it passes inspection/validation.
        Moves the spec status to "shipped".

        Args:
            spec_id: ID of the RoutineSpec.
            attempt_id: ID of the RoutineAttempt to ship.
            when_to_use: Guidance for the user on when to use this routine.
            parameters_summary: Human-readable parameter descriptions.
        """
        spec = self._ledger.get_spec(spec_id)
        if spec is None:
            return {"error": f"No spec found with ID: {spec_id}"}

        attempt = self._ledger.get_attempt(attempt_id)
        if attempt is None:
            return {"error": f"No attempt found with ID: {attempt_id}"}

        spec.status = RoutineSpecStatus.SHIPPED
        spec.shipped_attempt_id = attempt_id

        self._persist(f"shipped_{spec.name}")
        return {
            "ok": True,
            "shipped": spec.name,
            "attempt_id": attempt_id,
        }

    @agent_tool()
    def _mark_routine_failed(
        self,
        spec_id: str,
        reason: str,
    ) -> dict[str, Any]:
        """
        Give up on a specific routine. Records why it failed.

        Args:
            spec_id: ID of the RoutineSpec.
            reason: Why this routine can't be built.
        """
        spec = self._ledger.get_spec(spec_id)
        if spec is None:
            return {"error": f"No spec found with ID: {spec_id}"}

        # Guardrail: require minimum experimentation before giving up
        spec_experiments = self._ledger.get_experiments_for_spec(spec_id)
        if len(spec_experiments) < 2:
            return {
                "error": (
                    f"Cannot mark routine '{spec.name}' as failed after only "
                    f"{len(spec_experiments)} experiment(s). Try at least 2 experiments "
                    "with different approaches before giving up. Consider: CDP-level "
                    "intercepts, direct navigation to API URLs, or checking the "
                    "captured session data for working request patterns."
                )
            }

        spec.status = RoutineSpecStatus.FAILED
        spec.failure_reason = reason

        self._persist(f"failed_{spec.name}")
        return {"ok": True, "failed": spec.name, "reason": reason}

    # ===================================================================
    # DASHBOARD TOOL
    # ===================================================================

    @agent_tool()
    def _get_ledger(self) -> dict[str, Any]:
        """
        Read the full Discovery Ledger — routine specs, experiments, proven
        artifacts, attempts, and unresolved questions. Use this to review
        progress and decide what to work on next.
        """
        return {
            "summary": self._ledger.to_summary(),
            "total_specs": len(self._ledger.routine_specs),
            "shipped": sum(
                1 for s in self._ledger.routine_specs
                if s.status == RoutineSpecStatus.SHIPPED
            ),
            "failed": sum(
                1 for s in self._ledger.routine_specs
                if s.status == RoutineSpecStatus.FAILED
            ),
            "total_experiments": len(self._ledger.experiments),
            "confirmed": len(self._ledger.get_confirmed_experiments()),
            "total_attempts": len(self._ledger.attempts),
            "proven_fetches": len(self._ledger.proven.fetches),
            "proven_navigations": len(self._ledger.proven.navigations),
            "proven_tokens": len(self._ledger.proven.tokens),
            "proven_parameters": len(self._ledger.proven.parameters),
            "unresolved": self._ledger.unresolved,
        }

    # ===================================================================
    # TERMINATION TOOLS
    # ===================================================================

    @agent_tool()
    def _mark_complete(self, usage_guide: str) -> dict[str, Any]:
        """
        Signal that the pipeline is done. Call this when ALL routines
        have been addressed (shipped or failed).

        Provides a usage_guide string explaining how to use the routines
        together and when to use each one. Builds the final RoutineCatalog.

        Args:
            usage_guide: How to use these routines together. Include:
                - What each routine does
                - When to use each one
                - How they relate to each other
                - What parameters each expects
        """
        # Guardrail: reject if routines are still unaddressed
        unaddressed = [
            s for s in self._ledger.routine_specs
            if s.status not in (RoutineSpecStatus.SHIPPED, RoutineSpecStatus.FAILED)
        ]
        shipped_count = sum(
            1 for s in self._ledger.routine_specs
            if s.status == RoutineSpecStatus.SHIPPED
        )
        if unaddressed:
            unaddressed_names = [f"{s.name} ({s.status.value})" for s in unaddressed]
            return {
                "error": (
                    f"Cannot mark complete — {len(unaddressed)} routine(s) are still unaddressed: "
                    f"{', '.join(unaddressed_names)}. "
                    "Each routine must be either shipped (via submit_routine → mark_routine_shipped) "
                    "or explicitly failed (via mark_routine_failed) before calling mark_complete. "
                    "You must build and submit actual routine JSON with test_parameters for each routine."
                )
            }

        # Guardrail: reject if nothing was shipped at all
        if shipped_count == 0:
            return {
                "error": (
                    "Cannot mark complete with 0 shipped routines. At least one routine "
                    "must be successfully built, submitted, and shipped. Use submit_routine "
                    "with a complete routine_json and test_parameters to create routine attempts."
                )
            }

        catalog = self._build_catalog(usage_guide)
        self._ledger.catalog = catalog
        self._pipeline_result = catalog
        self._is_done = True

        self._persist("complete")
        return {
            "status": "complete",
            "routines_shipped": len(catalog.routines),
            "routines_failed": len(catalog.failed_routines),
            "total_experiments": catalog.total_experiments,
            "total_attempts": catalog.total_attempts,
        }

    @agent_tool()
    def _mark_failed(self, reason: str) -> dict[str, Any]:
        """
        Signal that the pipeline has failed — can't build ANY routines at all.

        Args:
            reason: Why construction failed entirely.
        """
        # Guardrail: prevent premature pipeline abandonment
        total_experiments = len(self._ledger.experiments)
        unaddressed_specs = [
            s for s in self._ledger.routine_specs
            if s.status not in (RoutineSpecStatus.SHIPPED, RoutineSpecStatus.FAILED)
        ]
        if total_experiments < 5 and unaddressed_specs:
            return {
                "error": (
                    f"Cannot mark pipeline as failed after only {total_experiments} experiment(s). "
                    f"You have {len(unaddressed_specs)} unaddressed routine(s). "
                    "Try alternative approaches: use capture_search_transactions to find working "
                    "request patterns, use browser_cdp_command for CDP-level intercepts, or "
                    "navigate directly to API URLs. Mark individual routines as failed with "
                    "mark_routine_failed if they truly can't be built, then call mark_complete."
                )
            }

        self._is_done = True
        self._pipeline_result = None
        logger.warning("PI marked pipeline as failed: %s", reason)

        self._persist("failed")
        return {"status": "failed", "reason": reason}

    # ===================================================================
    # Internal — catalog building
    # ===================================================================

    def _build_catalog(self, usage_guide: str) -> RoutineCatalog:
        """Build a RoutineCatalog from the current ledger state."""
        shipped_routines: list[ShippedRoutine] = []
        failed_routines: list[dict[str, Any]] = []

        for spec in self._ledger.routine_specs:
            if spec.status == RoutineSpecStatus.SHIPPED and spec.shipped_attempt_id:
                attempt = self._ledger.get_attempt(spec.shipped_attempt_id)
                if attempt:
                    shipped_routines.append(ShippedRoutine(
                        routine_spec_id=spec.id,
                        routine_json=attempt.routine_json,
                        name=spec.name,
                        description=spec.description,
                        when_to_use=f"Use to {spec.description.lower()}",
                        parameters_summary=[],
                        inspection_score=attempt.inspection_result.get("overall_score", 0)
                        if attempt.inspection_result else 0,
                    ))
            elif spec.status == RoutineSpecStatus.FAILED:
                failed_routines.append({
                    "name": spec.name,
                    "description": spec.description,
                    "reason": spec.failure_reason or "Unknown",
                })

        # Infer site from exploration summaries or first URL
        site = "unknown"
        for summary_text in self._exploration_summaries.values():
            if "://" in summary_text:
                # Try to extract domain
                match = re.search(r'https?://([^/\s]+)', summary_text)
                if match:
                    site = match.group(1)
                    break

        return RoutineCatalog(
            site=site,
            user_task=self._task,
            routines=shipped_routines,
            usage_guide=usage_guide,
            failed_routines=failed_routines,
            total_experiments=len(self._ledger.experiments),
            total_attempts=len(self._ledger.attempts),
        )

    def _build_partial_catalog(self) -> RoutineCatalog | None:
        """Build a partial catalog from whatever has been shipped so far."""
        shipped = [
            s for s in self._ledger.routine_specs
            if s.status == RoutineSpecStatus.SHIPPED
        ]
        if not shipped:
            return None
        return self._build_catalog(
            "Pipeline hit iteration limit. These routines were completed."
        )

    # ===================================================================
    # Internal — worker management
    # ===================================================================

    def _create_worker(self) -> ExperimentWorker:
        """Create a new ExperimentWorker instance with all available context."""
        return ExperimentWorker(
            emit_message_callable=self._emit_message_callable,
            # Browser context
            remote_debugging_address=self._remote_debugging_address,
            # Capture data loaders
            network_data_loader=self._network_data_loader,
            storage_data_loader=self._storage_data_loader,
            dom_data_loader=self._dom_data_loader,
            window_property_data_loader=self._window_property_data_loader,
            # Config
            llm_model=self._worker_llm_model,
            run_mode=RunMode.AUTONOMOUS,
        )

    def _create_inspector(self) -> RoutineInspector:
        """Create a new RoutineInspector instance."""
        return RoutineInspector(
            emit_message_callable=self._emit_message_callable,
            llm_model=self._worker_llm_model,
            run_mode=RunMode.AUTONOMOUS,
        )

    def _get_or_create_agent(self, task: Task) -> AbstractSpecialist:
        """
        Get existing agent instance or create/reuse one for the task.

        Workers are capped at num_workers. Once the pool is full, new tasks
        are assigned round-robin to existing workers (each gets a fresh
        autonomous run but the PI can still follow_up on the same worker).
        """
        if task.agent_id and task.agent_id in self._agent_instances:
            return self._agent_instances[task.agent_id]

        # Check if we can reuse an existing worker (pool is full)
        worker_ids = [
            sid for sid, agent in self._agent_instances.items()
            if isinstance(agent, ExperimentWorker)
        ]

        if len(worker_ids) >= self._num_workers:
            # Round-robin to existing workers
            reuse_id = worker_ids[self._worker_counter % len(worker_ids)]
            self._worker_counter += 1
            task.agent_id = reuse_id
            subagent = self._orchestration_state.subagents.get(reuse_id)
            if subagent:
                subagent.task_ids.append(task.id)
            # Close old browser tab — _ensure_browser will create a fresh one
            worker = self._agent_instances[reuse_id]
            if isinstance(worker, ExperimentWorker):
                worker.close()
            return worker

        # Create new worker
        agent = self._create_worker()

        subagent = SubAgent(
            type=task.agent_type,
            llm_model=self._worker_llm_model.value,
        )
        self._orchestration_state.subagents[subagent.id] = subagent
        self._agent_instances[subagent.id] = agent

        task.agent_id = subagent.id
        subagent.task_ids.append(task.id)

        return agent

    def _get_or_create_inspector(self) -> RoutineInspector:
        """
        Get an existing inspector instance or create one.

        Inspectors are capped at num_inspectors. Once the pool is full,
        existing inspectors are reused round-robin (each gets a fresh
        autonomous run via reset).
        """
        inspector_ids = [
            sid for sid, agent in self._agent_instances.items()
            if isinstance(agent, RoutineInspector)
        ]

        if len(inspector_ids) < self._num_inspectors:
            # Pool not full — create a new inspector
            inspector = self._create_inspector()
            subagent = SubAgent(
                type=SpecialistAgentType.ROUTINE_INSPECTOR,
                llm_model=self._worker_llm_model.value,
            )
            self._orchestration_state.subagents[subagent.id] = subagent
            self._agent_instances[subagent.id] = inspector
            return inspector

        # Pool full — reuse round-robin with fresh conversation
        reuse_id = inspector_ids[self._inspector_counter % len(inspector_ids)]
        self._inspector_counter += 1
        inspector = self._agent_instances[reuse_id]
        assert isinstance(inspector, RoutineInspector)
        inspector.reset()
        return inspector

    # ===================================================================
    # Internal — routine execution and inspection
    # ===================================================================

    def _execute_routine_with_params(
        self,
        routine: Routine,
        test_parameters: dict[str, Any] | None,
    ) -> RoutineExecutionResultWithMetadata | None:
        """Execute a routine with test parameters in a live browser."""
        if not self._remote_debugging_address:
            logger.warning("No remote_debugging_address — skipping routine execution")
            return None

        try:
            result = routine.execute(
                parameters_dict=test_parameters,
                remote_debugging_address=self._remote_debugging_address,
                timeout=120.0,
                close_tab_when_done=True,
                incognito=True,
            )
            return result
        except Exception as e:
            logger.error("Routine execution failed: %s", e)
            return None

    def _run_inspection(
        self,
        routine: Routine,
        execution_result: RoutineExecutionResultWithMetadata | None,
        spec: RoutineSpec,
    ) -> dict[str, Any] | None:
        """Run a RoutineInspector on a routine + execution result."""
        inspector = self._get_or_create_inspector()

        # Build inspection prompt with all context
        prompt_parts: list[str] = [
            f"## User Task\n{self._task}\n",
            f"## Routine Name\n{spec.name}\n",
            f"## Routine Description\n{spec.description}\n",
            f"## Routine JSON\n```json\n{json.dumps(routine.model_dump(), indent=2, default=str)}\n```\n",
        ]

        if execution_result is not None:
            exec_data = execution_result.model_dump()
            prompt_parts.append(
                f"## Execution Result\n```json\n{json.dumps(exec_data, indent=2, default=str)}\n```\n"
            )
        else:
            prompt_parts.append("## Execution Result\nNot available (no browser or execution failed).\n")

        # Add exploration summaries for cross-reference
        if self._exploration_summaries:
            prompt_parts.append("## Exploration Summaries\n")
            for domain, summary in self._exploration_summaries.items():
                prompt_parts.append(f"### {domain}\n{summary}\n")

        inspection_prompt = "\n".join(prompt_parts)

        try:
            config = AutonomousConfig(min_iterations=1, max_iterations=5)
            result = inspector.run_autonomous(
                task=inspection_prompt,
                config=config,
                output_schema=INSPECTION_OUTPUT_SCHEMA,
                output_description="RoutineInspectionResult with scores, blocking issues, and verdict",
            )

            # Dump inspector thread
            self._dump_agent_thread(f"inspector_{spec.name}", inspector)

            if result is not None:
                return result.model_dump() if isinstance(result, BaseModel) else result
            return None
        except Exception as e:
            logger.error("Inspection failed for %s: %s", spec.name, e)
            return None

    def _execute_task(self, task: Task) -> dict[str, Any]:
        """Execute a task using an ExperimentWorker."""
        task.status = TaskStatus.IN_PROGRESS
        task.started_at = datetime.now()

        try:
            agent = self._get_or_create_agent(task)

            remaining_loops = task.max_loops - task.loops_used
            if remaining_loops <= 0:
                task.status = TaskStatus.FAILED
                task.error = "No loops remaining"
                return {"success": False, "error": "No loops remaining"}

            config = AutonomousConfig(
                min_iterations=1,
                max_iterations=remaining_loops,
            )

            result = agent.run_autonomous(
                task=task.prompt,
                config=config,
                output_schema=task.output_schema,
                output_description=task.output_description,
            )

            task.loops_used += agent.autonomous_iteration

            # Dump the worker's full message history for debugging
            self._dump_agent_thread(f"worker_{task.agent_id}", agent)

            if result is not None:
                task.status = TaskStatus.COMPLETED
                task.completed_at = datetime.now()
                task.result = result.model_dump() if isinstance(result, BaseModel) else result
                return {"success": True, "result": task.result}
            else:
                if task.loops_used < task.max_loops:
                    task.status = TaskStatus.PAUSED
                    return {"success": False, "status": "paused", "loops_used": task.loops_used}
                else:
                    task.status = TaskStatus.FAILED
                    task.error = "Max loops reached without result"
                    return {"success": False, "error": task.error}

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.completed_at = datetime.now()
            logger.error("Task %s failed: %s", task.id, e)
            return {"success": False, "error": str(e)}

    def close(self) -> None:
        """Clean up all worker agent instances."""
        for agent in self._agent_instances.values():
            if hasattr(agent, "close"):
                try:
                    agent.close()
                except Exception:
                    pass
        self._agent_instances.clear()
