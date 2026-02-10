"""
bluebox/agents/super_discovery_agent.py

SuperDiscoveryAgent - orchestrator for routine discovery.

This agent coordinates specialist subagents (JSSpecialist, NetworkSpecialist, etc.)
to discover routines from CDP captures. It delegates specific tasks to specialists
while managing the overall discovery workflow:

1. PLANNING: Analyze task, plan approach
2. DISCOVERING: Delegate discovery tasks to specialists
3. CONSTRUCTING: Build routine from discoveries
4. VALIDATING: Test the constructed routine
5. COMPLETE/FAILED: Finish discovery

The agent inherits from AbstractAgent for LLM/chat/tool infrastructure.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from textwrap import dedent
from typing import Any, Callable

from pydantic import BaseModel

from bluebox.agents.abstract_agent import AbstractAgent, AgentCard, agent_tool
from bluebox.agents.specialists.abstract_specialist import AbstractSpecialist, AutonomousConfig, RunMode
from bluebox.agents.specialists.js_specialist import JSSpecialist
from bluebox.agents.specialists.network_specialist import NetworkSpecialist
from bluebox.agents.specialists.value_trace_resolver_specialist import ValueTraceResolverSpecialist
from bluebox.agents.specialists.interaction_specialist import InteractionSpecialist
from bluebox.data_models.llms.interaction import (
    Chat,
    ChatRole,
    ChatThread,
    EmittedMessage,
    ChatResponseEmittedMessage,
    ErrorEmittedMessage,
)
from bluebox.data_models.llms.vendors import LLMModel, OpenAIModel
from bluebox.data_models.orchestration.task import Task, SubAgent, TaskStatus, SpecialistAgentType
from bluebox.data_models.orchestration.state import AgentOrchestrationState
from bluebox.data_models.routine.endpoint import HTTPMethod
from bluebox.data_models.routine.routine import Routine
from bluebox.data_models.routine_discovery.state import RoutineDiscoveryState, DiscoveryPhase
from bluebox.data_models.routine_discovery.llm_responses import (
    TransactionIdentificationResponse,
    Variable,
    VariableType,
    ExtractedVariableResponse,
    ResolvedVariableResponse,
    SessionStorageSource,
    TransactionSource,
    WindowPropertySource,
    SessionStorageType,
)
from bluebox.llms.data_loaders.documentation_data_loader import DocumentationDataLoader
from bluebox.llms.data_loaders.interactions_data_loader import InteractionsDataLoader
from bluebox.llms.data_loaders.js_data_loader import JSDataLoader
from bluebox.llms.data_loaders.network_data_loader import NetworkDataLoader
from bluebox.llms.data_loaders.storage_data_loader import StorageDataLoader
from bluebox.llms.data_loaders.window_property_data_loader import WindowPropertyDataLoader
from bluebox.utils.data_utils import resolve_dotted_path
from bluebox.utils.logger import get_logger

logger = get_logger(name=__name__)


class SuperDiscoveryAgent(AbstractAgent):
    """
    Orchestrator agent that coordinates specialist subagents for routine discovery.

    Unlike specialists which do focused work, this agent plans and delegates:
    - Creates tasks for specialists to handle
    - Runs tasks and collects results
    - Uses results to construct routines
    """

    AGENT_CARD = AgentCard(
        description=(
            "Orchestrates routine discovery by coordinating specialist subagents. "
            "Delegates network analysis, value tracing, JS generation, and interaction "
            "analysis to specialists, then assembles the results into a routine."
        ),
    )

    ## System prompts — phase-scoped sections

    # Core identity + delegation rules (included in every phase)
    PROMPT_CORE: str = dedent("""\
        You are an expert at analyzing network traffic and building web automation routines.
        You coordinate specialist agents to discover and construct routines.

        ## Your Task
        Analyze captured browser network data to create a reusable routine that accomplishes the user's task.

        ## CRITICAL: You MUST Delegate to Specialists

        **DO NOT** try to do everything yourself with direct tools. You are an ORCHESTRATOR.
        Your job is to coordinate specialists, not to manually inspect every transaction.

        **How to delegate:**
        1. `create_task(agent_type="network_specialist", prompt="...")`
        2. `run_pending_tasks()`
        3. `get_task_result(task_id)` to review findings

        ## Important Notes
        - Focus on the user's INTENT, not literal wording
        - Keep parameters MINIMAL - only what the user MUST provide
        - If only one value was observed and it could be hardcoded, hardcode it
        - Credentials for fetch operations: same-origin > include > omit
    """)

    # Phase-specific instructions (only the active phase's block is included)
    PROMPT_PLANNING: str = dedent("""\
        ## Current Phase: PLANNING — Identify the Target Endpoint

        1. **REQUIRED**: Create a task for network_specialist to find the endpoint:
           ```
           create_task(
               agent_type="network_specialist",
               prompt="Find the API endpoint that accomplishes: <user's task>. Search for relevant keywords."
           )
           ```
        2. Call `run_pending_tasks()` to execute
        3. Review results with `get_task_result(task_id)`
        4. Use `record_identified_endpoint` with the specialist's findings
    """)

    PROMPT_DISCOVERING: str = dedent("""\
        ## Current Phase: DISCOVERING — Process Transactions (BFS Queue)

        For each transaction in the queue:
        1. Use `get_transaction` to see full details
        2. Use `record_extracted_variable` to log variables found in the request
        3. **For DYNAMIC_TOKENs — DELEGATE TO value_trace_resolver**:
           ```
           create_task(
               agent_type="value_trace_resolver",
               prompt="Trace the origin of value '<observed_value>' (variable: <name>). Find where it comes from."
           )
           ```
        4. Call `run_pending_tasks()` then `get_task_result(task_id)` to get findings
        5. Use `record_resolved_variable` to record where each token comes from
           - If source is another transaction, it will be auto-added to the queue
           - PREFER NETWORK SOURCES: When a value appears in both session storage AND a prior
             transaction response, use source_type='transaction' as the PRIMARY source.
             Session storage may be empty in a fresh session.
        6. Use `mark_transaction_processed` when done with a transaction
        7. Continue until queue is empty

        ## Variable Classification Rules

        **PARAMETER** (requires_dynamic_resolution=false):
        - Values the user explicitly provides as input
        - Examples: search_query, item_id, page_number, username
        - Rule: If the user wouldn't directly provide this value, it's NOT a parameter

        **DYNAMIC_TOKEN** (requires_dynamic_resolution=true):
        - Auth/session values that change per session
        - Examples: CSRF tokens, JWTs, session_id, visitorData, auth headers
        - Also: trace IDs, request IDs, correlation IDs
        - Rule: If it looks like a generated ID or security token, it's a DYNAMIC_TOKEN

        **STATIC_VALUE** (requires_dynamic_resolution=false):
        - Constants that don't change between sessions
        - Examples: App version, User-Agent, clientName, timeZone, language codes
        - Rule: If you can hardcode it and it will work across sessions, it's STATIC
    """)

    PROMPT_CONSTRUCTING: str = dedent("""\
        ## Current Phase: CONSTRUCTING — Build the Routine

        1. Use `get_discovery_context` to see all processed data (includes CRITICAL_OBSERVED_VALUES)
        2. Review the **Routine Schema Reference** below for required fields and operation types
        3. Use `construct_routine` with the routine definition:
           - `routine`: the routine definition (name, description, parameters, operations)

        **If browser is connected (validation available):**
        4. After constructing, use `validate_routine` with test_parameters (observed values)
        5. Use `analyze_validation` to reflect on results (REQUIRED before done)

        **If NO browser connected:**
        4. Call `done` directly after construct_routine
    """)

    PLACEHOLDER_INSTRUCTIONS: str = (
        "## Placeholder Syntax\n"
        "- PARAMS: {{param_name}} (NO prefix, name matches parameter definition)\n"
        "- SOURCES (use dot paths): {{cookie:name}}, {{sessionStorage:path.to.value}}, "
        "{{localStorage:key}}, {{windowProperty:obj.key}}\n\n"
        "JSON VALUE RULES (TWO sets of quotes needed for strings!):\n"
        '- String: "key": \\"{{x}}\\"  (OUTER quotes = JSON string, INNER \\" = escaped quotes)\n'
        '- Number/bool/null: "key": "{{x}}"  (only outer quotes, they get stripped)\n'
        '- Inside larger string: "prefix\\"{{x}}\\"suffix"  (escaped quotes wrap placeholder)\n\n'
        "EXAMPLES:\n"
        '1. String param:     "name": \\"{{username}}\\"           -> "name": "john"\n'
        '2. Number param:     "count": "{{limit}}"                -> "count": 50\n'
        '3. Bool param:       "active": "{{is_active}}"           -> "active": true\n'
        '4. String in string: "msg_\\"{{id}}\\""                  -> "msg_abc"\n'
        '5. Number in string: "page\\"{{num}}\\""                 -> "page5"\n'
        '6. URL with param:   "/api/\\"{{user_id}}\\"/data"       -> "/api/123/data"\n'
        '7. Session storage:  "token": \\"{{sessionStorage:auth.access_token}}\\"\n'
        '8. Cookie:           "sid": \\"{{cookie:session_id}}\\"\n'
        "IMPORTANT: YOU MUST ENSURE THAT EACH PLACEHOLDER IS SURROUNDED BY QUOTES OR ESCAPED QUOTES!"
    )

    PROMPT_VALIDATING: str = dedent("""\
        ## Current Phase: VALIDATING — Test the Routine

        1. Review the `validate_routine` execution results
        2. Use `analyze_validation` to reflect:
           - `analysis`: What worked and what failed
           - `data_matches_task`: Does the returned data accomplish the user's original task?
           - `next_action`: "done" | "fix_routine" | "retry_validation"
        3. Based on your analysis:
           - If data_matches_task=True and next_action="done": call `done`
           - If data_matches_task=False: set next_action="fix_routine", then use construct_routine to fix and re-validate
    """)

    ## Magic methods

    def __init__(
        self,
        emit_message_callable: Callable[[EmittedMessage], None],
        network_data_loader: NetworkDataLoader,
        task: str,
        storage_data_loader: StorageDataLoader | None = None,
        window_property_data_loader: WindowPropertyDataLoader | None = None,
        js_data_loader: JSDataLoader | None = None,
        interaction_data_loader: InteractionsDataLoader | None = None,
        documentation_data_loader: DocumentationDataLoader | None = None,
        llm_model: LLMModel = OpenAIModel.GPT_5_1,
        subagent_llm_model: LLMModel | None = None,
        max_iterations: int = 50,
        remote_debugging_address: str | None = None,
        persist_chat_callable: Callable[[Chat], Chat] | None = None,
        persist_chat_thread_callable: Callable[[ChatThread], ChatThread] | None = None,
        stream_chunk_callable: Callable[[str], None] | None = None,
        chat_thread: ChatThread | None = None,
        existing_chats: list[Chat] | None = None,
    ) -> None:
        """
        Initialize the SuperDiscoveryAgent.

        Args:
            emit_message_callable: Callback to emit messages to the host.
            network_data_loader: NetworkDataLoader with network traffic data.
            task: The discovery task description.
            storage_data_loader: Optional StorageDataLoader for browser storage.
            window_property_data_loader: Optional WindowPropertyDataLoader for window properties.
            js_data_loader: Optional JSDataLoader for JavaScript files.
            interaction_data_loader: Optional InteractionsDataLoader for interaction events.
            documentation_data_loader: Optional DocumentationDataLoader for docs and code files.
            llm_model: LLM model for the orchestrator.
            subagent_llm_model: LLM model for subagents (defaults to orchestrator's model).
            max_iterations: Maximum iterations for the main loop.
            remote_debugging_address: Chrome remote debugging address for validation.
            persist_chat_callable: Optional callback to persist Chat objects.
            persist_chat_thread_callable: Optional callback to persist ChatThread.
            stream_chunk_callable: Optional callback for streaming text chunks.
            chat_thread: Existing ChatThread to continue, or None for new.
            existing_chats: Existing Chat messages if loading from persistence.
        """
        self._network_data_loader = network_data_loader
        self._storage_data_loader = storage_data_loader
        self._window_property_data_loader = window_property_data_loader
        self._js_data_loader = js_data_loader
        self._interaction_data_loader = interaction_data_loader
        self._documentation_data_loader = documentation_data_loader
        self._task = task
        self._subagent_llm_model = subagent_llm_model or llm_model
        self._max_iterations = max_iterations
        self._remote_debugging_address = remote_debugging_address

        # Internal state
        self._orchestration_state = AgentOrchestrationState()
        self._discovery_state = RoutineDiscoveryState(phase=DiscoveryPhase.PLANNING)
        self._agent_instances: dict[str, AbstractSpecialist] = {}  # agent_id -> instance

        # Result tracking
        self._final_routine: Routine | None = None
        self._failure_reason: str | None = None

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

    ## Abstract method implementations

    def _get_system_prompt(self) -> str:
        """Build the system prompt scoped to the current phase."""
        phase = self._discovery_state.phase

        # Core identity + delegation rules (always included)
        prompt_parts = [self.PROMPT_CORE]

        # Inject specialist descriptions from AgentCard metadata
        specialist_lines = [
            f"- `{agent_type.value}`: {cls.AGENT_CARD.description}"
            for agent_type, cls in (
                (SpecialistAgentType.NETWORK_SPECIALIST, NetworkSpecialist),
                (SpecialistAgentType.VALUE_TRACE_RESOLVER, ValueTraceResolverSpecialist),
                (SpecialistAgentType.JS_SPECIALIST, JSSpecialist),
                (SpecialistAgentType.INTERACTION_SPECIALIST, InteractionSpecialist),
            )
        ]
        prompt_parts.append("\n\n**Available specialists:**\n" + "\n".join(specialist_lines))

        # Phase-specific instructions
        if phase == DiscoveryPhase.PLANNING:
            prompt_parts.append(self.PROMPT_PLANNING)
        elif phase == DiscoveryPhase.DISCOVERING:
            prompt_parts.append(self.PROMPT_DISCOVERING)
        elif phase == DiscoveryPhase.CONSTRUCTING:
            prompt_parts.append(self.PROMPT_CONSTRUCTING)
            prompt_parts.append(self.PLACEHOLDER_INSTRUCTIONS)
            prompt_parts.append(Routine.model_schema_markdown())
        elif phase == DiscoveryPhase.VALIDATING:
            prompt_parts.append(self.PROMPT_VALIDATING)
            prompt_parts.append(self.PLACEHOLDER_INSTRUCTIONS)  # needed if fix_routine
            prompt_parts.append(Routine.model_schema_markdown())  # needed if fix_routine

        # Add data store summaries
        data_store_info = []
        if self._network_data_loader:
            stats = self._network_data_loader.stats
            data_store_info.append(f"Network: {stats.total_requests} transactions")
        if self._storage_data_loader:
            stats = self._storage_data_loader.stats
            data_store_info.append(f"Storage: {stats.total_events} events")
        if self._window_property_data_loader:
            stats = self._window_property_data_loader.stats
            data_store_info.append(f"Window: {stats.total_events} events")
        if self._js_data_loader:
            data_store_info.append("JS files: available")
        if self._documentation_data_loader:
            summary = self._documentation_data_loader.stats.to_summary()
            data_store_info.append(f"Documentation: {summary}")

        if data_store_info:
            prompt_parts.append(f"\n\n## Data Sources\n{', '.join(data_store_info)}")

        # Add current state
        status = self._orchestration_state.get_queue_status()
        prompt_parts.append(dedent(f"""\

            ## Current State
            - Phase: {self._discovery_state.phase.value}
            - Pending tasks: {status['pending_tasks']}
            - In-progress tasks: {status['in_progress_tasks']}
            - Completed tasks: {status['completed_tasks']}
            - Failed tasks: {status['failed_tasks']}
        """))

        # Add discovery state tracking info
        discovery_status = self._discovery_state.get_queue_status()
        if self._discovery_state.root_transaction or self._discovery_state.processed_transactions:
            prompt_parts.append(dedent(f"""\

            ## Discovery Progress
            - Root transaction: {"Set" if self._discovery_state.root_transaction else "Not set"}
            - Transaction queue: {discovery_status['pending_count']} pending, {discovery_status['processed_count']} processed
            - Resolved variables: {len(self._discovery_state.all_resolved_variables)}
            - Routine: {"Constructed" if self._discovery_state.production_routine else "Not constructed"}
            """))

        if self._remote_debugging_address:
            prompt_parts.append("\n- Browser: Connected (validation available)")
        else:
            prompt_parts.append("\n- Browser: Not connected (skip validation)")

        return "".join(prompt_parts)

    ## Public API

    def run(self) -> Routine | None:
        """
        Run the discovery to completion.

        Returns:
            The discovered Routine, or None if discovery failed.
        """
        # Seed the conversation with emphasis on delegation
        initial_message = (
            f"TASK: {self._task}\n\n"
            "IMPORTANT: Start by delegating to network_specialist to find the relevant endpoint. "
            "Call create_task(agent_type='network_specialist', prompt='Find the API endpoint for: <task>') "
            "then run_pending_tasks(). DO NOT manually browse transactions yourself."
        )
        self._add_chat(ChatRole.USER, initial_message)

        # Run the main loop
        for iteration in range(self._max_iterations):
            logger.debug("SuperDiscovery iteration %d/%d, phase: %s",
                        iteration + 1, self._max_iterations, self._discovery_state.phase.value)

            # Check for completion
            if self._discovery_state.phase == DiscoveryPhase.COMPLETE:
                return self._final_routine

            if self._discovery_state.phase == DiscoveryPhase.FAILED:
                logger.error("Discovery failed: %s", self._failure_reason)
                return None

            # Run agent loop iteration
            messages = self._build_messages_for_llm()
            try:
                response = self._call_llm(
                    messages,
                    self._get_system_prompt(),
                    tool_choice="required",
                )

                if response.response_id:
                    self._previous_response_id = response.response_id

                if response.content or response.tool_calls:
                    chat = self._add_chat(
                        ChatRole.ASSISTANT,
                        response.content or "",
                        tool_calls=response.tool_calls if response.tool_calls else None,
                        llm_provider_response_id=response.response_id,
                    )
                    if response.content:
                        self._emit_message(
                            ChatResponseEmittedMessage(
                                content=response.content,
                                chat_id=chat.id,
                                chat_thread_id=self._thread.id,
                            )
                        )

                if response.tool_calls:
                    self._process_tool_calls(response.tool_calls)
                else:
                    # Prompt the agent to continue if no tool calls - provide phase-specific guidance
                    phase = self._discovery_state.phase
                    if phase == DiscoveryPhase.PLANNING:
                        guidance = (
                            "Phase: PLANNING. You MUST delegate to specialists! "
                            "Call create_task(agent_type='network_specialist', prompt='Find the API endpoint for: <task>') "
                            "then run_pending_tasks(). DO NOT use list_transactions or get_transaction directly."
                        )
                    elif phase == DiscoveryPhase.DISCOVERING:
                        task_status = self._orchestration_state.get_queue_status()
                        if task_status["pending_tasks"] > 0:
                            guidance = (
                                f"Phase: DISCOVERING. You have {task_status['pending_tasks']} pending tasks. "
                                "Call run_pending_tasks() to execute them."
                            )
                        elif task_status["completed_tasks"] > 0:
                            guidance = (
                                "Phase: DISCOVERING. Tasks completed. Review results with get_task_result(task_id), "
                                "then record findings using record_identified_endpoint, record_extracted_variable. "
                                "For DYNAMIC_TOKENs, delegate to value_trace_resolver - don't use scan_for_value directly."
                            )
                        else:
                            guidance = (
                                "Phase: DISCOVERING. No tasks created yet! You MUST delegate: "
                                "create_task(agent_type='network_specialist', prompt='...') then run_pending_tasks(). "
                                "DO NOT manually inspect transactions - let specialists do the work."
                            )
                    elif phase == DiscoveryPhase.CONSTRUCTING:
                        if not self._discovery_state.production_routine:
                            guidance = (
                                "Phase: CONSTRUCTING. Call get_discovery_context to see all discovered data, "
                                "then use construct_routine to build the routine."
                            )
                        else:
                            guidance = (
                                "Phase: CONSTRUCTING. Routine already constructed. "
                                "Proceed to validation or mark as done."
                            )
                    elif phase == DiscoveryPhase.VALIDATING:
                        guidance = (
                            "Phase: VALIDATING. Review the construct_routine execution results. "
                            "If execution_success=True, call done. If execution_success=False, "
                            "fix the issues and call construct_routine again."
                        )
                    else:
                        guidance = f"Phase: {phase.value}. Use tools to make progress."

                    self._add_chat(ChatRole.SYSTEM, f"[ACTION REQUIRED] {guidance}")

            except Exception as e:
                logger.exception("Error in SuperDiscovery loop: %s", e)
                self._emit_message(ErrorEmittedMessage(error=str(e)))
                self._discovery_state.phase = DiscoveryPhase.FAILED
                self._failure_reason = str(e)
                return None

        logger.warning("SuperDiscovery hit max iterations (%d)", self._max_iterations)
        self._discovery_state.phase = DiscoveryPhase.FAILED
        self._failure_reason = f"Max iterations ({self._max_iterations}) reached"
        return None

    ## Internal methods

    def _get_or_create_agent(self, task: Task) -> AbstractSpecialist:
        """Get existing agent instance or create new one for the task."""
        # Check if task specifies an existing agent
        if task.agent_id and task.agent_id in self._agent_instances:
            return self._agent_instances[task.agent_id]

        # Create new agent based on type
        agent_type = task.agent_type
        agent = self._create_specialist(agent_type)

        # Create SubAgent record and store instance
        subagent = SubAgent(
            type=agent_type,
            llm_model=self._subagent_llm_model.value,
        )
        self._orchestration_state.subagents[subagent.id] = subagent
        self._agent_instances[subagent.id] = agent

        # Update task with agent_id
        task.agent_id = subagent.id
        subagent.task_ids.append(task.id)

        return agent

    def _create_specialist(self, agent_type: SpecialistAgentType) -> AbstractSpecialist:
        """Create a specialist instance based on type."""
        if agent_type == SpecialistAgentType.JS_SPECIALIST:
            return JSSpecialist(
                emit_message_callable=self._emit_message_callable,
                llm_model=self._subagent_llm_model,
                documentation_data_loader=self._documentation_data_loader,
                network_data_store=self._network_data_loader,
                js_data_store=None,  # NOTE: this is intentionally left None for now
                remote_debugging_address=self._remote_debugging_address,
                run_mode=RunMode.AUTONOMOUS,
            )

        elif agent_type == SpecialistAgentType.VALUE_TRACE_RESOLVER:
            return ValueTraceResolverSpecialist(
                emit_message_callable=self._emit_message_callable,
                llm_model=self._subagent_llm_model,
                network_data_store=self._network_data_loader,
                storage_data_store=self._storage_data_loader,
                window_property_data_store=self._window_property_data_loader,
                run_mode=RunMode.AUTONOMOUS,
            )

        elif agent_type == SpecialistAgentType.NETWORK_SPECIALIST:
            if not self._network_data_loader:
                raise ValueError(
                    "network_specialist requires network_data_loader, "
                    "but it was not provided to SuperDiscoveryAgent"
                )
            return NetworkSpecialist(
                emit_message_callable=self._emit_message_callable,
                llm_model=self._subagent_llm_model,
                network_data_store=self._network_data_loader,
                run_mode=RunMode.AUTONOMOUS,
            )

        elif agent_type == SpecialistAgentType.INTERACTION_SPECIALIST:
            if not self._interaction_data_loader:
                raise ValueError(
                    "interaction_specialist requires interaction_data_loader, "
                    "but it was not provided to SuperDiscoveryAgent"
                )
            return InteractionSpecialist(
                emit_message_callable=self._emit_message_callable,
                interaction_data_store=self._interaction_data_loader,
                llm_model=self._subagent_llm_model,
                run_mode=RunMode.AUTONOMOUS,
            )

        else:
            raise NotImplementedError(
                f"Agent type {agent_type.value} is not yet supported. "
                f"Available types: js_specialist, network_specialist, value_trace_resolver, interaction_specialist"
            )

    def _execute_task(self, task: Task) -> dict[str, Any]:
        """Execute a task using the appropriate specialist."""
        task.status = TaskStatus.IN_PROGRESS
        task.started_at = datetime.now()

        try:
            agent = self._get_or_create_agent(task)

            # Calculate remaining loops
            remaining_loops = task.max_loops - task.loops_used
            if remaining_loops <= 0:
                task.status = TaskStatus.FAILED
                task.error = "No loops remaining"
                return {"success": False, "error": "No loops remaining"}

            # Run autonomous with config - pass output schema here (not before)
            # so it doesn't get cleared by _reset_autonomous_state()
            config = AutonomousConfig(
                min_iterations=1,  # Allow immediate finalization for resumed tasks
                max_iterations=remaining_loops,
            )

            result = agent.run_autonomous(
                task=task.prompt,
                config=config,
                output_schema=task.output_schema,
                output_description=task.output_description,
            )

            # Update loops used
            task.loops_used += agent.autonomous_iteration

            if result is not None:
                task.status = TaskStatus.COMPLETED
                task.completed_at = datetime.now()
                task.result = result.model_dump() if isinstance(result, BaseModel) else result
                return {"success": True, "result": task.result}
            else:
                # Agent hit max iterations without finalizing
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

    def _validate_discovery_completeness(self) -> tuple[bool, list[str]]:
        """
        Check if discovery state is complete enough to construct routine.

        Returns:
            Tuple of (is_complete, list_of_blockers).
            If is_complete is False, blockers explain what's missing.
        """
        blockers = []

        # Check if root transaction is set
        if not self._discovery_state.root_transaction:
            blockers.append("No root transaction recorded")

        # Check for unresolved dynamic tokens
        unresolved_tokens = []
        for tx_id, tx_data in self._discovery_state.transaction_data.items():
            if tx_data.get("extracted_variables"):
                extracted = tx_data["extracted_variables"]
                resolved_names = {
                    rv.variable.name
                    for rv in tx_data.get("resolved_variables", [])
                }
                for var in extracted.variables:
                    if var.requires_dynamic_resolution and var.name not in resolved_names:
                        unresolved_tokens.append(f"{var.name} (in {tx_id})")

        if unresolved_tokens:
            blockers.append(f"Unresolved dynamic tokens: {', '.join(unresolved_tokens)}")

        # Check if transaction queue is not empty
        if self._discovery_state.transaction_queue:
            blockers.append(
                f"Transaction dependencies pending: {self._discovery_state.transaction_queue}"
            )

        is_complete = len(blockers) == 0
        return is_complete, blockers

    def _get_discovery_summary(self) -> str:
        """
        Get a human-readable summary of the current discovery state.

        Returns:
            Formatted string summarizing discovery progress.
        """
        lines = []
        lines.append("=== Discovery State Summary ===")

        # Root transaction
        if self._discovery_state.root_transaction:
            root = self._discovery_state.root_transaction
            lines.append(f"Root Transaction: {root.url} ({root.method.value})")
        else:
            lines.append("Root Transaction: Not set")

        # Transaction processing
        status = self._discovery_state.get_queue_status()
        lines.append(
            f"Transactions: {status['processed_count']} processed, "
            f"{status['pending_count']} pending"
        )

        # Variables
        params = [
            rv.variable for rv in self._discovery_state.all_resolved_variables
            if rv.variable.type == VariableType.PARAMETER
        ]
        tokens = [
            rv.variable for rv in self._discovery_state.all_resolved_variables
            if rv.variable.type == VariableType.DYNAMIC_TOKEN
        ]
        statics = [
            rv.variable for rv in self._discovery_state.all_resolved_variables
            if rv.variable.type == VariableType.STATIC_VALUE
        ]

        lines.append(f"Parameters: {len(params)} ({', '.join(p.name for p in params) if params else 'none'})")
        lines.append(f"Dynamic Tokens: {len(tokens)} ({', '.join(t.name for t in tokens) if tokens else 'none'})")
        lines.append(f"Static Values: {len(statics)}")

        # Routine status
        if self._discovery_state.production_routine:
            routine = self._discovery_state.production_routine
            lines.append(
                f"Routine: Constructed ({len(routine.parameters)} params, "
                f"{len(routine.operations)} operations)"
            )
        else:
            lines.append("Routine: Not constructed")

        # Completeness check
        is_complete, blockers = self._validate_discovery_completeness()
        if is_complete:
            lines.append("Status: Ready to construct routine")
        else:
            lines.append(f"Status: Not ready - {'; '.join(blockers)}")

        return "\n".join(lines)

    ## Tools - Task Management

    # Available agent types for task creation
    AVAILABLE_AGENT_TYPES = {
        SpecialistAgentType.JS_SPECIALIST,
        SpecialistAgentType.NETWORK_SPECIALIST,
        SpecialistAgentType.VALUE_TRACE_RESOLVER,
        SpecialistAgentType.INTERACTION_SPECIALIST,
    }

    @agent_tool(
        description="Create a new task for a specialist subagent (network_specialist, value_trace_resolver, js_specialist, interaction_specialist).",
        parameters={
            "type": "object",
            "properties": {
                "agent_type": {
                    "type": "string",
                    "enum": ["network_specialist", "value_trace_resolver", "js_specialist", "interaction_specialist"],
                    "description": "Type of specialist agent"
                },
                "prompt": {
                    "type": "string",
                    "description": "Task instructions for the specialist"
                },
                "agent_id": {
                    "type": "string",
                    "description": "Optional ID of existing agent to reuse"
                },
                "max_loops": {
                    "type": "integer",
                    "default": 15,
                    "description": "Maximum LLM iterations for this task"
                },
                "output_schema": {
                    "type": "object",
                    "description": "JSON Schema defining expected output structure"
                },
                "output_description": {
                    "type": "string",
                    "description": "Human-readable description of expected output"
                },
                "context": {
                    "type": "object",
                    "description": "Additional context data for the specialist"
                }
            },
            "required": ["agent_type", "prompt"]
        },
        availability=True,
    )
    def _create_task(
        self,
        agent_type: str,
        prompt: str,
        agent_id: str | None = None,
        max_loops: int = 15,
        output_schema: dict[str, Any] | None = None,
        output_description: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Create a new task for a specialist subagent.

        Args:
            agent_type: Type of specialist (js_specialist, network_specialist, value_trace_resolver, interaction_specialist).
            prompt: Task instructions for the specialist.
            agent_id: Optional ID of existing agent to reuse (preserves context).
            max_loops: Maximum LLM iterations for this task (default 15).
            output_schema: JSON Schema defining expected output structure.
            output_description: Human-readable description of expected output.
            context: Additional context data for the specialist.
        """
        try:
            parsed_type = SpecialistAgentType(agent_type)
        except ValueError:
            valid_types = [t.value for t in self.AVAILABLE_AGENT_TYPES]
            return {"error": f"Invalid agent_type. Must be one of: {valid_types}"}

        if parsed_type not in self.AVAILABLE_AGENT_TYPES:
            valid_types = [t.value for t in self.AVAILABLE_AGENT_TYPES]
            return {"error": f"Agent type '{agent_type}' not available. Use: {valid_types}"}

        task = Task(
            agent_type=parsed_type,
            agent_id=agent_id,
            prompt=prompt,
            max_loops=max_loops,
            output_schema=output_schema,
            output_description=output_description,
            context=context or {},
        )

        self._orchestration_state.add_task(task)
        self._discovery_state.phase = DiscoveryPhase.DISCOVERING

        result: dict[str, Any] = {
            "success": True,
            "task_id": task.id,
            "agent_type": agent_type,
            "message": "Task created. Use run_pending_tasks to execute.",
        }
        if output_schema:
            result["output_schema_set"] = True
        if output_description:
            result["output_description_set"] = True

        return result

    @agent_tool(
        description="List all tasks and their current status.",
        parameters={"type": "object", "properties": {}, "required": []},
        availability=True,
    )
    def _list_tasks(self) -> dict[str, Any]:
        """List all tasks and their current status."""
        tasks_summary = []
        for task in self._orchestration_state.tasks.values():
            tasks_summary.append({
                "id": task.id,
                "agent_type": task.agent_type,
                "status": task.status.value,
                "prompt": task.prompt[:100] + "..." if len(task.prompt) > 100 else task.prompt,
                "loops_used": task.loops_used,
                "max_loops": task.max_loops,
            })

        return {
            "total": len(tasks_summary),
            "pending": len(self._orchestration_state.get_pending_tasks()),
            "in_progress": len(self._orchestration_state.get_in_progress_tasks()),
            "completed": len(self._orchestration_state.get_completed_tasks()),
            "failed": len(self._orchestration_state.get_failed_tasks()),
            "tasks": tasks_summary,
        }

    @agent_tool(
        description="Get the result of a completed task.",
        parameters={
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "string",
                    "description": "The ID of the task to get results for"
                }
            },
            "required": ["task_id"]
        },
        availability=True,
    )
    def _get_task_result(self, task_id: str) -> dict[str, Any]:
        """
        Get the result of a completed task.

        Args:
            task_id: The ID of the task to get results for.
        """
        task = self._orchestration_state.tasks.get(task_id)
        if not task:
            return {"error": f"Task {task_id} not found"}

        return {
            "task_id": task.id,
            "status": task.status.value,
            "result": task.result,
            "error": task.error,
            "loops_used": task.loops_used,
        }

    @agent_tool(
        description="Execute all pending tasks and return their results.",
        parameters={"type": "object", "properties": {}, "required": []},
        availability=True,
    )
    def _run_pending_tasks(self) -> dict[str, Any]:
        """Execute all pending tasks concurrently and return their results."""
        pending = self._orchestration_state.get_pending_tasks()
        if not pending:
            return {"message": "No pending tasks", "results": []}

        if len(pending) == 1:
            # Single task — no threading overhead
            task = pending[0]
            result = self._execute_task(task)
            results = [{"task_id": task.id, "agent_type": task.agent_type, **result}]
        else:
            # Multiple independent tasks — run in parallel
            results = []
            with ThreadPoolExecutor(max_workers=len(pending)) as executor:
                future_to_task = {
                    executor.submit(self._execute_task, task): task
                    for task in pending
                }
                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    try:
                        result = future.result()
                    except Exception as e:
                        logger.error("Task %s raised exception: %s", task.id, e)
                        result = {"success": False, "error": str(e)}
                    results.append({
                        "task_id": task.id,
                        "agent_type": task.agent_type,
                        **result,
                    })
            # Preserve original task order for deterministic output
            task_order = {task.id: i for i, task in enumerate(pending)}
            results.sort(
                key=lambda r: task_order.get(r["task_id"], 0)
            )

        # Check if all tasks are done and update phase
        phase_message = None
        if not self._orchestration_state.get_pending_tasks() and not self._orchestration_state.get_in_progress_tasks():
            if self._orchestration_state.get_failed_tasks():
                phase_message = "Some tasks failed. Review results and decide next steps."
            else:
                # All tasks completed successfully
                # Check if we can transition to CONSTRUCTING
                can_construct = True
                construction_blockers = []

                # Check if root transaction is set
                if not self._discovery_state.root_transaction:
                    construction_blockers.append("No root transaction recorded (use record_identified_endpoint)")

                # Check if any unresolved dynamic tokens exist
                unresolved_tokens = []
                for tx_id, tx_data in self._discovery_state.transaction_data.items():
                    if tx_data.get("extracted_variables"):
                        extracted = tx_data["extracted_variables"]
                        resolved_names = {
                            rv.variable.name
                            for rv in tx_data.get("resolved_variables", [])
                        }
                        for var in extracted.variables:
                            if var.requires_dynamic_resolution and var.name not in resolved_names:
                                unresolved_tokens.append(var.name)

                if unresolved_tokens:
                    construction_blockers.append(
                        f"Unresolved dynamic tokens: {unresolved_tokens} "
                        f"(use value_trace_resolver and record_resolved_variable)"
                    )

                # Check if transaction queue is not empty (dependencies pending)
                if self._discovery_state.transaction_queue:
                    construction_blockers.append(
                        f"Transaction queue not empty: {self._discovery_state.transaction_queue} "
                        f"(process dependencies first)"
                    )

                if construction_blockers:
                    can_construct = False
                    phase_message = (
                        "All tasks completed, but cannot construct routine yet. Blockers: " +
                        "; ".join(construction_blockers)
                    )
                else:
                    # Can transition to CONSTRUCTING
                    self._discovery_state.phase = DiscoveryPhase.CONSTRUCTING
                    phase_message = (
                        "All tasks completed and discovery is complete! "
                        "Use get_discovery_context to see all discovered data, "
                        "then construct_routine to build the routine."
                    )

        result = {
            "executed": len(results),
            "results": results,
            "phase": self._discovery_state.phase.value,
        }

        if phase_message:
            result["phase_message"] = phase_message

        return result

    ## Tools - Data Access

    @agent_tool(
        description="[PREFER network_specialist] List transaction IDs. For finding the RIGHT endpoint, delegate to network_specialist instead - it can search semantically.",
        parameters={"type": "object", "properties": {}, "required": []},
        availability=lambda self: self._network_data_loader is not None,
    )
    def _list_transactions(self) -> dict[str, Any]:
        """List all available transaction IDs from the network captures."""
        if not self._network_data_loader:
            return {"error": "No network data store available"}

        entries = self._network_data_loader.entries
        # Filter to likely-useful API entries (skip static assets)
        static_extensions = ('.js', '.css', '.png', '.jpg', '.jpeg', '.gif', '.svg', '.ico', '.woff', '.woff2', '.ttf')
        api_entries = [e for e in entries if not any(e.url.split('?')[0].endswith(ext) for ext in static_extensions)]
        tx_summaries = [
            {"id": e.request_id, "method": e.method, "url": e.url[:100]}
            for e in api_entries
        ]
        return {
            "transactions": tx_summaries,
            "count": len(entries),
            "showing": len(tx_summaries),
            "filtered_out": len(entries) - len(api_entries),
        }

    @agent_tool(
        description="Get full details of a transaction. Use AFTER network_specialist identifies the right transaction ID.",
        parameters={
            "type": "object",
            "properties": {
                "transaction_id": {
                    "type": "string",
                    "description": "The ID of the transaction to retrieve"
                }
            },
            "required": ["transaction_id"]
        },
        availability=lambda self: self._network_data_loader is not None,
    )
    def _get_transaction(self, transaction_id: str) -> dict[str, Any]:
        """
        Get full details of a transaction.

        Args:
            transaction_id: The ID of the transaction to retrieve.
        """
        if not self._network_data_loader:
            return {"error": "No network data store available"}

        entry = self._network_data_loader.get_entry(transaction_id)
        if not entry:
            # Show some available IDs as hints
            available = [e.request_id for e in self._network_data_loader.entries[:10]]
            return {"error": f"Transaction {transaction_id} not found. Sample IDs: {available}"}

        max_body_len = 5_000
        response_body = entry.response_body
        truncated = False
        original_length = 0
        if response_body:
            original_length = len(response_body)
            if original_length > max_body_len:
                response_body = response_body[:max_body_len]
                truncated = True

        result: dict[str, Any] = {
            "transaction_id": transaction_id,
            "method": entry.method,
            "url": entry.url,
            "status": entry.status,
            "request_headers": entry.request_headers,
            "post_data": entry.post_data,
            "response_headers": entry.response_headers,
            "response_body": response_body,
        }
        if truncated:
            result["response_body_truncated"] = True
            result["response_body_full_length"] = original_length
            result["response_body_note"] = (
                f"Response body truncated to {max_body_len} chars "
                f"(full length: {original_length}). "
                f"Delegate to network_specialist for full body search."
            )
        return result

    @agent_tool(
        description=(
            "[PREFER value_trace_resolver SPECIALIST] Basic value search. "
            "For DYNAMIC_TOKENs, delegate to value_trace_resolver instead - it has deeper analysis capabilities."
        ),
        parameters={
            "type": "object",
            "properties": {
                "value": {
                    "type": "string",
                    "description": "The value to search for"
                },
                "exclude_transaction_id": {
                    "type": "string",
                    "description": "Transaction ID to exclude from search (usually the one containing the value)"
                }
            },
            "required": ["value"]
        },
        availability=True,
    )
    def _scan_for_value(
        self,
        value: str,
        exclude_transaction_id: str | None = None
    ) -> dict[str, Any]:
        """
        Search for a value across all data sources.

        Args:
            value: The value to search for.
            exclude_transaction_id: Transaction ID to exclude from search.
        """
        results: dict[str, Any] = {
            "value": value,
            "found_in": [],
        }

        # Search network transactions
        if self._network_data_loader:
            for entry in self._network_data_loader.entries:
                if exclude_transaction_id and entry.request_id == exclude_transaction_id:
                    continue

                # Search response body
                if entry.response_body and value in entry.response_body:
                    results["found_in"].append({
                        "source_type": "transaction",
                        "transaction_id": entry.request_id,
                        "location": "response_body",
                        "url": entry.url[:100],
                    })

                # Search response headers
                if entry.response_headers:
                    for header_name, header_value in entry.response_headers.items():
                        if value in str(header_value):
                            results["found_in"].append({
                                "source_type": "transaction",
                                "transaction_id": entry.request_id,
                                "location": f"response_header:{header_name}",
                                "url": entry.url[:100],
                            })

                # Search request headers
                if entry.request_headers:
                    for header_name, header_value in entry.request_headers.items():
                        if value in str(header_value):
                            results["found_in"].append({
                                "source_type": "transaction",
                                "transaction_id": entry.request_id,
                                "location": f"request_header:{header_name}",
                                "url": entry.url[:100],
                            })

                # Search request body (post_data)
                if entry.post_data:
                    post_data_str = entry.post_data if isinstance(entry.post_data, str) else json.dumps(entry.post_data)
                    if value in post_data_str:
                        results["found_in"].append({
                            "source_type": "transaction",
                            "transaction_id": entry.request_id,
                            "location": "request_body",
                            "url": entry.url[:100],
                        })

        # Search storage
        if self._storage_data_loader:
            for event in self._storage_data_loader.entries:
                if hasattr(event, 'value') and event.value and value in str(event.value):
                    results["found_in"].append({
                        "source_type": "storage",
                        "storage_type": event.storage_type if hasattr(event, 'storage_type') else "unknown",
                        "key": event.key if hasattr(event, 'key') else "unknown",
                    })

        # Search window properties
        if self._window_property_data_loader:
            for event in self._window_property_data_loader.entries:
                if hasattr(event, 'value') and event.value and value in str(event.value):
                    results["found_in"].append({
                        "source_type": "window_property",
                        "path": event.path if hasattr(event, 'path') else "unknown",
                    })

        results["total_matches"] = len(results["found_in"])
        return results

    ## Tools - State Population

    @agent_tool(
        description="Record the main transaction identified (root transaction for routine).",
        parameters={
            "type": "object",
            "properties": {
                "request_id": {
                    "type": "string",
                    "description": "The transaction ID (HAR entry ID)"
                },
                "url": {
                    "type": "string",
                    "description": "The URL of the endpoint"
                },
                "method": {
                    "type": "string",
                    "enum": ["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"],
                    "description": "HTTP method"
                },
                "description": {
                    "type": "string",
                    "description": "What this transaction does"
                }
            },
            "required": ["request_id", "url", "method", "description"]
        },
        availability=lambda self: self._network_data_loader is not None,
    )
    def _record_identified_endpoint(
        self,
        request_id: str,
        url: str,
        method: str,
        description: str
    ) -> dict[str, Any]:
        """
        Record the main transaction identified by network_specialist.
        This becomes the root_transaction in discovery state.

        Args:
            request_id: The HAR entry ID from network_specialist results.
            url: The URL of the endpoint.
            method: HTTP method (GET, POST, etc).
            description: What this transaction does.
        """
        # Validate request_id exists in network data
        if not self._network_data_loader:
            return {"error": "No network data loader available"}

        entry = self._network_data_loader.get_entry(request_id)
        if not entry:
            available_ids = [e.request_id for e in self._network_data_loader.entries[:10]]
            return {
                "error": f"Request ID '{request_id}' not found",
                "sample_ids": available_ids
            }

        # Parse HTTP method
        try:
            http_method = HTTPMethod(method.upper())
        except ValueError:
            return {"error": f"Invalid HTTP method '{method}'. Use GET, POST, PUT, DELETE, etc."}

        # Create TransactionIdentificationResponse
        root_transaction = TransactionIdentificationResponse(
            transaction_id=request_id,
            description=description,
            url=url,
            method=http_method,
            short_explanation=f"Main endpoint for {description}"
        )

        # Store in discovery state
        self._discovery_state.root_transaction = root_transaction

        # Add to transaction queue
        added, position = self._discovery_state.add_to_queue(request_id)

        # Initialize transaction data
        self._discovery_state.store_transaction_data(
            transaction_id=request_id,
            request={
                "url": entry.url,
                "method": entry.method,
                "headers": entry.request_headers,
                "body": entry.post_data,
            }
        )

        # Transition to DISCOVERING phase
        self._discovery_state.phase = DiscoveryPhase.DISCOVERING

        return {
            "success": True,
            "transaction_id": request_id,
            "added_to_queue": added,
            "queue_position": position,
            "message": f"Recorded root transaction: {url}"
        }

    @agent_tool(
        description="Record a variable discovered from analyzing a transaction (parameter, dynamic_token, or static_value).",
        parameters={
            "type": "object",
            "properties": {
                "transaction_id": {
                    "type": "string",
                    "description": "The transaction this variable belongs to"
                },
                "name": {
                    "type": "string",
                    "description": "Variable name (e.g., 'origin_city', 'x-trace-id')"
                },
                "type": {
                    "type": "string",
                    "enum": ["parameter", "dynamic_token", "static_value"],
                    "description": "Variable type"
                },
                "observed_value": {
                    "type": "string",
                    "description": "The actual value seen in the capture"
                },
                "requires_dynamic_resolution": {
                    "type": "boolean",
                    "description": "True if value must be resolved at runtime"
                },
                "values_to_scan_for": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of values to search for"
                }
            },
            "required": ["transaction_id", "name", "type", "observed_value", "requires_dynamic_resolution"]
        },
        availability=lambda self: self._discovery_state.root_transaction is not None,
    )
    def _record_extracted_variable(
        self,
        transaction_id: str,
        name: str,
        type: str,
        observed_value: str,
        requires_dynamic_resolution: bool,
        values_to_scan_for: list[str] | None = None
    ) -> dict[str, Any]:
        """
        Record a variable discovered from analyzing a transaction.

        Args:
            transaction_id: The transaction this variable belongs to.
            name: Variable name (e.g., "origin_city", "x-trace-id").
            type: Variable type - "parameter", "dynamic_token", or "static_value".
            observed_value: The actual value seen in the capture.
            requires_dynamic_resolution: True if value must be resolved at runtime.
            values_to_scan_for: Optional list of values to search for (defaults to [observed_value]).
        """
        # Validate variable type
        try:
            var_type = VariableType(type)
        except ValueError:
            return {
                "error": f"Invalid variable type '{type}'. Use: parameter, dynamic_token, or static_value"
            }

        # Create Variable object
        variable = Variable(
            type=var_type,
            requires_dynamic_resolution=requires_dynamic_resolution,
            name=name,
            observed_value=observed_value,
            values_to_scan_for=values_to_scan_for or [observed_value]
        )

        # Check if transaction_data exists for this transaction
        if transaction_id not in self._discovery_state.transaction_data:
            self._discovery_state.transaction_data[transaction_id] = {
                "request": None,
                "extracted_variables": None,
                "resolved_variables": []
            }

        # Get or create ExtractedVariableResponse
        tx_data = self._discovery_state.transaction_data[transaction_id]
        if tx_data.get("extracted_variables") is None:
            extracted = ExtractedVariableResponse(
                transaction_id=transaction_id,
                variables=[variable]
            )
            tx_data["extracted_variables"] = extracted
        else:
            # Add to existing variables
            tx_data["extracted_variables"].variables.append(variable)

        return {
            "success": True,
            "transaction_id": transaction_id,
            "variable_name": name,
            "variable_type": type,
            "requires_resolution": requires_dynamic_resolution,
            "message": f"Recorded variable '{name}' for transaction {transaction_id}"
        }

    @agent_tool(
        description="Record how to resolve a dynamic token (storage, window_property, or transaction source). Auto-adds dependency transactions.",
        parameters={
            "type": "object",
            "properties": {
                "variable_name": {
                    "type": "string",
                    "description": "Name of the variable being resolved"
                },
                "transaction_id": {
                    "type": "string",
                    "description": "The transaction this variable belongs to"
                },
                "source_type": {
                    "type": "string",
                    "enum": ["storage", "window_property", "transaction"],
                    "description": "Where the value comes from"
                },
                "storage_source": {
                    "type": "object",
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": ["cookie", "localStorage", "sessionStorage"]
                        },
                        "dot_path": {"type": "string"}
                    },
                    "description": "For storage source"
                },
                "window_property_source": {
                    "type": "object",
                    "properties": {
                        "dot_path": {"type": "string"}
                    },
                    "description": "For window source"
                },
                "transaction_source": {
                    "type": "object",
                    "properties": {
                        "transaction_id": {"type": "string"},
                        "dot_path": {"type": "string"}
                    },
                    "description": "For transaction source"
                }
            },
            "required": ["variable_name", "transaction_id", "source_type"]
        },
        availability=lambda self: self._discovery_state.root_transaction is not None,
    )
    def _record_resolved_variable(
        self,
        variable_name: str,
        transaction_id: str,
        source_type: str,
        storage_source: dict[str, str] | None = None,
        window_property_source: dict[str, str] | None = None,
        transaction_source: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """
        Record how to resolve a dynamic token.

        Args:
            variable_name: Name of the variable being resolved.
            transaction_id: The transaction this variable belongs to.
            source_type: Where the value comes from ("storage", "window_property", "transaction").
            storage_source: For storage source - {"type": "cookie|localStorage|sessionStorage", "dot_path": "path"}.
            window_property_source: For window property source - {"dot_path": "path"}.
            transaction_source: For transaction source - {"transaction_id": "id", "dot_path": "path"}.
        """
        # Get the variable from extracted variables
        tx_data = self._discovery_state.transaction_data.get(transaction_id)
        if not tx_data or not tx_data.get("extracted_variables"):
            return {"error": f"No extracted variables found for transaction {transaction_id}"}

        extracted = tx_data["extracted_variables"]
        variable = None
        for var in extracted.variables:
            if var.name == variable_name:
                variable = var
                break

        if not variable:
            available = [v.name for v in extracted.variables]
            return {
                "error": f"Variable '{variable_name}' not found in transaction {transaction_id}",
                "available_variables": available
            }

        # Build the source object based on source_type
        source = None
        dependency_added = False

        if source_type == "storage":
            if not storage_source:
                return {"error": "storage_source required for source_type='storage'"}
            try:
                storage_type = SessionStorageType(storage_source["type"])
            except (KeyError, ValueError):
                return {"error": "storage_source must have 'type' (cookie, localStorage, sessionStorage) and 'dot_path'"}
            source = SessionStorageSource(
                type=storage_type,
                dot_path=storage_source.get("dot_path", "")
            )

        elif source_type == "window_property":
            if not window_property_source:
                return {"error": "window_property_source required for source_type='window_property'"}
            source = WindowPropertySource(
                dot_path=window_property_source.get("dot_path", "")
            )

        elif source_type == "transaction":
            if not transaction_source:
                return {"error": "transaction_source required for source_type='transaction'"}
            source_tx_id = transaction_source.get("transaction_id")
            if not source_tx_id:
                return {"error": "transaction_source must have 'transaction_id' and 'dot_path'"}

            dot_path = transaction_source.get("dot_path", "")

            # Validate that dot_path resolves in the source transaction's response
            if dot_path and self._network_data_loader:
                source_entry = self._network_data_loader.get_entry(source_tx_id)
                if source_entry and source_entry.response_body:
                    resolved_value = resolve_dotted_path(logger, source_entry.response_body, dot_path)
                    if resolved_value is None:
                        return {
                            "error": (
                                f"dot_path '{dot_path}' does not resolve to a value in transaction {source_tx_id}'s "
                                "response body. Verify the path is correct."
                            )
                        }

            source = TransactionSource(
                transaction_id=source_tx_id,
                dot_path=dot_path,
            )

            # Auto-add dependency transaction to queue
            added, position = self._discovery_state.add_to_queue(source_tx_id)
            if added:
                dependency_added = True
                # Initialize transaction data for dependency if not exists
                if source_tx_id not in self._discovery_state.transaction_data:
                    entry = self._network_data_loader.get_entry(source_tx_id) if self._network_data_loader else None
                    if entry:
                        self._discovery_state.store_transaction_data(
                            transaction_id=source_tx_id,
                            request={
                                "url": entry.url,
                                "method": entry.method,
                                "headers": entry.request_headers,
                                "body": entry.post_data,
                            }
                        )

        else:
            return {"error": f"Invalid source_type '{source_type}'. Use: storage, window_property, transaction"}

        # Create ResolvedVariableResponse
        resolved = ResolvedVariableResponse(
            variable=variable,
            source=source
        )

        # Store in transaction data
        if "resolved_variables" not in tx_data:
            tx_data["resolved_variables"] = []
        tx_data["resolved_variables"].append(resolved)

        result = {
            "success": True,
            "variable_name": variable_name,
            "source_type": source_type,
            "message": f"Recorded resolution for '{variable_name}'"
        }

        if dependency_added:
            result["dependency_added"] = source_tx_id
            result["message"] += f" (dependency transaction {source_tx_id} added to queue)"

        return result

    @agent_tool(
        description="Mark a transaction as fully processed (all variables extracted and resolved). Removes from queue.",
        parameters={
            "type": "object",
            "properties": {
                "transaction_id": {
                    "type": "string",
                    "description": "The transaction ID to mark as processed"
                }
            },
            "required": ["transaction_id"]
        },
        availability=lambda self: self._discovery_state.root_transaction is not None,
    )
    def _mark_transaction_processed(self, transaction_id: str) -> dict[str, Any]:
        """
        Mark a transaction as fully processed.

        Call this when you've extracted all variables and resolved all dynamic tokens
        for a transaction. This removes it from the queue and adds it to processed list.

        Args:
            transaction_id: The transaction ID to mark as processed.
        """
        # Check if transaction exists in our data
        if transaction_id not in self._discovery_state.transaction_data:
            return {"error": f"Transaction {transaction_id} not found in discovery data"}

        # Check for unresolved dynamic tokens
        tx_data = self._discovery_state.transaction_data[transaction_id]
        unresolved = []
        if tx_data.get("extracted_variables"):
            resolved_names = {
                rv.variable.name
                for rv in tx_data.get("resolved_variables", [])
            }
            for var in tx_data["extracted_variables"].variables:
                if var.requires_dynamic_resolution and var.name not in resolved_names:
                    unresolved.append(var.name)

        if unresolved:
            return {
                "error": f"Cannot mark as processed - unresolved dynamic tokens: {unresolved}",
                "hint": "Use scan_for_value and record_resolved_variable for each token first"
            }

        # Remove from queue if present
        if transaction_id in self._discovery_state.transaction_queue:
            self._discovery_state.transaction_queue.remove(transaction_id)

        # Mark as processed
        self._discovery_state.mark_transaction_complete(transaction_id)

        # Get next transaction in queue
        queue_status = self._discovery_state.get_queue_status()

        return {
            "success": True,
            "transaction_id": transaction_id,
            "message": f"Transaction {transaction_id} marked as processed",
            "remaining_queue": queue_status["pending"],
            "processed_count": queue_status["processed_count"],
        }

    @agent_tool()
    def _get_discovery_context(self) -> dict[str, Any]:
        """Get complete discovery context for routine construction."""
        # Build CRITICAL observed values reminder - this goes at the TOP
        observed_values_for_params: dict[str, str] = {}
        for tx_id, tx_data in self._discovery_state.transaction_data.items():
            if tx_data.get("extracted_variables"):
                for var in tx_data["extracted_variables"].variables:
                    if var.type == VariableType.PARAMETER and var.observed_value:
                        observed_values_for_params[var.name] = var.observed_value

        context: dict[str, Any] = {
            "phase": self._discovery_state.phase.value,
            "CRITICAL_OBSERVED_VALUES": {
                "message": "YOU MUST INCLUDE THESE observed_value FIELDS WHEN CONSTRUCTING ROUTINE PARAMETERS!",
                "parameters_with_observed_values": observed_values_for_params,
            },
            "root_transaction": None,
            "processed_transactions": [],
            "all_variables": {
                "parameters": [],
                "dynamic_tokens": [],
                "static_values": [],
            },
            "resolution_map": {},
            "summary": self._get_discovery_summary(),
        }

        # Root transaction
        if self._discovery_state.root_transaction:
            root = self._discovery_state.root_transaction
            context["root_transaction"] = {
                "transaction_id": root.transaction_id,
                "url": root.url,
                "method": root.method.value,
                "description": root.description,
            }

        # Process all transaction data
        for tx_id, tx_data in self._discovery_state.transaction_data.items():
            tx_summary = {
                "transaction_id": tx_id,
                "request": tx_data.get("request"),
                "variables": [],
            }

            if tx_data.get("extracted_variables"):
                for var in tx_data["extracted_variables"].variables:
                    var_info = {
                        "name": var.name,
                        "type": var.type.value,
                        "observed_value": var.observed_value,
                        "requires_resolution": var.requires_dynamic_resolution,
                    }
                    tx_summary["variables"].append(var_info)

                    # Categorize by type
                    if var.type == VariableType.PARAMETER:
                        context["all_variables"]["parameters"].append(var_info)
                    elif var.type == VariableType.DYNAMIC_TOKEN:
                        context["all_variables"]["dynamic_tokens"].append(var_info)
                    else:
                        context["all_variables"]["static_values"].append(var_info)

            # Add resolution info
            if tx_data.get("resolved_variables"):
                for resolved in tx_data["resolved_variables"]:
                    source_info = {}
                    if isinstance(resolved.source, SessionStorageSource):
                        source_info = {
                            "type": "storage",
                            "storage_type": resolved.source.type.value,
                            "dot_path": resolved.source.dot_path,
                        }
                    elif isinstance(resolved.source, WindowPropertySource):
                        source_info = {
                            "type": "window_property",
                            "dot_path": resolved.source.dot_path,
                        }
                    elif isinstance(resolved.source, TransactionSource):
                        source_info = {
                            "type": "transaction",
                            "transaction_id": resolved.source.transaction_id,
                            "dot_path": resolved.source.dot_path,
                        }

                    context["resolution_map"][resolved.variable.name] = source_info

            context["processed_transactions"].append(tx_summary)

        # Completeness check
        is_complete, blockers = self._validate_discovery_completeness()
        context["is_complete"] = is_complete
        context["blockers"] = blockers

        return context

    ## Tools - Routine Construction

    @agent_tool(
        description="Construct a routine from discovered data. After constructing, use validate_routine to test it.",
        parameters={
            "type": "object",
            "properties": {
                "routine": {
                    "type": "object",
                    "description": "The routine to construct.",
                    "properties": {
                        "name": {"type": "string", "description": "Routine name"},
                        "description": {"type": "string", "description": "What the routine does"},
                        "parameters": {
                            "type": "array",
                            "description": "Input parameters. Each needs: name, type (string|number|boolean|date|enum), description.",
                            "items": {"type": "object"},
                        },
                        "operations": {
                            "type": "array",
                            "description": (
                                "Ordered operations. Each needs a 'type' field: "
                                "navigate|fetch|return|sleep|click|input_text|press|"
                                "wait_for_url|scroll|get_cookies|download|return_html|js_evaluate. "
                                "Key schemas — navigate: {type, url}. "
                                "fetch: {type, endpoint: {url, method, headers?, body?}, session_storage_key}. "
                                "return: {type, session_storage_key, tables?}. "
                                "Use {{paramName}} placeholders in URLs/bodies for parameters."
                            ),
                            "items": {"type": "object"},
                        },
                    },
                    "required": ["name", "description", "parameters", "operations"],
                },
            },
            "required": ["routine"],
        },
        availability=lambda self: (
            self._discovery_state.root_transaction is not None and
            not self._discovery_state.transaction_queue
        ),
    )
    def _construct_routine(
        self,
        routine: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Construct a routine from discovered data (no execution).

        After constructing, use validate_routine to test it with parameters.

        Args:
            routine: The routine dict with name, description, parameters, and operations.
        """
        self._discovery_state.phase = DiscoveryPhase.CONSTRUCTING
        self._discovery_state.construction_attempts += 1

        # Reset validation state when routine is (re)constructed
        self._discovery_state.last_validation_result = None
        self._discovery_state.validation_analyzed = False
        self._discovery_state.last_analysis = None

        try:
            routine_obj = Routine.model_validate(routine)
        except Exception as e:
            return {
                "error": f"Invalid routine structure: {e}",
                "message": "Failed to parse routine. Check schema in the docs and try again.",
            }

        # Get structure warnings (errors are already caught by model validation above)
        structure_warnings = routine_obj.get_structure_warnings()

        try:
            self._discovery_state.production_routine = routine_obj

            return {
                "success": True,
                "routine_name": routine_obj.name,
                "parameter_count": len(routine_obj.parameters),
                "operation_count": len(routine_obj.operations),
                "warnings": structure_warnings,
                "message": "Routine constructed. Now use validate_routine with test_parameters to execute and verify it works.",
            }

        except Exception as e:
            return {
                "error": str(e),
                "message": "Failed to construct routine. Check schema in the docs and try again.",
            }

    @agent_tool(
        description=(
            "Execute the constructed routine with test parameters to validate it works. "
            "Only available when browser is connected."
        ),
        parameters={
            "type": "object",
            "properties": {
                "test_parameters": {
                    "type": "object",
                    "description": (
                        "Test parameter values from observed data. "
                        "Map of parameter_name -> observed_value. "
                        "Example: {\"origin\": \"NYC\", \"destination\": \"BOS\"}. "
                        "Get these from the extracted variables' observed_value fields."
                    ),
                    "additionalProperties": {"type": "string"},
                },
            },
            "required": ["test_parameters"],
        },
        availability=lambda self: (
            self._discovery_state.production_routine is not None and
            self._remote_debugging_address is not None  # Require browser connection
        ),
    )
    def _validate_routine(
        self,
        test_parameters: dict[str, str],
    ) -> dict[str, Any]:
        """
        Execute the constructed routine with test parameters to validate it works.

        After validation, use analyze_validation to reflect on results before calling done.

        Args:
            test_parameters: Map of parameter names to observed values for testing.
        """
        if not self._discovery_state.production_routine:
            return {"error": "No routine constructed. Use construct_routine first."}

        self._discovery_state.phase = DiscoveryPhase.VALIDATING
        self._discovery_state.validation_attempts += 1

        # Store test_parameters in discovery state
        self._discovery_state.test_parameters = test_parameters

        # Reset analysis state
        self._discovery_state.validation_analyzed = False
        self._discovery_state.last_analysis = None

        routine_obj = self._discovery_state.production_routine

        # Import here to avoid circular dependency
        from bluebox.llms.tools.execute_routine_tool import execute_routine

        result = execute_routine(
            routine=routine_obj.model_dump(),
            parameters=test_parameters,
            remote_debugging_address=self._remote_debugging_address,
            timeout=60,
            close_tab_when_done=True,
        )

        # Store full result for analysis
        if result.get("success"):
            exec_result = result.get("result")
            self._discovery_state.last_validation_result = {
                "success": True,
                "exec_result": exec_result.model_dump() if exec_result else None,
                "data_returned": exec_result.data is not None if exec_result else False,
            }

            if exec_result and exec_result.ok and exec_result.data is not None:
                return {
                    "routine_name": routine_obj.name,
                    "execution_success": True,
                    "data_returned": True,
                    "data_preview": str(exec_result.data)[:500],
                    "message": "Routine executed successfully with data. Use analyze_validation to reflect on results.",
                }
            else:
                return {
                    "routine_name": routine_obj.name,
                    "execution_success": True,
                    "data_returned": False,
                    "exec_result": exec_result.model_dump() if exec_result else None,
                    "message": (
                        "Routine executed but 'data' field is missing or empty. "
                        "Use analyze_validation to decide next steps."
                    ),
                }
        else:
            self._discovery_state.last_validation_result = {
                "success": False,
                "error": result.get("error", "Unknown error"),
            }
            return {
                "routine_name": routine_obj.name,
                "execution_success": False,
                "error": result.get("error", "Unknown error"),
                "message": "Routine execution failed. Use analyze_validation to decide next steps.",
            }

    @agent_tool(
        description="Analyze validation results and decide next steps. REQUIRED before calling done().",
        parameters={
            "type": "object",
            "properties": {
                "analysis": {
                    "type": "string",
                    "description": "Your analysis of what worked and what failed in the validation.",
                },
                "data_matches_task": {
                    "type": "boolean",
                    "description": "Does the returned data accomplish the original task the user requested?",
                },
                "next_action": {
                    "type": "string",
                    "enum": ["done", "fix_routine", "retry_validation"],
                    "description": (
                        "What to do next: 'done' if successful, 'fix_routine' to modify routine, "
                        "'retry_validation' to re-run."
                    ),
                },
            },
            "required": ["analysis", "data_matches_task", "next_action"],
        },
        availability=lambda self: (
            self._discovery_state.last_validation_result is not None and
            not self._discovery_state.validation_analyzed
        ),
    )
    def _analyze_validation(
        self,
        analysis: str,
        data_matches_task: bool,
        next_action: str,
    ) -> dict[str, Any]:
        """
        Analyze validation results and decide next steps. Required before calling done().

        Args:
            analysis: Your analysis of what worked and what failed.
            data_matches_task: Does the returned data accomplish the original task?
            next_action: What to do next - 'done', 'fix_routine', or 'retry_validation'.
        """
        if self._discovery_state.last_validation_result is None:
            return {"error": "No validation result to analyze. Use validate_routine first."}

        # Validate next_action
        valid_actions = ["done", "fix_routine", "retry_validation"]
        if next_action not in valid_actions:
            return {"error": f"Invalid next_action. Must be one of: {valid_actions}"}

        # Store the analysis
        self._discovery_state.last_analysis = {
            "analysis": analysis,
            "data_matches_task": data_matches_task,
            "next_action": next_action,
        }
        self._discovery_state.validation_analyzed = True

        # Check for inconsistency: can't say "done" if data doesn't match task
        if next_action == "done" and not data_matches_task:
            return {
                "error": "Inconsistent analysis: next_action is 'done' but data_matches_task is False.",
                "message": "If data doesn't match the task, you must fix the routine first.",
                "hint": "Set next_action to 'fix_routine' and update the routine to return correct data.",
            }

        # Check validation result
        validation_result = self._discovery_state.last_validation_result
        validation_failed = not validation_result.get("success", False) and not validation_result.get("skipped", False)

        if next_action == "done" and validation_failed:
            return {
                "error": "Cannot mark as done when validation failed.",
                "message": "Fix the routine and re-validate before completing.",
            }

        # Return guidance based on next_action
        if next_action == "done":
            return {
                "success": True,
                "message": "Analysis recorded. You may now call done() to complete discovery.",
                "analysis_summary": {
                    "analysis": analysis,
                    "data_matches_task": data_matches_task,
                },
            }
        elif next_action == "fix_routine":
            return {
                "success": True,
                "message": "Analysis recorded. Use construct_routine to fix the routine, then validate_routine again.",
                "analysis_summary": {
                    "analysis": analysis,
                    "data_matches_task": data_matches_task,
                },
            }
        else:  # retry_validation
            # Reset for retry
            self._discovery_state.validation_analyzed = False
            self._discovery_state.last_analysis = None
            return {
                "success": True,
                "message": "Analysis recorded. Use validate_routine to retry validation.",
                "analysis_summary": {
                    "analysis": analysis,
                    "data_matches_task": data_matches_task,
                },
            }

    ## Tools - Completion

    def _can_complete(self) -> bool:
        """Check if discovery can be marked complete."""
        # Must have a routine
        if not self._discovery_state.production_routine:
            return False

        # If no browser connected, can complete without validation
        # (we can't execute routines without a browser)
        if not self._remote_debugging_address:
            return True

        # With browser: must have validated and analyzed successfully
        if not self._discovery_state.validation_analyzed:
            return False

        analysis = self._discovery_state.last_analysis
        if not analysis:
            return False

        return analysis.get("data_matches_task", False)

    @agent_tool(
        availability=lambda self: self._can_complete(),
    )
    def _done(self) -> dict[str, Any]:
        """Mark discovery as complete. Available after construct_routine (no browser) or successful analyze_validation (with browser)."""
        if not self._discovery_state.production_routine:
            return {"error": "No routine constructed. Use construct_routine first."}

        # If browser connected, require successful validation analysis
        if self._remote_debugging_address:
            if not self._discovery_state.validation_analyzed:
                return {"error": "Validation not analyzed. Use validate_routine then analyze_validation first."}

            analysis = self._discovery_state.last_analysis
            if not analysis:
                return {"error": "No analysis found. Use analyze_validation first."}

            if not analysis.get("data_matches_task", False):
                return {
                    "error": "Cannot complete when data doesn't match task.",
                    "message": "Fix the routine with construct_routine, then validate_routine and analyze_validation again.",
                }

        self._discovery_state.phase = DiscoveryPhase.COMPLETE
        self._final_routine = self._discovery_state.production_routine

        # Note if routine was not validated
        message = "Discovery completed"
        if not self._remote_debugging_address:
            message += " (routine not validated - no browser connected)"

        return {
            "success": True,
            "message": message,
            "routine_name": self._final_routine.name,
        }

    @agent_tool(
        availability=lambda self: (
            self._discovery_state.root_transaction is None
            or self._discovery_state.construction_attempts >= 5
        ),
    )
    def _fail(self, reason: str) -> dict[str, Any]:
        """
        Mark discovery as failed.

        Args:
            reason: Why discovery could not be completed.
        """
        self._discovery_state.phase = DiscoveryPhase.FAILED
        self._failure_reason = reason
        return {
            "success": False,
            "message": "Discovery marked as failed",
            "reason": reason,
        }
