"""
bluebox/agents/super_discovery_agent.py

SuperDiscoveryAgent - orchestrator for routine discovery.

This agent coordinates specialist subagents (JSSpecialist, NetworkSpyAgent, etc.)
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

from datetime import datetime
from textwrap import dedent
from typing import Any, Callable

from pydantic import BaseModel

from bluebox.agents.abstract_agent import AbstractAgent, agent_tool
from bluebox.agents.specialists.abstract_specialist import AbstractSpecialist, AutonomousConfig
from bluebox.agents.specialists.docs_digger_agent import DocsDiggerAgent
from bluebox.agents.specialists.js_specialist import JSSpecialist
from bluebox.agents.specialists.network_spy_agent import NetworkSpyAgent
from bluebox.agents.specialists.trace_hound_agent import TraceHoundAgent
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
from bluebox.llms.data_loaders.js_data_loader import JSDataLoader
from bluebox.llms.data_loaders.network_data_loader import NetworkDataLoader
from bluebox.llms.data_loaders.storage_data_loader import StorageDataLoader
from bluebox.llms.data_loaders.window_property_data_loader import WindowPropertyDataLoader
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

    ## System prompts

    PLACEHOLDER_INSTRUCTIONS: str = (
        "PLACEHOLDER SYNTAX:\n"
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

    SYSTEM_PROMPT: str = dedent("""\
        You are an expert at analyzing network traffic and building web automation routines.
        You coordinate specialist agents to help you discover and construct routines.

        ## Your Task
        Analyze captured browser network data to create a reusable routine that accomplishes the user's task.

        ## Workflow
        Follow these phases in order:

        ### Phase 1: Identify Transaction
        1. Use `list_transactions` to see available transactions
        2. Use `get_transaction` to examine promising candidates
        3. Use `record_identified_endpoint` when you find the transaction that accomplishes the user's task

        ### Phase 2: Process Transactions (BFS Queue)
        For each transaction in the queue:
        1. Use `get_transaction` to see full details
        2. Use `record_extracted_variable` to log variables found in the request:
           - PARAMETER: User input (search_query, item_id) - things the user explicitly provides
           - DYNAMIC_TOKEN: Auth/session values (CSRF, JWT, session_id) - require resolution
           - STATIC_VALUE: Constants (app version, User-Agent) - can be hardcoded
        3. For each DYNAMIC_TOKEN, use `scan_for_value` to find its source
        4. Use `record_resolved_variable` to record where each token comes from
           - If source is another transaction, it will be auto-added to the queue
           - IMPORTANT: If value is found in BOTH storage AND a prior transaction,
             use source_type='transaction' as the primary source. Session storage may
             be empty in a fresh session - prefer network sources for reliability.
        5. Continue until queue is empty

        ### Phase 3: Construct and Finalize Routine
        1. Use `get_discovery_context` to see all processed data
        2. **IMPORTANT**: If you need help with routine structure, use docs_digger specialist:
           - create_task(agent_type="docs_digger", prompt="Find complete example routines with fetch operations. Show the full structure including parameters and operations.")
           - run_pending_tasks() then get_task_result() to get examples
           - This is MUCH better than search_documentation which may return 0 results
        3. Use `construct_routine` to build the routine from all processed data
           - For each parameter, specify `observed_value` from the extracted variable's observed_value
           - Example: parameter name "origin" with observed_value "LAX" (from extracted variable's observed_value)
           - Observed values are embedded in the routine for immediate testing
        4. construct_routine AUTOMATICALLY executes the routine and returns results
        5. Review execution results:
           - If execution_success=True: call `done`
           - If execution_success=False: fix issues and call construct_routine again

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

        ## Specialist Agents (Optional - Use When Needed)

        You have 4 specialist agents available. Delegate to them when you need help:

        ### 1. network_spy - Endpoint Discovery Expert
        - Searches network traffic by keywords to find relevant API endpoints
        - Use when you need help finding which endpoint does what

        ### 2. trace_hound - Value Origin Detective
        - Traces where values come from across transactions
        - Use when scan_for_value isn't enough or you need deep analysis

        ### 3. js_specialist - Browser JavaScript Expert
        - Writes IIFE JavaScript for DOM manipulation and extraction
        - Use when routine needs browser-side JavaScript execution

        ### 4. docs_digger - Schema & Documentation Expert
        - Searches documentation for schemas and examples
        - Use when you need to understand routine structure or see examples

        To delegate: create_task(agent_type="network_spy", prompt="..."), then run_pending_tasks()

        ## Important Notes
        - Focus on the user's INTENT, not literal wording
        - Keep parameters MINIMAL - only what the user MUST provide
        - If only one value was observed and it could be hardcoded, hardcode it
        - Credentials for fetch operations: same-origin > include > omit
        - PREFER NETWORK SOURCES: When a value appears in both session storage AND a prior
          transaction response, use source_type='transaction' as the PRIMARY source, not storage.
          Session storage may be empty in a fresh session, so the routine must fetch via network.

        {placeholder_instructions}
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
        self._documentation_data_loader = documentation_data_loader
        self._task = task
        self._subagent_llm_model = subagent_llm_model or llm_model
        self._max_iterations = max_iterations
        self._remote_debugging_address = remote_debugging_address

        # Internal state
        self._orchestration_state = AgentOrchestrationState()
        self._discovery_state = RoutineDiscoveryState()
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
        )

    ## Abstract method implementations

    def _get_system_prompt(self) -> str:
        """Build the complete system prompt with current state."""
        prompt_parts = [self.SYSTEM_PROMPT.format(
            placeholder_instructions=self.PLACEHOLDER_INSTRUCTIONS
        )]

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
            stats = self._documentation_data_loader.stats
            data_store_info.append(f"Documentation: {stats.total_docs} docs, {stats.total_code} code files")

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
        # Seed the conversation
        initial_message = f"TASK: {self._task}\n\nAnalyze the network captures and build a routine."
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
                response = self._call_llm(messages, self._get_system_prompt())

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
                            "Phase: PLANNING. Skip manual inspection and delegate immediately. "
                            "Create specialist tasks (network_spy, trace_hound) to analyze the data."
                        )
                    elif phase == DiscoveryPhase.DISCOVERING:
                        task_status = self._orchestration_state.get_queue_status()
                        if task_status["pending_tasks"] > 0:
                            guidance = (
                                f"Phase: DISCOVERING. You have {task_status['pending_tasks']} pending tasks. "
                                "Call run_pending_tasks to execute them."
                            )
                        elif task_status["completed_tasks"] > 0:
                            guidance = (
                                "Phase: DISCOVERING. Tasks completed. Review results with get_task_result, "
                                "then record findings using record_identified_endpoint, record_extracted_variable, "
                                "and record_resolved_variable tools."
                            )
                        else:
                            guidance = (
                                "Phase: DISCOVERING. No tasks created yet. "
                                "Create specialist tasks to analyze the data."
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
                remote_debugging_address=self._remote_debugging_address,
            )

        elif agent_type == SpecialistAgentType.TRACE_HOUND:
            return TraceHoundAgent(
                emit_message_callable=self._emit_message_callable,
                llm_model=self._subagent_llm_model,
                network_data_store=self._network_data_loader,
                storage_data_store=self._storage_data_loader,
                window_property_data_store=self._window_property_data_loader,
            )

        elif agent_type == SpecialistAgentType.NETWORK_SPY:
            if not self._network_data_loader:
                raise ValueError(
                    "network_spy specialist requires network_data_store, "
                    "but it was not provided to SuperDiscoveryAgent"
                )
            return NetworkSpyAgent(
                emit_message_callable=self._emit_message_callable,
                llm_model=self._subagent_llm_model,
                network_data_store=self._network_data_loader,
            )

        elif agent_type == SpecialistAgentType.DOCS_DIGGER:
            if not self._documentation_data_loader:
                raise ValueError(
                    "DocsDiggerAgent requires documentation_data_store. "
                    "Ensure SuperDiscoveryAgent was initialized with documentation_data_store."
                )
            return DocsDiggerAgent(
                emit_message_callable=self._emit_message_callable,
                llm_model=self._subagent_llm_model,
                documentation_data_store=self._documentation_data_loader,
            )

        else:
            # For other agent types that aren't fully implemented yet
            raise NotImplementedError(
                f"Agent type {agent_type.value} is not yet supported. "
                f"Available types: js_specialist, trace_hound, docs_digger, network_spy"
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

            # Run autonomous with config
            config = AutonomousConfig(
                min_iterations=1,  # Allow immediate finalization for resumed tasks
                max_iterations=remaining_loops,
            )

            result = agent.run_autonomous(task.prompt, config)

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
        SpecialistAgentType.TRACE_HOUND,
        SpecialistAgentType.DOCS_DIGGER,
        SpecialistAgentType.NETWORK_SPY,
    }

    @agent_tool(
        description="Create a new task for a specialist subagent (network_spy, trace_hound, js_specialist, docs_digger).",
        parameters={
            "type": "object",
            "properties": {
                "agent_type": {
                    "type": "string",
                    "enum": ["network_spy", "trace_hound", "js_specialist", "docs_digger"],
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
    ) -> dict[str, Any]:
        """
        Create a new task for a specialist subagent.

        Args:
            agent_type: Type of specialist (js_specialist, trace_hound, network_spy, docs_digger).
            prompt: Task instructions for the specialist.
            agent_id: Optional ID of existing agent to reuse (preserves context).
            max_loops: Maximum LLM iterations for this task (default 15).
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
        )

        self._orchestration_state.add_task(task)
        self._discovery_state.phase = DiscoveryPhase.DISCOVERING

        return {
            "success": True,
            "task_id": task.id,
            "agent_type": agent_type,
            "message": f"Task created. Use run_pending_tasks to execute.",
        }

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
                "agent_type": task.agent_type.value,
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
        """Execute all pending tasks and return their results."""
        pending = self._orchestration_state.get_pending_tasks()
        if not pending:
            return {"message": "No pending tasks", "results": []}

        results = []
        for task in pending:
            result = self._execute_task(task)
            results.append({
                "task_id": task.id,
                "agent_type": task.agent_type.value,
                **result,
            })

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
                        f"(use trace_hound and record_resolved_variable)"
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
        Record the main transaction identified by network_spy.
        This becomes the root_transaction in discovery state.

        Args:
            request_id: The HAR entry ID from network_spy results.
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
        Typically called when reviewing NetworkSpyAgent findings.

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
        storage_source: dict[str, Any] | None = None,
        window_property_source: dict[str, Any] | None = None,
        transaction_source: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Record how to resolve a dynamic token based on TraceHoundAgent findings.
        Auto-adds dependency transactions to the queue.

        Args:
            variable_name: Name of the variable being resolved.
            transaction_id: The transaction this variable belongs to.
            source_type: Where the value comes from - "storage", "window_property", or "transaction".
            storage_source: For storage source - {"type": "cookie|localStorage|sessionStorage", "dot_path": "path"}.
            window_property_source: For window source - {"dot_path": "path"}.
            transaction_source: For transaction source - {"transaction_id": "id", "dot_path": "path"}.
        """
        # Find the variable in extracted variables
        tx_data = self._discovery_state.transaction_data.get(transaction_id)
        if not tx_data or not tx_data.get("extracted_variables"):
            return {
                "error": f"No extracted variables found for transaction {transaction_id}. "
                        f"Call record_extracted_variable first."
            }

        extracted = tx_data["extracted_variables"]
        variable = None
        for v in extracted.variables:
            if v.name == variable_name:
                variable = v
                break

        if not variable:
            available = [v.name for v in extracted.variables]
            return {
                "error": f"Variable '{variable_name}' not found in transaction {transaction_id}. "
                        f"Available variables: {available}"
            }

        # Build source objects based on source_type
        session_storage_src = None
        transaction_src = None
        window_property_src = None
        dependency_tx_id = None

        if source_type == "storage" and storage_source:
            try:
                storage_type = SessionStorageType(storage_source["type"])
                session_storage_src = SessionStorageSource(
                    type=storage_type,
                    dot_path=storage_source["dot_path"]
                )
            except (KeyError, ValueError) as e:
                return {"error": f"Invalid storage_source format: {e}"}

        elif source_type == "window_property" and window_property_source:
            try:
                window_property_src = WindowPropertySource(
                    dot_path=window_property_source["dot_path"]
                )
            except KeyError as e:
                return {"error": f"Invalid window_property_source format: {e}"}

        elif source_type == "transaction" and transaction_source:
            try:
                transaction_src = TransactionSource(
                    transaction_id=transaction_source["transaction_id"],
                    dot_path=transaction_source["dot_path"]
                )
                # Auto-add dependency to queue
                dep_tx_id = transaction_source["transaction_id"]
                if dep_tx_id not in self._discovery_state.processed_transactions:
                    added, _ = self._discovery_state.add_to_queue(dep_tx_id)
                    if added:
                        dependency_tx_id = dep_tx_id
            except KeyError as e:
                return {"error": f"Invalid transaction_source format: {e}"}
        else:
            return {
                "error": f"Invalid source_type '{source_type}' or missing source data. "
                        f"Provide storage_source, window_property_source, or transaction_source."
            }

        # Create ResolvedVariableResponse
        resolved = ResolvedVariableResponse(
            variable=variable,
            session_storage_source=session_storage_src,
            transaction_source=transaction_src,
            window_property_source=window_property_src,
            short_explanation=f"Resolved {variable_name} from {source_type}"
        )

        # Store in state
        self._discovery_state.store_transaction_data(
            transaction_id=transaction_id,
            resolved_variable=resolved
        )

        result = {
            "success": True,
            "variable_name": variable_name,
            "source_type": source_type,
            "added_to_all_resolved": True
        }

        if dependency_tx_id:
            result["dependency_added"] = True
            result["dependency_transaction_id"] = dependency_tx_id
            result["message"] = f"Resolved '{variable_name}' and added dependency {dependency_tx_id} to queue"
        else:
            result["message"] = f"Resolved '{variable_name}' from {source_type}"

        return result

    @agent_tool(
        description="Get formatted summary of everything discovered so far (transactions, variables, resolution info). Use before constructing routine.",
        parameters={"type": "object", "properties": {}, "required": []},
        availability=True,
    )
    def _get_discovery_context(self) -> dict[str, Any]:
        """
        Get formatted summary of everything discovered so far.
        Use this before constructing the routine to see all available data.

        Returns:
            Comprehensive discovery context including transactions, variables, and resolution info.
        """
        context: dict[str, Any] = {
            "root_transaction": None,
            "processed_transactions": [],
            "all_parameters": [],
            "all_dynamic_tokens": [],
            "all_static_values": [],
            "transaction_order": []
        }

        # Root transaction
        if self._discovery_state.root_transaction:
            context["root_transaction"] = {
                "id": self._discovery_state.root_transaction.transaction_id,
                "url": self._discovery_state.root_transaction.url,
                "method": self._discovery_state.root_transaction.method.value,
                "description": self._discovery_state.root_transaction.description
            }

        # Processed transactions with their data
        for tx_id in self._discovery_state.processed_transactions:
            tx_data = self._discovery_state.transaction_data.get(tx_id, {})
            tx_info: dict[str, Any] = {"id": tx_id}

            # Add request info
            if tx_data.get("request"):
                req = tx_data["request"]
                tx_info["url"] = req.get("url", "")
                tx_info["method"] = req.get("method", "")

            # Add extracted variables
            if tx_data.get("extracted_variables"):
                extracted = tx_data["extracted_variables"]
                tx_info["extracted_variables"] = [
                    {
                        "name": v.name,
                        "type": v.type.value,
                        "observed_value": v.observed_value,
                        "requires_resolution": v.requires_dynamic_resolution
                    }
                    for v in extracted.variables
                ]

            # Add resolved variables
            if tx_data.get("resolved_variables"):
                tx_info["resolved_variables"] = [
                    {
                        "name": rv.variable.name,
                        "source_type": (
                            "storage" if rv.session_storage_source else
                            "transaction" if rv.transaction_source else
                            "window_property" if rv.window_property_source else
                            "unknown"
                        ),
                        "explanation": rv.short_explanation
                    }
                    for rv in tx_data["resolved_variables"]
                ]

            context["processed_transactions"].append(tx_info)

        # Categorize all resolved variables by type
        for resolved in self._discovery_state.all_resolved_variables:
            var = resolved.variable
            var_dict = {
                "name": var.name,
                "observed_value": var.observed_value,
                "source": (
                    f"storage:{resolved.session_storage_source.dot_path}" if resolved.session_storage_source else
                    f"transaction:{resolved.transaction_source.transaction_id}:{resolved.transaction_source.dot_path}" if resolved.transaction_source else
                    f"window:{resolved.window_property_source.dot_path}" if resolved.window_property_source else
                    "unknown"
                )
            }

            if var.type == VariableType.PARAMETER:
                context["all_parameters"].append(var_dict)
            elif var.type == VariableType.DYNAMIC_TOKEN:
                context["all_dynamic_tokens"].append(var_dict)
            elif var.type == VariableType.STATIC_VALUE:
                context["all_static_values"].append(var_dict)

        # Transaction execution order (dependencies first)
        ordered = self._discovery_state.get_ordered_transactions()
        context["transaction_order"] = list(ordered.keys())

        # Add summary stats
        context["summary"] = {
            "total_transactions": len(self._discovery_state.processed_transactions),
            "total_parameters": len(context["all_parameters"]),
            "total_dynamic_tokens": len(context["all_dynamic_tokens"]),
            "total_static_values": len(context["all_static_values"]),
            "queue_pending": len(self._discovery_state.transaction_queue),
            "current_phase": self._discovery_state.phase.value
        }

        return context

    ## Tools - Data Store Access

    @agent_tool(
        description="List all available transaction IDs.",
        parameters={"type": "object", "properties": {}, "required": []},
        availability=lambda self: self._network_data_loader is not None,
    )
    def _list_transactions(self) -> dict[str, Any]:
        """List all available transaction IDs."""
        entries = self._network_data_loader.entries
        tx_ids = [e.request_id for e in entries]
        return {
            "transaction_ids": tx_ids,
            "count": len(tx_ids),
        }

    @agent_tool(
        description="Get full details of a transaction.",
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
        entry = self._network_data_loader.get_entry(transaction_id)
        if not entry:
            # Show some available IDs as hints
            available = [e.request_id for e in self._network_data_loader.entries[:10]]
            return {"error": f"Transaction {transaction_id} not found. Available: {available}..."}

        return {
            "transaction_id": transaction_id,
            "request": {
                "method": entry.method,
                "url": entry.url,
                "headers": entry.request_headers,
                "body": entry.post_data,
            },
            "response": {
                "status": entry.status,
                "headers": entry.response_headers,
                "body": entry.response_body[:5000] if entry.response_body else None,  # Truncate large bodies
            },
        }

    @agent_tool(
        description="Scan storage, window properties, and transactions for a value.",
        parameters={
            "type": "object",
            "properties": {
                "value": {
                    "type": "string",
                    "description": "The value to search for"
                },
                "before_transaction_id": {
                    "type": "string",
                    "description": "Optional transaction ID to limit search to events before this transaction"
                }
            },
            "required": ["value"]
        },
        availability=lambda self: self._network_data_loader is not None,
    )
    def _scan_for_value(
        self,
        value: str,
        before_transaction_id: str | None = None
    ) -> dict[str, Any]:
        """
        Scan storage, window properties, and transactions for a value.

        Args:
            value: The value to search for.
            before_transaction_id: Optional transaction ID to limit search to events before this transaction.
        """
        max_timestamp = None
        if before_transaction_id:
            entry = self._network_data_loader.get_entry(before_transaction_id)
            if entry:
                max_timestamp = entry.timestamp

        # Scan storage
        storage_sources = []
        if self._storage_data_loader:
            storage_sources = self._storage_data_loader.scan_for_value(value, max_timestamp=max_timestamp)

        # Scan window properties
        window_sources = []
        if self._window_property_data_loader:
            window_sources = self._window_property_data_loader.scan_for_value(value, max_timestamp=max_timestamp)

        # Scan transaction responses
        transaction_sources = []
        for entry in self._network_data_loader.entries:
            if max_timestamp and entry.timestamp >= max_timestamp:
                continue
            if entry.response_body and value in str(entry.response_body):
                transaction_sources.append({
                    "transaction_id": entry.request_id,
                    "url": entry.url,
                    "timestamp": entry.timestamp,
                })

        return {
            "storage_sources": storage_sources[:5],  # Limit results
            "window_property_sources": window_sources[:5],
            "transaction_sources": transaction_sources[:5],
            "found_count": len(storage_sources) + len(window_sources) + len(transaction_sources),
        }

    @agent_tool(
        description="Add a transaction to the processing queue.",
        parameters={
            "type": "object",
            "properties": {
                "transaction_id": {
                    "type": "string",
                    "description": "The transaction ID to add to the queue"
                },
                "reason": {
                    "type": "string",
                    "description": "Why this transaction is being added"
                }
            },
            "required": ["transaction_id", "reason"]
        },
        availability=lambda self: (
            self._network_data_loader is not None and
            self._discovery_state.root_transaction is not None
        ),
    )
    def _add_transaction_to_queue(
        self,
        transaction_id: str,
        reason: str
    ) -> dict[str, Any]:
        """
        Add a transaction to the processing queue.

        Args:
            transaction_id: The transaction ID to add to the queue.
            reason: Why this transaction is being added.
        """
        entry = self._network_data_loader.get_entry(transaction_id)
        if not entry:
            return {"success": False, "error": f"Transaction {transaction_id} not found"}

        added, position = self._discovery_state.add_to_queue(transaction_id)
        return {
            "success": True,
            "added": added,
            "queue_position": position,
            "already_processed": transaction_id in self._discovery_state.processed_transactions,
            "reason": reason,
        }

    @agent_tool(
        description="Get current queue status.",
        parameters={"type": "object", "properties": {}, "required": []},
        availability=lambda self: self._discovery_state.root_transaction is not None,
    )
    def _get_queue_status(self) -> dict[str, Any]:
        """Get current queue status."""
        return self._discovery_state.get_queue_status()

    @agent_tool(
        description="Mark a transaction as complete and get the next one.",
        parameters={
            "type": "object",
            "properties": {
                "transaction_id": {
                    "type": "string",
                    "description": "The transaction ID to mark as complete"
                }
            },
            "required": ["transaction_id"]
        },
        availability=lambda self: (
            self._discovery_state.root_transaction is not None and
            (self._discovery_state.current_transaction is not None or len(self._discovery_state.transaction_queue) > 0)
        ),
    )
    def _mark_transaction_complete(self, transaction_id: str) -> dict[str, Any]:
        """
        Mark a transaction as complete and get the next one.

        Args:
            transaction_id: The transaction ID to mark as complete.
        """
        self._discovery_state.mark_transaction_complete(transaction_id)
        next_tx = self._discovery_state.pop_next_transaction()

        # Check if we should advance phase
        if not next_tx and not self._discovery_state.transaction_queue:
            self._discovery_state.phase = DiscoveryPhase.CONSTRUCTING

        return {
            "success": True,
            "next_transaction": next_tx,
            "remaining_count": len(self._discovery_state.transaction_queue),
            "phase": self._discovery_state.phase.value,
        }

    @agent_tool(
        description="Search documentation and code files for schema definitions, examples, and implementation details.",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "String to search for (e.g., 'Parameter schema', 'fetch operation')"
                },
                "file_type": {
                    "type": "string",
                    "enum": ["documentation", "code"],
                    "description": "Optional filter"
                }
            },
            "required": ["query"]
        },
        availability=lambda self: self._documentation_data_loader is not None,
    )
    def _search_documentation(
        self,
        query: str,
        file_type: str | None = None,
    ) -> dict[str, Any]:
        """
        Search documentation and code files for a query string.

        Use this to find schema definitions, examples, and implementation details
        when you need to understand how to construct parameters or operations.

        Args:
            query: String to search for (e.g., "Parameter schema", "fetch operation").
            file_type: Optional filter - "documentation" or "code".
        """
        if not self._documentation_data_loader:
            return {"error": "No documentation data store available"}

        from bluebox.llms.data_loaders.documentation_data_loader import FileType

        file_type_enum = None
        if file_type:
            try:
                file_type_enum = FileType(file_type)
            except ValueError:
                return {"error": f"Invalid file_type. Use 'documentation' or 'code'"}

        results = self._documentation_data_loader.search_content_with_lines(
            query=query,
            file_type=file_type_enum,
            case_sensitive=False,
            max_matches_per_file=5,
        )

        return {
            "query": query,
            "results_count": len(results),
            "results": results[:10],  # Limit to top 10 files
        }

    @agent_tool(
        description="Read the full content of a documentation or code file.",
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path (supports partial matching)"
                },
                "start_line": {
                    "type": "integer",
                    "description": "Optional starting line (1-indexed, inclusive)"
                },
                "end_line": {
                    "type": "integer",
                    "description": "Optional ending line (1-indexed, inclusive)"
                }
            },
            "required": ["path"]
        },
        availability=lambda self: self._documentation_data_loader is not None,
    )
    def _read_documentation_file(
        self,
        path: str,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> dict[str, Any]:
        """
        Read the full content of a documentation or code file.

        Args:
            path: File path (supports partial matching).
            start_line: Optional starting line (1-indexed, inclusive).
            end_line: Optional ending line (1-indexed, inclusive).
        """
        if not self._documentation_data_loader:
            return {"error": "No documentation data store available"}

        result = self._documentation_data_loader.get_file_lines(
            path=path,
            start_line=start_line,
            end_line=end_line,
        )

        if result is None:
            return {"error": f"File not found: {path}"}

        content, total_lines = result
        return {
            "path": path,
            "content": content,
            "total_lines": total_lines,
            "showing_lines": f"{start_line or 1}-{end_line or total_lines}",
        }

    ## Tools - Routine Construction

    @agent_tool(
        description="""Construct and auto-execute a routine. Pass complete routine dict with name, description, parameters, operations.

CRITICAL PLACEHOLDER FORMAT:
❌ WRONG: /seasons/{{season_id}}/standings
❌ WRONG: /seasons/'{{season_id}}'/standings
✅ RIGHT: /seasons/\\"{{season_id}}\\"/standings

The backslash-quote \\"{{param}}\\" is MANDATORY! Parameter names must match exactly.""",
        parameters={
            "type": "object",
            "properties": {
                "routine": {
                    "type": "object",
                    "description": "Complete routine dictionary with name, description, parameters, and operations",
                    "properties": {
                        "name": {"type": "string"},
                        "description": {"type": "string"},
                        "parameters": {
                            "type": "array",
                            "items": {"type": "object"},
                            "description": "List of parameter objects"
                        },
                        "operations": {
                            "type": "array",
                            "items": {"type": "object"},
                            "description": "List of operation objects"
                        }
                    },
                    "required": ["name", "description", "operations"]
                }
            },
            "required": ["routine"]
        },
        availability=lambda self: (
            # Must have root transaction
            self._discovery_state.root_transaction is not None and
            # Transaction queue must be empty (all dependencies processed)
            len(self._discovery_state.transaction_queue) == 0 and
            # Must have processed at least the root transaction
            len(self._discovery_state.processed_transactions) > 0
        ),
    )
    def _construct_routine(
        self,
        routine: dict[str, Any],
    ) -> dict[str, Any]:

        self._discovery_state.construction_attempts += 1
        self._discovery_state.phase = DiscoveryPhase.CONSTRUCTING

        # Extract from routine dict
        name = routine["name"]
        description = routine["description"]
        parameters = routine.get("parameters", [])
        operations = routine["operations"]

        # Optional: Provide guidance if discovery state has data
        warnings = []
        if self._discovery_state.all_resolved_variables:
            # Check if routine uses discovered variables
            discovered_param_names = {
                rv.variable.name for rv in self._discovery_state.all_resolved_variables
                if rv.variable.type == VariableType.PARAMETER
            }
            discovered_token_names = {
                rv.variable.name for rv in self._discovery_state.all_resolved_variables
                if rv.variable.type == VariableType.DYNAMIC_TOKEN
            }

            routine_param_names = {p["name"] for p in parameters}

            # Warn about discovered parameters not used in routine
            unused_params = discovered_param_names - routine_param_names
            if unused_params:
                warnings.append(
                    f"Discovered parameters not used in routine: {unused_params}. "
                    f"Consider if these should be included."
                )

            # Note: We don't validate token usage here since they're embedded in placeholders

        # Suggest transaction order if multiple transactions were processed
        if len(self._discovery_state.processed_transactions) > 1:
            ordered_txs = list(self._discovery_state.get_ordered_transactions().keys())
            warnings.append(
                f"Multiple transactions were processed. Suggested operation order "
                f"(dependencies first): {ordered_txs}"
            )

        try:
            routine = Routine(
                name=name,
                description=description,
                parameters=parameters,
                operations=operations,
            )

            self._discovery_state.production_routine = routine
            self._discovery_state.phase = DiscoveryPhase.VALIDATING

            # AUTOMATICALLY execute the routine and return results
            if not self._remote_debugging_address:
                return {
                    "success": False,
                    "error": "Cannot construct routine without browser connection for validation. "
                            "Provide --remote-debugging-address to enable routine execution and validation."
                }

            # Execute routine automatically with observed parameter values
            from bluebox.llms.tools.execute_routine_tool import execute_routine

            test_params = {}
            for param in routine.parameters:
                if param.observed_value:
                    test_params[param.name] = param.observed_value

            exec_result = execute_routine(
                routine=routine.model_dump(),
                parameters=test_params,
                remote_debugging_address=self._remote_debugging_address,
                timeout=60,
                close_tab_when_done=True,
            )

            # Build result with execution feedback
            result = {
                "success": True,
                "routine_constructed": True,
                "routine_name": routine.name,
                "operations_count": len(routine.operations),
                "parameters_count": len(routine.parameters),
            }

            if warnings:
                result["warnings"] = warnings

            # Add execution results
            if not exec_result.get("success"):
                result["execution_success"] = False
                result["execution_error"] = exec_result.get("error", "Unknown error")
                result["message"] = (
                    "Routine constructed and executed, but execution FAILED. "
                    "Review the error, fix issues, and call construct_routine again. "
                    "DO NOT call done until execution succeeds."
                )
                return result

            routine_exec_result = exec_result.get("result")
            if not routine_exec_result or not routine_exec_result.ok:
                failed_placeholders = []
                if routine_exec_result and routine_exec_result.placeholder_resolution:
                    failed_placeholders = [
                        k for k, v in routine_exec_result.placeholder_resolution.items() if v is None
                    ]

                result["execution_success"] = False
                result["execution_error"] = routine_exec_result.error if routine_exec_result else "Execution failed"
                result["failed_placeholders"] = failed_placeholders
                result["message"] = (
                    "Routine constructed and executed, but execution FAILED. "
                    "Review the error and failed placeholders, fix issues, and call construct_routine again. "
                    "DO NOT call done until execution succeeds."
                )
                return result

            # Execution ran without errors, but check if data was actually returned
            if routine_exec_result.data is None:
                result["execution_success"] = False
                result["execution_error"] = "Routine executed but returned NO DATA. The return operation is likely missing or incorrect."
                result["message"] = (
                    "Routine constructed and executed, but returned NO DATA (data=None). "
                    "This means the routine is incomplete or has a missing/incorrect return operation. "
                    "Fix the routine to actually return data, then call construct_routine again. "
                    "DO NOT call done until data is returned!"
                )
                return result

            # Execution SUCCESS with data!
            result["execution_success"] = True
            result["execution_data_preview"] = str(routine_exec_result.data)[:500]
            result["message"] = (
                "Routine constructed and executed SUCCESSFULLY! "
                "The routine works correctly and returned data. You can now call done to complete discovery."
            )
            return result

        except Exception as e:
            error_msg = str(e)

            # Build helpful error message with documentation suggestions
            help_text = (
                "Construction failed. To fix this:\n\n"
                "1. Search documentation for examples:\n"
                "   - search_documentation(query='Routine schema')\n"
                "   - search_documentation(query='Operation types')\n"
                "   - search_documentation(query='Parameter definition')\n"
                "   - search_documentation(query='Placeholder format')\n\n"
                "2. Look for example routines:\n"
                "   - search_documentation(query='example_routines')\n"
                "   - search_documentation(query='fetch operation example')\n\n"
            )

            # Add specific error-based hints
            if "operation" in error_msg.lower():
                help_text += "3. Your error is related to OPERATIONS. Search: search_documentation(query='operations schema')\n"
            elif "parameter" in error_msg.lower():
                help_text += "3. Your error is related to PARAMETERS. Search: search_documentation(query='parameters schema')\n"
            elif "placeholder" in error_msg.lower():
                help_text += "3. Your error is related to PLACEHOLDERS. Search: search_documentation(query='placeholder resolution')\n"

            # Provide state-aware help
            if self._discovery_state.all_resolved_variables:
                help_text += (
                    f"\n4. You have {len(self._discovery_state.all_resolved_variables)} resolved variables available. "
                    f"Call get_discovery_context to see all discovered data."
                )
            elif self._discovery_state.root_transaction:
                help_text += (
                    "\n4. You have a root transaction but no resolved variables. "
                    "Use trace_hound to resolve dynamic tokens before constructing."
                )

            return {
                "success": False,
                "error": error_msg,
                "attempt": self._discovery_state.construction_attempts,
                "help": help_text,
            }

    @agent_tool(
        description="Execute the constructed routine to validate it works.",
        parameters={
            "type": "object",
            "properties": {
                "parameters": {
                    "type": "object",
                    "description": "Optional test parameters for execution"
                }
            },
            "required": []
        },
        availability=lambda self: self._remote_debugging_address is not None,
    )
    def _execute_routine(self, parameters: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Execute the constructed routine to validate it works.

        Args:
            parameters: Optional test parameters for execution.
        """
        if not self._discovery_state.production_routine:
            return {"error": "No routine constructed yet. Use construct_routine first."}

        if not self._remote_debugging_address:
            return {"error": "No browser connection for validation."}

        self._discovery_state.validation_attempts += 1

        # Import here to avoid circular dependency
        from bluebox.llms.tools.execute_routine_tool import execute_routine

        test_params = parameters or {}

        # If no params provided, extract from routine's observed values
        if not test_params:
            for param in self._discovery_state.production_routine.parameters:
                if param.observed_value:
                    test_params[param.name] = param.observed_value

        result = execute_routine(
            routine=self._discovery_state.production_routine.model_dump(),
            parameters=test_params,
            remote_debugging_address=self._remote_debugging_address,
            timeout=60,
            close_tab_when_done=True,
        )

        if not result.get("success"):
            return {
                "success": False,
                "error": result.get("error", "Unknown error"),
                "attempt": self._discovery_state.validation_attempts,
            }

        exec_result = result.get("result")
        if exec_result is None:
            return {"success": False, "error": "No result returned"}

        if not exec_result.ok:
            failed_placeholders = [
                k for k, v in exec_result.placeholder_resolution.items() if v is None
            ]
            return {
                "success": False,
                "error": exec_result.error or "Execution failed",
                "failed_placeholders": failed_placeholders,
                "attempt": self._discovery_state.validation_attempts,
            }

        # Success! Mark validation as succeeded
        self._validation_succeeded = True
        return {
            "success": True,
            "message": "Routine validated successfully! You can now call done to complete discovery.",
            "data_preview": str(exec_result.data)[:500] if exec_result.data else None,
            "next_step": "Call done to mark discovery as complete"
        }

    ## Tools - Completion

    @agent_tool(
        description="Mark discovery as complete. Call this ONLY after construct_routine shows execution_success=True.",
        parameters={"type": "object", "properties": {}, "required": []},
        availability=lambda self: (
            self._discovery_state.production_routine is not None and
            self._discovery_state.phase == DiscoveryPhase.VALIDATING
        ),
    )
    def _done(self) -> dict[str, Any]:
        """Mark discovery as complete. Call this ONLY after construct_routine shows execution_success=True."""
        if not self._discovery_state.production_routine:
            return {"error": "No routine constructed. Use construct_routine first."}

        self._discovery_state.phase = DiscoveryPhase.COMPLETE
        self._final_routine = self._discovery_state.production_routine
        return {
            "success": True,
            "message": "Discovery completed with validated routine",
            "routine_name": self._final_routine.name,
        }

    @agent_tool(
        description="Mark discovery as failed.",
        parameters={
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "Why discovery could not be completed"
                }
            },
            "required": ["reason"]
        },
        availability=True,
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
            "success": True,
            "message": "Discovery marked as failed",
            "reason": reason,
        }
