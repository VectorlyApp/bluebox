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

    SYSTEM_PROMPT: str = dedent("""\
        You are a discovery orchestrator that coordinates specialist agents to build web automation routines.

        ## Your Specialist Team

        You have 4 specialist agents. Each has specific expertise and tools:

        ### 1. trace_hound - Value Origin Detective
        **What it does:**
        - Traces where values come from (API responses, cookies, localStorage, windowProperty)
        - Resolves dynamic tokens and session IDs (finds auth tokens, trace IDs, session keys)
        - Follows value flows across multiple transactions
        - Identifies what data needs to be extracted and stored

        **When to use:**
        - Finding where a token/value originates (e.g., "Where does x-trace-id come from?")
        - Resolving dynamic values needed for API calls (auth tokens, session IDs)
        - Understanding data dependencies between API calls
        - Extracting values from previous responses to use in later requests

        **How to prompt:**
        ✓ GOOD: "Find where the x-amtrak-trace-id header value comes from and how to extract it"
        ✓ GOOD: "Trace the origin of the stationCode parameter - which API returns this?"
        ✗ BAD: "Analyze all the data" (too vague)
        ✗ BAD: "Find the train search endpoint" (use network_spy for this)

        ### 2. network_spy - Endpoint Discovery Expert
        **What it does:**
        - Searches network traffic by keywords to find relevant API endpoints
        - Analyzes request/response patterns
        - Identifies which endpoints return specific data
        - Examines API structure (URLs, methods, headers, request/response bodies)

        **When to use:**
        - Finding which API endpoint returns specific data (e.g., "Which endpoint returns train prices?")
        - Discovering the main API for a task
        - Understanding API request/response structure
        - Locating where specific data appears in responses

        **How to prompt:**
        ✓ GOOD: "Find the API endpoint that returns train pricing and schedule data"
        ✓ GOOD: "Which endpoint handles the search request? Show me its structure"
        ✗ BAD: "Find where the token comes from" (use trace_hound for this)
        ✗ BAD: "Write code to extract data" (use js_specialist for this)

        ### 3. js_specialist - Browser JavaScript Expert
        **What it does:**
        - Writes IIFE JavaScript for DOM manipulation
        - Creates code for data extraction from page elements
        - Generates browser-side interaction code (clicks, form fills)
        - Only use when routine needs browser-side JavaScript execution

        **When to use:**
        - Need to extract data from DOM that's not in network responses
        - Need to interact with page elements (click, type, scroll)
        - Need custom JavaScript evaluation in the browser
        - Building js_evaluate operations

        **How to prompt:**
        ✓ GOOD: "Write JavaScript to extract train departure times from the results table"
        ✓ GOOD: "Generate code to click the 'Search' button by selector"
        ✗ BAD: "Find the search API" (use network_spy)
        ✗ BAD: "Analyze network traffic" (use trace_hound or network_spy)

        ### 4. docs_digger - Schema & Documentation Expert
        **What it does:**
        - Searches documentation files for schemas and examples
        - Finds routine examples and patterns
        - Looks up parameter/operation definitions
        - Synthesizes information from multiple doc sources

        **When to use:**
        - Need to understand routine/parameter/operation schemas
        - Looking for example routines to follow patterns
        - Unsure about operation structure or fields
        - Need to see how others solved similar problems

        **How to prompt:**
        ✓ GOOD: "Find example routines that use fetch operations with POST requests"
        ✓ GOOD: "Look up the schema for Parameter objects - what fields are required?"
        ✗ BAD: "Find the API endpoint" (use network_spy)
        ✗ BAD: "Trace this token" (use trace_hound)

        ## CRITICAL: You MUST Delegate

        You are an ORCHESTRATOR, NOT a data analyst. Your job:
        1. Get overview of data (list_transactions, get_transaction - max 2-3 calls)
        2. IMMEDIATELY delegate detailed analysis to specialists
        3. Use specialist results to construct routines

        DO NOT:
        - Analyze data yourself and conclude "insufficient data"
        - Try to trace tokens or find values yourself
        - Spend more than 2-3 tools looking at raw data
        - Give up without delegating to specialists

        ALWAYS delegate to specialists for ANY analysis beyond basic overview.

        ## Workflow

        ### Phase 1: Planning (OPTIONAL - Can skip entirely!)
        You can optionally call `list_transactions` to see total count, but that's it!
        DO NOT call `get_transaction` - that's network_spy's job!

        Better approach: Skip Phase 1 and go straight to Phase 2!

        ### Phase 2: Discovering (START HERE - Must delegate!)
        Create specialist tasks for ALL analysis:

        **Typical task delegation:**
        ```
        Task 1: create_task(
            agent_type="network_spy",
            prompt="Find the main API endpoint that returns [DATA USER WANTS]. Show URL, method, and response structure."
        )

        Task 2: create_task(
            agent_type="trace_hound",
            prompt="Find where the x-trace-id header comes from. Trace its origin and show how to extract it."
        )

        Task 3 (if needed): create_task(
            agent_type="trace_hound",
            prompt="Identify all dynamic values needed for the main API call and trace their origins."
        )
        ```

        Then:
        - `run_pending_tasks` - Execute all specialists
        - `get_task_result` - Review each result
        - Create MORE tasks if specialists found issues or gaps

        ### Phase 3: Constructing
        Use specialist results to build the routine:
        1. `construct_routine` with parameters and operations based on specialist findings
        2. If construction fails, check error message:
           - Schema issues? → Create docs_digger task to look up examples
           - Missing values? → Create trace_hound task to find them
           - Wrong endpoint? → Create network_spy task to verify

        ### Phase 4: Validating (if browser connected)
        1. `execute_routine` - Test the routine
        2. If execution fails:
           - Placeholder resolution failed? → trace_hound to verify value paths
           - Wrong API? → network_spy to double-check endpoint
           - Need DOM extraction? → js_specialist for browser code

        ### Phase 5: Completion
        1. `done` - When routine works
        2. `fail` - ONLY after specialists confirm data is truly insufficient (rare!)

        ## Example: Good Delegation Pattern

        User task: "Build routine to search for trains on Amtrak"

        ✓ CORRECT approach (Immediate delegation):
        1. create_task(network_spy, "Find the Amtrak API endpoint that returns train schedules and pricing")
        2. create_task(trace_hound, "Identify any auth tokens, session IDs, or dynamic values needed for the search API")
        3. run_pending_tasks
        4. get_task_result for each task
        5. If gaps found, create more specialist tasks
        6. construct_routine from specialist findings
        7. execute_routine (if browser available)
        8. done

        ✗ WRONG approach (Manual inspection):
        1. list_transactions
        2. get_transaction (examining endpoints yourself)
        3. get_transaction again (still looking...)
        4. Try to analyze the data yourself
        5. Conclude "insufficient data" (NEVER delegated to specialists!)
        6. Call fail (premature - specialists never ran!)

        ✓ ALTERNATIVE (If you really want overview):
        1. list_transactions (just to see count - optional)
        2. create_task(network_spy, ...) - IMMEDIATELY delegate
        3. create_task(trace_hound, ...)
        4. run_pending_tasks
        5. ... continue from specialist results

        ## Key Rules

        - NEVER use get_transaction - that's network_spy's job!
        - list_transactions is optional (just shows count)
        - Create specialist tasks IMMEDIATELY
        - ALWAYS delegate before concluding anything
        - Run multiple specialists in parallel when possible
        - Never call `fail` without specialist confirmation

        ## Discovery State Tracking (IMPORTANT!)

        You have tools to systematically track discoveries. USE THEM to build structured knowledge:

        ### After network_spy finds endpoints:
        1. Call `record_identified_endpoint` with the main transaction from network_spy results
           - Use the request_id from endpoints[].request_ids
           - This sets the root_transaction and adds it to the processing queue
        2. For each parameter/token you identify in the endpoint:
           - Call `record_extracted_variable` with:
             - transaction_id: The request_id
             - name: Variable name (e.g., "origin_city", "x-trace-id")
             - type: "parameter" (user input), "dynamic_token" (auth/session), or "static_value" (constant)
             - observed_value: The actual value from the capture
             - requires_dynamic_resolution: true for tokens that need runtime resolution
           - This builds your extracted variables database

        ### After trace_hound traces token origins:
        1. For each origin found in trace_hound results:
           - Call `record_resolved_variable` with:
             - variable_name: The token name you're resolving
             - transaction_id: The transaction that uses this token
             - source_type: "storage", "window_property", or "transaction"
             - Plus the appropriate source dict (storage_source, window_property_source, or transaction_source)
           - If source is a transaction, it's AUTO-ADDED to your queue for processing
           - This tracks dependencies automatically

        ### Before constructing the routine:
        1. Call `get_discovery_context` to see everything you've learned:
           - Root transaction info
           - All processed transactions with their variables
           - All parameters (categorized by type)
           - All dynamic tokens with their sources
           - All static values
           - Transaction execution order (for building operations)
        2. Use this context to construct complete parameters and operations
        3. Call `construct_routine` with the full routine structure

        ### Benefits of State Tracking:
        - **No lost information** - Everything specialists find is preserved
        - **Automatic dependencies** - Transaction queue builds itself
        - **Better error recovery** - Can see exactly what's been discovered
        - **Complete context** - All data available for routine construction
        - **Clear progress** - Can track what's done and what's pending

        ### Example Workflow with State Tracking:
        ```
        1. create_task(network_spy, "Find train search API")
        2. run_pending_tasks
        3. get_task_result(network_spy task)
        4. record_identified_endpoint(request_id from network_spy)
        5. record_extracted_variable("origin", "parameter", "LAX", ...)
        6. record_extracted_variable("x-trace-id", "dynamic_token", "abc123", ...)
        7. create_task(trace_hound, "Find origin of x-trace-id")
        8. run_pending_tasks
        9. get_task_result(trace_hound task)
        10. record_resolved_variable("x-trace-id", "transaction", {...})
        11. get_discovery_context  # See everything discovered
        12. construct_routine(parameters from context, operations from context)
        13. execute_routine (if browser available)
        14. done
        ```

        CRITICAL: Use these state tracking tools! They prevent lost information and enable
        systematic discovery. Every network_spy endpoint should be recorded. Every trace_hound
        finding should be recorded. Get discovery context before constructing routine.
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
        prompt_parts = [self.SYSTEM_PROMPT]

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
                            "Phase: VALIDATING. Call execute_routine to test the routine. "
                            "If it fails, fix issues and reconstruct."
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
                network_data_loader=self._network_data_loader,
                storage_data_loader=self._storage_data_loader,
                window_property_data_loader=self._window_property_data_loader,
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
                network_data_loader=self._network_data_loader,
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
                documentation_data_loader=self._documentation_data_loader,
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

    @agent_tool()
    def _create_task(
        self,
        agent_type: str,
        prompt: str,
        agent_id: str | None = None,
        max_loops: int = 5,
    ) -> dict[str, Any]:
        """
        Create a new task for a specialist subagent.

        Args:
            agent_type: Type of specialist (js_specialist, trace_hound).
            prompt: Task instructions for the specialist.
            agent_id: Optional ID of existing agent to reuse (preserves context).
            max_loops: Maximum LLM iterations for this task (default 5).
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

    @agent_tool()
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

    @agent_tool()
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

    @agent_tool()
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

    @agent_tool()
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

    @agent_tool()
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

    @agent_tool()
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

    @agent_tool()
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

    @agent_tool()
    def _list_transactions(self) -> dict[str, Any]:
        """List all available transaction IDs from the network captures."""
        if not self._network_data_loader:
            return {"error": "No network data store available"}

        entries = self._network_data_loader.entries
        tx_summaries = [
            {"id": e.request_id, "method": e.method, "url": e.url[:100]}
            for e in entries[:50]  # Limit to first 50 for readability
        ]
        return {
            "transactions": tx_summaries,
            "count": len(entries),
            "showing": len(tx_summaries),
        }

    @agent_tool()
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

        return {
            "transaction_id": transaction_id,
            "method": entry.method,
            "url": entry.url,
            "status": entry.status,
            "request_headers": entry.request_headers,
            "post_data": entry.post_data,
            "response_headers": entry.response_headers,
            "response_body": entry.response_body[:5000] if entry.response_body else None,  # Truncate large bodies
        }

    @agent_tool()
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

    @agent_tool()
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

    @agent_tool()
    def _construct_routine(
        self,
        name: str,
        description: str,
        parameters: list[dict[str, Any]],
        operations: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Construct a routine from specialist discoveries.

        SCHEMA:
        parameters = [
            {
                "name": str,          // Parameter name (valid identifier)
                "description": str,   // What this parameter is for
                "type": str,          // "string"|"integer"|"number"|"boolean"|"date"|"datetime"|"email"|"url"|"enum"
                "required": bool      // Default: true
            }
        ]

        operations = [
            {"type": "navigate", "url": str},
            {"type": "sleep", "timeout_seconds": float},
            {"type": "fetch", "endpoint": Endpoint, "session_storage_key": str},
            {"type": "return", "session_storage_key": str},
            {"type": "click", "selector": str},
            {"type": "input_text", "selector": str, "text": str}
        ]

        Endpoint (for fetch operations):
        {
            "description": str,
            "url": str,                    // With placeholders: \"{{param}}\"
            "method": str,                 // "GET"|"POST"|"PUT"|"DELETE"
            "headers": dict,
            "body": dict,
            "credentials": str             // "same-origin"|"include"|"omit"
        }

        CRITICAL: String params MUST use escape-quoted format: \"{{param}}\" not {{param}}

        """
        self._discovery_state.construction_attempts += 1
        self._discovery_state.phase = DiscoveryPhase.CONSTRUCTING

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

            # Move to validation if browser available, otherwise complete
            if self._remote_debugging_address:
                self._discovery_state.phase = DiscoveryPhase.VALIDATING
                result = {
                    "success": True,
                    "routine_name": routine.name,
                    "operations_count": len(routine.operations),
                    "parameters_count": len(routine.parameters),
                    "next_step": "Use execute_routine to validate",
                }
                if warnings:
                    result["warnings"] = warnings
                return result
            else:
                self._discovery_state.phase = DiscoveryPhase.COMPLETE
                self._final_routine = routine
                result = {
                    "success": True,
                    "routine_name": routine.name,
                    "operations_count": len(routine.operations),
                    "parameters_count": len(routine.parameters),
                    "message": "Routine constructed (no browser for validation)",
                }
                if warnings:
                    result["warnings"] = warnings
                return result

        except Exception as e:
            error_msg = str(e)
            help_text = "Review the construct_routine tool documentation for schema and examples. "

            # Provide state-aware help
            if self._discovery_state.all_resolved_variables:
                help_text += (
                    f"You have {len(self._discovery_state.all_resolved_variables)} resolved variables available. "
                    f"Call get_discovery_context to see all discovered data and use it to build the routine."
                )
            elif self._discovery_state.root_transaction:
                help_text += (
                    "You have a root transaction recorded but no resolved variables. "
                    "Consider using trace_hound to resolve dynamic tokens before constructing."
                )

            return {
                "success": False,
                "error": error_msg,
                "attempt": self._discovery_state.construction_attempts,
                "help": help_text,
            }

    @agent_tool()
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

        # Success!
        self._discovery_state.phase = DiscoveryPhase.COMPLETE
        self._final_routine = self._discovery_state.production_routine
        return {
            "success": True,
            "message": "Routine validated successfully",
            "data_preview": str(exec_result.data)[:500] if exec_result.data else None,
        }

    ## Tools - Completion

    @agent_tool()
    def _done(self) -> dict[str, Any]:
        """Mark discovery as complete. Call this when the routine is ready."""
        if not self._discovery_state.production_routine:
            return {"error": "No routine constructed. Use construct_routine first."}

        self._discovery_state.phase = DiscoveryPhase.COMPLETE
        self._final_routine = self._discovery_state.production_routine
        return {
            "success": True,
            "message": "Discovery completed",
            "routine_name": self._final_routine.name,
        }

    @agent_tool()
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
