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
from bluebox.data_models.orchestration.state import SuperDiscoveryState, SuperDiscoveryPhase
from bluebox.data_models.routine.routine import Routine
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
        self._state = SuperDiscoveryState()
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
        status = self._state.get_queue_status()
        prompt_parts.append(dedent(f"""\

            ## Current State
            - Phase: {status['phase']}
            - Pending tasks: {status['pending_tasks']}
            - In-progress tasks: {status['in_progress_tasks']}
            - Completed tasks: {status['completed_tasks']}
            - Failed tasks: {status['failed_tasks']}
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
                        iteration + 1, self._max_iterations, self._state.phase.value)

            # Check for completion
            if self._state.phase == SuperDiscoveryPhase.COMPLETE:
                return self._final_routine

            if self._state.phase == SuperDiscoveryPhase.FAILED:
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
                    # Prompt the agent to continue if no tool calls
                    self._add_chat(
                        ChatRole.SYSTEM,
                        f"[ACTION REQUIRED] Phase: {self._state.phase.value}. Use tools to make progress."
                    )

            except Exception as e:
                logger.exception("Error in SuperDiscovery loop: %s", e)
                self._emit_message(ErrorEmittedMessage(error=str(e)))
                self._state.phase = SuperDiscoveryPhase.FAILED
                self._failure_reason = str(e)
                return None

        logger.warning("SuperDiscovery hit max iterations (%d)", self._max_iterations)
        self._state.phase = SuperDiscoveryPhase.FAILED
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
        self._state.subagents[subagent.id] = subagent
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

        self._state.add_task(task)
        self._state.phase = SuperDiscoveryPhase.DISCOVERING

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
        for task in self._state.tasks.values():
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
            "pending": len(self._state.get_pending_tasks()),
            "in_progress": len(self._state.get_in_progress_tasks()),
            "completed": len(self._state.get_completed_tasks()),
            "failed": len(self._state.get_failed_tasks()),
            "tasks": tasks_summary,
        }

    @agent_tool()
    def _get_task_result(self, task_id: str) -> dict[str, Any]:
        """
        Get the result of a completed task.

        Args:
            task_id: The ID of the task to get results for.
        """
        task = self._state.tasks.get(task_id)
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
        pending = self._state.get_pending_tasks()
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

        # Check if all tasks are done
        if not self._state.get_pending_tasks() and not self._state.get_in_progress_tasks():
            if self._state.get_failed_tasks():
                pass  # Some failed, let orchestrator decide
            else:
                self._state.phase = SuperDiscoveryPhase.CONSTRUCTING

        return {
            "executed": len(results),
            "results": results,
            "phase": self._state.phase.value,
        }

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
        self._state.construction_attempts += 1
        self._state.phase = SuperDiscoveryPhase.CONSTRUCTING

        try:
            routine = Routine(
                name=name,
                description=description,
                parameters=parameters,
                operations=operations,
            )

            self._state.current_routine = routine

            # Move to validation if browser available, otherwise complete
            if self._remote_debugging_address:
                self._state.phase = SuperDiscoveryPhase.VALIDATING
                return {
                    "success": True,
                    "routine_name": routine.name,
                    "operations_count": len(routine.operations),
                    "parameters_count": len(routine.parameters),
                    "next_step": "Use execute_routine to validate",
                }
            else:
                self._state.phase = SuperDiscoveryPhase.COMPLETE
                self._final_routine = routine
                return {
                    "success": True,
                    "routine_name": routine.name,
                    "operations_count": len(routine.operations),
                    "parameters_count": len(routine.parameters),
                    "message": "Routine constructed (no browser for validation)",
                }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "attempt": self._state.construction_attempts,
                "help": (
                    "Review the construct_routine tool documentation for schema and examples. "
                ),
            }

    @agent_tool()
    def _execute_routine(self, parameters: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Execute the constructed routine to validate it works.

        Args:
            parameters: Optional test parameters for execution.
        """
        if not self._state.current_routine:
            return {"error": "No routine constructed yet. Use construct_routine first."}

        if not self._remote_debugging_address:
            return {"error": "No browser connection for validation."}

        self._state.validation_attempts += 1

        # Import here to avoid circular dependency
        from bluebox.llms.tools.execute_routine_tool import execute_routine

        test_params = parameters or {}

        # If no params provided, extract from routine's observed values
        if not test_params:
            for param in self._state.current_routine.parameters:
                if param.observed_value:
                    test_params[param.name] = param.observed_value

        result = execute_routine(
            routine=self._state.current_routine.model_dump(),
            parameters=test_params,
            remote_debugging_address=self._remote_debugging_address,
            timeout=60,
            close_tab_when_done=True,
        )

        if not result.get("success"):
            return {
                "success": False,
                "error": result.get("error", "Unknown error"),
                "attempt": self._state.validation_attempts,
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
                "attempt": self._state.validation_attempts,
            }

        # Success!
        self._state.phase = SuperDiscoveryPhase.COMPLETE
        self._final_routine = self._state.current_routine
        return {
            "success": True,
            "message": "Routine validated successfully",
            "data_preview": str(exec_result.data)[:500] if exec_result.data else None,
        }

    ## Tools - Completion

    @agent_tool()
    def _done(self) -> dict[str, Any]:
        """Mark discovery as complete. Call this when the routine is ready."""
        if not self._state.current_routine:
            return {"error": "No routine constructed. Use construct_routine first."}

        self._state.phase = SuperDiscoveryPhase.COMPLETE
        self._final_routine = self._state.current_routine
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
        self._state.phase = SuperDiscoveryPhase.FAILED
        self._failure_reason = reason
        return {
            "success": True,
            "message": "Discovery marked as failed",
            "reason": reason,
        }
