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
from bluebox.agents.specialists.js_specialist import JSSpecialist
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
from bluebox.data_models.orchestration.task import Task, SubAgent, TaskStatus, AgentType
from bluebox.data_models.orchestration.state import SuperDiscoveryState, SuperDiscoveryPhase
from bluebox.data_models.routine.routine import Routine
from bluebox.llms.infra.data_store import DiscoveryDataStore
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

        ## Your Role

        You analyze network traffic captures and delegate specific tasks to specialist agents:
        - **js_specialist**: Writes IIFE JavaScript for DOM manipulation and data extraction
        - **trace_hound**: Traces value origins across transactions, resolves dynamic tokens

        NOTE: You have direct access to transaction data via list_transactions, get_transaction, and scan_for_value.
        Use these tools directly rather than delegating simple lookups to specialists.

        ## Workflow

        ### Phase 1: Planning
        1. Use `list_transactions` to see available network data
        2. Use `get_transaction` to examine promising endpoints
        3. Plan which specialists to deploy and what tasks to give them

        ### Phase 2: Discovering
        1. Use `create_task` to delegate work to specialists
        2. Use `run_pending_tasks` to execute delegated tasks
        3. Use `get_task_result` to review completed work
        4. Iterate until you have all the information needed

        ### Phase 3: Constructing
        1. Use `construct_routine` to build the routine from discoveries
        2. If validation fails, create more specialist tasks to fix issues

        ### Phase 4: Validating (if browser available)
        1. Use `execute_routine` to test the routine
        2. If execution fails, analyze errors and fix with specialists

        ### Phase 5: Completion
        1. Use `done` when the routine is ready
        2. Use `fail` if discovery cannot be completed

        ## Guidelines

        - Keep specialist tasks focused and specific
        - Don't give one specialist work better suited for another
        - Use trace_hound for token resolution, not js_specialist
        - Use js_specialist only when DOM manipulation is actually needed
        - Monitor task status and handle failures gracefully
    """)

    ## Magic methods

    def __init__(
        self,
        emit_message_callable: Callable[[EmittedMessage], None],
        data_store: DiscoveryDataStore,
        task: str,
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
            data_store: Data store with CDP captures.
            task: The discovery task description.
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
        self._data_store = data_store
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

        # Add data store context
        if self._data_store:
            data_store_prompt = self._data_store.generate_data_store_prompt()
            if data_store_prompt:
                prompt_parts.append(f"\n\n{data_store_prompt}")

        # Add current state
        status = self._state.get_queue_status()
        prompt_parts.append(f"""

## Current State
- Phase: {status['phase']}
- Pending tasks: {status['pending_tasks']}
- In-progress tasks: {status['in_progress_tasks']}
- Completed tasks: {status['completed_tasks']}
- Failed tasks: {status['failed_tasks']}
""")

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

    def _create_specialist(self, agent_type: AgentType) -> AbstractSpecialist:
        """Create a specialist instance based on type."""
        if agent_type == AgentType.JS_SPECIALIST:
            return JSSpecialist(
                emit_message_callable=self._emit_message_callable,
                llm_model=self._subagent_llm_model,
                remote_debugging_address=self._remote_debugging_address,
            )

        elif agent_type == AgentType.TRACE_HOUND:
            # TraceHoundAgent needs specific data stores - we'll pass what we have
            # from the DiscoveryDataStore (or None if not available)
            return TraceHoundAgent(
                emit_message_callable=self._emit_message_callable,
                llm_model=self._subagent_llm_model,
                # TraceHoundAgent accepts these as optional
                network_data_store=None,  # Would need NetworkDataStore
                storage_data_store=None,  # Would need StorageDataStore
                window_property_data_store=None,  # Would need WindowPropertyDataStore
            )

        else:
            # For other agent types that aren't fully implemented yet
            raise NotImplementedError(
                f"Agent type {agent_type.value} is not yet supported. "
                f"Available types: js_specialist, trace_hound"
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
    AVAILABLE_AGENT_TYPES = {AgentType.JS_SPECIALIST, AgentType.TRACE_HOUND}

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
            parsed_type = AgentType(agent_type)
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
        """List all available transaction IDs from the CDP captures."""
        tx_ids = self._data_store.get_all_transaction_ids()
        return {
            "transaction_ids": tx_ids,
            "count": len(tx_ids),
        }

    @agent_tool()
    def _get_transaction(self, transaction_id: str) -> dict[str, Any]:
        """
        Get full details of a transaction.

        Args:
            transaction_id: The ID of the transaction to retrieve.
        """
        all_ids = self._data_store.get_all_transaction_ids()
        if transaction_id not in all_ids:
            return {"error": f"Transaction {transaction_id} not found. Available: {all_ids[:10]}..."}

        tx = self._data_store.get_transaction_by_id(transaction_id)
        return {
            "transaction_id": transaction_id,
            "request": tx.get("request", {}),
            "response": tx.get("response", {}),
        }

    @agent_tool()
    def _scan_for_value(self, value: str, before_transaction_id: str | None = None) -> dict[str, Any]:
        """
        Scan storage, window properties, and transactions for a value.

        Args:
            value: The value to search for.
            before_transaction_id: Optional - only search transactions before this one.
        """
        max_timestamp = None
        if before_transaction_id:
            max_timestamp = self._data_store.get_transaction_timestamp(before_transaction_id)

        storage_sources = self._data_store.scan_storage_for_value(value)
        window_sources = self._data_store.scan_window_properties_for_value(value)
        tx_sources = self._data_store.scan_transaction_responses(value, max_timestamp=max_timestamp)

        return {
            "storage_sources": storage_sources[:5],
            "window_property_sources": window_sources[:5],
            "transaction_sources": tx_sources[:5],
            "found_count": len(storage_sources) + len(window_sources) + len(tx_sources),
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
        Construct a routine from the discovery results.

        Args:
            name: Name of the routine.
            description: Description of what the routine does.
            parameters: List of parameter definitions.
            operations: List of operation definitions.
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
