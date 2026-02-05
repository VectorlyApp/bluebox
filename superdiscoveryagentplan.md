# SuperDiscoveryAgent Implementation Plan

## Overview
Create a `SuperDiscoveryAgent` that orchestrates specialist agents as subagents for routine discovery. This involves refactoring the agent hierarchy to extract a common `AbstractAgent` base class.

## Architecture

```
AbstractAgent (NEW - extracted from AbstractSpecialist)
├── AbstractSpecialist (REFACTORED - inherits AbstractAgent)
│   ├── NetworkSpyAgent
│   ├── TraceHoundAgent
│   ├── JSSpecialist
│   ├── InteractionSpecialist
│   └── DocsDiggerAgent
└── SuperDiscoveryAgent (NEW)
```

---

## Phase 1: Create AbstractAgent Base Class

**File:** `bluebox/agents/abstract_agent.py` (NEW)

Extract from `abstract_specialist.py` (lines 127-739):
- `@agent_tool` decorator (renamed from `@agent_tool`) and `_ToolMeta` dataclass (lines 67-124)
- LLM client initialization and management
- Chat/thread state management (`_thread`, `_chats`, `_previous_response_id`)
- Tool registration: `_sync_tools()`, `_execute_tool()`, `_collect_tools()` (lines 381-479)
- Message building: `_build_messages_for_llm()` (lines 659-689)
- LLM calling: `_call_llm()`, `_process_streaming_response()` (lines 582-617)
- Chat helpers: `_emit_message()`, `_add_chat()` (lines 621-657)
- Tool execution: `_auto_execute_tool()`, `_process_tool_calls()` (lines 694-738)
- Properties: `chat_thread_id`

**AbstractAgent abstract method:**
```python
@abstractmethod
def _get_system_prompt(self) -> str: ...
```

---

## Phase 2: Refactor AbstractSpecialist

**File:** `bluebox/agents/specialists/abstract_specialist.py` (MODIFY)

Keep in AbstractSpecialist (inherits AbstractAgent):
- `RunMode` enum, `AutonomousConfig` namedtuple
- Autonomous-specific state: `_autonomous_iteration`, `_autonomous_config`
- `can_finalize` property (lines 274-284)
- Abstract methods for autonomous mode:
  - `_get_autonomous_system_prompt()`
  - `_get_autonomous_initial_message()`
  - `_check_autonomous_completion()`
  - `_get_autonomous_result()`
  - `_reset_autonomous_state()`
- `run_autonomous()` method (lines 299-336)
- `_run_autonomous_loop()` method (lines 525-578)

**Changes to autonomous mode:**
1. Rename `max_iterations` semantics: "max iterations THIS run" (for pause/resume)
2. Add `loops_used` tracking for paused tasks
3. Allow `min_iterations=1` for resumed tasks (finalize available immediately)

---

## Phase 3: Data Models

### File: `bluebox/data_models/orchestration/__init__.py` (NEW)
```python
from .task import Task, SubAgent, TaskStatus, AgentType
from .state import SuperDiscoveryState, SuperDiscoveryPhase
```

### File: `bluebox/data_models/orchestration/task.py` (NEW)

```python
class TaskStatus(StrEnum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    PAUSED = "paused"           # Hit max_loops, can resume
    COMPLETED = "completed"
    FAILED = "failed"

class AgentType(StrEnum):
    JS_SPECIALIST = "js_specialist"
    NETWORK_SPY = "network_spy"
    TRACE_HOUND = "trace_hound"
    INTERACTION_SPECIALIST = "interaction_specialist"
    DOCS_DIGGER = "docs_digger"

class Task(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    agent_type: AgentType
    agent_id: str | None = None       # None = create new instance
    prompt: str
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: str | None = None
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    max_loops: int = 5                # Max LLM iterations before returning
    loops_used: int = 0
    context: dict[str, Any] = Field(default_factory=dict)

class SubAgent(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    type: AgentType
    llm_model: str
    task_ids: list[str] = Field(default_factory=list)
    created_at: datetime
```

### File: `bluebox/data_models/orchestration/state.py` (NEW)

```python
class SuperDiscoveryPhase(StrEnum):
    PLANNING = "planning"
    DISCOVERING = "discovering"
    CONSTRUCTING = "constructing"
    VALIDATING = "validating"
    COMPLETE = "complete"
    FAILED = "failed"

class SuperDiscoveryState(BaseModel):
    phase: SuperDiscoveryPhase = SuperDiscoveryPhase.PLANNING

    # Transaction tracking (like RoutineDiscoveryState)
    root_transaction_id: str | None = None
    queued_transaction_ids: list[str] = Field(default_factory=list)
    processed_transaction_ids: list[str] = Field(default_factory=list)

    # Placeholder tracking
    queued_placeholders: list[Placeholder] = Field(default_factory=list)
    processed_placeholders: list[Placeholder] = Field(default_factory=list)

    # Task/Subagent management
    tasks: dict[str, Task] = Field(default_factory=dict)
    subagents: dict[str, SubAgent] = Field(default_factory=dict)

    # Routine
    current_routine: Routine | None = None
    current_test_parameters: dict[str, Any] = Field(default_factory=dict)

    # Methods: add_task(), get_pending_tasks(), get_in_progress_tasks()
```

---

## Phase 4: SuperDiscoveryAgent Implementation

**File:** `bluebox/agents/super_discovery_agent.py` (NEW)

```python
class SuperDiscoveryAgent(AbstractAgent):
    """Orchestrator that coordinates specialist subagents."""

    def __init__(
        self,
        emit_message_callable,
        data_store: DiscoveryDataStore,
        task: str,
        llm_model: LLMModel = OpenAIModel.GPT_5_1,
        subagent_llm_model: LLMModel | None = None,
        max_iterations: int = 50,
        remote_debugging_address: str | None = None,
    ): ...

    # Internal state (not in model)
    _state: SuperDiscoveryState
    _agent_instances: dict[str, AbstractSpecialist]  # agent_id -> instance
```

### SuperDiscoveryAgent Tools

**Task Management:**
```python
@agent_tool()
def _create_subagent_task(self, agent_type: str, prompt: str,
                          agent_id: str | None = None, max_loops: int = 5) -> dict: ...

@agent_tool()
def _list_tasks(self) -> dict: ...

@agent_tool()
def _get_task_result(self, task_id: str) -> dict: ...

@agent_tool()
def _run_pending_tasks(self) -> dict: ...
```

**Data Store (reuse from RoutineDiscoveryAgent):**
```python
@agent_tool()
def _list_transactions(self) -> dict: ...

@agent_tool()
def _get_transaction(self, transaction_id: str) -> dict: ...

@agent_tool()
def _scan_for_value(self, value: str, before_transaction_id: str | None = None) -> dict: ...
```

**Routine Construction (adapt from RoutineDiscoveryAgent):**
```python
@agent_tool()
def _construct_routine(self, name: str, description: str,
                       parameters: list, operations: list) -> dict: ...

@agent_tool()
def _execute_routine(self, parameters: dict | None = None) -> dict: ...
```

**Completion:**
```python
@agent_tool()
def _done(self) -> dict: ...

@agent_tool()
def _fail(self, reason: str) -> dict: ...
```

### Key Internal Methods

```python
def _execute_task(self, task: Task) -> dict:
    """Run a task using the appropriate specialist."""
    agent = self._get_or_create_agent(task)
    config = AutonomousConfig(min_iterations=1, max_iterations=task.max_loops - task.loops_used)
    result = agent.run_autonomous(task.prompt, config)
    task.loops_used += agent.autonomous_iteration
    # Handle completion, pause, or failure

def _get_or_create_agent(self, task: Task) -> AbstractSpecialist:
    """Get existing agent instance or create new one."""
    # Map AgentType to specialist class, instantiate with proper kwargs

def run(self) -> Routine:
    """Main entry point - run discovery to completion."""
```

---

## Critical Files Summary

| File | Action | Purpose |
|------|--------|---------|
| `bluebox/agents/abstract_agent.py` | CREATE | Base class with LLM/chat/tool infrastructure |
| `bluebox/agents/specialists/abstract_specialist.py` | MODIFY | Inherit AbstractAgent, keep autonomous mode |
| `bluebox/data_models/orchestration/task.py` | CREATE | Task and SubAgent models |
| `bluebox/data_models/orchestration/state.py` | CREATE | SuperDiscoveryState model |
| `bluebox/agents/super_discovery_agent.py` | CREATE | Orchestrator agent |

**Reference files (patterns to reuse):**
- `bluebox/agents/routine_discovery_agent.py` - Tool implementations for transactions, scanning, routine construction
- `bluebox/data_models/routine_discovery/state.py` - BFS queue management pattern

---

## Verification Plan

1. **Unit tests for AbstractAgent extraction:**
   - Verify existing specialists still work after refactor
   - Test tool registration/execution

2. **Unit tests for new data models:**
   - Task/SubAgent serialization
   - State transitions

3. **Integration test for SuperDiscoveryAgent:**
   - Use existing CDP captures from `tests/data/`
   - Verify task delegation to specialists
   - Verify routine construction

4. **Manual test:**
   ```bash
   # Run super discovery with test captures
   python -c "
   from bluebox.agents.super_discovery_agent import SuperDiscoveryAgent
   # ... test with real data store
   "
   ```

---

## Implementation Order

1. **AbstractAgent extraction** (safest first - no new functionality)
2. **Data models** (simple Pydantic, no dependencies)
3. **SuperDiscoveryAgent** (depends on 1 & 2)
4. **Specialist modifications** (pause/resume support)
5. **CLI entry point** (`bluebox-super-discover`)
