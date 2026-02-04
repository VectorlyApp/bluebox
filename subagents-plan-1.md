# SuperRoutineDiscoveryAgent Implementation Plan

## Overview

Create a new `SuperRoutineDiscoveryAgent` that uses **both openai-agents and claude-agent-sdk** via a unified abstraction layer to orchestrate existing specialist subagents in parallel. This agent will perform the same workflow as `RoutineDiscoveryAgent` but delegate to specialized subagents for different phases.

## Architecture Decision

**Unified AgentClient abstraction with vendor-specific implementations:**
- Create `AbstractAgentOrchestrator` base class (parallel to `AbstractLLMVendorClient`)
- Implement `OpenAIAgentOrchestrator` using openai-agents SDK
- Implement `AnthropicAgentOrchestrator` using claude-agent-sdk
- Create `AgentClient` facade (parallel to `LLMClient`) for vendor selection
- Existing specialists remain unchanged - wrapped as tools for each SDK

```
┌─────────────────────────────────────────────────────────┐
│                     AgentClient                          │
│              (Unified Facade - like LLMClient)           │
└──────────────────────────┬──────────────────────────────┘
                           │
        ┌──────────────────┴──────────────────┐
        ▼                                      ▼
┌───────────────────┐               ┌───────────────────┐
│OpenAIAgentOrchest.│               │AnthropicAgentOrch.│
│(openai-agents SDK)│               │(claude-agent-sdk) │
└─────────┬─────────┘               └─────────┬─────────┘
          │                                    │
          └────────────────┬───────────────────┘
                           │ tool calls
    ┌──────────────┬───────┴───────┬──────────────┐
    ▼              ▼               ▼              ▼
┌────────┐  ┌────────────┐  ┌──────────┐  ┌─────────┐
│Network │  │Interaction │  │TraceHound│  │   JS    │
│SpyAgent│  │Specialist  │  │  Agent   │  │Specialist│
└────────┘  └────────────┘  └──────────┘  └─────────┘
          (existing specialists - unchanged)
```

## Files to Create

### 1. `bluebox/agents/orchestration/discovery_context.py` (NEW)
Shared context dataclass used by both SDK implementations.

```python
from dataclasses import dataclass
from bluebox.data_models.routine_discovery.state import RoutineDiscoveryState
from bluebox.llms.infra.network_data_store import NetworkDataStore
from bluebox.llms.infra.storage_data_store import StorageDataStore
from bluebox.llms.infra.window_property_data_store import WindowPropertyDataStore
from bluebox.llms.infra.interactions_data_store import InteractionsDataStore

@dataclass
class DiscoveryContext:
    state: RoutineDiscoveryState
    network_data_store: NetworkDataStore
    storage_data_store: StorageDataStore
    window_property_data_store: WindowPropertyDataStore
    interactions_data_store: InteractionsDataStore
    remote_debugging_address: str | None = None
```

### 2. `bluebox/agents/orchestration/abstract_agent_orchestrator.py` (NEW)
Base class for vendor-specific orchestrators (parallel to `AbstractLLMVendorClient`).

```python
from abc import ABC, abstractmethod
from bluebox.data_models.routine.routine import Routine

class AbstractAgentOrchestrator(ABC):
    """Base class for agent orchestrators that delegate to specialist subagents."""

    def __init__(self, context: DiscoveryContext, model: str):
        self._context = context
        self._model = model

    @abstractmethod
    async def run(self, task: str) -> Routine | None:
        """Execute the orchestrated workflow and return the resulting routine."""
        ...

    @property
    def context(self) -> DiscoveryContext:
        return self._context
```

### 3. `bluebox/agents/orchestration/openai_agent_orchestrator.py` (NEW)
OpenAI implementation using openai-agents SDK.

```python
from agents import Agent, Runner, RunConfig, function_tool, RunContextWrapper

class OpenAIAgentOrchestrator(AbstractAgentOrchestrator):
    def __init__(self, context: DiscoveryContext, model: str = "gpt-5.1"):
        super().__init__(context, model)
        self._agent = Agent(
            name="RoutineDiscoveryOrchestrator",
            instructions=ORCHESTRATOR_INSTRUCTIONS,
            model=model,
            tools=self._create_tools(),
        )

    def _create_tools(self) -> list:
        # Tool wrappers using @function_tool decorator
        return [discover_endpoints, trace_token_origin, discover_parameters, ...]

    async def run(self, task: str) -> Routine | None:
        result = await Runner.run(
            self._agent,
            input=f"Task: {task}",
            context=self._context,
            run_config=RunConfig(max_turns=50),
        )
        return self._context.state.production_routine
```

### 4. `bluebox/agents/orchestration/anthropic_agent_orchestrator.py` (NEW)
Anthropic implementation using claude-agent-sdk.

**Important:** The claude-agent-sdk uses a different paradigm than openai-agents:
- Uses `query()` streaming function with built-in tools (Read, Grep, Glob, Bash)
- Subagents are defined via `AgentDefinition` with prompts, not Python function wrappers
- Specialists must be defined as subagents with prompts that use built-in tools

```python
from claude_agent_sdk import query, ClaudeAgentOptions, AgentDefinition

class AnthropicAgentOrchestrator(AbstractAgentOrchestrator):
    def __init__(self, context: DiscoveryContext, model: str = "claude-sonnet-4-20250514"):
        super().__init__(context, model)
        self._model = model

    def _create_subagent_definitions(self) -> dict[str, AgentDefinition]:
        """Define specialists as subagents with prompts that use built-in tools."""
        return {
            "network-spy": AgentDefinition(
                description="Searches HAR/network capture files to find API endpoints",
                prompt=self._get_network_spy_prompt(),  # detailed prompt for HAR analysis
                tools=["Read", "Glob", "Grep"],
                model="sonnet",
            ),
            "trace-hound": AgentDefinition(
                description="Traces token origins across network, storage, and JS properties",
                prompt=self._get_trace_hound_prompt(),
                tools=["Read", "Glob", "Grep"],
            ),
            "interaction-analyst": AgentDefinition(
                description="Discovers user input parameters from interaction data",
                prompt=self._get_interaction_prompt(),
                tools=["Read", "Glob", "Grep"],
            ),
            # ... etc
        }

    async def run(self, task: str) -> Routine | None:
        result_text = None
        async for message in query(
            prompt=f"Discover a routine for: {task}",
            options=ClaudeAgentOptions(
                allowed_tools=["Read", "Glob", "Grep", "Task"],
                agents=self._create_subagent_definitions(),
                # Pass working directory with CDP captures
                cwd=str(self._context.cdp_captures_dir),
            )
        ):
            if hasattr(message, "result"):
                result_text = message.result
        # Parse result_text to build routine (requires structured output parsing)
        return self._parse_routine_from_result(result_text)
```

**Trade-off:** The Anthropic approach requires re-implementing specialist logic as prompts that use built-in tools, rather than wrapping existing Python code. This means:
- Specialists run as Claude agents reading files directly (not calling Python code)
- Results are text-based (need parsing) rather than structured Pydantic models
- Logic is duplicated between openai-agents (Python wrappers) and claude-agent-sdk (prompts)

### 5. `bluebox/agents/orchestration/agent_client.py` (NEW)
Unified facade for vendor selection (parallel to `LLMClient`).

```python
from enum import Enum

class AgentVendor(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"

class AgentClient:
    """Unified facade for agent orchestration across vendors."""

    def __init__(
        self,
        context: DiscoveryContext,
        vendor: AgentVendor = AgentVendor.OPENAI,
        model: str | None = None,
    ):
        self._vendor = vendor
        if vendor == AgentVendor.OPENAI:
            self._orchestrator = OpenAIAgentOrchestrator(
                context, model or "gpt-5.1"
            )
        else:
            self._orchestrator = AnthropicAgentOrchestrator(
                context, model or "claude-sonnet-4-20250514"
            )

    async def run(self, task: str) -> Routine | None:
        return await self._orchestrator.run(task)
```

### 6. `bluebox/agents/super_routine_discovery_agent.py` (NEW)
Main agent using the unified AgentClient.

```python
class SuperRoutineDiscoveryAgent:
    """Orchestrates specialist subagents to discover routines."""

    def __init__(
        self,
        context: DiscoveryContext,
        vendor: AgentVendor = AgentVendor.OPENAI,
        model: str | None = None,
    ):
        self._client = AgentClient(context, vendor, model)

    async def run(self, task: str) -> Routine | None:
        return await self._client.run(task)
```

### 7. `bluebox/scripts/run_super_routine_discovery_agent.py` (NEW)
Demo/test script with vendor selection.

```python
"""
Usage:
    python -m bluebox.scripts.run_super_routine_discovery_agent \
        --cdp-captures-dir ./cdp_captures \
        --task "Search for trains from NYC to Boston" \
        --vendor openai \
        --model gpt-5.1
"""

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cdp-captures-dir", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--vendor", choices=["openai", "anthropic"], default="openai")
    parser.add_argument("--model")  # uses vendor default if not specified
    parser.add_argument("--remote-debugging-address")
    args = parser.parse_args()

    context = DiscoveryContext(
        state=RoutineDiscoveryState(),
        network_data_store=NetworkDataStore.from_jsonl(...),
        ...
    )

    agent = SuperRoutineDiscoveryAgent(
        context,
        vendor=AgentVendor(args.vendor),
        model=args.model,
    )
    routine = await agent.run(args.task)

    if routine:
        print(routine.model_dump_json(indent=2))
```

## Existing Code to Reuse

| Component | File Path | What to Reuse |
|-----------|-----------|---------------|
| State model | `bluebox/data_models/routine_discovery/state.py` | `RoutineDiscoveryState`, `DiscoveryPhase` |
| Network data | `bluebox/llms/infra/network_data_store.py` | `NetworkDataStore.from_jsonl()`, `NetworkStats` |
| Storage data | `bluebox/llms/infra/storage_data_store.py` | `StorageDataStore` |
| Window props | `bluebox/llms/infra/window_property_data_store.py` | `WindowPropertyDataStore` |
| Interactions | `bluebox/llms/infra/interactions_data_store.py` | `InteractionsDataStore` |
| NetworkSpyAgent | `bluebox/agents/network_spy_agent.py` | `run_autonomous()`, result models |
| TraceHoundAgent | `bluebox/agents/trace_hound_agent.py` | `run_autonomous()`, result models |
| InteractionSpecialist | `bluebox/agents/specialists/interaction_specialist.py` | `run_autonomous()` |
| JSSpecialist | `bluebox/agents/specialists/js_specialist.py` | `run_autonomous()` |
| DocsDiggerAgent | `bluebox/agents/docs_digger_agent.py` | `run_autonomous()` |
| Routine models | `bluebox/data_models/routine/routine.py` | `Routine` |
| DevRoutine | `bluebox/data_models/routine/dev_routine.py` | `DevRoutine` |
| Execute routine | `bluebox/llms/tools/execute_routine_tool.py` | `execute_routine()` |

## Implementation Steps

### Step 1: Create shared context (`discovery_context.py`)
1. Define `DiscoveryContext` dataclass holding state + data stores
2. Include paths to CDP capture files for file-based access

### Step 2: Create OpenAI orchestrator (`openai_agent_orchestrator.py`)
1. Create `@function_tool` wrappers for each specialist:
   - `discover_endpoints` → `NetworkSpyAgent.run_autonomous()`
   - `trace_token_origin` → `TraceHoundAgent.run_autonomous()`
   - `discover_parameters` → `InteractionSpecialist.run_autonomous()`
   - `write_js_code` → `JSSpecialist.run_autonomous()`
   - `search_documentation` → `DocsDiggerAgent.run_autonomous()`
2. Create direct tools for routine construction/validation:
   - `construct_routine` - adapts logic from `RoutineDiscoveryAgent._tool_construct_routine()`
   - `execute_routine` - wraps `execute_routine_tool.execute_routine()`
3. Implement `OpenAIAgentOrchestrator` class using openai-agents SDK
4. Use `asyncio.to_thread()` to bridge sync specialists to async tools

### Step 3: Create Anthropic orchestrator (`anthropic_agent_orchestrator.py`)
1. Define subagent prompts that replicate specialist logic using built-in tools:
   - `network-spy`: Prompt to search HAR files using Read/Glob/Grep
   - `trace-hound`: Prompt to trace tokens in storage/network files
   - `interaction-analyst`: Prompt to find parameters in interaction logs
   - `js-writer`: Prompt to write IIFE JavaScript code
   - `docs-digger`: Prompt to search documentation files
2. Create `AgentDefinition` for each subagent
3. Implement `AnthropicAgentOrchestrator` using claude-agent-sdk `query()` streaming
4. Implement result parsing from text output to `Routine` model

### Step 4: Create unified facade (`agent_client.py`)
1. Define `AgentVendor` enum (OPENAI, ANTHROPIC)
2. Implement `AgentClient` that selects orchestrator based on vendor
3. Expose unified `run(task: str)` interface

### Step 5: Create `super_routine_discovery_agent.py`
1. Simple wrapper around `AgentClient`
2. Define orchestrator instructions for 4-phase workflow
3. Handle vendor selection from constructor args

### Step 6: Create demo script (`run_super_routine_discovery_agent.py`)
1. Parse CLI arguments (cdp-captures-dir, task, vendor, model)
2. Load data stores from CDP captures directory
3. Create `DiscoveryContext` with state + data stores + file paths
4. Instantiate and run `SuperRoutineDiscoveryAgent`
5. Output resulting routine as JSON

## Verification

### Unit Testing
```bash
# Run existing specialist tests to ensure they still work
pytest tests/unit/test_network_spy_agent.py -v
pytest tests/unit/test_interaction_specialist.py -v
```

### Integration Testing (OpenAI)
```bash
# Test with sample CDP captures using openai-agents
python -m bluebox.scripts.run_super_routine_discovery_agent \
    --cdp-captures-dir example_data/cdp_captures/amtrak \
    --task "Search for one-way trains from Boston to New York" \
    --vendor openai \
    --model gpt-5.1
```

### Integration Testing (Anthropic)
```bash
# Test with sample CDP captures using claude-agent-sdk
python -m bluebox.scripts.run_super_routine_discovery_agent \
    --cdp-captures-dir example_data/cdp_captures/amtrak \
    --task "Search for one-way trains from Boston to New York" \
    --vendor anthropic \
    --model claude-sonnet-4-20250514
```

### Expected Output
- Routine JSON with:
  - `name`, `description`
  - `parameters` with `observed_value` populated
  - `operations` (navigate → fetch → return sequence)
- No errors from subagent failures
- Both vendors should produce equivalent results (same routine structure)

## Key Architectural Trade-off

**The two SDKs have fundamentally different patterns:**

| Aspect | openai-agents | claude-agent-sdk |
|--------|---------------|------------------|
| Tool definition | `@function_tool` on Python functions | Built-in tools (Read, Grep, Glob, Bash) |
| Custom logic | Wrap existing Python code directly | Define as prompts for subagents |
| Output format | Structured (Pydantic models) | Text (requires parsing) |
| State management | `RunContextWrapper` with context | File-based (CDP files in cwd) |

**Implication:** OpenAI approach wraps existing specialists cleanly. Anthropic approach requires re-implementing specialist logic as prompts + built-in tools, with text output parsing.

**Recommendation:** Implement OpenAI orchestrator first (straightforward), then Anthropic (more work, different paradigm).

## Resolved Questions

1. **Parallel execution?** → Use `asyncio.gather()` for independent tool calls within orchestrator
2. **State sharing?** → OpenAI: `RunContextWrapper`, Anthropic: files in working directory
3. **Error handling?** → Tools return `{"success": False, "error": ...}` on failure
4. **Async/sync bridging?** → `asyncio.to_thread()` for specialists in openai-agents tools
