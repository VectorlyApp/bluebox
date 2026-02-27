# bluebox Development Guide

This file provides context and guidelines for working with the bluebox codebase.

## Bash Commands

### Development Setup

- `uv venv bluebox-env && source bluebox-env/bin/activate` - Create and activate virtual environment (recommended)
- `python3 -m venv bluebox-env && source bluebox-env/bin/activate` - Alternative venv creation
- `uv pip install -e .` - Install package in editable mode (faster with uv)
- `pip install -e .` - Install package in editable mode (standard)

### Testing

- `pytest tests/ -v` - Run all tests with verbose output
- `pytest tests/unit/test_js_utils.py -v` - Run specific test file
- `pytest tests/unit/test_js_utils.py::test_function_name -v` - Run specific test
- `python scripts/dev/run_benchmarks.py` - Run routine discovery benchmarks
- `python scripts/dev/run_benchmarks.py -v` - Run benchmarks with verbose output

### CLI Tools

- `bluebox-monitor --host 127.0.0.1 --port 9222 --output-dir ./cdp_captures --url about:blank --incognito` - Start browser monitoring
- `bluebox-discover --task "your task description" --cdp-captures-dir ./cdp_captures --output-dir ./routine_discovery_output --llm-model gpt-5.2` - Discover routines from captures
- `bluebox-execute --routine-path example_data/example_routines/amtrak_one_way_train_search_routine.json --parameters-path example_data/example_routines/amtrak_one_way_train_search_input.json` - Execute a routine
- `bluebox-api-index --cdp-captures-dir ./cdp_captures --task "your task" --output-dir ./api_indexing_output --model gpt-5.2 --post-run-analysis` - Run the API indexing pipeline (exploration + routine construction)
- `bluebox-agent-adapter --agent NetworkSpecialist --cdp-captures-dir ./cdp_captures` - Start HTTP adapter for programmatic agent interaction (see Agent HTTP Adapter section below)
- `bluebox-agent-adapter --list-agents` - List all available agents and their required data

### Chrome Debug Mode

- macOS: `/Applications/Google Chrome.app/Contents/MacOS/Google Chrome --remote-debugging-address=127.0.0.1 --remote-debugging-port=9222 --user-data-dir="$HOME/tmp/chrome" --remote-allow-origins='*' --no-first-run --no-default-browser-check`
- Verify: `curl http://127.0.0.1:9222/json/version`

### Type Checking & Linting

- `pylint bluebox/` - Run pylint (uses .pylintrc config)

## Code Style

### Type Hints

- **IMPORTANT**: Every function and method MUST have type hints
- Use `-> ReturnType` for return types
- Use `param: Type` for parameters
- Use `Optional[Type]` or `Type | None` for nullable types
- Use `list[Type]` instead of `List[Type]` (Python 3.9+ style)

### Imports

- **IMPORTANT**: NO lazy imports! All imports must be at the top of the file
- Use absolute imports from `bluebox.*`
- Group imports: stdlib, third-party, local (with blank lines between groups)

### Python Version

- Requires Python 3.12+ (specifically `>=3.12.3,<3.13`)
- Use modern Python features (type hints, f-strings, dataclasses, etc.)

### Data Models

- Use Pydantic `BaseModel` for all data models (see `bluebox/data_models/`)
- Use `Field()` for field descriptions and defaults
- Use `model_validator` for custom validation logic
- All models should be in `bluebox/data_models/` directory

### Error Handling

- Use custom exceptions from `bluebox.utils.exceptions`
- Return `RoutineExecutionResult` for routine execution results
- Log errors using `bluebox.utils.logger.get_logger()`

### JavaScript Code Generation

- All JavaScript code should be generated through functions in `bluebox/utils/js_utils.py`
- JavaScript code must be wrapped in IIFE format: `(function() { ... })()`
- Use helper functions from `_get_placeholder_resolution_js_helpers()` for placeholder resolution

## Workflow

### Development Process

1. **Explore**: Read relevant files before coding
2. **Plan**: Make a plan before implementing (use "think" for complex problems)
3. **Code**: Implement with type hints and proper error handling
4. **Test**: Write and run tests
5. **Commit**: Use descriptive commit messages

### Routine Development Workflow

1. Launch Chrome in debug mode (or use quickstart.py)
2. Run `bluebox-monitor` and perform actions manually
3. Run `bluebox-discover` with task description
4. Review generated `routine.json`
5. Test with `bluebox-execute`
6. Review generated `routine.json` for correct parameter types and placeholder usage

## Core Files and Utilities

### Key Modules

- `bluebox/data_models/routine/routine.py` - Main Routine model
- `bluebox/data_models/routine/operation.py` - Operation types and execution
- `bluebox/data_models/routine/parameter.py` - Parameter definitions
- `bluebox/data_models/routine/placeholder.py` - Placeholder resolution
- `bluebox/cdp/connection.py` - Chrome DevTools Protocol connection
- `bluebox/utils/js_utils.py` - JavaScript code generation
- `bluebox/utils/web_socket_utils.py` - WebSocket utilities for CDP
- `bluebox/sdk/client.py` - Main SDK client
- `bluebox/workspace.py` - Agent workspace (artifact-oriented file I/O with provenance tracking)

### Agents

AI agents that power routine discovery, API indexing, and conversational interactions. All agents inherit from `AbstractAgent` (`bluebox/agents/abstract_agent.py`).

**Core agents:**
- `bluebox/agents/routine_discovery_agent.py` - Analyzes CDP captures to generate routines (identifies transactions, extracts/resolves variables, constructs operations)
- `bluebox/agents/guide_agent.py` - Conversational agent for guiding users through routine creation/editing (maintains chat history, dynamic tool registration)
- `bluebox/agents/bluebox_agent.py` - General-purpose conversational agent

**API Indexing Pipeline agents:**
- `bluebox/agents/principal_investigator.py` - Orchestrator: plans routine catalog, dispatches experiments to workers, reviews results, assembles and ships routines
- `bluebox/agents/workers/experiment_worker.py` - Browser-capable execution agent: live browser tools + recorded capture lookup tools, executes experiments
- `bluebox/agents/routine_inspector.py` - Independent quality gate: scores routines on 6 dimensions, hard-fails on 4xx/5xx or unresolved placeholders

**Specialists** (domain-specific agents for exploration):
- `bluebox/agents/specialists/network_specialist.py` - Network traffic analysis
- `bluebox/agents/specialists/dom_specialist.py` - DOM structure analysis
- `bluebox/agents/specialists/interaction_specialist.py` - UI interaction analysis
- `bluebox/agents/specialists/js_specialist.py` - JavaScript file analysis
- `bluebox/agents/specialists/value_trace_resolver_specialist.py` - Storage & window property analysis

**Agent HTTP Adapter** (`bluebox/scripts/agent_http_adapter.py`):

HTTP wrapper that exposes any `AbstractAgent` subclass as a JSON API, enabling programmatic interaction via curl. Agents are auto-discovered at runtime — adding a new `AbstractAgent` subclass makes it available with zero adapter changes.

```bash
# Start adapter with a specific agent
bluebox-agent-adapter --agent NetworkSpecialist --cdp-captures-dir ./cdp_captures

# Agents with no data requirements (e.g. BlueBoxAgent) don't need --cdp-captures-dir
bluebox-agent-adapter --agent BlueBoxAgent
```

Endpoints:
- `GET /health` — liveness check
- `GET /status` — agent type, chat state, discovery support
- `POST /chat {"message": "..."}` — send a chat message (all agents)
- `POST /discover {"task": "..."}` — run discovery/autonomous mode
- `GET /routine` — retrieve discovered routine JSON

**Best practices when calling from Claude Code or scripts:**
- **Use `--max-time 300` (5 min) on curl calls.** The first `/chat` or `/discover` request triggers a cold start (agent construction + first LLM round-trip) that can take 2+ minutes. Subsequent requests are fast since the agent stays in memory.
- **Start the adapter in the background** and poll `/health` until ready before sending requests.
- **Use `-q` (quiet mode)** to suppress Bluebox logging noise from the adapter process.
- **Save responses to files** (`-o /tmp/response.json`) rather than piping directly, to avoid losing data on timeout.
- Constructor params are auto-wired via `inspect.signature` — the adapter maps data loader param names (handling the `_loader`/`_store` naming split) to canonical keys automatically.

**LLM Infrastructure:**
- `bluebox/llms/data_loaders/` - Specialized data loaders for CDP capture analysis:
  - `NetworkDataLoader` - HTTP request/response transactions
  - `DOMDataLoader` - DOM snapshots (string-interning tables, element classification by tag family)
  - `JSDataLoader` - JavaScript files
  - `StorageDataLoader` - Cookies, localStorage, sessionStorage, IndexedDB
  - `WindowPropertyDataLoader` - Window property changes
  - `InteractionsDataLoader` - UI interaction events
  - `DocumentationDataLoader` - Documentation files
- `bluebox/llms/infra/data_store.py` - Legacy data stores (soon to be deprecated)

**Import patterns:**
```python
from bluebox.agents.abstract_agent import AbstractAgent, agent_tool, AgentCard
from bluebox.agents.guide_agent import GuideAgent
from bluebox.agents.routine_discovery_agent import RoutineDiscoveryAgent
from bluebox.agents.principal_investigator import PrincipalInvestigator
from bluebox.agents.workers.experiment_worker import ExperimentWorker
from bluebox.agents.routine_inspector import RoutineInspector
from bluebox.workspace import AgentWorkspace, LocalAgentWorkspace
from bluebox.llms.data_loaders.network_data_loader import NetworkDataLoader
from bluebox.llms.data_loaders.dom_data_loader import DOMDataLoader
from bluebox.llms.data_loaders.js_data_loader import JSDataLoader
```

### Workspace

The workspace (`bluebox/workspace.py`) is an artifact-oriented file I/O system attached to agents. Each workspace has a strict directory layout:

- `raw/` (read-only): tool result artifacts and mounted external files
- `output/`: agent-generated deliverables
- `context/`: reusable notes/context saved for later use in the same run
- `meta/`: system-managed metadata (`manifest.jsonl`, `input_mounts.jsonl`) — not editable
- `scratch/`: ephemeral scratch space

External files (e.g. CDP capture JSONL) can be mounted into `raw/` via hardlinks using `attach_input_file()`. The `save_artifact()` API records provenance in `meta/manifest.jsonl` (SHA-256, size, content type, timestamp).

### API Indexing Pipeline

End-to-end pipeline (`bluebox-api-index`) that turns raw CDP captures into a catalog of executable routines.

**Phase 1 — Exploration** (4 specialists in parallel): Network, Storage, DOM, and UI specialists each produce a structured exploration summary.

**Phase 2 — Routine Construction**: PrincipalInvestigator reads summaries, dispatches ExperimentWorker agents, reviews results, assembles routines, submits to RoutineInspector for quality gating. Incremental persistence to disk. PI crash recovery via DiscoveryLedger.

**Data models:**
- `bluebox/data_models/orchestration/` - `DiscoveryLedger`, `ExperimentEntry`, `RoutineSpec`, `RoutineAttempt`, `RoutineCatalog`, `RoutineInspectionResult`
- `bluebox/data_models/api_indexing/` - `NetworkExplorationSummary`, `StorageExplorationSummary`, `DOMExplorationSummary`, `UIExplorationSummary`

### Important Patterns

- **Routine Execution**: Operations execute sequentially, maintaining state via `RoutineExecutionContext`
- **Placeholder Resolution**: All parameters use `{{paramName}}` format; `Parameter.type` drives coercion at runtime
- **Session Storage**: Use `session_storage_key` to store and retrieve data between operations
- **CDP Sessions**: Use flattened sessions for multiplexing via `session_id`
- **Agent Tools**: Decorate with `@agent_tool()`. Supports `persist` (`NEVER`/`ALWAYS`/`OVERFLOW`), `max_characters`, and `token_optimized` parameters
- **Agent Card**: Every concrete `AbstractAgent` subclass must declare an `AGENT_CARD`

### Common Gotchas

- All parameters use uniform `"field": "{{param}}"` format; `Parameter.type` handles coercion
- Chrome must be running in debug mode on `127.0.0.1:9222` before executing routines
- Runtime placeholders (`sessionStorage`, `localStorage`, `cookie`, `meta`) are resolved in the browser, not Python-side
- All parameters must be used in the routine (validation enforces this)
- Builtin parameters (`epoch_milliseconds`, `uuid`) don't need to be defined in parameters list

## Testing

### Test Structure

- Unit tests in `tests/unit/`
- Test data in `tests/data/input/` and `tests/data/expected_output/`
- Use `pytest` fixtures from `tests/conftest.py`

### Writing Tests

- Test files should start with `test_`
- Test functions should start with `test_`
- Use descriptive test names
- Test both success and failure cases
- Test edge cases and validation

### Running Tests

- Prefer running single tests or test files for faster iteration
- Run full test suite before committing: `pytest tests/ -v`
- Benchmarks validate routine discovery pipeline: `python scripts/dev/run_benchmarks.py`

## Environment Setup

### Prerequisites

- Python 3.12.3+ (use `pyenv install 3.12.3` if needed)
- Google Chrome (stable)
- uv (recommended) or pip
- OpenAI API key (set `OPENAI_API_KEY` environment variable)

### Virtual Environment

- Always use a virtual environment
- Activate before working: `source bluebox-env/bin/activate`
- Install dependencies: `uv pip install -e .` or `pip install -e .`

### Environment Variables

- `OPENAI_API_KEY` - Required for routine discovery
- Can use `.env` file with `python-dotenv` (loaded automatically with `uv run`)

## Repository Etiquette

### Branch Naming

- Use descriptive branch names
- Examples: `feature/add-new-operation`, `fix/placeholder-resolution`, `refactor/cdp-connection`

### Commits

- Write clear, descriptive commit messages
- Reference issues when applicable
- Separate logical changes into different commits

### Pull Requests

- Add tests for new features
- Ensure all tests pass: `pytest tests/ -v`
- Update documentation if needed
- Follow existing code style

## Example Routines

- `example_data/example_routines/amtrak_one_way_train_search_routine.json` - Train search example
- `example_data/example_routines/download_arxive_paper_routine.json` - Paper download example
- `example_data/example_routines/massachusetts_corp_search_routine.json` - Corporate search example

Use these as references when creating new routines or understanding the routine format.
