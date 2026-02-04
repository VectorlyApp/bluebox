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
- `python bluebox/scripts/run_benchmarks.py` - Run routine discovery benchmarks
- `python bluebox/scripts/run_benchmarks.py -v` - Run benchmarks with verbose output

### CLI Tools

- `bluebox-monitor --host 127.0.0.1 --port 9222 --output-dir ./cdp_captures --url about:blank --incognito` - Start browser monitoring
- `bluebox-discover --task "your task description" --cdp-captures-dir ./cdp_captures --output-dir ./routine_discovery_output --llm-model gpt-5.1` - Discover routines from captures
- `bluebox-execute --routine-path example_data/example_routines/amtrak_one_way_train_search_routine.json --parameters-path example_data/example_routines/amtrak_one_way_train_search_input.json` - Execute a routine

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
6. Fix any placeholder escaping issues (string params need `\"{{param}}\"`)

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

### Agents

AI agents that power routine discovery and conversational interactions:

- `bluebox/agents/routine_discovery_agent.py` - Analyzes CDP captures to generate routines (identifies transactions, extracts/resolves variables, constructs operations)
- `bluebox/agents/guide_agent.py` - Conversational agent for guiding users through routine creation/editing (maintains chat history, dynamic tool registration)

**LLM Infrastructure:**
- `bluebox/llms/infra/data_store.py` - Data stores for CDP captures and vectorstore management (used by both agents)

**Import patterns:**
```python
from bluebox.agents.guide_agent import GuideAgent
from bluebox.agents.routine_discovery_agent import RoutineDiscoveryAgent
from bluebox.llms.infra.data_store import DiscoveryDataStore, LocalDiscoveryDataStore
```

### Important Patterns

- **Routine Execution**: Operations execute sequentially, maintaining state via `RoutineExecutionContext`
- **Placeholder Resolution**: String parameters MUST use escape-quoted format: `\"{{paramName}}\"` in JSON bodies
- **Session Storage**: Use `session_storage_key` to store and retrieve data between operations
- **CDP Sessions**: Use flattened sessions for multiplexing via `session_id`

### Common Gotchas

- String parameters in JSON bodies need double escaping: `"field": "\"{{param}}\""` not `"field": "{{param}}"`
- Chrome must be running in debug mode on `127.0.0.1:9222` before executing routines
- Placeholder resolution for `sessionStorage`, `localStorage`, `cookie`, `meta` only works in fetch `headers` and `body` (not in URLs yet)
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
- Benchmarks validate routine discovery pipeline: `python bluebox/scripts/run_benchmarks.py`

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
