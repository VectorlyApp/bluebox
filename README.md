<p align="center">
  <a href="https://www.vectorly.app/"><img src="https://img.shields.io/badge/Website-Vectorly.app-0ea5e9?style=for-the-badge&logo=googlechrome&logoColor=white" /></a>
  <a href="https://console.vectorly.app"><img src="https://img.shields.io/badge/Console-console.vectorly.app-8b5cf6?style=for-the-badge&logo=googlechrome&logoColor=white" /></a>
  <a href="https://www.youtube.com/@VectorlyAI"><img src="https://img.shields.io/badge/YouTube-@VectorlyAI-ff0000?style=for-the-badge&logo=youtube&logoColor=white" /></a>
  <a href="https://www.apache.org/licenses/LICENSE-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-10b981?style=for-the-badge&logo=apache&logoColor=white" /></a>
</p>

# bluebox-sdk 🟦

REVERSE ENGINEER ANY WEB APP! ⚡️

**Why "Blue Box"?** Named after the [phone phreaking devices](https://en.wikipedia.org/wiki/Blue_box) that let tech enthusiasts in the 1960s and 70s explore telephone networks.

**You are in the right place if you ...**

* want your AI agent to take real actions on the web
* never want to pay for an API (except for OpenAI... shouldn't piss them off...)
* are tired of complicated, endless API integrations
* dealing with closed APIs

Welcome to Vectorly's Bluebox... **No API? No Problem!**

## Our Process ᯓ ✈︎`

1) Launch Chrome in debug mode (enable DevTools protocol on `127.0.0.1:9222`).
2) Run the browser monitor and manually perform the target actions to capture browser state.
3) Specify your task and run the routine discovery script; the agent reverse‑engineers the API flow.
4) Review and run/test the generated routine JSON (locally).
5) Go to [console.vectorly.app](https://console.vectorly.app) and productionize your routines!

## What is a *Routine*?

> A [**Routine**](https://vectorly.app/docs/routines/overview) is a portable automation recipe that captures how to perform a specific task in any web app.

Define once. Reuse everywhere. Automate anything you can do in a browser.

Each Routine includes:

- **name** — a human-readable identifier
- **description** — what the Routine does
- **parameters** — input values the Routine needs to run (e.g. URLs, credentials, text)
- **operations** — the ordered browser actions that perform the automation

Example:

> Navigate to a dashboard, search based on keywords, and return results — all as a reusable Routine.

### Quickstart

<p align="center">
  <video src="https://github.com/user-attachments/assets/1b239ba2-45fd-4098-96c0-6d8f97e5e66b" width="760" controls autoplay loop muted>
    Video not supported? [Watch the demo on YouTube](https://youtu.be/s4Xe_2pXcSQ)
  </video>
</p>

### Parameters

- Defined as typed inputs (see [`Parameter`](https://github.com/VectorlyApp/bluebox/blob/main/src/data_models/production_routine.py) class).
- Each parameter has required `name` and `description` fields. Optional fields include `type` (defaults to `string`), `required` (defaults to `true`), `default`, and `examples`.
- Parameters are referenced inside `operations` using placeholder tokens like `"{{paramName}}"` or `\"{{paramName}}\"` (see [Placeholder Interpolation](#placeholder-interpolation-) below).
- **Parameter Types**: Supported types include `string`, `integer`, `number`, `boolean`, `date`, `datetime`, `email`, `url`, and `enum`.
- **Parameter Validation**: Parameters support validation constraints such as `min_length`, `max_length`, `min_value`, `max_value`, `pattern` (regex), `enum_values`, and `format`.
- **Reserved Prefixes**: Parameter names cannot start with reserved prefixes: `sessionStorage`, `localStorage`, `cookie`, `meta`, `uuid`, `epoch_milliseconds`.

### Operations

Operations define the executable steps of a Routine. They are represented as a **typed list** (see [`RoutineOperationUnion`](https://github.com/VectorlyApp/bluebox/blob/main/bluebox/data_models/routine/operation.py)) and are executed sequentially by a browser.

Each operation specifies a `type` and its parameters:

#### Navigation

- **navigate** — open a URL in the browser.
  ```json
  { "type": "navigate", "url": "https://example.com", "sleep_after_navigation_seconds": 3.0 }
  ```
- **sleep** — pause execution for a given duration (in seconds).
  ```json
  { "type": "sleep", "timeout_seconds": 1.5 }
  ```
- **wait_for_url** — wait for the current URL to match a regex pattern.
  ```json
  { "type": "wait_for_url", "url_regex": ".*dashboard.*", "timeout_ms": 20000 }
  ```

#### Network

- **fetch** — perform an HTTP request defined by an `endpoint` object (method, URL, headers, body, credentials). Optionally, store the response under a `session_storage_key`.
  ```json
  {
    "type": "fetch",
    "endpoint": {
      "method": "GET",
      "url": "https://api.example.com",
      "headers": {},
      "body": {},
      "credentials": "same-origin"
    },
    "session_storage_key": "userData"
  }
  ```
- **download** — download a file and return it as base64-encoded content.
  ```json
  {
    "type": "download",
    "endpoint": {
      "method": "GET",
      "url": "https://example.com/report.pdf",
      "headers": {},
      "body": {}
    },
    "filename": "report.pdf"
  }
  ```
- **get_cookies** — retrieve all cookies (including HttpOnly) via CDP and store them in session storage.
  ```json
  { "type": "get_cookies", "session_storage_key": "allCookies", "domain_filter": "*" }
  ```

#### Interaction

- **click** — click on an element by CSS selector. Automatically validates visibility to avoid honeypot traps.
  ```json
  { "type": "click", "selector": "#submit-button", "button": "left", "ensure_visible": true }
  ```
- **input_text** — type text into an input element. Validates visibility before typing.
  ```json
  { "type": "input_text", "selector": "#username", "text": "\"{{username}}\"", "clear": false }
  ```
- **press** — press a keyboard key (enter, tab, escape, etc.).
  ```json
  { "type": "press", "key": "enter" }
  ```
- **scroll** — scroll the page or a specific element.
  ```json
  { "type": "scroll", "selector": "#content", "delta_y": 500, "behavior": "auto" }
  ```

#### Code Execution

- **js_evaluate** — evaluate custom JavaScript code in the browser context. Must be wrapped in an IIFE format.
  ```json
  {
    "type": "js_evaluate",
    "js": "(function() { return document.title; })()",
    "timeout_seconds": 5.0,
    "session_storage_key": "pageTitle"
  }
  ```

#### Data

- **return** — return the value previously stored under a `session_storage_key`.
  ```json
  { "type": "return", "session_storage_key": "userData" }
  ```
- **return_html** — return HTML content from the page or a specific element.
  ```json
  { "type": "return_html", "scope": "page" }
  ```

Example sequence:

```json
[
  { "type": "navigate", "url": "https://example.com/login" },
  { "type": "sleep", "timeout_seconds": 1 },
  { 
    "type": "fetch", 
    "endpoint": { 
      "method": "POST", 
      "url": "/auth", 
      "body": { "username": "\"{{user}}\"", "password": "\"{{pass}}\"" } 
    }, 
    "session_storage_key": "token" 
  },
  { "type": "return", "session_storage_key": "token" }
]
```

This defines a deterministic flow: open → wait → authenticate → return a session token.

### Placeholder Interpolation `{{...}}`

Placeholders inside operation fields are resolved at runtime:

- **Parameter placeholders**: `"{{paramName}}"` or `\"{{paramName}}\"` → substituted from routine parameters
- **Storage placeholders** (read values from the current session):
  - `{{sessionStorage:myKey.path.to.value}}` — access nested values in sessionStorage
  - `{{localStorage:myKey}}` — access localStorage values
  - `{{cookie:CookieName}}` — read cookie values
  - `{{meta:name}}` — read meta tag content (e.g., `<meta name="csrf-token">`)

**Important:** Currently, `sessionStorage`, `localStorage`, `cookie`, and `meta` placeholder resolution is supported only inside fetch `headers` and `body`. Future versions will support interpolation anywhere in operations.

Interpolation occurs before an operation executes. For example, a fetch endpoint might be:

```json
{
  "type": "fetch",
  "endpoint": {
    "method": "GET",
    "url": "https://api.example.com/search?paramName1=\"{{paramName1}}\"&paramName2=\"{{paramName1}}\"",
    "headers": {
      "Authorization": "Bearer {{cookie:auth_token}}"
    },
    "body": {}
  },
  "session_storage_key": "result_key"
}
```

This substitutes parameter values and injects `auth_token` from cookies. The JSON response is stored under `sessionStorage['result_key']` and can be returned by a final `return` operation using the matching `session_storage_key`.

## Prerequisites

- Python 3.12+
- Google Chrome (stable)
- [uv (Python package manager)](https://github.com/astral-sh/uv) (optional, for development)
  - macOS/Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`
  - Windows (PowerShell): `iwr https://astral.sh/uv/install.ps1 -UseBasicParsing | iex`
- OpenAI API key

## Installation

### From PyPI (Recommended)

**Note:** We recommend using a virtual environment to avoid dependency conflicts.

```bash
# Create and activate a virtual environment
# Option 1: Using uv (recommended - handles Python version automatically)
uv venv bluebox-env
source bluebox-env/bin/activate  # On Windows: bluebox-env\Scripts\activate
uv pip install bluebox-sdk

# Option 2: Using python3 (if Python 3.12+ is your default)
python3 -m venv bluebox-env
source bluebox-env/bin/activate  # On Windows: bluebox-env\Scripts\activate
pip install bluebox-sdk

# Option 3: Using pyenv (if you need a specific Python version)
pyenv install 3.12.3  # if not already installed
pyenv local 3.12.3
python -m venv bluebox-env
source bluebox-env/bin/activate  # On Windows: bluebox-env\Scripts\activate
pip install bluebox-sdk

# Troubleshooting: If pip is not found, recreate the venv or use:
python -m ensurepip --upgrade  # Install pip in the venv
pip install bluebox-sdk
```

### From Source (Development)

For development or if you want the latest code:

```bash
# Clone the repository
git clone https://github.com/VectorlyApp/bluebox.git
cd bluebox

# Create and activate virtual environment
python3 -m venv bluebox-env
source bluebox-env/bin/activate  # On Windows: bluebox-env\Scripts\activate

# Install in editable mode
pip install -e .

# Or using uv (faster)
uv venv bluebox-env
source bluebox-env/bin/activate
uv pip install -e .
```

## Quickstart (Easiest Way) 🚀

The fastest way to get started is using the quickstart script, which automates the entire workflow:

```bash
# Make sure bluebox-sdk is installed
pip install bluebox-sdk 
# Or install from the latest code
# pip install "git+https://github.com/VectorlyApp/bluebox-sdk.git"

# Set your OpenAI API key
export OPENAI_API_KEY="sk-..."

# Run the quickstart script
python quickstart.py
```

The quickstart script will:

1. ✅ Automatically launch Chrome in debug mode
2. 📊 Start browser monitoring (you perform actions)
3. 🤖 Discover routines from captured data
4. 📝 Show you how to execute the discovered routine

**Note:** The quickstart script is included in the repository. If you installed from PyPI, you can download it from the [GitHub repository](https://github.com/VectorlyApp/bluebox/blob/main/quickstart.py).

## Guide Agent Terminal (Interactive Mode)

For iterative routine development, debugging, and more complex workflows, use the Guide Agent terminal app. It provides a full chat interface with an LLM-powered agent that can help you create, edit, and refine routines interactively.

<p align="center">
  <img src="https://github.com/user-attachments/assets/bb2df1e0-46e2-456e-b991-6583fc9038da" alt="Guide Agent Terminal" width="824" />
</p>

```bash
# Set your OpenAI API key
export OPENAI_API_KEY="sk-..."

# Run the guide agent terminal
python scripts/run_guide_agent.py
```

**Features:**

- **Interactive chat** with streaming LLM responses
- **Load/edit routines** with hot-reload on file changes
- **Browser monitoring** directly from the terminal (`/monitor`)
- **Routine discovery** with agent-guided task description
- **Suggested edits** with diff/accept/reject workflow
- **Routine validation and execution** (`/validate`, `/execute`)

**Commands:**

| Command | Description |
|---------|-------------|
| `/load <file.json>` | Load a routine file (auto-reloads on edits) |
| `/execute [params.json]` | Execute the loaded routine |
| `/monitor` | Start browser monitoring session |
| `/validate` | Validate the current routine |
| `/diff`, `/accept`, `/reject` | Review agent-suggested edits |
| `/show`, `/status` | Display routine details and state |
| `/help` | Show all commands |

**When to use Guide Agent vs Quickstart:**

- **Quickstart (`quickstart.py`)**: First-time users, demos, simple one-off automation tasks
- **Guide Agent (`scripts/run_guide_agent.py`)**: Iterative routine development, debugging, complex workflows requiring back-and-forth with the agent

## Specialized Agents (Beta)

In addition to the Guide Agent, we provide specialized agents for analyzing captured browser data:

| Agent | Purpose |
|-------|---------|
| **Network Spy** | Search and analyze network traffic. Find API endpoints, inspect request/response patterns, discover authentication flows. |
| **Trace Hound** | Trace where tokens and values originate. Search across network, cookies, localStorage, sessionStorage, and window properties to find the source of any value. |
| **Docs Digger** | Search through documentation and code files. Find relevant docs, examples, and implementation details in a codebase. |

These agents are in beta and available in `bluebox/agents/`. They can be used programmatically or integrated into custom workflows for deeper analysis of captured browser sessions.

## Reverse Engineer Web Apps

The reverse engineering process follows a simple three-step workflow:

1. **Monitor** — Capture network traffic, storage events, and interactions while you manually perform the target task in Chrome
2. **Discover** — Let the AI agent analyze the captured data and generate a reusable Routine
3. **Execute** — Run the discovered Routine with different parameters to automate the task

### Legal & Privacy Notice

Reverse-engineering and automating a website can violate terms of service. Store captures securely and scrub any sensitive fields before sharing.

### Quick Start (Recommended)

**Easiest way:** Use the [quickstart script](#quickstart-easiest-way-🚀) which automates the entire workflow.

### Manual Workflow (Step-by-Step)

#### 0. Launch Chrome in Debug Mode

> 💡 **Tip:** The [quickstart script](#quickstart-easiest-way-🚀) automatically launches Chrome for you. You only need these manual instructions if you're not using the quickstart script.

##### macOS

```bash
# Create temporary Chrome user directory
mkdir -p $HOME/tmp/chrome

# Launch Chrome in debug mode
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  --remote-debugging-address=127.0.0.1 \
  --remote-debugging-port=9222 \
  --user-data-dir="$HOME/tmp/chrome" \
  --remote-allow-origins='*' \
  --no-first-run \
  --no-default-browser-check

# Verify Chrome is running
curl http://127.0.0.1:9222/json/version
```

##### Windows

```powershell
# Create temporary Chrome user directory
New-Item -ItemType Directory -Force -Path "$env:USERPROFILE\tmp\chrome" | Out-Null

# Locate Chrome
$chrome = "C:\Program Files\Google\Chrome\Application\chrome.exe"
if (!(Test-Path $chrome)) {
  $chrome = "C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"
}

# Launch Chrome in debug mode
& $chrome `
  --remote-debugging-address=127.0.0.1 `
  --remote-debugging-port=9222 `
  --user-data-dir="$env:USERPROFILE\tmp\chrome" `
  --remote-allow-origins=* `
  --no-first-run `
  --no-default-browser-check

# Verify Chrome is running
(Invoke-WebRequest http://127.0.0.1:9222/json/version).Content
```

##### Linux

```bash
# Create temporary Chrome user directory
mkdir -p $HOME/tmp/chrome

# Launch Chrome in debug mode (adjust path if needed)
google-chrome \
  --remote-debugging-address=127.0.0.1 \
  --remote-debugging-port=9222 \
  --user-data-dir="$HOME/tmp/chrome" \
  --remote-allow-origins='*' \
  --no-first-run \
  --no-default-browser-check

# Verify Chrome is running
curl http://127.0.0.1:9222/json/version
```

#### 1. Monitor Browser While Performing Some Task

Use the CDP browser monitor to block trackers and capture network, storage, and interaction data while you manually perform the task in Chrome.

**Run this command to start monitoring:**

```bash
bluebox-monitor --host 127.0.0.1 --port 9222 --output-dir ./cdp_captures --url about:blank --incognito
```

The script will open a new tab (starting at `about:blank`). Navigate to your target website, then manually perform the actions you want to automate (e.g., search, login, export report). Keep Chrome focused during this process. Press `Ctrl+C` and the script will consolidate transactions and produce a HAR automatically.

**Output structure** (under `--output-dir`, default `./cdp_captures`):

```
cdp_captures/
├── session_summary.json
├── network/
│   ├── consolidated_transactions.json
│   ├── network.har
│   └── transactions/
│       └── <timestamp_url_id>/
│           ├── request.json
│           ├── response.json
│           └── response_body.[ext]
└── storage/
    └── events.jsonl
```

Tip: Keep Chrome focused while monitoring and perform the target flow (search, checkout, etc.). Press Ctrl+C to stop; the script will consolidate transactions and produce a HTTP Archive (HAR) automatically.

#### 2. Run Routine-Discovery Agent (Our Very Smart AI with Very Good Prompts🔮)🤖

Use the **routine-discovery pipeline** to analyze captured data and synthesize a reusable Routine (`navigate → fetch → return`).

**Prerequisites:** You’ve already captured a session with the browser monitor (`./cdp_captures` exists).

**Run the discovery agent:**

> ⚠️ **Important:** You must specify your own `--task` parameter. The example below is just for demonstration—replace it with a description of what you want to automate.

**Linux/macOS (bash):**

```bash
bluebox-discover \
  --task "Recover API endpoints for searching for trains and their prices" \
  --cdp-captures-dir ./cdp_captures \
  --output-dir ./routine_discovery_output \
  --llm-model gpt-5.1
```

**Windows (PowerShell):**

```powershell
# Simple task (no quotes inside):
bluebox-discover --task "Recover the API endpoints for searching for trains and their prices" --cdp-captures-dir ./cdp_captures --output-dir ./routine_discovery_output --llm-model gpt-5.1
```

**Example tasks:**

- `"recover the api endpoints for searching for trains and their prices"` (shown above)
- `"discover how to search for flights and get pricing"`
- `"find the API endpoint for user authentication"`
- `"extract the endpoint for submitting a job application"`

Arguments:

- **--task**: A clear description of what you want to automate. This guides the AI agent to identify which network requests to extract and convert into a Routine. Examples: searching for products, booking appointments, submitting forms, etc.
- **--cdp-captures-dir**: Root of prior CDP capture output (default: `./cdp_captures`)
- **--output-dir**: Directory to write results (default: `./routine_discovery_output`)
- **--llm-model**: LLM to use for reasoning/parsing (default: `gpt-5.1`)

Outputs (under `--output-dir`):

```
routine_discovery_output/
├── identified_transactions.json    # Chosen transaction id/url
├── routine_transactions.json       # Slimmed request/response samples given to LLM
├── resolved_variables.json         # Resolution hints for cookies/tokens (if any)
└── routine.json                    # Final Routine model (name, parameters, operations)
```

#### 3. Execute the Discovered Routines 🏃

⚠️ **Prerequisite:** Make sure Chrome is still running in debug mode (see [Launch Chrome in Debug Mode](#launch-chrome-in-debug-mode-🐞) above). The routine execution script connects to the same Chrome debug session on `127.0.0.1:9222`.

⚠️ **Important:** If you have a string-typed parameter used in a JSON body field, it may need to be escaped. When the agent generates routines, string parameters are sometimes placed as `"{{PARAM}}"` when they should be `"\"{{PARAM}}\""` to ensure proper JSON string escaping.

**Example:** If you see:

```json
"field": "{{paramName}}"
```

And `paramName` is a string parameter, manually change it to:

```json
"field": "\"{{paramName}}\""
```

This ensures the parameter value is properly quoted as a JSON string when substituted.

Run the example routine:

```bash
# Using a parameters file:

bluebox-execute \
  --routine-path example_data/example_routines/amtrak_one_way_train_search_routine.json \
  --parameters-path example_data/example_routines/amtrak_one_way_train_search_input.json

# Or pass parameters inline (JSON string):

bluebox-execute \
  --routine-path example_data/example_routines/amtrak_one_way_train_search_routine.json \
  --parameters-dict '{"origin": "BOS", "destination": "NYP", "departureDate": "2026-03-22"}'
```

Run a discovered routine:

```bash
bluebox-execute \
  --routine-path routine_discovery_output/routine.json \
  --parameters-path routine_discovery_output/test_parameters.json
```

**Note:** Routines execute in a new incognito tab by default (controlled by the routine's `incognito` field). This ensures clean sessions for each execution.

**Alternative:** Deploy your routine to [console.vectorly.app](https://console.vectorly.app) to expose it as an API endpoint or MCP tool for use in production environments.

## Common Issues ⚠️

- Chrome not detected / cannot connect to DevTools

  - Ensure Chrome is launched in debug mode and `http://127.0.0.1:9222/json/version` returns JSON.
  - Check `--host`/`--port` flags match your Chrome launch args.
- `OPENAI_API_KEY` not set

  - Export the key in your shell or create a `.env` file and run via `uv run` (dotenv is loaded).
- `No such file or directory: './cdp_captures/network/transactions/N/A'` or similar transaction path errors

  - The agent cannot find any network transactions relevant to your task. This usually means:

    - The `--task` description doesn't match what you actually performed during monitoring
    - The relevant network requests weren't captured (they may have been blocked or filtered)
    - The task description is too vague or too specific
  - **Fix:** Reword your `--task` parameter to more accurately describe what you did during the monitoring step, or re-run the browser monitor and ensure you perform the exact actions you want to automate.

## Coming Soon 🔮

### Pipeline Improvements

- **Integration of routine testing into the agentic pipeline**
  - The agent will execute discovered routines, detect failures, and automatically suggest/fix issues to make routines more robust and efficient.
- **Checkpointing progress and resumability**
  - Avoid re-running the entire discovery pipeline after exceptions; the agent will checkpoint progress and resume from the last successful stage.
- **Parameter resolution visibility**
  - During execution, show which placeholders (e.g., `{{sessionStorage:...}}`, `{{cookie:...}}`, `{{localStorage:...}}`) resolved successfully and which failed.

### Additional Operations (Not Yet Implemented)

#### Navigation

- **wait_for_title** — wait for the page title to match a regex pattern

#### Network

- **network_sniffing** (background operation) — intercept and capture network requests matching a URL pattern in the background while other operations execute. Useful for capturing API calls triggered by UI interactions.
  - Supports different capture modes: `list` (all matching requests), `first` (only first match), `last` (only last match)
  - Can capture request, response, or body data

#### Interaction

- **hover** — move mouse over an element to trigger hover states
- **wait_for_selector** — wait for an element to reach a specific state (visible, hidden, attached, detached)
- **set_files** — set file paths for file input elements (for file uploads)

#### Data

- **return_screenshot** — capture and return a screenshot of the page as base64

## Running Benchmarks 📊

Benchmarks validate the routine discovery pipeline against known ground-truth routines. They run both deterministic tests (checking routine structure) and LLM-based tests (evaluating semantic correctness).

```bash
# Run all benchmarks
python bluebox/scripts/run_benchmarks.py

# With verbose output (shows each test result as it runs)
python bluebox/scripts/run_benchmarks.py -v

# Use a specific model
python bluebox/scripts/run_benchmarks.py --model gpt-4.1

# Custom output directory
python bluebox/scripts/run_benchmarks.py --output-dir ./my_benchmarks
```

Results are saved to the output directory:
- `{benchmark_name}.json` — Full evaluation results for each benchmark
- `_summary.json` — Aggregated summary of all benchmark runs

## Contributing 🤝

We welcome contributions! Here's how to get started:

1. **Report bugs or request features** — Open an [issue](https://github.com/VectorlyApp/bluebox/issues)
2. **Submit code** — Fork the repo and open a [pull request](https://github.com/VectorlyApp/bluebox/pulls)
3. **Test your code** — Add unit tests and make sure all tests pass:

```bash
pytest tests/ -v
```

Please follow existing code style and include tests for new features.