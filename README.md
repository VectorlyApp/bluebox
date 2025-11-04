<p align="center">
  <a href="https://www.vectorly.app/"><img src="https://img.shields.io/badge/Website-Vectorly.app-0ea5e9?style=for-the-badge&logo=googlechrome&logoColor=white" /></a>
  <a href="https://console.vectorly.app"><img src="https://img.shields.io/badge/Console-console.vectorly.app-8b5cf6?style=for-the-badge&logo=googlechrome&logoColor=white" /></a>
  <a href="https://www.youtube.com/@VectorlyAI"><img src="https://img.shields.io/badge/YouTube-@VectorlyAI-ff0000?style=for-the-badge&logo=youtube&logoColor=white" /></a>
  <a href="https://www.apache.org/licenses/LICENSE-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-10b981?style=for-the-badge&logo=apache&logoColor=white" /></a>
</p>

# web-hacker

REVERSE ENGINEER ANY WEB APP! ⚡️

**You are in the right place if you ...**

* want your AI agent to take real actions on the web
* never want to pay for an API (except for OpenAI... shouldn't piss them off...)
* are tired of complicated, endless API integrations
* dealing with closed APIs

Welcome to Vectorly's Web Hacker... **No API? No Problem!**

## Our Process ᯓ ✈︎

1) Launch Chrome in debug mode (enable DevTools protocol on `127.0.0.1:9222`).
2) Run the browser monitor and manually perform the target actions to capture browser state.
3) Specify your task and run the routine discovery script; the agent reverse‑engineers the API flow.
4) Review and run/test the generated routine JSON (locally).
5) Go to [console.vectorly.app](https://console.vectorly.app) and productionize your routines!

## What is a *Routine*?

> A **Routine** is a portable automation recipe that captures how to perform a specific task in any web app.

Define once. Reuse everywhere. Automate anything you can do in a browser.

Each Routine includes:
- **name** — a human-readable identifier
- **description** — what the Routine does
- **parameters** — input values the Routine needs to run (e.g. URLs, credentials, text)
- **operations** — the ordered browser actions that perform the automation

Example:
> Navigate to a dashboard, search based on keywords, and return results — all as a reusable Routine.

### Parameters

- Defined as typed inputs (see [`Parameter`](https://github.com/VectorlyApp/web-hacker/blob/main/src/data_models/production_routine.py) class).
- Each parameter has required `name` and `description` fields. Optional fields include `type` (defaults to `string`), `required` (defaults to `true`), `default`, and `examples`.
- Parameters are referenced inside `operations` using placeholder tokens like `"{{paramName}}"` or `\"{{paramName}}\"` (see [Placeholder Interpolation](#placeholder-interpolation-) below).
- **Parameter Types**: Supported types include `string`, `integer`, `number`, `boolean`, `date`, `datetime`, `email`, `url`, and `enum`.
- **Parameter Validation**: Parameters support validation constraints such as `min_length`, `max_length`, `min_value`, `max_value`, `pattern` (regex), `enum_values`, and `format`.
- **Reserved Prefixes**: Parameter names cannot start with reserved prefixes: `sessionStorage`, `localStorage`, `cookie`, `meta`, `uuid`, `epoch_milliseconds`.


### Operations

Operations define the executable steps of a Routine. They are represented as a **typed list** (see [`RoutineOperationUnion`](https://github.com/VectorlyApp/web-hacker/blob/main/src/data_models/production_routine.py)) and are executed sequentially by a browser.

Each operation specifies a `type` and its parameters:

- **navigate** — open a URL in the browser.  
  ```json
  { "type": "navigate", "url": "https://example.com" }
  ```
- **sleep** — pause execution for a given duration (in seconds).  
  ```json
  { "type": "sleep", "timeout_seconds": 1.5 }
  ```
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
- **return** — return the value previously stored under a `session_storage_key`.  
  ```json
  { "type": "return", "session_storage_key": "userData" }
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
- [uv (Python package manager)](https://github.com/astral-sh/uv)
  - macOS/Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`
  - Windows (PowerShell): `iwr https://astral.sh/uv/install.ps1 -UseBasicParsing | iex`
- OpenAI API key

## Set up Your Environment 🔧

### Linux

```bash
# 1) Clone and enter the repo
git clone https://github.com/VectorlyApp/web-hacker.git
cd web-hacker

# 2) Create & activate virtual environment (uv)
uv venv --prompt web-hacker
source .venv/bin/activate   # Windows: .venv\\Scripts\\activate

# 3) Install in editable mode via uv (pip-compatible interface)
uv pip install -e .

# 4) Configure environment
cp .env.example .env  # then edit values
# or set directly
export OPENAI_API_KEY="sk-..."
```

### Windows

```powershell
# 1) Clone and enter the repo
git clone https://github.com/VectorlyApp/web-hacker.git
cd web-hacker

# 2) Install uv (if not already installed)
iwr https://astral.sh/uv/install.ps1 -UseBasicParsing | iex

# 3) Create & activate virtual environment (uv)
uv venv --prompt web-hacker
.venv\Scripts\activate

# 4) Install in editable mode via uv (pip-compatible interface)
uv pip install -e .

# 5) Configure environment
copy .env.example .env  # then edit values
# or set directly
$env:OPENAI_API_KEY="sk-..."
```

## Launch Chrome in Debug Mode 🐞

### Instructions for MacOS

```
# You should see JSON containing a webSocketDebuggerUrl like:
# ws://127.0.0.1:9222/devtools/browser/*************************************# Create temporary chrome user directory
mkdir $HOME/tmp
mkdir $HOME/tmp/chrome

# Launch Chrome app in debug mode (this exposes websocket for controlling and monitoring the browser)
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  --remote-debugging-address=127.0.0.1 \
  --remote-debugging-port=9222 \
  --user-data-dir="$HOME/tmp/chrome" \
  '--remote-allow-origins=*' \
  --no-first-run \
  --no-default-browser-check


# Verify chrome is running in debug mode
curl http://127.0.0.1:9222/json/version

# You should see JSON containing a webSocketDebuggerUrl like:
# ws://127.0.0.1:9222/devtools/browser/*************************************
```

### Instructions for Windows

```
# Create temporary Chrome user directory
New-Item -ItemType Directory -Force -Path "$env:USERPROFILE\\tmp\\chrome" | Out-Null

# Locate Chrome (adjust path if Chrome is installed elsewhere)
$chrome = "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe"
if (!(Test-Path $chrome)) {
  $chrome = "C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe"
}

# Launch Chrome in debug mode (exposes DevTools WebSocket)
& $chrome `
  --remote-debugging-address=127.0.0.1 `
  --remote-debugging-port=9222 `
  --user-data-dir="$env:USERPROFILE\\tmp\\chrome" `
  --remote-allow-origins=* `
  --no-first-run `
  --no-default-browser-check


# Verify Chrome is running in debug mode
(Invoke-WebRequest http://127.0.0.1:9222/json/version).Content

# You should see JSON containing a webSocketDebuggerUrl like:
# ws://127.0.0.1:9222/devtools/browser/*************************************
```

## HACK (reverse engineer) WEB APPS 👨🏻‍💻

The reverse engineering process follows a simple three-step workflow:

1. **Monitor** — Capture network traffic, storage events, and interactions while you manually perform the target task in Chrome
2. **Discover** — Let the AI agent analyze the captured data and generate a reusable Routine
3. **Execute** — Run the discovered Routine with different parameters to automate the task

Each step is detailed below. Start by ensuring Chrome is running in debug mode (see [Launch Chrome in Debug Mode](#launch-chrome-in-debug-mode-🐞) above).

### 0. Legal & Privacy Notice ⚠️
Reverse-engineering and automating a website can violate terms of service. Store captures securely and scrub any sensitive fields before sharing.

### 1. Monitor Browser While Performing Some Task

Use the CDP browser monitor to block trackers and capture network, storage, and interaction data while you manually perform the task in Chrome.

**Run this command to start monitoring:**

```bash
python scripts/browser_monitor.py --host 127.0.0.1 --port 9222 --output-dir ./cdp_captures --url about:blank --incognito
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

### 2. Run Routine-Discovery Agent (Our Very Smart AI with Very Good Prompts🔮)🤖

Use the **routine-discovery pipeline** to analyze captured data and synthesize a reusable Routine (`navigate → fetch → return`).

**Prerequisites:** You’ve already captured a session with the browser monitor (`./cdp_captures` exists).

**Run the discovery agent:**

> ⚠️ **Important:** You must specify your own `--task` parameter. The example below is just for demonstration—replace it with a description of what you want to automate.

**Linux/macOS (bash):**
```bash
python scripts/discover_routines.py \
  --task "recover the api endpoints for searching for trains and their prices" \
  --cdp-captures-dir ./cdp_captures \
  --output-dir ./routine_discovery_output \
  --llm-model gpt-5
```

**Windows (PowerShell):**
```powershell
# Simple task (no quotes inside):
python scripts/discover_routines.py --task "Recover the API endpoints for searching for trains and their prices" --cdp-captures-dir ./cdp_captures --output-dir ./routine_discovery_output --llm-model gpt-5
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
- **--llm-model**: LLM to use for reasoning/parsing (default: `gpt-5`)

Outputs (under `--output-dir`):

```
routine_discovery_output/
├── identified_transactions.json    # Chosen transaction id/url
├── routine_transactions.json       # Slimmed request/response samples given to LLM
├── resolved_variables.json         # Resolution hints for cookies/tokens (if any)
└── routine.json                    # Final Routine model (name, parameters, operations)
```

### 3. Execute the Discovered Routines 🏃

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

python scripts/execute_routine.py \
  --routine-path example_routines/amtrak_one_way_train_search_routine.json \
  --parameters-path example_routines/amtrak_one_way_train_search_input.json

# Or pass parameters inline (JSON string):

python scripts/execute_routine.py \
  --routine-path example_routines/amtrak_one_way_train_search_routine.json \
  --parameters-dict '{"origin": "BOS", "destination": "NYP", "departureDate": "2026-03-22"}'
```

Run a discovered routine:

```bash
python scripts/execute_routine.py \
  --routine-path routine_discovery_output/routine.json \
  --parameters-path routine_discovery_output/test_parameters.json
```

**Note:** Routines execute in a new incognito tab by default (controlled by the routine's `incognito` field). This ensures clean sessions for each execution.

**Alternative:** Deploy your routine to [console.vectorly.app](https://console.vectorly.app) to expose it as an API endpoint or MCP server for use in production environments.

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

- Integration of routine testing into the agentic pipeline

  - The agent will execute discovered routines, detect failures, and automatically suggest/fix issues to make routines more robust and efficient.
- Checkpointing progress and resumability

  - Avoid re-running the entire discovery pipeline after exceptions; the agent will checkpoint progress and resume from the last successful stage.
- Context overflow management

  - On detection of context overflow, the agent will checkpoint state, summarize findings, and spawn a continuation agent to proceed with discovery without losing context.
- Parameter resolution visibility

  - During execution, show which placeholders (e.g., `{{sessionStorage:...}}`, `{{cookie:...}}`, `{{localStorage:...}}` resolved successfully and which failed
