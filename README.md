<p align="center">
  <a href="https://www.vectorly.app/"><img src="https://img.shields.io/badge/Website-Vectorly.app-0ea5e9?style=for-the-badge&logo=googlechrome&logoColor=white" /></a>
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

> A Routine is a portable recipe for automating a web flow. It has:

- name, description
- parameters: input values the routine needs
- operations: ordered steps the browser executes

### Parameters

- Defined as typed inputs (see `src/data_models/production_routine.py:Parameter`).
- Each parameter has a `name`, `type`, `required`, and optional `default`/`examples`.
- Parameters are referenced inside operations using placeholder tokens like `{{argument_1}}`, `{{argument_2}}`.

### Operations

Operations are a typed list (see `RoutineOperationUnion`) executed in order:

- navigate: `{ "type": "navigate", "url": "https://example.com" }`
- sleep: `{ "type": "sleep", "timeout_seconds": 1.5 }`
- fetch: performs an HTTP request described by an `endpoint` object (method, url, headers, body, credentials) and can store results under a `session_storage_key`.
- return: returns the value previously stored under a `session_storage_key`.

### Placeholder Interpolation `{{...}}`

Placeholders inside operation fields are resolved at runtime:

- Parameter placeholders: `{{paramName}}` → substituted from routine parameters
- Storage placeholders (read values from the current session):
  - `{{sessionStorage:myKey.path.to.value}}`
  - `{{localStorage:myKey}}`
  - `{{cookie:CookieName}}`

**Important:** Currently, `sessionStorage`, `localStorage`, and `cookie` placeholder resolution is supported only inside fetch `headers` and `body`. Future versions will support interpolation anywhere in operations.

Interpolation occurs before an operation executes. For example, a fetch endpoint might be:

```
{
  "type": "fetch",
  "endpoint": {
    "method": "GET",
    "url": "https://api.example.com/search?arg1={{argument_1}}&arg2={{argument_2}}",
    "headers": {
      "Authorization": "Bearer {{cookie:auth_token}}"
    },
    "body": {}
  },
  "session_storage_key": "result_key"
}
```

This substitutes parameter values and injects `auth_token` from cookies. The JSON response is stored under `sessionStorage['result_key']` and can be returned by a final `return` operation using the matching `session_storage_key`.

## Prerequisits

- Python 3.11+
- Google Chrome (stable)
- uv (Python package manager)
  - macOS/Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`
  - Windows (PowerShell): `iwr https://astral.sh/uv/install.ps1 -UseBasicParsing | iex`
- OpenAI API key

## Setup Your Environment 🔧

```bash
# 1) Clone and enter the repo
git clone https://github.com/VectorlyApp/web-hacker.git
cd web-hacker

# 2) Create & activate virtual environment (uv)
uv venv .venv
source .venv/bin/activate   # Windows: .venv\\Scripts\\activate

# 3) Install in editable mode via uv (pip-compatible interface)
uv pip install -e .

# 4) Configure environment
cp .env.example .env  # then edit values
# or set directly
export OPENAI_API_KEY="sk-..."
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

### Monitor Browser While Performing Some Task

Use the CDP browser monitor to block trackers and capture network, storage and interaction data while you manually perform tasks in Chrome.

Prereq: Chrome running in debug mode (see above). Get a `TAB_ID` from `chrome://inspect/#devices` or `http://127.0.0.1:9222/json`.

Basic usage:

```
python scripts/browser_monitor.py \
  --host 127.0.0.1 \
  --port 9222 \
  --output-dir ./cdp_captures \
  --url https://www.example.com
```

Attach to existing tab:

```
python scripts/browser_monitor.py <TAB_ID>
# or
python scripts/browser_monitor.py --tab-id <TAB_ID>
```

Create a new tab automatically:

```
python scripts/browser_monitor.py --url https://example.com
```

Incognito new tab (only when not supplying TAB_ID):

```
python scripts/browser_monitor.py --incognito --url https://example.com
```

Attach without navigating (keep current page):

```
python scripts/browser_monitor.py --tab-id <TAB_ID> --no-navigate
```

Control output directory behavior:

```
# default is to clear; to keep previous outputs
python scripts/browser_monitor.py --keep-output
```

Select which resource types to capture (default: XHR, Fetch):

```
python scripts/browser_monitor.py --tab-id <TAB_ID> \
  --capture-resources XHR Fetch
```

Disable clearing cookies/storage (cleared by default):

```
python scripts/browser_monitor.py --tab-id <TAB_ID> --no-clear-all
# or granular
python scripts/browser_monitor.py --tab-id <TAB_ID> --no-clear-cookies
python scripts/browser_monitor.py --tab-id <TAB_ID> --no-clear-storage
```

Output structure (under `--output-dir`, default `./cdp_captures`):

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
├── storage/
│   └── events.jsonl
```

Tip: Keep Chrome focused while monitoring and perform the target flow (search, checkout, etc.). Press Ctrl+C to stop; the script will consolidate transactions and produce a HAR automatically.

### Run Routine Discovery Agent (Our Very Smart AI with Very Good Prompt🔮)🤖

Use the routine discovery pipeline to generate a reusable Routine (navigate → fetch → return) from your captured network data.

Prereq: You have already captured data with the browser monitor (see above) and have `./cdp_captures` populated.

Basic usage:

```
python scripts/discover_routines.py \
  --task "recover the api endpoints for searching for trains and their prices" \
  --cdp-captures-dir ./cdp_captures \
  --output-dir ./routine_discovery_output \
  --llm-model gpt-5
```

Arguments:

- **--task**: What you want to achieve? What API endpoint should it discover?
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

### Execute the Discovered Routines 🏃

Run the example routine: 



```
# Using a parameters file (see examples in `scripts/execute_routine.py`):

python scripts/execute_routine.py \
  --routine-path example_data/amtrak_one_way_train_search_routine.json \
  --parameters-path example_data/amtrak_one_way_train_search_input.json


# Or pass parameters inline (JSON string) — matches the script’s examples:

python scripts/execute_routine.py \
  --routine-path example_data/amtrak_one_way_train_search_routine.json \
  --parameters-dict '{"origin": "boston", "destination": "new york", "departureDate": "2026-03-22"}'
```

Once you have a routine JSON, run it in a real browser session (same Chrome debug session):

```
python scripts/execute_routine.py \
        --routine-path routine_discovery_output/routine.json \
        --parameters-path routine_discovery_output/test_parameters.json
```

## Common Issues ⚠️

- Chrome not detected / cannot connect to DevTools

  - Ensure Chrome is launched in debug mode and `http://127.0.0.1:9222/json/version` returns JSON.
  - Check `--host`/`--port` flags match your Chrome launch args.
- `OPENAI_API_KEY` not set

  - Export the key in your shell or create a `.env` file and run via `uv run` (dotenv is loaded).

## Coming Soon 🔮

- Integration of routine testing into the agentic pipeline

  - The agent will execute discovered routines, detect failures, and automatically suggest/fix issues to make routines more robust and efficient.
- Checkpointing progress and resumability

  - Avoid re-running the entire discovery pipeline after exceptions; the agent will checkpoint progress and resume from the last successful stage.
- Context overflow management

  - On detection of context overflow, the agent will checkpoint state, summarize findings, and spawn a continuation agent to proceed with discovery without losing context.
- Parameter resolution visibility

  - During execution, show which placeholders (e.g., `{{sessionStorage:...}}`, `{{cookie:...}}`, `{{localStorage:...}}` resolved successfully and which failed
