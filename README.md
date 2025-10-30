# web-hacker

REVERSE ENGINEER ANY WEB APP!

## Overview of Our Process

1. Launch Chrome in debug mode (enable DevTools protocol on 127.0.0.1:9222).
2. Run the browser monitor and manually perform the target actions to capture browser state.
3. Specify your task and run the routine discovery script; the agent reverse‑engineers the API flow.
4. Review and test the generated routine JSON to automate the workflow.
5. Go to [console.vectorly.app](https://console.vectorly.app) and productionize your routines!

## Prerequisits

- Python 3.11+
- Google Chrome (stable)
- uv (Python package manager)
  - macOS/Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`
  - Windows (PowerShell): `iwr https://astral.sh/uv/install.ps1 -UseBasicParsing | iex`
- OpenAI API key

## Setup Your Environment

```bash
# 1) Clone and enter the repo
git clone <repo-url>
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

## Launch Chrome in Debug Mode

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

## Reverse Engineer!

## Monitor Browser While Performing Some Task

Use the CDP browser monitor to block trackers and capture network, storage and interaction data while you manually perform tasks in Chrome.

Prereq: Chrome running in debug mode (see above). Get a `TAB_ID` from `chrome://inspect/#devices` or `http://127.0.0.1:9222/json`.

Basic usage:

```
uv run python scripts/browser_monitor.py \
  --host 127.0.0.1 \
  --port 9222 \
  --output-dir ./cdp_captures \
  --url https://www.example.com
```

Attach to existing tab:

```
uv run python scripts/browser_monitor.py <TAB_ID>
# or
uv run python scripts/browser_monitor.py --tab-id <TAB_ID>
```

Create a new tab automatically:

```
uv run python scripts/browser_monitor.py --url https://example.com
```

Incognito new tab (only when not supplying TAB_ID):

```
uv run python scripts/browser_monitor.py --incognito --url https://example.com
```

Attach without navigating (keep current page):

```
uv run python scripts/browser_monitor.py --tab-id <TAB_ID> --no-navigate
```

Control output directory behavior:

```
# default is to clear; to keep previous outputs
uv run python scripts/browser_monitor.py --keep-output
```

Select which resource types to capture (default: XHR, Fetch):

```
uv run python scripts/browser_monitor.py --tab-id <TAB_ID> \
  --capture-resources XHR Fetch
```

Disable clearing cookies/storage (cleared by default):

```
uv run python scripts/browser_monitor.py --tab-id <TAB_ID> --no-clear-all
# or granular
uv run python scripts/browser_monitor.py --tab-id <TAB_ID> --no-clear-cookies
uv run python scripts/browser_monitor.py --tab-id <TAB_ID> --no-clear-storage
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

## Run Routine Discovery Pipeline

Use the routine discovery pipeline to generate a reusable Routine (navigate → fetch → return) from your captured network data.

Prereq: You have already captured data with the browser monitor (see above) and have `./cdp_captures` populated.

Basic usage:

```
uv run python scripts/discover_routines.py \
  --task-description "recover the api endpoints for searching for trains and their prices" \
  --cdp-captures-dir ./cdp_captures \
  --output-dir ./routine_discovery_output \
  --llm-model gpt-5
```

Arguments:

- **--task-description**: What you want to achieve? What API endpoint should it discover?
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
