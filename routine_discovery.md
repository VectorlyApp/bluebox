# Creating Your Own Routines

Everything below covers how to reverse-engineer web apps and build your own routines from scratch. If you just want to run existing routines, the [BlueBox Agent](README.md#bluebox-agent) is all you need.

**Additional prerequisites for routine creation:**

- Google Chrome (stable)
- OpenAI API key (`export OPENAI_API_KEY="sk-..."`)

### What is a *Routine*?

> A [**Routine**](https://vectorly.app/docs/routines/overview) is a portable automation recipe that captures how to perform a specific task in any web app.

Define once. Reuse everywhere. Automate anything you can do in a browser.

Each Routine includes:

- **name** — a human-readable identifier
- **description** — what the Routine does
- **parameters** — input values the Routine needs to run (e.g. URLs, credentials, text)
- **operations** — the ordered browser actions that perform the automation

Example:

> Navigate to a dashboard, search based on keywords, and return results — all as a reusable Routine.

### Quickstart (Easiest Way) 🚀

<p align="center">
  <video src="https://github.com/user-attachments/assets/1b239ba2-45fd-4098-96c0-6d8f97e5e66b" width="760" controls autoplay loop muted>
    Video not supported? [Watch the demo on YouTube](https://youtu.be/s4Xe_2pXcSQ)
  </video>
</p>

The fastest way to get started is using the quickstart script, which automates the entire workflow:

```bash
# Make sure bluebox-lib is installed
pip install bluebox-lib
# Or install from the latest code
# pip install "git+https://github.com/VectorlyApp/bluebox.git"

# Set your OpenAI API key
export OPENAI_API_KEY="sk-..."

# Run the quickstart script
python quickstart.py
```

The quickstart script will:

1. 📊 **Monitor** — Launch Chrome (if needed) and capture browser activity while you perform actions
2. 🤖 **Discover** — Analyze captured data and generate a reusable Routine
3. 🏃 **Execute** — Run the discovered routine with your parameters

### Guide Agent Terminal (Interactive Mode)

For iterative routine development, debugging, and more complex workflows, use the Guide Agent terminal app. It provides a full chat interface with an LLM-powered agent that can help you create, edit, and refine routines interactively.

<p align="center">
  <img src="https://github.com/user-attachments/assets/bb2df1e0-46e2-456e-b991-6583fc9038da" alt="Guide Agent Terminal" width="824" />
</p>

```bash
# Set your OpenAI API key
export OPENAI_API_KEY="sk-..."

# Run the guide agent terminal
bluebox-guide
```

**Features:**

- **Interactive chat** with streaming LLM responses
- **Load/edit routines** with hot-reload on file changes
- **Browser monitoring** directly from the terminal (`/monitor`)
- **Routine discovery** with agent-guided task description
- **Suggested edits** with diff/accept/reject workflow
- **Routine validation and execution** (`/validate`, `/execute`)

**Commands:**

| Command                             | Description                                 |
| ----------------------------------- | ------------------------------------------- |
| `/load <file.json>`               | Load a routine file (auto-reloads on edits) |
| `/unload`                         | Unload the current routine                  |
| `/execute [params.json]`          | Execute the loaded routine                  |
| `/monitor`                        | Start browser monitoring session            |
| `/validate`                       | Validate the current routine                |
| `/diff`, `/accept`, `/reject` | Review agent-suggested edits                |
| `/show`, `/status`              | Display routine details and state           |
| `/chats`                          | Show all messages in the thread             |
| `/reset`                          | Start a new conversation                    |
| `/help`                           | Show all commands                           |
| `/quit`                           | Exit                                        |

**When to use Guide Agent vs Quickstart:**

- **Quickstart (`quickstart.py`)**: First-time users, demos, simple one-off automation tasks
- **Guide Agent (`bluebox-guide`)**: Iterative routine development, debugging, complex workflows

### Manual Workflow (Step-by-Step)

The reverse engineering process follows three steps: **Monitor** (capture browser traffic) → **Discover** (AI generates a Routine) → **Execute** (run with different parameters).

> **Legal & Privacy Notice:** Reverse-engineering and automating a website can violate terms of service. Store captures securely and scrub any sensitive fields before sharing.

##### 0. Launch Chrome in Debug Mode

> 💡 **Tip:** The [quickstart script](#quickstart-easiest-way-) automatically launches Chrome for you. You only need these manual instructions if you're not using the quickstart script.

<details>
<summary><strong>macOS</strong></summary>

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

</details>

<details>
<summary><strong>Windows</strong></summary>

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

</details>

<details>
<summary><strong>Linux</strong></summary>

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

</details>

##### 1. Monitor Browser While Performing Some Task

Use the CDP browser monitor to block trackers and capture network, storage, and interaction data while you manually perform the task in Chrome.

**Run this command to start monitoring:**

```bash
bluebox-monitor --host 127.0.0.1 --port 9222 --output-dir ./cdp_captures --url about:blank --incognito
```

Navigate to your target website, perform the actions you want to automate, then press `Ctrl+C`. The script consolidates transactions and produces a HAR automatically.

**Output structure** (under `--output-dir`, default `./cdp_captures`):

```
cdp_captures/
├── session_summary.json
├── network/
│   ├── events.jsonl
│   ├── javascript_events.jsonl
├── storage/
│   └── events.jsonl
├── interaction/
│   └── events.jsonl
├── dom/
│   └── events.jsonl
└── window_properties/
    └── events.jsonl
```

##### 2. Run Routine-Discovery Agent 🤖

Use the **routine-discovery pipeline** to analyze captured data and synthesize a reusable Routine (`navigate → fetch → return`).

**Prerequisites:** You've already captured a session with the browser monitor (`./cdp_captures` exists).

**Run the discovery agent** (replace `--task` with your own description):

```bash
bluebox-discover \
  --task "Recover API endpoints for searching for trains and their prices" \
  --cdp-captures-dir ./cdp_captures \
  --output-dir ./routine_discovery_output \
  --llm-model gpt-5.1
```

Arguments:

- **--task**: A clear description of what you want to automate. This guides the AI agent to identify which network requests to extract and convert into a Routine. Examples: searching for products, booking appointments, submitting forms, etc.
- **--cdp-captures-dir**: Root of prior CDP capture output (default: `./cdp_captures`)
- **--output-dir**: Directory to write results (default: `./routine_discovery_output`)
- **--llm-model**: LLM to use for reasoning/parsing (default: `gpt-5.1`)

Outputs (under `--output-dir`):

```
routine_discovery_output/
├── routine.json                    # Final Routine model (name, parameters, operations)
├── dev_routine.json                # Intermediate DevRoutine (debugging)
├── root_transaction.json           # Identified root endpoint
├── identified_transactions.json    # Chosen transaction id/url
├── routine_transactions.json       # Slimmed request/response samples given to LLM
├── resolved_variables.json         # Resolution hints for cookies/tokens (if any)
├── message_history.json            # Full agent conversation transcript
└── transaction_N/                  # Per-transaction extracted/resolved data
    ├── extracted_variables.json
    └── resolved_variables.json
```

##### 3. Execute the Discovered Routines 🏃

⚠️ **Prerequisite:** Chrome must still be running in debug mode (`127.0.0.1:9222`).

```bash
# Run an example routine (from a parameters file):
bluebox-execute \
  --routine-path example_data/example_routines/amtrak_one_way_train_search_routine.json \
  --parameters-path example_data/example_routines/amtrak_one_way_train_search_input.json

# Or pass parameters inline:
bluebox-execute \
  --routine-path example_data/example_routines/amtrak_one_way_train_search_routine.json \
  --parameters-dict '{"origin": "BOS", "destination": "NYP", "departureDate": "2026-03-22"}'

# Run a discovered routine:
bluebox-execute \
  --routine-path routine_discovery_output/routine.json \
  --parameters-path routine_discovery_output/test_parameters.json
```

Routines execute in a new incognito tab by default. You can also deploy routines to [console.vectorly.app](https://console.vectorly.app) to expose them as API endpoints or MCP tools.

### Specialized Agents (Beta)

In addition to the Guide Agent, we provide specialized agents for analyzing captured browser data:

| Agent                            | Purpose                                                                                                                                                       |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Network Specialist**     | Search and analyze network traffic. Find API endpoints, inspect request/response patterns, discover authentication flows.                                     |
| **Value Trace Resolver**   | Trace where tokens and values originate. Search across network, cookies, localStorage, sessionStorage, and window properties to find the source of any value. |
| **JS Specialist**          | Interactive JavaScript code generation and analysis. Analyzes DOM snapshots, JS files, and network traffic to generate custom code for data extraction.       |
| **Interaction Specialist** | User interaction analysis and parameter discovery. Analyzes interaction events (clicks, inputs, form submissions) to discover parameters and workflows.       |

```bash
# Network Specialist - analyze captured network traffic
bluebox-network-specialist --jsonl-path ./cdp_captures/network/events.jsonl

# Value Trace Resolver - trace token origins across network, storage, and window properties
bluebox-value-trace-resolver-specialist \
    --network-jsonl ./cdp_captures/network/events.jsonl \
    --storage-jsonl ./cdp_captures/storage/events.jsonl \
    --window-props-jsonl ./cdp_captures/window_properties/events.jsonl

# JS Specialist - interactive JavaScript code generation
bluebox-js-specialist \
    --dom-snapshots-path ./cdp_captures/dom/events.jsonl \
    --javascript-events-jsonl-path ./cdp_captures/network/javascript_events.jsonl \
    --network-events-jsonl-path ./cdp_captures/network/events.jsonl \
    --remote-debugging-address 127.0.0.1:9222

# Interaction Specialist - analyze user interactions and discover parameters
bluebox-interaction-specialist --jsonl-path ./cdp_captures/interaction/events.jsonl
```

### Routine Reference

<details>
<summary><strong>Parameters, Operations, and Placeholder Interpolation</strong></summary>

#### Parameters

- Defined as typed inputs (see [`Parameter`](https://github.com/VectorlyApp/bluebox/blob/main/bluebox/data_models/routine/parameter.py) class).
- Each parameter has required `name` and `description` fields. Optional fields include `type` (defaults to `string`), `required` (defaults to `true`), `default`, and `examples`.
- Parameters are referenced inside `operations` using `"{{paramName}}"` placeholder tokens (see [Placeholder Interpolation](#placeholder-interpolation) below). `Parameter.type` drives coercion at runtime.
- **Parameter Types**: Supported types include `string`, `integer`, `number`, `boolean`, `date`, `datetime`, `email`, `url`, and `enum`.
- **Parameter Validation**: Parameters support validation constraints such as `min_length`, `max_length`, `min_value`, `max_value`, `pattern` (regex), `enum_values`, and `format`.
- **Reserved Prefixes**: Parameter names cannot start with reserved prefixes: `sessionStorage`, `localStorage`, `cookie`, `meta`, `windowProperty`, `uuid`, `epoch_milliseconds`.

#### Operations

Operations define the executable steps of a Routine. They are represented as a **typed list** (see [`RoutineOperationUnion`](https://github.com/VectorlyApp/bluebox/blob/main/bluebox/data_models/routine/operation.py)) and are executed sequentially by a browser.

Each operation specifies a `type` and its parameters:

**Navigation**

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

**Network**

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

**Interaction**

- **click** — click on an element by CSS selector. Automatically validates visibility to avoid honeypot traps.
  ```json
  { "type": "click", "selector": "#submit-button", "button": "left", "ensure_visible": true }
  ```
- **input_text** — type text into an input element. Validates visibility before typing.
  ```json
  { "type": "input_text", "selector": "#username", "text": "{{username}}", "clear": false }
  ```
- **press** — press a keyboard key (enter, tab, escape, etc.).
  ```json
  { "type": "press", "key": "enter" }
  ```
- **scroll** — scroll the page or a specific element.
  ```json
  { "type": "scroll", "selector": "#content", "delta_y": 500, "behavior": "auto" }
  ```

**Code Execution**

- **js_evaluate** — evaluate custom JavaScript code in the browser context. Must be wrapped in an IIFE format.
  ```json
  {
    "type": "js_evaluate",
    "js": "(function() { return document.title; })()",
    "timeout_seconds": 5.0,
    "session_storage_key": "pageTitle"
  }
  ```

**Data**

- **return** — return the value previously stored under a `session_storage_key`.
  ```json
  { "type": "return", "session_storage_key": "userData" }
  ```
- **return_html** — return HTML content from the page or a specific element.
  ```json
  { "type": "return_html", "scope": "page" }
  ```

**Example sequence:**

```json
[
  { "type": "navigate", "url": "https://example.com/login" },
  { "type": "sleep", "timeout_seconds": 1 },
  {
    "type": "fetch",
    "endpoint": {
      "method": "POST",
      "url": "/auth",
      "body": { "username": "{{user}}", "password": "{{pass}}" }
    },
    "session_storage_key": "token"
  },
  { "type": "return", "session_storage_key": "token" }
]
```

This defines a deterministic flow: open → wait → authenticate → return a session token.

#### Placeholder Interpolation

Placeholders inside operation fields are resolved at runtime:

- **Parameter placeholders**: `"{{paramName}}"` → substituted from routine parameters. `Parameter.type` drives coercion (standalone `"{{param}}"` → typed value; substring `"prefix {{param}}"` → string)
- **Storage placeholders** (read values from the current session):
  - `{{sessionStorage:myKey.path.to.value}}` — access nested values in sessionStorage
  - `{{localStorage:myKey}}` — access localStorage values
  - `{{cookie:CookieName}}` — read cookie values
  - `{{meta:name}}` — read meta tag content (e.g., `<meta name="csrf-token">`)
  - `{{windowProperty:path.to.value}}` — access window property values

**Important:** Currently, `sessionStorage`, `localStorage`, `cookie`, `meta`, and `windowProperty` placeholder resolution is supported only inside fetch `headers` and `body`.

Interpolation occurs before an operation executes. For example, a fetch endpoint might be:

```json
{
  "type": "fetch",
  "endpoint": {
    "method": "GET",
    "url": "https://api.example.com/search?paramName1={{paramName1}}&paramName2={{paramName1}}",
    "headers": {
      "Authorization": "Bearer {{cookie:auth_token}}"
    },
    "body": {}
  },
  "session_storage_key": "result_key"
}
```

This substitutes parameter values and injects `auth_token` from cookies. The JSON response is stored under `sessionStorage['result_key']` and can be returned by a final `return` operation using the matching `session_storage_key`.

</details>

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

## Running Benchmarks 📊

Benchmarks validate the routine discovery pipeline against known ground-truth routines. They run both deterministic tests (checking routine structure) and LLM-based tests (evaluating semantic correctness).

```bash
# Run all benchmarks
bluebox-benchmarks

# With verbose output (shows each test result as it runs)
bluebox-benchmarks -v

# Use a specific model
bluebox-benchmarks --model gpt-4.1

# Custom output directory
bluebox-benchmarks --output-dir ./my_benchmarks
```

Results are saved to the output directory:

- `{benchmark_name}.json` — Full evaluation results for each benchmark
- `_summary.json` — Aggregated summary of all benchmark runs
