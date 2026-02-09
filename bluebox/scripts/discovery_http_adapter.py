"""
bluebox/scripts/discovery_http_adapter.py

HTTP adapter for the SuperDiscoveryAgent.

Exposes the agent as a simple JSON API so external tools (like Claude Code)
can interact with it programmatically via curl.

Usage:
    # Start the server (loads CDP data, then listens on port 8765)
    python -m bluebox.scripts.discovery_http_adapter --cdp-captures-dir ./cdp_captures

    # Interact via curl:
    curl localhost:8765/status
    curl -X POST localhost:8765/discover -d '{"task": "Find train search endpoint"}'
    curl -X POST localhost:8765/chat -d '{"message": "What endpoints did you find?"}'
    curl localhost:8765/routine
"""

import argparse
import json
import logging
import sys
import threading
import traceback
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Any

from bluebox.agents.super_discovery_agent import SuperDiscoveryAgent
from bluebox.config import Config
from bluebox.data_models.llms.interaction import (
    BaseEmittedMessage,
    ChatResponseEmittedMessage,
    ChatRole,
    ErrorEmittedMessage,
    ToolInvocationResultEmittedMessage,
)
from bluebox.data_models.llms.vendors import OpenAIModel
from bluebox.data_models.routine.routine import Routine
from bluebox.llms.data_loaders.documentation_data_loader import DocumentationDataLoader
from bluebox.llms.data_loaders.interactions_data_loader import InteractionsDataLoader
from bluebox.llms.data_loaders.js_data_loader import JSDataLoader
from bluebox.llms.data_loaders.network_data_loader import NetworkDataLoader
from bluebox.llms.data_loaders.storage_data_loader import StorageDataLoader
from bluebox.llms.data_loaders.window_property_data_loader import WindowPropertyDataLoader
from bluebox.utils.logger import get_logger

BLUEBOX_PACKAGE_ROOT = Path(__file__).resolve().parent.parent

logger = get_logger(__name__)


class AgentState:
    """Holds the long-lived agent and accumulated responses."""

    def __init__(
        self,
        network_data_loader: NetworkDataLoader,
        storage_data_loader: StorageDataLoader | None = None,
        window_property_data_loader: WindowPropertyDataLoader | None = None,
        js_data_loader: JSDataLoader | None = None,
        interaction_data_loader: InteractionsDataLoader | None = None,
        documentation_data_loader: DocumentationDataLoader | None = None,
        llm_model: OpenAIModel = OpenAIModel.GPT_5_1,
        subagent_llm_model: OpenAIModel | None = None,
        max_iterations: int = 50,
        remote_debugging_address: str | None = None,
    ) -> None:
        self._network_data_loader = network_data_loader
        self._storage_data_loader = storage_data_loader
        self._window_property_data_loader = window_property_data_loader
        self._js_data_loader = js_data_loader
        self._interaction_data_loader = interaction_data_loader
        self._documentation_data_loader = documentation_data_loader
        self._llm_model = llm_model
        self._subagent_llm_model = subagent_llm_model
        self._max_iterations = max_iterations
        self._remote_debugging_address = remote_debugging_address

        self.discovery_agent: SuperDiscoveryAgent | None = None
        self.chat_agent: SuperDiscoveryAgent | None = None
        self.discovered_routine: Routine | None = None

        # Accumulator for messages emitted during a single request
        self._lock = threading.Lock()
        self._collected: list[dict[str, Any]] = []
        self._streamed_text: list[str] = []

    # -- callbacks given to the agent ------------------------------------------

    def _on_message(self, message: BaseEmittedMessage) -> None:
        if isinstance(message, ChatResponseEmittedMessage):
            self._collected.append({"type": "assistant", "content": message.content})
        elif isinstance(message, ErrorEmittedMessage):
            self._collected.append({"type": "error", "content": message.error})
        elif isinstance(message, ToolInvocationResultEmittedMessage):
            self._collected.append({
                "type": "tool",
                "tool": message.tool_invocation.tool_name,
                "status": message.tool_invocation.status.value,
                "args": message.tool_invocation.tool_arguments,
                "result": _truncate(message.tool_result, 2000),
            })

    def _on_stream_chunk(self, chunk: str) -> None:
        self._streamed_text.append(chunk)

    # -- helpers ---------------------------------------------------------------

    def _flush(self) -> list[dict[str, Any]]:
        """Return collected messages and reset accumulators."""
        msgs = list(self._collected)
        self._collected.clear()
        self._streamed_text.clear()
        return msgs

    def _make_agent(self, task: str) -> SuperDiscoveryAgent:
        return SuperDiscoveryAgent(
            emit_message_callable=self._on_message,
            stream_chunk_callable=self._on_stream_chunk,
            network_data_loader=self._network_data_loader,
            task=task,
            storage_data_loader=self._storage_data_loader,
            window_property_data_loader=self._window_property_data_loader,
            js_data_loader=self._js_data_loader,
            interaction_data_loader=self._interaction_data_loader,
            documentation_data_loader=self._documentation_data_loader,
            llm_model=self._llm_model,
            subagent_llm_model=self._subagent_llm_model,
            max_iterations=self._max_iterations,
            remote_debugging_address=self._remote_debugging_address,
        )

    # -- public operations -----------------------------------------------------

    def discover(self, task: str) -> dict[str, Any]:
        """Run full discovery for the given task. Blocking."""
        with self._lock:
            self._flush()
            self.discovery_agent = self._make_agent(task)
            routine = self.discovery_agent.run()
            messages = self._flush()

            if routine:
                self.discovered_routine = routine
                return {
                    "ok": True,
                    "routine": routine.model_dump(),
                    "test_parameters": self.discovery_agent._discovery_state.test_parameters,
                    "messages": messages,
                }
            return {"ok": False, "messages": messages}

    def chat(self, message: str) -> dict[str, Any]:
        """Send a chat message to the agent."""
        with self._lock:
            if not self.chat_agent:
                self.chat_agent = self._make_agent(
                    "Help the user understand their CDP captures and answer questions."
                )
            self._flush()
            self.chat_agent.process_new_message(message, ChatRole.USER)
            messages = self._flush()
            return {"ok": True, "messages": messages}

    def status(self) -> dict[str, Any]:
        """Return current state summary."""
        result: dict[str, Any] = {"routine_discovered": self.discovered_routine is not None}
        if self.discovered_routine:
            r = self.discovered_routine
            result["routine_name"] = r.name
            result["operations"] = len(r.operations)
            result["parameters"] = [p.name for p in r.parameters]
        if self.discovery_agent:
            ds = self.discovery_agent._discovery_state
            result["phase"] = ds.phase.value
            result["processed_transactions"] = len(ds.processed_transactions)
            result["queued_transactions"] = len(ds.transaction_queue)
        return result

    def get_routine(self) -> dict[str, Any]:
        """Return the discovered routine as JSON."""
        if not self.discovered_routine:
            return {"ok": False, "error": "No routine discovered yet"}
        return {"ok": True, "routine": self.discovered_routine.model_dump()}


def _truncate(value: Any, max_len: int) -> Any:
    """Truncate large values for JSON responses."""
    if isinstance(value, str) and len(value) > max_len:
        return value[:max_len] + f"... ({len(value)} chars total)"
    return value


# -- HTTP handler --------------------------------------------------------------


def make_handler(state: AgentState) -> type[BaseHTTPRequestHandler]:
    """Create a request handler class bound to the given AgentState."""

    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, data: dict[str, Any], status: int = 200) -> None:
            body = json.dumps(data, indent=2, default=str).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", 0))
            if length == 0:
                return {}
            raw = self.rfile.read(length)
            return json.loads(raw)

        def do_GET(self) -> None:
            if self.path == "/status":
                self._send_json(state.status())
            elif self.path == "/routine":
                self._send_json(state.get_routine())
            elif self.path == "/health":
                self._send_json({"ok": True})
            else:
                self._send_json({"error": f"Unknown GET endpoint: {self.path}"}, 404)

        def do_POST(self) -> None:
            try:
                body = self._read_body()

                if self.path == "/discover":
                    task = body.get("task", "")
                    if not task:
                        self._send_json({"error": "Missing 'task' field"}, 400)
                        return
                    result = state.discover(task)
                    self._send_json(result)

                elif self.path == "/chat":
                    message = body.get("message", "")
                    if not message:
                        self._send_json({"error": "Missing 'message' field"}, 400)
                        return
                    result = state.chat(message)
                    self._send_json(result)

                else:
                    self._send_json({"error": f"Unknown POST endpoint: {self.path}"}, 404)

            except Exception as e:
                self._send_json({"error": str(e), "traceback": traceback.format_exc()}, 500)

        def log_message(self, format: str, *args: Any) -> None:
            # Quiet down the default stderr logging
            logger.debug(format, *args)

    return Handler


# -- data loading (reused from run_super_discovery_agent.py) -------------------


def load_data(args: argparse.Namespace) -> dict[str, Any]:
    """Load all CDP data loaders from CLI args. Returns dict of loader kwargs."""
    network_jsonl = args.network_jsonl
    storage_jsonl = args.storage_jsonl
    window_props_jsonl = args.window_props_jsonl
    js_jsonl = args.js_jsonl
    interaction_jsonl = args.interaction_jsonl

    if args.cdp_captures_dir:
        cdp_dir = Path(args.cdp_captures_dir)
        mapping = [
            ("network_jsonl", cdp_dir / "network" / "events.jsonl"),
            ("storage_jsonl", cdp_dir / "storage" / "events.jsonl"),
            ("window_props_jsonl", cdp_dir / "window_properties" / "events.jsonl"),
            ("js_jsonl", cdp_dir / "network" / "javascript_events.jsonl"),
            ("interaction_jsonl", cdp_dir / "interaction" / "events.jsonl"),
        ]
        local_vars = {
            "network_jsonl": network_jsonl,
            "storage_jsonl": storage_jsonl,
            "window_props_jsonl": window_props_jsonl,
            "js_jsonl": js_jsonl,
            "interaction_jsonl": interaction_jsonl,
        }
        for name, candidate in mapping:
            if not local_vars[name] and candidate.exists():
                local_vars[name] = str(candidate)
        network_jsonl = local_vars["network_jsonl"]
        storage_jsonl = local_vars["storage_jsonl"]
        window_props_jsonl = local_vars["window_props_jsonl"]
        js_jsonl = local_vars["js_jsonl"]
        interaction_jsonl = local_vars["interaction_jsonl"]

    if not network_jsonl:
        print("Error: No network data. Use --network-jsonl or --cdp-captures-dir", file=sys.stderr)
        sys.exit(1)

    loaders: dict[str, Any] = {}
    loaders["network_data_loader"] = NetworkDataLoader(network_jsonl)

    if storage_jsonl and Path(storage_jsonl).exists():
        loaders["storage_data_loader"] = StorageDataLoader(storage_jsonl)
    if window_props_jsonl and Path(window_props_jsonl).exists():
        loaders["window_property_data_loader"] = WindowPropertyDataLoader(window_props_jsonl)
    if js_jsonl and Path(js_jsonl).exists():
        loaders["js_data_loader"] = JSDataLoader(js_jsonl)
    if interaction_jsonl and Path(interaction_jsonl).exists():
        loaders["interaction_data_loader"] = InteractionsDataLoader.from_jsonl(interaction_jsonl)

    docs_dir = str(BLUEBOX_PACKAGE_ROOT / "agent_docs")
    code_paths = [
        str(BLUEBOX_PACKAGE_ROOT / "data_models" / "routine"),
        str(BLUEBOX_PACKAGE_ROOT / "data_models" / "ui_elements.py"),
        str(BLUEBOX_PACKAGE_ROOT / "agents" / "routine_discovery_agent.py"),
        str(BLUEBOX_PACKAGE_ROOT / "llms" / "infra" / "data_store.py"),
        str(BLUEBOX_PACKAGE_ROOT / "utils" / "js_utils.py"),
        str(BLUEBOX_PACKAGE_ROOT / "utils" / "data_utils.py"),
        "!" + str(BLUEBOX_PACKAGE_ROOT / "**" / "__init__.py"),
    ]
    loaders["documentation_data_loader"] = DocumentationDataLoader(
        documentation_paths=[docs_dir],
        code_paths=code_paths,
    )

    return loaders


def parse_model(model_str: str) -> OpenAIModel:
    """Parse a model string into an OpenAIModel enum value."""
    for model in OpenAIModel:
        if model.value == model_str or model.name == model_str:
            return model
    raise ValueError(f"Unknown model: {model_str}")


def main() -> None:
    """Entry point for the HTTP discovery adapter."""
    parser = argparse.ArgumentParser(description="HTTP adapter for SuperDiscoveryAgent")
    parser.add_argument("--cdp-captures-dir", type=str, default=None)
    parser.add_argument("--network-jsonl", type=str, default=None)
    parser.add_argument("--storage-jsonl", type=str, default=None)
    parser.add_argument("--window-props-jsonl", type=str, default=None)
    parser.add_argument("--js-jsonl", type=str, default=None)
    parser.add_argument("--interaction-jsonl", type=str, default=None)
    parser.add_argument("--llm-model", type=str, default=OpenAIModel.GPT_5_1.value)
    parser.add_argument("--subagent-llm-model", type=str, default=None)
    parser.add_argument("--max-iterations", type=int, default=50)
    parser.add_argument("--remote-debugging-address", type=str, default=None)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("-q", "--quiet", action="store_true")
    args = parser.parse_args()

    if args.quiet:
        logging.getLogger("bluebox").setLevel(logging.CRITICAL + 1)

    if Config.OPENAI_API_KEY is None:
        print("Error: OPENAI_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    print("Loading CDP data...", flush=True)
    loaders = load_data(args)

    llm_model = parse_model(args.llm_model)
    subagent_model = parse_model(args.subagent_llm_model) if args.subagent_llm_model else None

    state = AgentState(
        llm_model=llm_model,
        subagent_llm_model=subagent_model,
        max_iterations=args.max_iterations,
        remote_debugging_address=args.remote_debugging_address,
        **loaders,
    )

    server = HTTPServer(("127.0.0.1", args.port), make_handler(state))
    print(f"Ready on http://127.0.0.1:{args.port}", flush=True)
    print(f"  GET  /status   - agent state", flush=True)
    print(f"  GET  /routine  - discovered routine", flush=True)
    print(f"  POST /discover - start discovery (body: {{\"task\": \"...\"}})", flush=True)
    print(f"  POST /chat     - chat with agent (body: {{\"message\": \"...\"}})", flush=True)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
