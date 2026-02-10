"""
Generic HTTP adapter for Bluebox agents.

Works with any agent extending AbstractAgent, including all AbstractSpecialist
subclasses (auto-discovered at runtime via the specialist registry).

Uses inspect.signature to auto-wire each agent's constructor params to the
available data loaders, handling the _loader/_store naming split transparently.

Usage:
    bluebox-agent-adapter --list-agents
    bluebox-agent-adapter --cdp-captures-dir ./cdp_captures
    bluebox-agent-adapter --agent NetworkSpecialist --cdp-captures-dir ./cdp_captures
    bluebox-agent-adapter --agent BlueBoxAgent

Endpoints (all agents):
    GET  /health
    GET  /status
    POST /chat          {"message": "..."}

Agents with discovery support (specialists + SuperDiscoveryAgent):
    POST /discover      {"task": "..."}
    GET  /routine
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import logging
import pkgutil
import sys
import threading
import traceback
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from bluebox.agents.abstract_agent import AbstractAgent
from bluebox.agents.specialists.abstract_specialist import AbstractSpecialist
from bluebox.config import Config
from bluebox.data_models.llms.interaction import (
    BaseEmittedMessage,
    ChatResponseEmittedMessage,
    ChatRole,
    ErrorEmittedMessage,
    ToolInvocationResultEmittedMessage,
)
from bluebox.data_models.llms.vendors import OpenAIModel
from bluebox.llms.data_loaders.documentation_data_loader import DocumentationDataLoader
from bluebox.llms.data_loaders.interactions_data_loader import InteractionsDataLoader
from bluebox.llms.data_loaders.js_data_loader import JSDataLoader
from bluebox.llms.data_loaders.network_data_loader import NetworkDataLoader
from bluebox.llms.data_loaders.storage_data_loader import StorageDataLoader
from bluebox.llms.data_loaders.window_property_data_loader import WindowPropertyDataLoader
from bluebox.utils.logger import get_logger

BLUEBOX_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
logger = get_logger(__name__)

# Maps constructor param names → canonical data loader keys.
# Handles the _loader/_store naming split between agents and specialists.
_DATA_PARAM_TO_KEY: dict[str, str] = {
    "network_data_loader": "network",
    "network_data_store": "network",
    "storage_data_loader": "storage",
    "storage_data_store": "storage",
    "window_property_data_loader": "window_property",
    "window_property_data_store": "window_property",
    "js_data_loader": "js",
    "js_data_store": "js",
    "interaction_data_loader": "interaction",
    "interaction_data_loader": "interaction",
    "documentation_data_loader": "documentation",
}


# -- agent registry ----------------------------------------------------------------


def discover_agent_classes() -> dict[str, type]:
    """Build registry of all available AbstractAgent subclasses by class name."""
    from bluebox.agents.super_discovery_agent import SuperDiscoveryAgent
    from bluebox.agents.bluebox_agent import BlueBoxAgent

    # Import all specialist modules to trigger __init_subclass__ registration
    import bluebox.agents.specialists as specialists_pkg
    for _, module_name, _ in pkgutil.iter_modules(specialists_pkg.__path__):
        importlib.import_module(f"bluebox.agents.specialists.{module_name}")

    registry: dict[str, type] = {
        "SuperDiscoveryAgent": SuperDiscoveryAgent,
        "BlueBoxAgent": BlueBoxAgent,
    }
    for name in AbstractSpecialist.get_all_agent_types():
        cls = AbstractSpecialist.get_by_type(name)
        if cls is not None:
            registry[name] = cls
    return registry


def _accepts_param(cls: type, name: str) -> bool:
    """Check if cls.__init__ accepts a parameter with the given name."""
    return name in inspect.signature(cls.__init__).parameters


def _has_own_method(cls: type, name: str) -> bool:
    """Check if cls defines its own method (not just inherited)."""
    return name in cls.__dict__


# -- constructor auto-wiring -------------------------------------------------------


def wire_agent_kwargs(
    agent_class: type,
    data_loaders: dict[str, Any],
    common_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """
    Introspect agent_class.__init__ and build kwargs by matching params to
    available data loaders (by name→key mapping) and common kwargs.

    Only passes kwargs that the constructor actually accepts.
    Raises ValueError if a required data loader param has no matching data.
    """
    sig = inspect.signature(agent_class.__init__)
    kwargs: dict[str, Any] = {}
    missing: list[str] = []

    for name, param in sig.parameters.items():
        if name == "self":
            continue
        if name in common_kwargs:
            kwargs[name] = common_kwargs[name]
        elif name in _DATA_PARAM_TO_KEY:
            key = _DATA_PARAM_TO_KEY[name]
            if key in data_loaders:
                kwargs[name] = data_loaders[key]
            elif param.default is inspect.Parameter.empty:
                missing.append(f"{name} (needs '{key}' data)")
        # Params with defaults that we don't have are left to their defaults

    if missing:
        raise ValueError(
            f"{agent_class.__name__} requires: {', '.join(missing)}. "
            f"Available data: {sorted(data_loaders.keys())}. "
            f"Provide --cdp-captures-dir or explicit --*-jsonl flags."
        )
    return kwargs


def check_required_data(agent_class: type, data_loaders: dict[str, Any]) -> list[str]:
    """Return list of missing required data loaders for an agent class."""
    sig = inspect.signature(agent_class.__init__)
    missing = []
    for name, param in sig.parameters.items():
        if name in _DATA_PARAM_TO_KEY and param.default is inspect.Parameter.empty:
            key = _DATA_PARAM_TO_KEY[name]
            if key not in data_loaders:
                missing.append(f"'{key}' (constructor param: {name})")
    return missing


# -- agent state -------------------------------------------------------------------


class AgentState:
    """Manages agent lifecycle, message collection, and request dispatch."""

    def __init__(
        self,
        agent_class: type,
        data_loaders: dict[str, Any],
        llm_model: OpenAIModel = OpenAIModel.GPT_5_1,
        subagent_llm_model: OpenAIModel | None = None,
        max_iterations: int = 50,
        remote_debugging_address: str | None = None,
    ) -> None:
        self._agent_class = agent_class
        self._data_loaders = data_loaders
        self._llm_model = llm_model
        self._subagent_llm_model = subagent_llm_model
        self._max_iterations = max_iterations
        self._remote_debugging_address = remote_debugging_address

        self._chat_agent: Any = None
        self._discovered_routine: Any = None
        self._discovery_result: Any = None

        self._lock = threading.Lock()
        self._collected: list[dict[str, Any]] = []
        self._streamed_text: list[str] = []

    # -- callbacks -----------------------------------------------------------------

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
        else:
            self._collected.append({"type": message.__class__.__name__})

    def _on_stream_chunk(self, chunk: str) -> None:
        self._streamed_text.append(chunk)

    def _flush(self) -> list[dict[str, Any]]:
        msgs = list(self._collected)
        self._collected.clear()
        self._streamed_text.clear()
        return msgs

    # -- agent construction --------------------------------------------------------

    def _common_kwargs(self, **extra: Any) -> dict[str, Any]:
        """Build the non-data-loader kwargs for agent construction."""
        kw: dict[str, Any] = {
            "emit_message_callable": self._on_message,
            "stream_chunk_callable": self._on_stream_chunk,
            "llm_model": self._llm_model,
        }
        if self._subagent_llm_model is not None:
            kw["subagent_llm_model"] = self._subagent_llm_model
        if self._max_iterations is not None:
            kw["max_iterations"] = self._max_iterations
        if self._remote_debugging_address is not None:
            kw["remote_debugging_address"] = self._remote_debugging_address
        kw.update(extra)
        return kw

    def _make_agent(self, **extra: Any) -> Any:
        """Construct the agent, auto-wiring data loaders to constructor params."""
        common = self._common_kwargs(**extra)
        kwargs = wire_agent_kwargs(self._agent_class, self._data_loaders, common)
        return self._agent_class(**kwargs)

    # -- public operations ---------------------------------------------------------

    def chat(self, message: str) -> dict[str, Any]:
        """Send a chat message. Works with all AbstractAgent subclasses."""
        with self._lock:
            if not self._chat_agent:
                extra: dict[str, Any] = {}
                # SuperDiscoveryAgent requires a task constructor param
                if _accepts_param(self._agent_class, "task"):
                    extra["task"] = "Help the user understand their data and answer questions."
                self._chat_agent = self._make_agent(**extra)
            self._flush()
            self._chat_agent.process_new_message(message, ChatRole.USER)
            return {"ok": True, "messages": self._flush()}

    def discover(self, task: str) -> dict[str, Any]:
        """Run discovery. Specialists use run_autonomous(), others use run()."""
        with self._lock:
            self._flush()

            # Specialists: run_autonomous(task)
            if issubclass(self._agent_class, AbstractSpecialist):
                agent = self._make_agent()
                result = agent.run_autonomous(task)
                messages = self._flush()
                if result is not None:
                    self._discovery_result = result
                    result_data = result.model_dump() if isinstance(result, BaseModel) else result
                    return {"ok": True, "result": result_data, "messages": messages}
                return {"ok": False, "error": "Autonomous run finished without result", "messages": messages}

            # Non-specialist agents with their own run() (e.g. SuperDiscoveryAgent)
            if _has_own_method(self._agent_class, "run"):
                extra: dict[str, Any] = {}
                if _accepts_param(self._agent_class, "task"):
                    extra["task"] = task
                agent = self._make_agent(**extra)
                result = agent.run()
                messages = self._flush()
                if result is not None:
                    self._discovered_routine = result
                    response: dict[str, Any] = {
                        "ok": True,
                        "routine": result.model_dump() if isinstance(result, BaseModel) else result,
                        "messages": messages,
                    }
                    # Include test_parameters if the agent exposes them
                    ds = getattr(agent, "_discovery_state", None)
                    if ds and hasattr(ds, "test_parameters"):
                        response["test_parameters"] = ds.test_parameters
                    return response
                return {"ok": False, "messages": messages}

            return {"ok": False, "error": f"{self._agent_class.__name__} does not support /discover"}

    def status(self) -> dict[str, Any]:
        """Return agent-type-agnostic status."""
        info: dict[str, Any] = {
            "agent_type": self._agent_class.__name__,
            "chat_active": self._chat_agent is not None,
            "supports_discover": self.supports_discover,
        }
        if self._discovered_routine is not None:
            info["routine_discovered"] = True
            for attr in ("name", "operations", "parameters"):
                val = getattr(self._discovered_routine, attr, None)
                if val is not None:
                    if attr == "operations":
                        info[attr] = len(val)
                    elif attr == "parameters":
                        info[attr] = [p.name for p in val]
                    else:
                        info[attr] = val
        if self._discovery_result is not None:
            info["discovery_completed"] = True
        return info

    def get_routine(self) -> dict[str, Any]:
        """Return the discovered routine as JSON."""
        if self._discovered_routine is not None:
            return {"ok": True, "routine": self._discovered_routine.model_dump()}
        return {"ok": False, "error": "No routine discovered"}

    @property
    def supports_discover(self) -> bool:
        return (
            issubclass(self._agent_class, AbstractSpecialist)
            or _has_own_method(self._agent_class, "run")
        )


# -- utilities ---------------------------------------------------------------------


def _truncate(value: Any, max_len: int) -> Any:
    if isinstance(value, str) and len(value) > max_len:
        return value[:max_len] + f"... ({len(value)} chars total)"
    return value


# -- HTTP handler ------------------------------------------------------------------


def make_handler(state: AgentState) -> type[BaseHTTPRequestHandler]:
    """Create a request handler class bound to the given AgentState."""

    class Handler(BaseHTTPRequestHandler):
        def _json(self, data: dict[str, Any], status: int = 200) -> None:
            body = json.dumps(data, indent=2, default=str).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", 0))
            return json.loads(self.rfile.read(length)) if length else {}

        def do_GET(self) -> None:
            if self.path == "/health":
                self._json({"ok": True})
            elif self.path == "/status":
                self._json(state.status())
            elif self.path == "/routine":
                self._json(state.get_routine())
            else:
                self._json({"error": f"Unknown: {self.path}"}, 404)

        def do_POST(self) -> None:
            try:
                body = self._body()
                if self.path == "/chat":
                    msg = body.get("message", "")
                    if not msg:
                        self._json({"error": "Missing 'message'"}, 400)
                        return
                    self._json(state.chat(msg))
                elif self.path == "/discover":
                    task = body.get("task", "")
                    if not task:
                        self._json({"error": "Missing 'task'"}, 400)
                        return
                    self._json(state.discover(task))
                else:
                    self._json({"error": f"Unknown: {self.path}"}, 404)
            except Exception as e:
                self._json({"error": str(e), "traceback": traceback.format_exc()}, 500)

        def log_message(self, format: str, *args: Any) -> None:
            logger.debug(format, *args)

    return Handler


# -- data loading ------------------------------------------------------------------


def load_data(args: argparse.Namespace) -> dict[str, Any]:
    """Load all available CDP data loaders keyed by canonical names."""
    paths: dict[str, str | None] = {
        "network": args.network_jsonl,
        "storage": args.storage_jsonl,
        "window_property": args.window_props_jsonl,
        "js": args.js_jsonl,
        "interaction": args.interaction_jsonl,
    }

    if args.cdp_captures_dir:
        cdp_dir = Path(args.cdp_captures_dir)
        candidates = {
            "network": cdp_dir / "network" / "events.jsonl",
            "storage": cdp_dir / "storage" / "events.jsonl",
            "window_property": cdp_dir / "window_properties" / "events.jsonl",
            "js": cdp_dir / "network" / "javascript_events.jsonl",
            "interaction": cdp_dir / "interaction" / "events.jsonl",
        }
        for key, path in candidates.items():
            if not paths[key] and path.exists():
                paths[key] = str(path)

    loaders: dict[str, Any] = {}
    factories: dict[str, Any] = {
        "network": lambda p: NetworkDataLoader(p),
        "storage": lambda p: StorageDataLoader(p),
        "window_property": lambda p: WindowPropertyDataLoader(p),
        "js": lambda p: JSDataLoader(p),
        "interaction": lambda p: InteractionsDataLoader.from_jsonl(p),
    }
    for key, factory in factories.items():
        p = paths.get(key)
        if p and Path(p).exists():
            loaders[key] = factory(p)

    # Documentation is bundled with the package
    docs_dir = BLUEBOX_PACKAGE_ROOT / "agent_docs"
    if docs_dir.exists():
        code_paths = [
            str(BLUEBOX_PACKAGE_ROOT / "data_models" / "routine"),
            str(BLUEBOX_PACKAGE_ROOT / "data_models" / "ui_elements.py"),
            str(BLUEBOX_PACKAGE_ROOT / "agents" / "routine_discovery_agent.py"),
            str(BLUEBOX_PACKAGE_ROOT / "llms" / "infra" / "data_store.py"),
            str(BLUEBOX_PACKAGE_ROOT / "utils" / "js_utils.py"),
            str(BLUEBOX_PACKAGE_ROOT / "utils" / "data_utils.py"),
            "!" + str(BLUEBOX_PACKAGE_ROOT / "**" / "__init__.py"),
        ]
        loaders["documentation"] = DocumentationDataLoader(
            documentation_paths=[str(docs_dir)], code_paths=code_paths,
        )

    return loaders


def parse_model(model_str: str) -> OpenAIModel:
    """Parse a model string into an OpenAIModel enum value."""
    for model in OpenAIModel:
        if model.value == model_str or model.name == model_str:
            return model
    raise ValueError(f"Unknown model: {model_str}")


# -- main --------------------------------------------------------------------------


def main() -> None:
    """Entry point for the generic agent HTTP adapter."""
    registry = discover_agent_classes()

    parser = argparse.ArgumentParser(description="HTTP adapter for Bluebox agents")
    parser.add_argument("--agent", default="SuperDiscoveryAgent",
                        help="Agent class name (default: SuperDiscoveryAgent)")
    parser.add_argument("--list-agents", action="store_true",
                        help="List available agents and exit")
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

    if args.list_agents:
        print("Available agents:")
        for name, cls in sorted(registry.items()):
            required = []
            sig = inspect.signature(cls.__init__)
            for pname, param in sig.parameters.items():
                if pname in _DATA_PARAM_TO_KEY and param.default is inspect.Parameter.empty:
                    required.append(_DATA_PARAM_TO_KEY[pname])
            specialist = " (specialist)" if issubclass(cls, AbstractSpecialist) else ""
            req = f"  requires: {', '.join(required)}" if required else ""
            print(f"  {name}{specialist}{req}")
        return

    if args.agent not in registry:
        print(f"Unknown agent: {args.agent}", file=sys.stderr)
        print(f"Available: {', '.join(sorted(registry))}", file=sys.stderr)
        sys.exit(1)

    if args.quiet:
        logging.getLogger("bluebox").setLevel(logging.CRITICAL + 1)

    if Config.OPENAI_API_KEY is None:
        print("Error: OPENAI_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    agent_class = registry[args.agent]
    print(f"Agent: {args.agent}", flush=True)
    print("Loading data...", flush=True)
    loaders = load_data(args)
    print(f"Data loaded: {sorted(loaders.keys())}", flush=True)

    # Validate required data loaders before starting the server
    missing = check_required_data(agent_class, loaders)
    if missing:
        print(f"Error: {agent_class.__name__} requires: {', '.join(missing)}", file=sys.stderr)
        print("Provide --cdp-captures-dir or explicit --*-jsonl flags.", file=sys.stderr)
        sys.exit(1)

    state = AgentState(
        agent_class=agent_class,
        data_loaders=loaders,
        llm_model=parse_model(args.llm_model),
        subagent_llm_model=parse_model(args.subagent_llm_model) if args.subagent_llm_model else None,
        max_iterations=args.max_iterations,
        remote_debugging_address=args.remote_debugging_address,
    )

    server = HTTPServer(("127.0.0.1", args.port), make_handler(state))
    print(f"Ready on http://127.0.0.1:{args.port}", flush=True)
    print("  GET  /health", flush=True)
    print("  GET  /status", flush=True)
    print("  POST /chat     {\"message\": \"...\"}", flush=True)
    if state.supports_discover:
        print("  POST /discover {\"task\": \"...\"}", flush=True)
        print("  GET  /routine", flush=True)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
