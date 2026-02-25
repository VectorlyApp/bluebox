"""
bluebox/scripts/run_storage_exploration.py

Prototype script for Phase 1 (Exploration) — Storage & Window Properties domain.

Runs the ValueTraceResolverSpecialist in autonomous mode with an output schema
matching StorageExplorationSummary. The agent uses its existing tools
(search_everywhere, get_storage_entry, get_storage_by_key, get_window_prop_changes,
execute_python, etc.) to explore captured storage/window data and produce a
structured exploration summary focused on tokens and data blocks.

Usage:
    python -m bluebox.scripts.run_storage_exploration --cdp-captures-dir ./cdp_captures
    python -m bluebox.scripts.run_storage_exploration --cdp-captures-dir ./cdp_captures --model gpt-5.1
    python -m bluebox.scripts.run_storage_exploration --cdp-captures-dir ./cdp_captures --output /tmp/storage_exploration.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from bluebox.agents.abstract_agent import AgentExecutionMode, AutonomousRunConfig
from bluebox.agents.specialists.value_trace_resolver_specialist import ValueTraceResolverSpecialist
from bluebox.agents.workspace import LocalWorkspace
from bluebox.data_models.api_indexing.exploration import StorageExplorationSummary
from bluebox.data_models.llms.interaction import EmittedMessage
from bluebox.data_models.llms.vendors import LLMModel, OpenAIModel
from bluebox.llms.data_loaders.network_data_loader import NetworkDataLoader
from bluebox.llms.data_loaders.storage_data_loader import StorageDataLoader
from bluebox.llms.data_loaders.window_property_data_loader import WindowPropertyDataLoader
from bluebox.utils.logger import get_logger

logger = get_logger(name=__name__)

# ---------------------------------------------------------------------------
# Output schema — derived from StorageExplorationSummary
# ---------------------------------------------------------------------------

STORAGE_EXPLORATION_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "total_events": {
            "type": "integer",
            "description": "Total number of storage + window property events in the capture.",
        },
        "noise_filtered": {
            "type": "integer",
            "description": "Events discarded as noise (tracking cookies, analytics IDs, consent flags, etc.).",
        },
        "tokens": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Each entry is a freeform description of one discovered token. "
                "Include: where it lives (cookie/localStorage/sessionStorage/window property), "
                "the key name, what kind of token it looks like (JWT, session ID, "
                "CSRF, API key), rough size, and whether it changed during the session."
            ),
        },
        "data_blocks": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Each entry is a freeform description of one meaningful data block. "
                "Include: where it lives, the key name, what it contains, rough size."
            ),
        },
        "narrative": {
            "type": "string",
            "description": (
                "Freeform observations: auth patterns, storage lifecycle, "
                "oddities, cross-domain connections, anything else worth noting."
            ),
        },
    },
    "required": [
        "total_events",
        "noise_filtered",
        "tokens",
        "data_blocks",
        "narrative",
    ],
}

# ---------------------------------------------------------------------------
# Autonomous system prompt override — exploration-specific
# ---------------------------------------------------------------------------

EXPLORATION_SYSTEM_PROMPT = """
You are a storage & window property exploration agent. Your job is to survey ALL
browser storage (cookies, localStorage, sessionStorage, IndexedDB) and window
object properties to find tokens and data blocks.

## Your Mission

Scan everything in storage and window properties. Find two things:

1. **Tokens** — auth tokens (JWTs, session IDs), CSRF tokens, API keys, refresh
   tokens, OAuth codes. Anything an API endpoint might need to function.
2. **Data blocks** — large or structured values: cached API responses, user
   profiles, config objects, feature flags, search results.

Everything else (tracking cookies, analytics IDs, consent flags, ad pixels,
UI preferences) is noise — count it but don't catalog it.

## Process

1. **Survey**: Use `execute_python` to dump all storage keys and window property
   paths. Get the lay of the land — how many events, what origins, what keys exist.
2. **Identify tokens**: Look for values that look like auth material:
   - JWTs (three dot-separated base64 segments)
   - Session IDs (long opaque strings, often hex or base64)
   - CSRF/XSRF tokens (often in cookies or headers)
   - API keys (prefixed strings like `sk_live_...`, `pk_...`, `dk_...`)
   - OAuth tokens (access_token, refresh_token keys)
   Use `get_storage_by_key` or `search_in_storage` to inspect promising keys.
3. **Identify data blocks**: Look for large JSON values, objects with multiple
   keys, arrays of data. Use `get_storage_entry` to inspect them.
4. **Check window properties**: Use `search_in_window_props` or
   `get_window_prop_changes` to find tokens or config stashed on the window object.
5. **Count noise**: Tally up tracking cookies, analytics, consent, etc.
6. **Finalize**: Call `finalize_with_output(output={...})` with the COMPLETE JSON.

## CRITICAL: How to Finalize

When you call `finalize_with_output`, you MUST pass the ENTIRE output as a single
JSON object in the `output` parameter. Do NOT call it empty or with partial data.

```
finalize_with_output(output={
  "total_events": 45,
  "noise_filtered": 30,
  "tokens": [
    "sessionStorage['auth_token'] — JWT (~1.2kb), written once on page load, three dot-separated base64 segments",
    "cookie['XSRF-TOKEN'] — CSRF token (~64 chars hex), set by server on every response"
  ],
  "data_blocks": [
    "localStorage['user_profile'] — JSON object (~2kb) with keys: name, email, prefs, subscription"
  ],
  "narrative": "Auth uses JWT in sessionStorage + CSRF cookie. Token set on initial page load, never refreshed during session."
})
```

## Guidelines

- Be thorough — check ALL storage keys and window property paths, not just the first few.
- For tokens: describe what it IS, not just where it is. "Looks like a JWT" is useful.
  "Key exists" is not.
- For data blocks: describe the SHAPE (top-level keys, rough size), not the raw content.
- The narrative is your chance to tell the story: how does auth work? What gets
  cached? What's the lifecycle? Any surprises?
- Don't catalog noise items individually — just count them.
""".strip()


def _emit_message(msg: EmittedMessage) -> None:
    """Print emitted messages to stderr for visibility."""
    if hasattr(msg, "content") and msg.content:
        print(f"[agent] {msg.content}", file=sys.stderr)
    elif hasattr(msg, "error") and msg.error:
        print(f"[error] {msg.error}", file=sys.stderr)


def _resolve_model(model_str: str) -> LLMModel:
    """Resolve a model string to an LLMModel enum value."""
    for member in OpenAIModel:
        if member.value == model_str:
            return member
    raise ValueError(
        f"Unknown model: {model_str}. "
        f"Available: {[m.value for m in OpenAIModel]}"
    )


def run_storage_exploration(
    cdp_captures_dir: Path,
    llm_model: LLMModel = OpenAIModel.GPT_5_1,
    min_iterations: int = 3,
    max_iterations: int = 15,
    workspace_dir: Path | None = None,
) -> StorageExplorationSummary | None:
    """
    Run storage exploration on a CDP captures directory.

    Args:
        cdp_captures_dir: Path to directory containing storage/events.jsonl
            and/or window_properties/events.jsonl.
        llm_model: LLM model to use.
        min_iterations: Minimum iterations before finalize is available.
        max_iterations: Maximum iterations before the loop exits.
        workspace_dir: Workspace directory for artifacts and mounted inputs.

    Returns:
        StorageExplorationSummary if successful, None if the agent failed or timed out.
    """
    # Try to load each data source — at least one of storage or window_properties must exist
    storage_loader: StorageDataLoader | None = None
    window_loader: WindowPropertyDataLoader | None = None
    network_loader: NetworkDataLoader | None = None

    storage_jsonl = cdp_captures_dir / "storage" / "events.jsonl"
    if storage_jsonl.exists():
        storage_loader = StorageDataLoader(jsonl_path=str(storage_jsonl))
        logger.info(
            "Loaded %d storage events (%d cookie, %d localStorage, %d sessionStorage)",
            storage_loader.stats.total_events,
            storage_loader.stats.cookie_events,
            storage_loader.stats.local_storage_events,
            storage_loader.stats.session_storage_events,
        )

    window_jsonl = cdp_captures_dir / "window_properties" / "events.jsonl"
    if window_jsonl.exists():
        window_loader = WindowPropertyDataLoader(jsonl_path=str(window_jsonl))
        logger.info(
            "Loaded %d window property events (%d unique paths)",
            window_loader.stats.total_events,
            window_loader.stats.unique_property_paths,
        )

    # Network is optional — if present, the agent can cross-reference tokens with requests
    network_jsonl = cdp_captures_dir / "network" / "events.jsonl"
    if network_jsonl.exists():
        network_loader = NetworkDataLoader(jsonl_path=str(network_jsonl))
        logger.info(
            "Loaded %d network entries (optional, for cross-referencing)",
            network_loader.stats.total_requests,
        )

    if storage_loader is None and window_loader is None:
        logger.error(
            "Neither storage/events.jsonl nor window_properties/events.jsonl found in %s",
            cdp_captures_dir,
        )
        return None

    workspace = LocalWorkspace.from_directory_path(
        workspace_dir or Path("./agent_workspace/storage_exploration"),
    )
    if storage_jsonl.exists():
        workspace.attach_input_file("storage_events", storage_jsonl)
    if window_jsonl.exists():
        workspace.attach_input_file("window_property_events", window_jsonl)
    if network_jsonl.exists():
        workspace.attach_input_file("network_events", network_jsonl)

    # Build the specialist
    specialist = ValueTraceResolverSpecialist(
        emit_message_callable=_emit_message,
        storage_data_loader=storage_loader,
        window_property_data_loader=window_loader,
        network_data_loader=network_loader,
        llm_model=llm_model,
        execution_mode=AgentExecutionMode.AUTONOMOUS,
        workspace=workspace,
    )

    # Bump max_output_tokens for the finalize call
    specialist.llm_client._client.DEFAULT_MAX_TOKENS = 16_384

    # Monkey-patch the autonomous system prompt for exploration
    def _exploration_system_prompt() -> str:
        context_parts: list[str] = [EXPLORATION_SYSTEM_PROMPT, "\n\n## Data Context"]

        if storage_loader:
            stats = storage_loader.stats
            context_parts.append(
                f"\n- Storage: {stats.total_events} events "
                f"(cookies: {stats.cookie_events}, localStorage: {stats.local_storage_events}, "
                f"sessionStorage: {stats.session_storage_events}, "
                f"IndexedDB: {stats.indexed_db_events}), "
                f"{stats.unique_origins} origins, {stats.unique_keys} unique keys"
            )
        else:
            context_parts.append("\n- Storage: Not available")

        if window_loader:
            stats = window_loader.stats
            context_parts.append(
                f"\n- Window Props: {stats.total_events} events, "
                f"{stats.unique_property_paths} unique paths"
            )
        else:
            context_parts.append("\n- Window Props: Not available")

        if network_loader:
            stats = network_loader.stats
            context_parts.append(
                f"\n- Network (optional): {stats.total_requests} requests, "
                f"{stats.unique_urls} unique URLs (available for cross-referencing)"
            )

        return (
            "".join(context_parts)
            + specialist._get_output_schema_prompt_section()
            + specialist._get_urgency_notice()
        )

    specialist._get_autonomous_system_prompt = _exploration_system_prompt  # type: ignore[assignment]

    # Run autonomous exploration
    task = (
        "Explore ALL storage and window property data in this capture. "
        "Find tokens (JWTs, session IDs, CSRF tokens, API keys) and data blocks "
        "(large JSON objects, cached responses, config). Count the noise. "
        "Then call finalize_with_output(output={...}) with a COMPLETE JSON object containing: "
        "total_events (int), noise_filtered (int), tokens (array of strings), "
        "data_blocks (array of strings), narrative (string)."
    )

    config = AutonomousRunConfig(
        min_iterations=min_iterations,
        max_iterations=max_iterations,
    )

    result = specialist.run_autonomous(
        task=task,
        config=config,
        output_schema=STORAGE_EXPLORATION_OUTPUT_SCHEMA,
        output_description=(
            "A StorageExplorationSummary with all discovered tokens and data blocks, "
            "noise count, and a narrative of findings."
        ),
    )

    if result is None:
        logger.warning("Storage exploration did not produce a result (timed out or failed)")
        return None

    if not result.success:
        logger.warning("Storage exploration failed: %s", result.failure_reason)
        return None

    # Parse into our Pydantic model
    try:
        summary = StorageExplorationSummary(**result.output)
        logger.info(
            "Storage exploration complete: %d tokens, %d data blocks discovered",
            len(summary.tokens),
            len(summary.data_blocks),
        )
        return summary
    except Exception as e:
        logger.error("Failed to parse exploration output: %s", e)
        logger.debug("Raw output: %s", result.output)
        return None


def main() -> None:
    """CLI entrypoint for storage exploration."""
    parser = argparse.ArgumentParser(
        description="Run Phase 1 Storage Exploration on CDP captures",
    )
    parser.add_argument(
        "--cdp-captures-dir",
        type=Path,
        required=True,
        help="Path to CDP captures directory (expects storage/events.jsonl and/or window_properties/events.jsonl inside)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5.1",
        help="LLM model to use (default: gpt-5.1)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("exploration_storage.json"),
        help="Path to write output JSON (default: exploration_storage.json)",
    )
    parser.add_argument(
        "--workspace-dir",
        type=Path,
        default=Path("./agent_workspace/storage_exploration"),
        help="Workspace directory for artifacts and mounted inputs.",
    )
    parser.add_argument(
        "--min-iterations",
        type=int,
        default=3,
        help="Minimum iterations before finalize is available (default: 3)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=15,
        help="Maximum iterations before the loop exits (default: 15)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        import logging
        logging.basicConfig(level=logging.DEBUG)

    if not args.cdp_captures_dir.exists():
        print(f"Error: {args.cdp_captures_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    llm_model = _resolve_model(args.model)

    summary = run_storage_exploration(
        cdp_captures_dir=args.cdp_captures_dir,
        llm_model=llm_model,
        min_iterations=args.min_iterations,
        max_iterations=args.max_iterations,
        workspace_dir=args.workspace_dir,
    )

    if summary is None:
        print("Exploration failed — no output produced.", file=sys.stderr)
        sys.exit(1)

    output_json = summary.model_dump_json(indent=2)

    if args.output:
        args.output.write_text(output_json)
        print(f"Exploration summary written to {args.output}", file=sys.stderr)
    else:
        print(output_json)


if __name__ == "__main__":
    main()
