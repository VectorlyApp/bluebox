"""
bluebox/scripts/api_indexing/run_network_exploration.py

Prototype script for Phase 1 (Exploration) — Network domain.

Runs the NetworkSpecialist in autonomous mode with an output schema
matching NetworkExplorationSummary. The agent uses its existing tools
(search_responses_by_terms, get_entry_detail, get_response_body_schema,
get_unique_urls, etc.) to explore captured traffic and produce a
structured exploration summary.

Usage:
    python -m bluebox.scripts.api_indexing.run_network_exploration --cdp-captures-dir ./cdp_captures
    python -m bluebox.scripts.api_indexing.run_network_exploration --cdp-captures-dir ./cdp_captures --model gpt-5.1
    python -m bluebox.scripts.api_indexing.run_network_exploration --cdp-captures-dir ./cdp_captures --output /tmp/exploration.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from bluebox.agents.abstract_agent import AutonomousRunConfig
from bluebox.agents.specialists.network_specialist import NetworkSpecialist
from bluebox.workspace import LocalAgentWorkspace
from bluebox.data_models.api_indexing.exploration import (
    EndpointCategory,
    EndpointCluster,
    InterestLevel,
    NetworkExplorationSummary,
)
from bluebox.data_models.llms.interaction import EmittedMessage
from bluebox.data_models.llms.vendors import LLMModel, OpenAIModel
from bluebox.llms.data_loaders.network_data_loader import NetworkDataLoader
from bluebox.utils.logger import get_logger

logger = get_logger(name=__name__)

# ---------------------------------------------------------------------------
# Output schema — derived from NetworkExplorationSummary
# ---------------------------------------------------------------------------

NETWORK_EXPLORATION_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "total_requests": {
            "type": "integer",
            "description": "Total number of requests in the capture.",
        },
        "endpoints": {
            "type": "array",
            "description": "Discovered endpoints, sorted by interest level (high first).",
            "items": {
                "type": "object",
                "properties": {
                    "url_pattern": {
                        "type": "string",
                        "description": "Deduplicated URL pattern, e.g. '/api/v2/search'.",
                    },
                    "method": {
                        "type": "string",
                        "description": "HTTP method, e.g. 'POST'.",
                    },
                    "category": {
                        "type": "string",
                        "enum": [e.value for e in EndpointCategory],
                        "description": "What role this endpoint plays: action, data, auth, or navigation.",
                    },
                    "hit_count": {
                        "type": "integer",
                        "description": "How many times this endpoint was called in the session.",
                    },
                    "description": {
                        "type": "string",
                        "description": "What this endpoint does.",
                    },
                    "interest": {
                        "type": "string",
                        "enum": [e.value for e in InterestLevel],
                        "description": "How relevant for routine construction: high (core business endpoints), medium (supporting/config), low (noise/analytics).",
                    },
                    "request_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "References to raw captured request IDs.",
                    },
                },
                "required": [
                    "url_pattern",
                    "method",
                    "category",
                    "hit_count",
                    "description",
                    "interest",
                    "request_ids",
                ],
            },
        },
        "auth_observations": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Observed auth patterns, e.g. 'Bearer JWT on all /api/ requests'.",
        },
        "narrative": {
            "type": "string",
            "description": "Free-form observations: oddities, patterns, anything that doesn't fit the structured fields.",
        },
    },
    "required": [
        "total_requests",
        "endpoints",
        "auth_observations",
        "narrative",
    ],
}

# ---------------------------------------------------------------------------
# Autonomous system prompt override — exploration-specific
# ---------------------------------------------------------------------------

EXPLORATION_SYSTEM_PROMPT = """
You are a network traffic exploration agent. Your job is to survey ALL captured
network traffic and produce a comprehensive exploration summary.

## Your Mission

Explore the full network capture and produce a structured summary that answers:
1. How many requests were captured? How many are noise (analytics, static assets, CDN)?
2. What are the distinct API endpoints? Group by URL pattern, not individual requests.
3. What HTTP methods and content types are used?
4. What auth patterns are visible? (Bearer tokens, cookies, CSRF headers, API keys)
5. Which endpoints are most interesting for automating user tasks?

## Process

1. **Survey**: Start with `get_unique_urls` to see all URLs in the capture.
2. **Categorize**: For each interesting URL pattern, use `get_entry_detail` to understand
   what it does. Distinguish between:
   - **action** endpoints (POST/PUT that do something — search, submit, create)
   - **data** endpoints (GET that return meaningful data — user info, config, prices)
   - **auth** endpoints (token endpoints, login, refresh, CSRF fetches)
   - **navigation** endpoints (HTML page loads, redirects)
3. **Investigate auth**: Look at request headers across multiple entries to identify
   auth patterns (Bearer tokens, cookies, CSRF headers, API keys).
4. **Rate interest**: Label each endpoint's interest level:
   - **high**: Core business endpoints the routine will actually call (search, submit, book)
   - **medium**: Supporting endpoints (reference data, config, session management, auth)
   - **low**: Noise (analytics, tracking, ads, bot detection, static assets, CDN)
5. **Filter noise**: Count requests that are clearly noise (analytics trackers, CDN
   resources, font/image loads, preflight OPTIONS requests).
6. **Finalize**: Call `finalize_with_output(output={...})` with the COMPLETE JSON object
   inline. The output must include ALL required fields: total_requests,
   endpoints (array), auth_observations (array), narrative (string).

## CRITICAL: How to Finalize

When you call `finalize_with_output`, you MUST pass the ENTIRE output as a single JSON
object in the `output` parameter. Do NOT call it empty or with partial data. Build the
full summary in your head first, then emit it in one call. Example:

```
finalize_with_output(output={
  "total_requests": 150,

  "endpoints": [
    {
      "url_pattern": "/api/search",
      "method": "POST",
      "category": "action",
      "hit_count": 2,
      "description": "Search endpoint",
      "interest": "high",
      "request_ids": ["req_123"]
    }
  ],
  "auth_observations": ["Bearer JWT on all /api/ requests"],
  "narrative": "The site uses a standard REST API pattern..."
})
```

Do NOT call finalize_failure unless you genuinely cannot find ANY endpoints in the data.

## Guidelines

- **Focus your investigation on high and medium interest endpoints.** These are the ones
  that matter for routine construction. Spend your tool calls understanding THESE in detail
  — inspect their request/response bodies, headers, auth patterns, and payloads.
- For low-interest endpoints (analytics, tracking, ads, CDN, bot detection), just list
  them briefly. Do NOT waste tool calls investigating Dynatrace beacons, Sojern pixels,
  or PerimeterX collectors — name them, mark them low, and move on.
- Be thorough — examine ALL distinct URL patterns, not just the first few.
- Group duplicate URLs (same path, different query params) into one endpoint cluster.
- Include request_ids for high and medium endpoints so we can drill in later.
  Low-interest endpoints can have empty request_ids.
- The narrative field is for anything that doesn't fit the structured fields:
  unexpected patterns, potential issues, rate limiting, error spikes, etc.
- Sort endpoints by interest level (high first, then medium, then low) in your output.
- Keep endpoint descriptions concise (one sentence each) to stay within token limits.
""".strip()


def _emit_message(msg: EmittedMessage) -> None:
    """Print emitted messages to stderr for visibility."""
    if hasattr(msg, "content") and msg.content:
        print(f"[agent] {msg.content}", file=sys.stderr)
    elif hasattr(msg, "error") and msg.error:
        print(f"[error] {msg.error}", file=sys.stderr)


def _resolve_model(model_str: str) -> LLMModel:
    """Resolve a model string to an LLMModel enum value."""
    # Try OpenAI models
    for member in OpenAIModel:
        if member.value == model_str:
            return member
    raise ValueError(
        f"Unknown model: {model_str}. "
        f"Available: {[m.value for m in OpenAIModel]}"
    )


def run_network_exploration(
    cdp_captures_dir: Path,
    llm_model: LLMModel = OpenAIModel.GPT_5_1,
    min_iterations: int = 3,
    max_iterations: int = 15,
    workspace_dir: Path | None = None,
    task: str | None = None,
) -> NetworkExplorationSummary | None:
    """
    Run network exploration on a CDP captures directory.

    Args:
        cdp_captures_dir: Path to directory containing network/events.jsonl.
        llm_model: LLM model to use.
        min_iterations: Minimum iterations before finalize is available.
        max_iterations: Maximum iterations before the loop exits.
        workspace_dir: Workspace directory for artifacts and mounted inputs.

    Returns:
        NetworkExplorationSummary if successful, None if the agent failed or timed out.
    """
    # Convention: cdp_captures/network/events.jsonl
    network_jsonl = cdp_captures_dir / "network" / "events.jsonl"
    if not network_jsonl.exists():
        logger.error("network/events.jsonl not found in %s", cdp_captures_dir)
        return None

    # Load network data
    loader = NetworkDataLoader(jsonl_path=str(network_jsonl))
    logger.info(
        "Loaded %d network entries (%d unique URLs, %d hosts)",
        loader.stats.total_requests,
        loader.stats.unique_urls,
        loader.stats.unique_hosts,
    )

    workspace = LocalAgentWorkspace.from_directory_path(
        workspace_dir or Path("./agent_workspace/network_exploration"),
    )
    workspace.attach_input_file("network_events", network_jsonl)

    # Build the specialist — override its autonomous prompt for exploration
    specialist = NetworkSpecialist(
        emit_message_callable=_emit_message,
        network_data_loader=loader,
        llm_model=llm_model,
        workspace=workspace,
    )

    # Bump max_output_tokens — the default 4096 is too small for the finalize_with_output
    # call which needs to emit the full exploration summary as a single JSON tool argument.
    specialist.llm_client._client.DEFAULT_MAX_TOKENS = 16_384

    # Monkey-patch the autonomous system prompt for exploration
    def _exploration_system_prompt() -> str:
        stats = loader.stats
        stats_context = (
            f"\n\n## Network Traffic Context\n"
            f"- Total Requests: {stats.total_requests}\n"
            f"- Unique URLs: {stats.unique_urls}\n"
            f"- Unique Hosts: {stats.unique_hosts}\n"
        )

        likely_urls = loader.api_urls
        if likely_urls:
            # Cap count and per-line length so system prompt stays under model context limits.
            trimmed_lines = []
            for url in likely_urls[:20]:
                short_url = url if len(url) <= 220 else url[:220] + "..."
                trimmed_lines.append(f"- {short_url}")
            urls_list = "\n".join(trimmed_lines)
            urls_context = f"\n\n## Likely API Endpoints\n{urls_list}"
        else:
            urls_context = ""

        host_stats = loader.get_host_stats()
        if host_stats:
            host_lines = []
            for hs in host_stats[:15]:
                methods_str = ", ".join(f"{m}:{c}" for m, c in sorted(hs["methods"].items()))
                host_lines.append(f"- {hs['host']}: {hs['request_count']} reqs ({methods_str})")
            host_context = f"\n\n## Host Statistics\n" + "\n".join(host_lines)
        else:
            host_context = ""

        return (
            EXPLORATION_SYSTEM_PROMPT
            + stats_context
            + host_context
            + urls_context
            + specialist._get_output_schema_prompt_section()
            + specialist._get_urgency_notice()
        )

    specialist._get_autonomous_system_prompt = _exploration_system_prompt  # type: ignore[assignment]

    # Override initial message so exploration framing is explicit.
    def _exploration_initial_message(task_text: str) -> str:
        return (
            f"NETWORK EXPLORATION TASK: {task_text}\n\n"
            "This is broad capture exploration. Survey URLs, clusters, auth signals, "
            "and endpoint categories across the full dataset, then finalize with the "
            "complete structured output."
        )

    specialist._get_autonomous_initial_message = _exploration_initial_message  # type: ignore[assignment]

    # Run autonomous exploration
    task_prefix = ""
    if task:
        task_prefix = (
            f"Here is the user's task: \"{task}\". "
            "Keep this in mind when exploring, but do not limit yourself to this task.\n\n"
        )
    exploration_task = (
        f"{task_prefix}"
        "Explore ALL network traffic in this capture. Use the tools to survey URLs, "
        "inspect interesting entries, and identify auth patterns. Then call "
        "finalize_with_output(output={...}) with a COMPLETE JSON object containing: "
        "total_requests (int), endpoints (array of endpoint objects), "
        "auth_observations (array of strings), and narrative (string). "
        "Each endpoint object needs: url_pattern, method, category (action|data|auth|navigation), "
        "hit_count, description, interest (high|medium|low), request_ids (array)."
    )

    config = AutonomousRunConfig(
        min_iterations=min_iterations,
        max_iterations=max_iterations,
    )

    result = specialist.run_autonomous(
        task=exploration_task,
        config=config,
        output_schema=NETWORK_EXPLORATION_OUTPUT_SCHEMA,
        output_description=(
            "A NetworkExplorationSummary with all discovered endpoints categorized "
            "and scored, auth observations, and a narrative of findings."
        ),
    )

    if result is None:
        logger.warning("Network exploration did not produce a result (timed out or failed)")
        return None

    # The result is a SpecialistResultWrapper — extract the output
    if not result.success:
        logger.warning("Network exploration failed: %s", result.failure_reason)
        return None

    # Parse into our Pydantic model
    try:
        summary = NetworkExplorationSummary(**result.output)
        logger.info(
            "Network exploration complete: %d endpoints discovered",
            len(summary.endpoints),
        )
        return summary
    except Exception as e:
        logger.error("Failed to parse exploration output: %s", e)
        logger.debug("Raw output: %s", result.output)
        return None


def main() -> None:
    """CLI entrypoint for network exploration."""
    parser = argparse.ArgumentParser(
        description="Run Phase 1 Network Exploration on CDP captures",
    )
    parser.add_argument(
        "--cdp-captures-dir",
        type=Path,
        required=True,
        help="Path to CDP captures directory (expects network/events.jsonl inside)",
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
        default=Path("exploration_network.json"),
        help="Path to write output JSON (default: exploration_network.json)",
    )
    parser.add_argument(
        "--workspace-dir",
        type=Path,
        default=Path("./agent_workspace/network_exploration"),
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
        logging.basicConfig(level=logging.DEBUG)

    if not args.cdp_captures_dir.exists():
        print(f"Error: {args.cdp_captures_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    llm_model = _resolve_model(args.model)

    summary = run_network_exploration(
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
