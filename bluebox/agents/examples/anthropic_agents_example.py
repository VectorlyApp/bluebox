"""
Anthropic Claude Agent SDK - Proper subagent patterns.

This file demonstrates the intended pattern for subagents:
- Define subagents via AgentDefinition with description, prompt, tools
- Main agent invokes subagents via the Task tool
- Subagents can ONLY use built-in tools (Read, Grep, Glob, Bash, etc.)
- Main orchestrator CAN use custom MCP tools via @tool decorator

Reference: https://platform.claude.com/docs/en/agent-sdk/subagents
Reference: https://platform.claude.com/docs/en/agent-sdk/custom-tools

python bluebox/agents/examples/anthropic_agents_example.py ./cdp_captures/
"""

import asyncio
import re
from typing import Any

from claude_agent_sdk import (
    query,
    ClaudeAgentOptions,
    AgentDefinition,
    tool,
    create_sdk_mcp_server,
)

from bluebox.utils.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# Custom MCP tools for the main orchestrator
# =============================================================================
# NOTE: These tools are only available to the main orchestrator, not subagents.
# Subagents (defined via AgentDefinition) can only use built-in tools.

@tool("extract_urls", "Extract all URLs from text data", {"text": str})
async def extract_urls(args: dict[str, Any]) -> dict[str, Any]:
    """Extract URLs from text using regex."""
    logger.info("[extract_urls] Extracting URLs from text...")
    urls = re.findall(r'https?://[^\s\'"<>]+', args["text"])
    logger.debug("[extract_urls] Found %d URLs", len(urls))
    return {
        "content": [{"type": "text", "text": f"Found URLs:\n" + "\n".join(urls) if urls else "No URLs found"}]
    }


@tool("count_patterns", "Count occurrences of a pattern in text", {"text": str, "pattern": str})
async def count_patterns(args: dict[str, Any]) -> dict[str, Any]:
    """Count pattern occurrences in text."""
    logger.info("[count_patterns] Counting pattern '%s'...", args["pattern"])
    count = len(re.findall(args["pattern"], args["text"], re.IGNORECASE))
    logger.debug("[count_patterns] Found %d matches", count)
    return {
        "content": [{"type": "text", "text": f"Pattern '{args['pattern']}' found {count} times"}]
    }


@tool("summarize_findings", "Create a summary from analysis results", {"findings": str})
async def summarize_findings(args: dict[str, Any]) -> dict[str, Any]:
    """Summarize findings (dummy implementation)."""
    logger.info("[summarize_findings] Creating summary...")
    lines = [l.strip() for l in args["findings"].split("\n") if l.strip()]
    logger.debug("[summarize_findings] Summarizing %d lines of findings", len(lines))
    return {
        "content": [{"type": "text", "text": f"Summary: Analyzed {len(lines)} findings. Key insights extracted."}]
    }


@tool("validate_routine", "Validate routine JSON structure", {"routine_json": str})
async def validate_routine(args: dict[str, Any]) -> dict[str, Any]:
    """Validate routine has required fields (dummy check)."""
    logger.info("[validate_routine] Validating routine structure...")
    required = ["name", "parameters", "operations"]
    missing = [f for f in required if f not in args["routine_json"]]
    if missing:
        logger.warning("[validate_routine] Missing fields: %s", missing)
        return {"content": [{"type": "text", "text": f"Validation failed: missing {missing}"}]}
    logger.debug("[validate_routine] Routine structure valid")
    return {"content": [{"type": "text", "text": "Routine structure is valid"}]}


# Create the MCP server with our custom tools
analysis_tools_server = create_sdk_mcp_server(
    name="analysis-tools",
    version="1.0.0",
    tools=[extract_urls, count_patterns, summarize_findings, validate_routine]
)


# =============================================================================
# Define subagents via AgentDefinition
# =============================================================================
# Each subagent is a separate Claude instance with its own prompt and tools.
# They can ONLY use built-in tools - no custom Python functions.

SUBAGENTS = {
    "network-analyzer": AgentDefinition(
        description="Analyzes HAR/network capture files to find API endpoints and patterns",
        prompt="""You are a network traffic analyst specializing in HAR file analysis.

Your task is to analyze network capture files and identify:
- API endpoints (URLs, HTTP methods)
- Request/response patterns
- Authentication mechanisms (headers, tokens)
- Dynamic values that change between requests

Use Glob to find .har or .jsonl files, Read to examine them, and Grep to search
for specific patterns like URLs or auth headers.

Output your findings in a structured format.""",
        tools=["Read", "Glob", "Grep"],
        model="sonnet",
    ),

    "interaction-analyzer": AgentDefinition(
        description="Analyzes user interaction logs to find form fields and input parameters",
        prompt="""You are a UI interaction analyst.

Your task is to analyze interaction log files and identify:
- Form fields and input elements
- User-provided values and their field names
- Click sequences and navigation patterns
- Parameter names that could become routine inputs

Use Glob to find interaction log files, Read to examine them, and Grep to search
for input events and form submissions.

Output parameter names with their observed values.""",
        tools=["Read", "Glob", "Grep"],
        model="sonnet",
    ),

    "token-tracer": AgentDefinition(
        description="Traces the origin of dynamic tokens across network, storage, and JS data",
        prompt="""You are a token origin specialist.

Given a specific token or dynamic value, trace where it came from:
- Previous API responses (search network logs)
- localStorage or sessionStorage (search storage dumps)
- Cookies (search cookie logs)
- JavaScript window properties (search JS property dumps)

Use Grep to search for the token value across all data files.
Report the exact source where the token first appeared.""",
        tools=["Read", "Glob", "Grep"],
        model="sonnet",
    ),

    "routine-builder": AgentDefinition(
        description="Constructs web automation routines from analysis results",
        prompt="""You are a routine construction specialist.

Given analyses from other specialists, construct a web automation routine:
1. Define parameters (user inputs identified by interaction-analyzer)
2. Define operations (API calls identified by network-analyzer)
3. Handle dynamic tokens (origins traced by token-tracer)

Output a JSON routine with:
- name, description
- parameters (name, type, description)
- operations (navigate, fetch with url/method/headers/body)

Ensure dynamic values use placeholder syntax: {{parameterName}}""",
        tools=["Read", "Glob", "Grep"],
        model="sonnet",
    ),
}


# =============================================================================
# Run the orchestrator with subagents
# =============================================================================

async def run_routine_discovery(
    task: str,
    cdp_captures_dir: str,
) -> str | None:
    """
    Run routine discovery using Claude Agent SDK with subagents.

    The main agent (orchestrator) will invoke subagents via the Task tool
    based on the task description and subagent descriptions.

    The orchestrator also has access to custom MCP tools (extract_urls,
    count_patterns, summarize_findings, validate_routine) for direct analysis.

    Args:
        task: Description of what routine to discover
        cdp_captures_dir: Directory containing CDP capture files

    Returns:
        The final result text from the orchestrator
    """
    logger.info("Starting routine discovery for task: %s", task)
    logger.debug("CDP captures directory: %s", cdp_captures_dir)

    result_text = None

    # Streaming input generator (required when using custom MCP tools)
    async def prompt_generator():
        yield {
            "type": "user",
            "message": {
                "role": "user",
                "content": f"""Discover a web automation routine for: {task}

The CDP capture files are in the current directory.

You have two types of tools available:

1. CUSTOM ANALYSIS TOOLS (use directly):
   - extract_urls: Extract URLs from text
   - count_patterns: Count pattern occurrences
   - summarize_findings: Create analysis summary
   - validate_routine: Validate routine JSON structure

2. SPECIALIST SUBAGENTS (invoke via Task tool):
   - network-analyzer: Analyzes HAR/network files for API endpoints
   - interaction-analyzer: Finds user input parameters
   - token-tracer: Traces dynamic token origins
   - routine-builder: Constructs the final routine JSON

Workflow:
1. Use network-analyzer and interaction-analyzer subagents to analyze the data
2. Use your extract_urls and count_patterns tools for quick analysis
3. If dynamic tokens are found, use token-tracer
4. Use routine-builder to construct the routine
5. Use validate_routine to check the result

Coordinate these tools and synthesize findings into a complete routine."""
            }
        }

    async for message in query(
        prompt=prompt_generator(),  # Streaming input for MCP tools
        options=ClaudeAgentOptions(
            # Built-in tools + Task for subagents + our custom MCP tools
            allowed_tools=[
                "Read", "Glob", "Grep", "Task",
                "mcp__analysis-tools__extract_urls",
                "mcp__analysis-tools__count_patterns",
                "mcp__analysis-tools__summarize_findings",
                "mcp__analysis-tools__validate_routine",
            ],
            # Register our custom MCP server
            mcp_servers={"analysis-tools": analysis_tools_server},
            # Define our specialist subagents
            agents=SUBAGENTS,
            # Set working directory to CDP captures
            cwd=cdp_captures_dir,
        )
    ):
        # Log subagent invocations and MCP tool calls
        if hasattr(message, 'content') and message.content:
            for block in message.content:
                if hasattr(block, 'name'):
                    if block.name == 'Task':
                        subagent = block.input.get('subagent_type', 'unknown')
                        logger.info("Orchestrator invoking subagent: %s", subagent)
                    elif block.name.startswith('mcp__'):
                        logger.info("Orchestrator using custom tool: %s", block.name)

        # Capture final result
        if hasattr(message, "result"):
            result_text = message.result

    logger.info("Routine discovery complete")
    return result_text


# =============================================================================
# Alternative: Explicit subagent invocation
# =============================================================================

async def run_with_explicit_subagent(
    subagent_name: str,
    task: str,
    working_dir: str,
) -> str | None:
    """
    Explicitly invoke a specific subagent by name.

    Use this when you know exactly which subagent should handle a task,
    rather than letting the orchestrator decide.
    """
    logger.info("Explicitly invoking subagent '%s' for task: %s", subagent_name, task)
    result_text = None

    async for message in query(
        # Explicitly name the subagent in the prompt
        prompt=f"Use the {subagent_name} agent to: {task}",
        options=ClaudeAgentOptions(
            allowed_tools=["Read", "Glob", "Grep", "Task"],
            agents=SUBAGENTS,
            cwd=working_dir,
        )
    ):
        if hasattr(message, "result"):
            result_text = message.result

    logger.debug("Subagent '%s' execution complete", subagent_name)
    return result_text


# =============================================================================
# Example usage
# =============================================================================

async def main():
    import sys

    if len(sys.argv) < 2:
        logger.error("Usage: python anthropic_agents_example.py <cdp_captures_dir>")
        logger.info("This will run routine discovery using Claude Agent SDK subagents.")
        return

    cdp_captures_dir = sys.argv[1]
    task = "Search for one-way trains from Boston to New York"

    logger.info("Starting routine discovery")
    logger.info("Task: %s", task)
    logger.info("CDP captures directory: %s", cdp_captures_dir)

    result = await run_routine_discovery(task, cdp_captures_dir)

    if result:
        logger.info("Final result:\n%s", result)
    else:
        logger.warning("No result returned from routine discovery")


if __name__ == "__main__":
    asyncio.run(main())
