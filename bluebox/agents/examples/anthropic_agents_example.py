"""
Anthropic Claude Agent SDK - Proper subagent patterns.

This file demonstrates the intended pattern for subagents:
- Define subagents via AgentDefinition with description, prompt, tools
- Main agent invokes subagents via the Task tool
- Subagents can ONLY use built-in tools (Read, Grep, Glob, Bash, etc.)

Reference: https://platform.claude.com/docs/en/agent-sdk/subagents

IMPORTANT: Unlike openai-agents, you cannot wrap arbitrary Python functions.
Subagents must accomplish their tasks using built-in file/system tools.

python bluebox/agents/examples/anthropic_agents_example.py ./cdp_captures/
"""

import asyncio
from claude_agent_sdk import query, ClaudeAgentOptions, AgentDefinition

from bluebox.utils.logger import get_logger

logger = get_logger(__name__)


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

    Args:
        task: Description of what routine to discover
        cdp_captures_dir: Directory containing CDP capture files

    Returns:
        The final result text from the orchestrator
    """
    logger.info("Starting routine discovery for task: %s", task)
    logger.debug("CDP captures directory: %s", cdp_captures_dir)

    result_text = None
    session_id = None

    async for message in query(
        prompt=f"""Discover a web automation routine for: {task}

The CDP capture files are in the current directory. Use your specialist subagents:

1. First, use network-analyzer to find API endpoints in the HAR/network files
2. Use interaction-analyzer to find user input parameters
3. If you find dynamic tokens, use token-tracer to find their origins
4. Finally, use routine-builder to construct the routine JSON

Coordinate these specialists and synthesize their findings into a complete routine.""",
        options=ClaudeAgentOptions(
            # Task tool is REQUIRED for subagent invocation
            allowed_tools=["Read", "Glob", "Grep", "Task"],
            # Define our specialist subagents
            agents=SUBAGENTS,
            # Set working directory to CDP captures
            cwd=cdp_captures_dir,
        )
    ):
        # Capture session_id for potential resumption
        if hasattr(message, "session_id"):
            session_id = message.session_id

        # Log subagent invocations
        if hasattr(message, 'content') and message.content:
            for block in message.content:
                if getattr(block, 'type', None) == 'tool_use' and block.name == 'Task':
                    subagent = block.input.get('subagent_type', 'unknown')
                    logger.info("Orchestrator invoking subagent: %s", subagent)

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
