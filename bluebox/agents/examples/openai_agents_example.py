"""
OpenAI Agents SDK - Proper subagent patterns.

This file demonstrates the TWO intended patterns for parallel/sub agents:

1. asyncio.gather() pattern - Run multiple agents concurrently, collect results
2. agent.as_tool() pattern - Register agents as tools for dynamic orchestration

Reference: https://cookbook.openai.com/examples/agents_sdk/parallel_agents
"""

import asyncio
from agents import Agent, Runner, ModelSettings

from bluebox.utils.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# Define specialized agents
# =============================================================================

network_analyzer = Agent(
    name="NetworkAnalyzer",
    instructions="""You are a network traffic analyst. Given HAR/network capture data,
identify API endpoints, their HTTP methods, request/response patterns, and any
authentication mechanisms used. Be specific about URLs and parameters.""",
)

interaction_analyzer = Agent(
    name="InteractionAnalyzer",
    instructions="""You are a UI interaction analyst. Given user interaction logs,
identify form fields, input parameters, button clicks, and the sequence of user
actions. Extract parameter names and their observed values.""",
)

token_tracer = Agent(
    name="TokenTracer",
    instructions="""You are a token origin specialist. Given a specific token or
dynamic value, trace its origin - whether it came from a previous API response,
localStorage, sessionStorage, cookies, or was generated client-side.""",
)

routine_constructor = Agent(
    name="RoutineConstructor",
    instructions="""You are a routine builder. Given analysis from network and
interaction specialists, construct a web automation routine that captures the
essential API calls and parameters needed to replicate a user workflow.""",
)


# =============================================================================
# Pattern 1: asyncio.gather() - Explicit parallel execution
# =============================================================================

async def run_parallel_analysis(input_data: str) -> str:
    """
    Run multiple specialist agents in parallel using asyncio.gather().

    This pattern is deterministic - you control exactly which agents run.
    Good for: Known workflows where you always need the same analyses.
    """

    async def run_agent(agent: Agent, data: str) -> str:
        logger.debug("Running agent: %s", agent.name)
        result = await Runner.run(agent, data)
        logger.debug("Agent %s completed", result.last_agent.name)
        return f"### {result.last_agent.name}\n{result.final_output}"

    # Run network and interaction analysis in parallel
    logger.info("Running NetworkAnalyzer and InteractionAnalyzer in parallel")
    responses = await asyncio.gather(
        run_agent(network_analyzer, input_data),
        run_agent(interaction_analyzer, input_data),
    )
    logger.info("Parallel analysis complete, passing to RoutineConstructor")

    # Combine results and pass to constructor
    combined = "\n\n".join(responses)
    final = await Runner.run(
        routine_constructor,
        f"Based on these analyses, construct a routine:\n\n{combined}"
    )

    return final.final_output


# =============================================================================
# Pattern 2: agent.as_tool() - Dynamic orchestration
# =============================================================================

# Create an orchestrator that can dynamically decide which agents to invoke
orchestrator = Agent(
    name="RoutineDiscoveryOrchestrator",
    instructions="""You coordinate specialist agents to build web automation routines.

Available specialists:
- network_analyzer: Analyzes HAR/network data to find API endpoints
- interaction_analyzer: Analyzes UI interactions to find parameters
- token_tracer: Traces origins of dynamic tokens/values
- routine_constructor: Builds the final routine from analyses

Workflow:
1. First, run network_analyzer and interaction_analyzer (can be parallel)
2. If dynamic tokens are found, use token_tracer to find their origins
3. Finally, use routine_constructor to build the routine

Call specialists as needed based on the task.""",
    model_settings=ModelSettings(parallel_tool_calls=True),  # Enable parallel tool calls
    tools=[
        network_analyzer.as_tool(
            tool_name="network_analyzer",
            tool_description="Analyze network/HAR data to identify API endpoints and patterns",
        ),
        interaction_analyzer.as_tool(
            tool_name="interaction_analyzer",
            tool_description="Analyze user interactions to identify form fields and parameters",
        ),
        token_tracer.as_tool(
            tool_name="token_tracer",
            tool_description="Trace the origin of a dynamic token or value",
        ),
        routine_constructor.as_tool(
            tool_name="routine_constructor",
            tool_description="Construct a routine from the collected analyses",
        ),
    ],
)


async def run_orchestrated_discovery(task: str, data: str) -> str:
    """
    Let the orchestrator dynamically decide which agents to invoke.

    This pattern is flexible - the orchestrator decides based on the task.
    Good for: Variable workflows where different tasks need different agents.
    """
    logger.info("Starting orchestrated discovery for task: %s", task)
    result = await Runner.run(
        orchestrator,
        f"Task: {task}\n\nData:\n{data}"
    )
    logger.info("Orchestrated discovery complete")
    return result.final_output


# =============================================================================
# Example usage
# =============================================================================

async def main():
    sample_data = """
    [Network Log]
    POST https://api.example.com/search
    Headers: Authorization: Bearer abc123
    Body: {"query": "boston", "date": "2024-01-15"}
    Response: {"results": [...]}

    [Interaction Log]
    User typed "boston" in #search-input
    User clicked #search-button
    """

    # Pattern 1: Explicit parallel
    logger.info("Running Pattern 1: asyncio.gather() - explicit parallel execution")
    result1 = await run_parallel_analysis(sample_data)
    logger.info("Pattern 1 result:\n%s", result1)

    # Pattern 2: Dynamic orchestration
    logger.info("Running Pattern 2: agent.as_tool() - dynamic orchestration")
    result2 = await run_orchestrated_discovery(
        task="Build a routine for searching locations",
        data=sample_data
    )
    logger.info("Pattern 2 result:\n%s", result2)


if __name__ == "__main__":
    asyncio.run(main())
