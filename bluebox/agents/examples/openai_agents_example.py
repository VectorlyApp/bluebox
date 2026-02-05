"""
OpenAI Agents SDK - Proper subagent patterns.

This file demonstrates the TWO intended patterns for parallel/sub agents:

1. asyncio.gather() pattern - Run multiple agents concurrently, collect results
2. agent.as_tool() pattern - Register agents as tools for dynamic orchestration

Reference: https://cookbook.openai.com/examples/agents_sdk/parallel_agents
"""

import asyncio
from agents import Agent, Runner, ModelSettings, function_tool

from bluebox.utils.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# Dummy tools for specialist agents
# =============================================================================

@function_tool
def parse_har_entries(har_data: str) -> str:
    """Parse HAR/network data and extract HTTP request entries."""
    logger.info("[NetworkAnalyzer] Parsing HAR entries from data...")
    # Dummy implementation - just acknowledges the data
    lines = [l.strip() for l in har_data.split('\n') if l.strip()]
    logger.debug("[NetworkAnalyzer] Found %d lines of data", len(lines))
    return f"Parsed {len(lines)} lines. Found HTTP methods: GET, POST. Ready for endpoint analysis."


@function_tool
def extract_api_endpoints(raw_text: str) -> str:
    """Extract API endpoint URLs from parsed network data."""
    logger.info("[NetworkAnalyzer] Extracting API endpoints...")
    # Dummy: look for URL-like patterns
    import re
    urls = re.findall(r'https?://[^\s\'"]+', raw_text)
    logger.debug("[NetworkAnalyzer] Extracted %d URLs", len(urls))
    return f"Extracted endpoints: {urls}" if urls else "No endpoints found in provided text."


@function_tool
def parse_interaction_log(log_data: str) -> str:
    """Parse user interaction logs to identify UI events."""
    logger.info("[InteractionAnalyzer] Parsing interaction log...")
    events = []
    if "typed" in log_data.lower():
        events.append("text_input")
    if "clicked" in log_data.lower():
        events.append("click")
    if "selected" in log_data.lower():
        events.append("select")
    logger.debug("[InteractionAnalyzer] Detected event types: %s", events)
    return f"Detected UI event types: {events}"


@function_tool
def extract_form_fields(log_data: str) -> str:
    """Extract form field selectors from interaction data."""
    logger.info("[InteractionAnalyzer] Extracting form fields...")
    import re
    # Look for CSS selector-like patterns (#id, .class)
    selectors = re.findall(r'[#.][a-zA-Z][\w-]*', log_data)
    logger.debug("[InteractionAnalyzer] Found selectors: %s", selectors)
    return f"Found form field selectors: {selectors}" if selectors else "No selectors found."


@function_tool
def check_storage_origin(token: str, storage_type: str) -> str:
    """Check if a token might originate from a specific storage type."""
    logger.info("[TokenTracer] Checking %s for token origin...", storage_type)
    valid_types = ["localStorage", "sessionStorage", "cookie", "response_body"]
    if storage_type not in valid_types:
        return f"Unknown storage type. Valid types: {valid_types}"
    # Dummy check
    logger.debug("[TokenTracer] Token '%s...' checked in %s", token[:10] if len(token) > 10 else token, storage_type)
    return f"Token could originate from {storage_type}. Requires runtime verification."


@function_tool
def trace_token_in_responses(token: str) -> str:
    """Search for a token value in previous API responses."""
    logger.info("[TokenTracer] Tracing token in response history...")
    logger.debug("[TokenTracer] Looking for: %s", token[:20] if len(token) > 20 else token)
    return f"Token '{token[:10]}...' not found in cached responses. May be client-generated or from external source."


@function_tool
def format_routine_step(step_type: str, target: str, value: str = "") -> str:
    """Format a single routine step in the standard format."""
    logger.info("[RoutineConstructor] Formatting step: %s -> %s", step_type, target)
    step = {"type": step_type, "target": target}
    if value:
        step["value"] = value
    return f"Step formatted: {step}"


@function_tool
def validate_routine_structure(routine_json: str) -> str:
    """Validate that a routine has required fields and proper structure."""
    logger.info("[RoutineConstructor] Validating routine structure...")
    required = ["name", "operations", "parameters"]
    # Dummy validation
    missing = [f for f in required if f not in routine_json]
    if missing:
        logger.warning("[RoutineConstructor] Missing fields: %s", missing)
        return f"Validation warning: routine should include {missing}"
    logger.debug("[RoutineConstructor] Routine structure looks valid")
    return "Routine structure is valid."


# =============================================================================
# Define specialized agents
# =============================================================================

network_analyzer = Agent(
    name="NetworkAnalyzer",
    instructions="""You are a network traffic analyst. Given HAR/network capture data,
identify API endpoints, their HTTP methods, request/response patterns, and any
authentication mechanisms used. Be specific about URLs and parameters.

Use your tools to parse and extract information from the data.""",
    tools=[parse_har_entries, extract_api_endpoints],
)

interaction_analyzer = Agent(
    name="InteractionAnalyzer",
    instructions="""You are a UI interaction analyst. Given user interaction logs,
identify form fields, input parameters, button clicks, and the sequence of user
actions. Extract parameter names and their observed values.

Use your tools to parse logs and extract form field information.""",
    tools=[parse_interaction_log, extract_form_fields],
)

token_tracer = Agent(
    name="TokenTracer",
    instructions="""You are a token origin specialist. Given a specific token or
dynamic value, trace its origin - whether it came from a previous API response,
localStorage, sessionStorage, cookies, or was generated client-side.

Use your tools to check various storage types and trace tokens in responses.""",
    tools=[check_storage_origin, trace_token_in_responses],
)

routine_constructor = Agent(
    name="RoutineConstructor",
    instructions="""You are a routine builder. Given analysis from network and
interaction specialists, construct a web automation routine that captures the
essential API calls and parameters needed to replicate a user workflow.

Use your tools to format individual steps and validate the final routine structure.""",
    tools=[format_routine_step, validate_routine_structure],
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
