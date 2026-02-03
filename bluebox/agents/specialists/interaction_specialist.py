"""
bluebox/agents/specialists/interaction_specialist.py

Interaction specialist agent.

Analyzes UI interaction recordings to discover routine parameters
(form inputs, typed values, dropdown selections, date pickers, etc.).
"""

from __future__ import annotations

from textwrap import dedent
from typing import Any, Callable

from pydantic import BaseModel, Field

from bluebox.agents.specialists.abstract_specialist import AbstractSpecialist, SpecialistMode, specialist_tool
from bluebox.data_models.llms.interaction import (
    Chat,
    ChatThread,
    EmittedMessage,
)
from bluebox.data_models.llms.vendors import LLMModel, OpenAIModel
from bluebox.llms.infra.interactions_data_store import InteractionsDataStore
from bluebox.utils.llm_utils import token_optimized
from bluebox.utils.logger import get_logger

logger = get_logger(name=__name__)


class DiscoveredParameter(BaseModel):
    """A single discovered routine parameter."""
    name: str = Field(description="snake_case parameter name")
    type: str = Field(description="ParameterType value (string, date, integer, etc.)")
    description: str = Field(description="Human-readable description of the parameter")
    examples: list[str] = Field(
        default_factory=list,
        description="Example values observed in interactions",
    )
    source_element_css_path: str | None = Field(
        default=None,
        description="CSS path of the source element",
    )
    source_element_tag: str | None = Field(
        default=None,
        description="HTML tag of the source element",
    )
    source_element_name: str | None = Field(
        default=None,
        description="Name attribute of the source element",
    )


class ParameterDiscoveryResult(BaseModel):
    """Successful parameter discovery result."""
    parameters: list[DiscoveredParameter] = Field(description="List of discovered parameters")


class ParameterDiscoveryFailureResult(BaseModel):
    """Failure result when parameters cannot be discovered."""
    reason: str = Field(description="Why parameters could not be discovered")
    interaction_summary: str = Field(description="Summary of interactions that were analyzed")


class InteractionSpecialist(AbstractSpecialist):
    """
    Interaction specialist agent.

    Analyzes recorded UI interactions to discover routine parameters.
    """

    SYSTEM_PROMPT: str = dedent("""\
        You are a UI interaction analyst specializing in discovering routine parameters from recorded browser interactions.

        ## Your Role

        You analyze recorded UI interactions (clicks, keypresses, form inputs, etc.) to identify which interactions represent parameterizable inputs for a routine.

        ## What to Look For

        - **Form inputs**: Text fields, search boxes, email/password fields
        - **Typed values**: Text entered by the user via keyboard
        - **Dropdown selections**: Select elements, custom dropdowns
        - **Date pickers**: Date/time inputs
        - **Checkboxes and toggles**: Boolean parameters

        ## What to Ignore

        - **Navigational clicks**: Clicks on links, buttons that just navigate
        - **Non-parameterizable interactions**: Scroll events, hover effects, focus/blur without input
        - **UI framework noise**: Internal framework events

        ## Parameter Requirements

        Each discovered parameter needs:
        - **name**: snake_case name (e.g., `search_query`, `departure_date`)
        - **type**: One of: string, integer, number, boolean, date, datetime, email, url, enum
        - **description**: Clear description of what the parameter represents
        - **examples**: Observed values from the interactions

        ## Tools

        - **get_interaction_summary**: Overview statistics of all interactions
        - **search_interactions_by_type**: Filter by interaction type (click, input, change, etc.)
        - **search_interactions_by_element**: Filter by element attributes (tag, id, class, type)
        - **get_interaction_detail**: Full detail of a specific interaction event
        - **get_form_inputs**: All input/change events with values
        - **get_unique_elements**: Deduplicated elements with interaction counts
    """)

    AUTONOMOUS_SYSTEM_PROMPT: str = dedent("""\
        You are a UI interaction analyst that autonomously discovers routine parameters from recorded browser interactions.

        ## Your Mission

        Analyze the recorded UI interactions to identify all parameterizable inputs, then produce a list of discovered parameters.

        ## Process

        1. **Survey**: Use `get_interaction_summary` to understand the overall interaction data
        2. **Focus on inputs**: Use `get_form_inputs` to find all form input events
        3. **Analyze elements**: Use `get_unique_elements` to see which elements were interacted with
        4. **Detail check**: Use `get_interaction_detail` for specific events needing closer inspection
        5. **Finalize**: Call `finalize_result` with discovered parameters

        ## Parameter Types

        - `string`: General text input
        - `date`: Date values (YYYY-MM-DD)
        - `datetime`: Date+time values
        - `integer`: Whole numbers
        - `number`: Decimal numbers
        - `boolean`: True/false (checkboxes, toggles)
        - `email`: Email addresses
        - `url`: URLs
        - `enum`: Selection from fixed options

        ## When finalize tools are available

        - **finalize_result**: Submit discovered parameters
        - **finalize_failure**: Report that no parameters could be discovered
    """)

    ## Magic methods

    def __init__(
        self,
        emit_message_callable: Callable[[EmittedMessage], None],
        interaction_data_store: InteractionsDataStore,
        persist_chat_callable: Callable[[Chat], Chat] | None = None,
        persist_chat_thread_callable: Callable[[ChatThread], ChatThread] | None = None,
        stream_chunk_callable: Callable[[str], None] | None = None,
        llm_model: LLMModel = OpenAIModel.GPT_5_1,
        chat_thread: ChatThread | None = None,
        existing_chats: list[Chat] | None = None,
    ) -> None:
        self._interaction_data_store = interaction_data_store

        # autonomous result state
        self._discovery_result: ParameterDiscoveryResult | None = None
        self._discovery_failure: ParameterDiscoveryFailureResult | None = None

        super().__init__(
            emit_message_callable=emit_message_callable,
            persist_chat_callable=persist_chat_callable,
            persist_chat_thread_callable=persist_chat_thread_callable,
            stream_chunk_callable=stream_chunk_callable,
            llm_model=llm_model,
            chat_thread=chat_thread,
            existing_chats=existing_chats,
        )
        logger.debug(
            "InteractionSpecialist initialized with %d events",
            len(interaction_data_store.events),
        )

    ## Abstract method implementations

    def _get_system_prompt(self) -> str:
        stats = self._interaction_data_store.stats
        context = (
            f"\n\n## Interaction Data Context\n"
            f"- Total Events: {stats.total_events}\n"
            f"- Unique URLs: {stats.unique_urls}\n"
            f"- Unique Elements: {stats.unique_elements}\n"
            f"- Events by Type: {stats.events_by_type}\n"
        )
        return self.SYSTEM_PROMPT + context

    def _get_autonomous_system_prompt(self) -> str:
        stats = self._interaction_data_store.stats
        context = (
            f"\n\n## Interaction Data Context\n"
            f"- Total Events: {stats.total_events}\n"
            f"- Unique URLs: {stats.unique_urls}\n"
            f"- Unique Elements: {stats.unique_elements}\n"
        )

        # urgency notices
        if self.mode == SpecialistMode.FINALIZING:
            remaining = self._max_iterations - self._autonomous_iteration
            if remaining <= 2:
                urgency = (
                    f"\n\n## CRITICAL: Only {remaining} iterations remaining!\n"
                    f"You MUST call finalize_result or finalize_failure NOW!"
                )
            elif remaining <= 4:
                urgency = (
                    f"\n\n## URGENT: Only {remaining} iterations remaining.\n"
                    f"Finalize your parameter discovery soon."
                )
            else:
                urgency = (
                    "\n\n## Finalize tools are now available.\n"
                    "Call finalize_result when you have identified all parameters."
                )
        else:
            urgency = (
                f"\n\n## Continue exploring (iteration {self._autonomous_iteration}).\n"
                "Finalize tools will become available after more exploration."
            )

        return self.AUTONOMOUS_SYSTEM_PROMPT + context + urgency

    # _register_tools and _execute_tool are provided by the base class
    # via @specialist_tool decorators below.

    def _get_autonomous_initial_message(self, task: str) -> str:
        return (
            f"TASK: {task}\n\n"
            "Analyze the recorded UI interactions to discover all parameterizable inputs. "
            "Focus on form inputs, typed values, dropdown selections, and date pickers. "
            "When confident, use finalize_result to report your findings."
        )

    def _check_autonomous_completion(self, tool_name: str) -> bool:
        if tool_name == "finalize_result" and self._discovery_result is not None:
            return True
        if tool_name == "finalize_failure" and self._discovery_failure is not None:
            return True
        return False

    def _get_autonomous_result(self) -> BaseModel | None:
        return self._discovery_result or self._discovery_failure

    def _reset_autonomous_state(self) -> None:
        self._discovery_result = None
        self._discovery_failure = None

    ## Tool handlers

    @specialist_tool(
        description="Get summary statistics of all recorded interactions.",
        parameters={"type": "object", "properties": {}},
    )
    @token_optimized
    def _get_interaction_summary(self, tool_arguments: dict[str, Any]) -> dict[str, Any]:
        stats = self._interaction_data_store.stats
        return {
            "total_events": stats.total_events,
            "unique_urls": stats.unique_urls,
            "unique_elements": stats.unique_elements,
            "events_by_type": stats.events_by_type,
        }


    @specialist_tool(
        description="Filter interactions by type (e.g., click, input, change, keydown, focus).",
        parameters={
            "type": "object",
            "properties": {
                "types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of InteractionType values to filter by.",
                }
            },
            "required": ["types"],
        },
    )
    @token_optimized
    def _search_interactions_by_type(self, tool_arguments: dict[str, Any]) -> dict[str, Any]:
        types = tool_arguments.get("types", [])
        if not types:
            return {"error": "types list is required"}

        events = self._interaction_data_store.filter_by_type(types)
        # Return summary to avoid overwhelming the LLM
        results = []
        for event in events[:50]:
            el = event.element
            results.append({
                "index": self._interaction_data_store.events.index(event),
                "type": event.type.value,
                "tag_name": el.tag_name,
                "element_id": el.id,
                "element_name": el.name,
                "value": el.value,
                "css_path": el.css_path,
                "url": event.url,
            })

        return {
            "total_matching": len(events),
            "showing": len(results),
            "results": results,
        }


    @specialist_tool(
        description="Filter interactions by element attributes (tag, id, class, type).",
        parameters={
            "type": "object",
            "properties": {
                "tag_name": {"type": "string", "description": "HTML tag name (e.g., input, select, button)."},
                "element_id": {"type": "string", "description": "Element ID attribute."},
                "class_name": {"type": "string", "description": "CSS class name (substring match)."},
                "type_attr": {"type": "string", "description": "Input type attribute (e.g., text, email, date)."},
            },
        },
    )
    @token_optimized
    def _search_interactions_by_element(self, tool_arguments: dict[str, Any]) -> dict[str, Any]:
        events = self._interaction_data_store.filter_by_element(
            tag_name=tool_arguments.get("tag_name"),
            element_id=tool_arguments.get("element_id"),
            class_name=tool_arguments.get("class_name"),
            type_attr=tool_arguments.get("type_attr"),
        )

        results = []
        for event in events[:50]:
            el = event.element
            results.append({
                "index": self._interaction_data_store.events.index(event),
                "type": event.type.value,
                "tag_name": el.tag_name,
                "element_id": el.id,
                "element_name": el.name,
                "type_attr": el.type_attr,
                "value": el.value,
                "css_path": el.css_path,
                "url": event.url,
            })

        return {
            "total_matching": len(events),
            "showing": len(results),
            "results": results,
        }


    @specialist_tool(
        description="Get full details of a specific interaction event by index.",
        parameters={
            "type": "object",
            "properties": {
                "index": {
                    "type": "integer",
                    "description": "Zero-based index of the interaction event.",
                }
            },
            "required": ["index"],
        },
    )
    @token_optimized
    def _get_interaction_detail(self, tool_arguments: dict[str, Any]) -> dict[str, Any]:
        index = tool_arguments.get("index")
        if index is None:
            return {"error": "index is required"}

        detail = self._interaction_data_store.get_event_detail(index)
        if detail is None:
            return {"error": f"Event index {index} out of range (0-{len(self._interaction_data_store.events) - 1})"}

        return detail


    @specialist_tool(
        description="Get all input/change events with their values and element info.",
        parameters={"type": "object", "properties": {}},
    )
    @token_optimized
    def _get_form_inputs(self, tool_arguments: dict[str, Any]) -> dict[str, Any]:
        inputs = self._interaction_data_store.get_form_inputs()
        return {
            "total_inputs": len(inputs),
            "inputs": inputs[:100],
        }


    @specialist_tool(
        description="Get deduplicated elements with interaction counts and types.",
        parameters={"type": "object", "properties": {}},
    )
    @token_optimized
    def _get_unique_elements(self, tool_arguments: dict[str, Any]) -> dict[str, Any]:
        elements = self._interaction_data_store.get_unique_elements()
        return {
            "total_unique_elements": len(elements),
            "elements": elements[:50],
        }


    @specialist_tool(
        description="Submit discovered parameters. Call when you have identified all parameterizable inputs.",
        availability=lambda self: self.mode == SpecialistMode.FINALIZING,
        parameters={
            "type": "object",
            "properties": {
                "parameters": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "snake_case parameter name."},
                            "type": {"type": "string", "description": "Parameter type (string, date, integer, etc.)."},
                            "description": {"type": "string", "description": "What the parameter represents."},
                            "examples": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Example values observed.",
                            },
                            "source_element_css_path": {"type": "string", "description": "CSS path of the source element."},
                            "source_element_tag": {"type": "string", "description": "HTML tag of the source element."},
                            "source_element_name": {"type": "string", "description": "Name attribute of the source element."},
                        },
                        "required": ["name", "type", "description"],
                    },
                    "description": "List of discovered parameters.",
                }
            },
            "required": ["parameters"],
        },
    )
    @token_optimized
    def _finalize_result(self, tool_arguments: dict[str, Any]) -> dict[str, Any]:
        params_data = tool_arguments.get("parameters", [])

        if not params_data:
            return {"error": "parameters list is required and cannot be empty"}

        discovered: list[DiscoveredParameter] = []
        for i, p in enumerate(params_data):
            name = p.get("name", "")
            param_type = p.get("type", "")
            description = p.get("description", "")

            if not name:
                return {"error": f"parameters[{i}].name is required"}
            if not param_type:
                return {"error": f"parameters[{i}].type is required"}
            if not description:
                return {"error": f"parameters[{i}].description is required"}

            discovered.append(DiscoveredParameter(
                name=name,
                type=param_type,
                description=description,
                examples=p.get("examples", []),
                source_element_css_path=p.get("source_element_css_path"),
                source_element_tag=p.get("source_element_tag"),
                source_element_name=p.get("source_element_name"),
            ))

        self._discovery_result = ParameterDiscoveryResult(parameters=discovered)

        logger.info("Parameter discovery completed: %d parameter(s)", len(discovered))
        for param in discovered:
            logger.info("  - %s (%s): %s", param.name, param.type, param.description)

        return {
            "status": "success",
            "message": f"Discovered {len(discovered)} parameter(s)",
            "result": self._discovery_result.model_dump(),
        }


    @specialist_tool(
        description="Report that no parameters could be discovered from the interactions.",
        availability=lambda self: self.mode == SpecialistMode.FINALIZING,
        parameters={
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "Why no parameters could be discovered.",
                },
                "interaction_summary": {
                    "type": "string",
                    "description": "Summary of what interactions were analyzed.",
                },
            },
            "required": ["reason", "interaction_summary"],
        },
    )
    @token_optimized
    def _finalize_failure(self, tool_arguments: dict[str, Any]) -> dict[str, Any]:
        reason = tool_arguments.get("reason", "")
        interaction_summary = tool_arguments.get("interaction_summary", "")

        if not reason:
            return {"error": "reason is required"}
        if not interaction_summary:
            return {"error": "interaction_summary is required"}

        self._discovery_failure = ParameterDiscoveryFailureResult(
            reason=reason,
            interaction_summary=interaction_summary,
        )

        logger.info("Parameter discovery failed: %s", reason)

        return {
            "status": "failure",
            "message": "Parameter discovery marked as failed",
            "result": self._discovery_failure.model_dump(),
        }
