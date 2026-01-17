"""
Chat Agent - Interactive Q&A about routines and browser captures.

Uses LocalDiscoveryDataStoreV2 to access:
- Agent documentation (routines, operations, parameters, placeholders)
- CDP captures (network transactions, storage, window properties)

This agent answers questions about:
- How routines work
- Operation types and their usage
- Parameter definitions
- What happened in a browser session
- Specific transaction details
"""

import json
import os
from pathlib import Path
from typing import Callable, Generator

from openai import OpenAI
from pydantic import BaseModel, Field, ConfigDict, ValidationError

from llm_context_manager_v3 import LLMContextManagerV3
from web_hacker.data_models.routine.routine import Routine
from web_hacker.data_models.routine.execution import RoutineExecutionResult
from web_hacker.routine_discovery.context_manager_v2 import LocalDiscoveryDataStoreV2
from web_hacker.utils.logger import get_logger

# Max result size to include directly in response (50KB)
MAX_INLINE_RESULT_SIZE = 50_000

logger = get_logger(__name__)


SYSTEM_PROMPT_BASE = """You are an expert assistant for the Web Hacker automation framework.

You have access to:
1. **Documentation** - Detailed docs about routines, operations, parameters, placeholders, endpoints, and execution
2. **CDP Captures** - Network transactions, storage events, and window properties from browser sessions
3. **validate_routine tool** - Validate routines by instantiating them. Use this to check if a routine is valid!
4. **execute_routine tool** - Execute a validated routine with parameters. Requires Chrome running on port 9222.

Your role is to:
- Answer questions about how routines work
- Explain operation types, parameters, and placeholder syntax
- Analyze browser captures to explain what happened
- Help debug routine issues
- Provide examples and best practices

When analyzing captures:
- Use file_search to find relevant transactions
- Explain the request/response flow
- Identify authentication patterns, API calls, and data flow

When explaining routines:
- Reference the documentation for accurate syntax
- Provide concrete examples
- Explain placeholder quoting rules (escape-quoted for strings: \\"{{param}}\\")

When asked to create or validate a routine:
- Use the validate_routine tool - pass the FULL routine JSON object as the "routine" parameter
- You MUST include: name, description, operations array (parameters optional)
- **VALIDATION IS CRITICAL** - If validation fails, READ THE ERROR CAREFULLY, fix the issues, and call validate_routine again
- **RETRY 2-4 TIMES** until validation succeeds - don't give up after one failure!
- Common fixes: escape-quoted strings `"\"{{param}}\""`, missing required fields, invalid operation types

When asked to execute a routine:
- First validate it with validate_routine (no permission needed for validation)
- Then use execute_routine with the routine and parameters
- The user will be asked for permission ONLY for execution, not validation

IMPORTANT: Be concise. Short answers. No fluff. But ALWAYS verify information is correct before responding - accuracy over speed."""

DOCS_INDEX_HEADER = """

## Documentation Index

Use file_search to access these docs:
"""

# Tool definition for validate_routine
VALIDATE_ROUTINE_TOOL = {
    "type": "function",
    "name": "validate_routine",
    "description": """Validate a routine JSON object. Pass the FULL routine object with all fields.

REQUIRED: You MUST pass the complete routine object, not just describe it. Example call:
{
  "routine": {
    "name": "my_routine",
    "description": "Does something",
    "parameters": [],
    "operations": [{"type": "navigate", "url": "https://example.com"}]
  }
}

Returns success=true with validated routine, or success=false with error details.""",
    "parameters": {
        "type": "object",
        "properties": {
            "routine": {
                "type": "object",
                "description": "The COMPLETE routine object with: name (string), description (string), operations (array), and optionally parameters (array)"
            }
        },
        "required": ["routine"]
    }
}

# Tool definition for execute_routine
EXECUTE_ROUTINE_TOOL = {
    "type": "function",
    "name": "execute_routine",
    "description": """Execute a routine against Chrome browser (must be running on port 9222).

Pass the complete routine object and any required parameters.

Example call:
{
  "routine": {
    "name": "my_routine",
    "description": "Does something",
    "parameters": [{"name": "query", "type": "string", "description": "Search term"}],
    "operations": [...]
  },
  "parameters": {"query": "test"}
}

Returns execution result with data, or error if execution fails.
NOTE: User will be prompted for permission before execution.""",
    "parameters": {
        "type": "object",
        "properties": {
            "routine": {
                "type": "object",
                "description": "The complete routine object to execute"
            },
            "parameters": {
                "type": "object",
                "description": "Parameter values for the routine (key-value pairs)"
            }
        },
        "required": ["routine"]
    }
}

# Tools that require user permission before execution
# Note: validate_routine does NOT require permission - only execute_routine does
TOOLS_REQUIRING_PERMISSION = {"execute_routine"}


def validate_routine(routine_dict: dict) -> dict:
    """
    Validate a routine by attempting to instantiate it.

    Args:
        routine_dict: The routine as a dictionary

    Returns:
        Dict with 'success' bool, and either 'routine' (JSON) or 'error' (string)
    """
    try:
        routine = Routine.model_validate(routine_dict)
        return {
            "success": True,
            "routine": routine.model_dump(),
            "message": "Routine is valid!"
        }
    except ValidationError as e:
        return {
            "success": False,
            "error": str(e),
            "message": "Routine validation failed"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "Unexpected error during validation"
        }


def execute_routine(
    routine_dict: dict,
    parameters: dict | None = None,
    remote_debugging_address: str = "http://127.0.0.1:9222",
) -> dict:
    """
    Execute a routine and return the result.

    Args:
        routine_dict: The routine as a dictionary
        parameters: Parameter values for execution
        remote_debugging_address: Chrome debugging address

    Returns:
        The execution result as a dict
    """
    try:
        routine = Routine.model_validate(routine_dict)
        result: RoutineExecutionResult = routine.execute(
            parameters_dict=parameters or {},
            remote_debugging_address=remote_debugging_address,
            timeout=180.0,
            close_tab_when_done=True,
        )
        return result.model_dump()

    except ValidationError as e:
        return {"success": False, "error": str(e)}
    except Exception as e:
        return {"success": False, "error": str(e)}


class ChatAgent(BaseModel):
    """
    Interactive chat agent for Q&A about routines and browser captures.
    """

    client: OpenAI
    context_manager: LocalDiscoveryDataStoreV2
    llm_model: str = Field(default="gpt-5.1")
    llm_context: LLMContextManagerV3 | None = Field(default=None)
    output_dir: str | None = Field(default=None)
    tools: list[dict] = Field(default_factory=list)
    initialized: bool = Field(default=False)

    # Permission callback: (tool_name, tool_args) -> bool
    # If None, tools execute without permission
    # If returns False, tool execution is denied
    permission_callback: Callable[[str, dict], bool] | None = Field(default=None, exclude=True)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context) -> None:
        """Initialize llm_context with summaries_dir if output_dir is set."""
        if self.llm_context is None:
            summaries_dir = Path(self.output_dir) / "chat_summaries" if self.output_dir else None
            # Chat agent needs larger context for docs + captures
            self.llm_context = LLMContextManagerV3(
                summaries_dir=summaries_dir,
                T_max=400_000,
                T_target=200_000,
                T_summary_max=50_000,
                checkpoint_interval=50_000,
            )

    def initialize(self) -> None:
        """Initialize the chat session."""
        if self.initialized:
            return

        # Ensure vectorstore exists (this also generates docs_index)
        if self.context_manager.vectorstore_id is None:
            logger.info("Creating vectorstore with agent docs and captures...")
            self.context_manager.make_vectorstore()

        # Set up tools
        self.tools = [
            {
                "type": "file_search",
                "vector_store_ids": [self.context_manager.vectorstore_id],
            },
            VALIDATE_ROUTINE_TOOL,
            EXECUTE_ROUTINE_TOOL,
        ]

        # Build system prompt with docs index
        system_prompt = SYSTEM_PROMPT_BASE
        if self.context_manager.docs_index:
            system_prompt += DOCS_INDEX_HEADER + self.context_manager.docs_index

        # Start LLM session
        self.llm_context.start_session(system_prompt)
        self.initialized = True

        logger.info("Chat agent initialized")

    def _handle_tool_call(self, tool_call: dict) -> dict:
        """
        Handle a single tool call and return the result.

        Returns:
            Dict with 'output' (string) and optionally 'file_id' for uploaded files
        """
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("arguments", "{}")

        logger.debug(f"Tool call raw arguments: {tool_args[:500] if isinstance(tool_args, str) else tool_args}")

        if isinstance(tool_args, str):
            tool_args = json.loads(tool_args)

        # Check permission for tools that require it
        if tool_name in TOOLS_REQUIRING_PERMISSION:
            if self.permission_callback is not None:
                if not self.permission_callback(tool_name, tool_args):
                    return {"output": json.dumps({
                        "success": False,
                        "error": "Permission denied by user",
                        "message": f"User denied permission to execute {tool_name}"
                    })}

        if tool_name == "validate_routine":
            routine_dict = tool_args.get("routine", {})
            logger.info(f"validate_routine called with keys: {list(routine_dict.keys()) if routine_dict else 'EMPTY'}")
            result = validate_routine(routine_dict)
            return {"output": json.dumps(result, indent=2)}

        if tool_name == "execute_routine":
            routine_dict = tool_args.get("routine", {})
            parameters = tool_args.get("parameters", {})

            result = execute_routine(routine_dict, parameters)

            # Save routine, parameters, and result to local file
            file_path = self._save_execution_result(
                result=result,
                routine_dict=routine_dict,
                parameters=parameters,
                routine_name=routine_dict.get("name", "routine"),
            )
            print(f"\n📄 Execution result saved to: {file_path}\n")

            # Upload to OpenAI so LLM can read full result
            file_id = self._upload_file_to_openai(file_path)

            # Build a summary for the tool output
            summary = self._build_execution_summary(result, file_path, file_id)
            return {"output": json.dumps(summary, indent=2, default=str), "file_id": file_id}

        return {"output": json.dumps({"error": f"Unknown tool: {tool_name}"})}

    def _upload_file_to_openai(self, file_path: str) -> str | None:
        """Upload a file to OpenAI and return the file_id."""
        try:
            with open(file_path, "rb") as f:
                uploaded = self.client.files.create(file=f, purpose="user_data")
            logger.info(f"Uploaded execution result to OpenAI: {uploaded.id}")
            return uploaded.id
        except Exception as e:
            logger.error(f"Failed to upload file to OpenAI: {e}")
            return None

    def _build_execution_summary(self, result: dict, file_path: str, file_id: str | None = None) -> dict:
        """
        Build a focused summary of execution results that highlights issues.

        Returns a dict with:
        - success status
        - DATA PREVIEW with warnings if empty/problematic
        - any errors from operations
        - console logs from js_evaluate (helpful for debugging)
        - reference to file for full details
        """
        data = result.get("data")
        ops_metadata = result.get("operations_metadata", [])

        # Check data status
        data_status = "OK"
        data_preview = data

        if data is None:
            data_status = "⚠️ WARNING: data is NULL"
        elif data == {}:
            data_status = "⚠️ WARNING: data is EMPTY OBJECT {}"
        elif data == []:
            data_status = "⚠️ WARNING: data is EMPTY ARRAY []"
        elif isinstance(data, dict) and len(data) == 0:
            data_status = "⚠️ WARNING: data is EMPTY OBJECT {}"
        elif isinstance(data, list) and len(data) == 0:
            data_status = "⚠️ WARNING: data is EMPTY ARRAY []"
        elif isinstance(data, str) and len(data) > 500:
            data_preview = data[:500] + f"... (truncated, {len(data)} chars total)"
        elif isinstance(data, (dict, list)):
            # Show preview of complex data
            data_str = json.dumps(data, indent=2, default=str)
            if len(data_str) > 1000:
                data_preview = f"(large data - {len(data_str)} chars, see file for full data)"
                if isinstance(data, list):
                    data_preview = f"Array with {len(data)} items. First item: {json.dumps(data[0], default=str)[:300] if data else 'N/A'}..."
                elif isinstance(data, dict):
                    data_preview = f"Object with keys: {list(data.keys())}"

        # Collect operation errors
        op_errors = []
        console_logs = []
        for op in ops_metadata:
            if op.get("error"):
                op_errors.append(f"{op.get('type')}: {op.get('error')}")
            # Get console logs from js_evaluate
            if op.get("type") == "js_evaluate":
                logs = op.get("details", {}).get("console_logs", [])
                if logs:
                    console_logs.extend(logs)
                exec_err = op.get("details", {}).get("execution_error")
                if exec_err:
                    op_errors.append(f"js_evaluate execution_error: {exec_err}")

        summary = {
            "success": result.get("success", False),
            "data_status": data_status,
            "data_preview": data_preview,
        }

        if op_errors:
            summary["operation_errors"] = op_errors

        if console_logs:
            summary["js_console_logs"] = console_logs

        if result.get("warnings"):
            summary["warnings"] = result.get("warnings")

        summary["full_details_file"] = file_path
        if file_id:
            summary["file_id"] = file_id
            summary["note"] = "FULL RESULT FILE ATTACHED. Review the data in the attached file. If data is empty or wrong, fix the routine."
        else:
            summary["note"] = "REVIEW THE DATA! If data is empty or wrong, check console_logs and fix the routine."

        return summary

    def _save_execution_result(
        self,
        result: dict,
        routine_dict: dict,
        parameters: dict,
        routine_name: str,
    ) -> str:
        """Save execution result with routine and parameters to a local file.

        Args:
            result: The execution result dict
            routine_dict: The routine that was executed
            parameters: The parameters passed to the routine
            routine_name: Name of the routine (for filename)

        Returns:
            Path to the saved file
        """
        from datetime import datetime

        # Determine output directory
        if self.output_dir:
            results_dir = Path(self.output_dir) / "execution_results"
        else:
            results_dir = Path.cwd() / "execution_results"

        results_dir.mkdir(parents=True, exist_ok=True)

        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in routine_name)
        file_path = results_dir / f"{safe_name}_{timestamp}.json"

        # Combine routine, parameters, and result into one object
        combined = {
            "routine": routine_dict,
            "parameters": parameters,
            "result": result,
        }

        # Write combined result
        file_path.write_text(json.dumps(combined, indent=2, default=str), encoding="utf-8")

        return str(file_path)

    def chat(self, user_message: str) -> str:
        """
        Send a message and get a response.

        Args:
            user_message: The user's question or message

        Returns:
            The assistant's response
        """
        if not self.initialized:
            self.initialize()

        # Add user message to context
        self.llm_context.add_user_message(user_message)

        # Get LLM input
        llm_input, previous_response_id = self.llm_context.get_llm_input()

        # Call LLM (loop to handle tool calls)
        while True:
            response = self.client.responses.create(
                model=self.llm_model,
                input=llm_input,
                previous_response_id=previous_response_id,
                tools=self.tools,
                tool_choice="auto",
            )

            # Check for function calls in output
            function_calls = [
                item for item in response.output
                if item.type == "function_call"
            ]

            if not function_calls:
                # No tool calls, we're done
                break

            # Process each function call
            tool_results = []
            file_ids = []
            for fc in function_calls:
                result = self._handle_tool_call({
                    "name": fc.name,
                    "arguments": fc.arguments,
                })
                tool_results.append({
                    "type": "function_call_output",
                    "call_id": fc.call_id,
                    "output": result["output"],
                })
                # Collect file_ids to include in next input
                if result.get("file_id"):
                    file_ids.append(result["file_id"])
                logger.debug(f"Tool call: {fc.name} completed ({len(result['output'])} chars)")

            # Continue conversation with tool results
            llm_input = tool_results

            # If we have files, add them as a user message so LLM can read them
            if file_ids:
                file_content = [{"type": "input_file", "file_id": fid} for fid in file_ids]
                file_content.append({
                    "type": "input_text",
                    "text": "Here is the full execution result file. Analyze the data and determine if the routine worked correctly."
                })
                llm_input.append({
                    "role": "user",
                    "content": file_content,
                })

            previous_response_id = response.id

        # Extract text from response
        response_text = response.output_text or ""

        # Add assistant response to context
        self.llm_context.add_assistant_message(response_text, response.id)

        logger.debug(f"Chat stats: {self.llm_context.get_stats()}")

        return response_text

    def chat_stream(self, user_message: str) -> Generator[str, None, None]:
        """
        Send a message and stream the response.

        Args:
            user_message: The user's question or message

        Yields:
            Chunks of the assistant's response
        """
        if not self.initialized:
            self.initialize()

        # Add user message to context
        self.llm_context.add_user_message(user_message)

        # Get LLM input
        llm_input, previous_response_id = self.llm_context.get_llm_input()

        # Call LLM with streaming
        full_response = ""
        response_id = None

        with self.client.responses.create(
            model=self.llm_model,
            input=llm_input,
            previous_response_id=previous_response_id,
            tools=self.tools,
            tool_choice="auto",
            stream=True,
        ) as stream:
            for event in stream:
                if hasattr(event, 'type'):
                    if event.type == 'response.output_text.delta':
                        chunk = event.delta
                        full_response += chunk
                        yield chunk
                    elif event.type == 'response.completed':
                        response_id = event.response.id

        # Add assistant response to context
        if response_id:
            self.llm_context.add_assistant_message(full_response, response_id)

        logger.debug(f"Chat stats: {self.llm_context.get_stats()}")

    def get_stats(self) -> dict:
        """Get current context stats."""
        return self.llm_context.get_stats() if self.llm_context else {}

    def cleanup(self) -> None:
        """Clean up resources."""
        if self.context_manager.vectorstore_id:
            try:
                self.context_manager.clean_up()
            except Exception as e:
                logger.warning(f"Failed to cleanup vectorstore: {e}")


def create_chat_agent(
    cdp_captures_dir: str,
    openai_api_key: str | None = None,
    llm_model: str = "gpt-5.1",
    output_dir: str | None = None,
) -> ChatAgent:
    """
    Factory function to create a ChatAgent from CDP captures directory.

    Args:
        cdp_captures_dir: Path to the CDP captures directory
        openai_api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
        llm_model: LLM model to use
        output_dir: Optional output directory for summaries

    Returns:
        Configured ChatAgent instance
    """
    client = OpenAI(api_key=openai_api_key or os.getenv("OPENAI_API_KEY"))

    # Build paths from captures dir
    cdp_path = Path(cdp_captures_dir)

    context_manager = LocalDiscoveryDataStoreV2(
        client=client,
        tmp_dir=str(cdp_path / "tmp"),
        transactions_dir=str(cdp_path / "network" / "transactions"),
        consolidated_transactions_path=str(cdp_path / "network" / "consolidated_transactions.json"),
        storage_jsonl_path=str(cdp_path / "storage" / "events.jsonl"),
        window_properties_path=str(cdp_path / "window_properties" / "window_properties.json"),
    )

    return ChatAgent(
        client=client,
        context_manager=context_manager,
        llm_model=llm_model,
        output_dir=output_dir,
    )
