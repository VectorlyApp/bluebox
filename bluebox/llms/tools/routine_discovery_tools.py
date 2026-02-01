"""
bluebox/llms/tools/routine_discovery_tools.py

Tool definitions and helper functions for the LLM-driven routine discovery agent.
"""

from typing import Any


# Tool definitions for registration with LLMClient
# Each tool has: name, description, parameters (JSON schema)

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    # === Data Access Tools ===
    {
        "name": "list_transactions",
        "description": (
            "Get a list of all available network transaction IDs from the captured CDP data. "
            "Use this at the start to understand what transactions are available for analysis."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_transaction",
        "description": (
            "Get the full details of a specific network transaction including request "
            "(URL, method, headers, body) and response (status, headers, body). "
            "Use this to examine a transaction in detail after identifying it."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "transaction_id": {
                    "type": "string",
                    "description": "The ID of the transaction to retrieve",
                },
            },
            "required": ["transaction_id"],
        },
    },
    {
        "name": "scan_for_value",
        "description": (
            "Scan storage (cookies, localStorage, sessionStorage), window properties, "
            "and prior transaction responses for a specific value. Use this to find where "
            "a dynamic token or variable originates. Returns sources where the value was found."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "value": {
                    "type": "string",
                    "description": "The exact value to search for (e.g., a token, cookie value, or dynamic ID)",
                },
                "before_transaction_id": {
                    "type": "string",
                    "description": "Only search in transactions that occurred before this transaction",
                },
            },
            "required": ["value"],
        },
    },

    # === Queue Management Tools ===
    {
        "name": "add_transaction_to_queue",
        "description": (
            "Add a transaction to the processing queue. Use this when you discover that "
            "a variable's source is another transaction that needs to be processed to build "
            "the dependency chain."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "transaction_id": {
                    "type": "string",
                    "description": "The transaction ID to add to the queue",
                },
                "reason": {
                    "type": "string",
                    "description": "Why this transaction is being added (e.g., 'Source for auth_token variable')",
                },
            },
            "required": ["transaction_id", "reason"],
        },
    },
    {
        "name": "get_queue_status",
        "description": (
            "Get the current status of the transaction processing queue. "
            "Shows pending transactions, processed transactions, and current state."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "mark_transaction_complete",
        "description": (
            "Mark the current transaction as fully processed. Call this after you have "
            "extracted all variables and resolved their sources for the current transaction."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "transaction_id": {
                    "type": "string",
                    "description": "The transaction ID that has been fully processed",
                },
            },
            "required": ["transaction_id"],
        },
    },

    # === State Recording Tools ===
    {
        "name": "record_identified_transaction",
        "description": (
            "Record the identified root transaction that corresponds to the user's task. "
            "This is the main transaction that accomplishes the user's intent. "
            "After recording, the transaction will be added to the processing queue."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "transaction_id": {
                    "type": "string",
                    "description": "The ID of the root transaction",
                },
                "description": {
                    "type": "string",
                    "description": "Brief description of what this transaction does",
                },
                "url": {
                    "type": "string",
                    "description": "The URL of the transaction",
                },
                "method": {
                    "type": "string",
                    "enum": ["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"],
                    "description": "HTTP method",
                },
            },
            "required": ["transaction_id", "description", "url", "method"],
        },
    },
    {
        "name": "record_extracted_variables",
        "description": (
            "Record extracted variables from a transaction's request. Variables can be:\n"
            "- PARAMETER: User input (search_query, item_id) - values the user provides\n"
            "- DYNAMIC_TOKEN: Auth/session values (CSRF, JWT, session_id) - need resolution\n"
            "- STATIC_VALUE: Constants (app version, User-Agent) - can be hardcoded\n\n"
            "Set requires_dynamic_resolution=true for DYNAMIC_TOKEN variables."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "transaction_id": {
                    "type": "string",
                    "description": "The transaction ID these variables belong to",
                },
                "variables": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {
                                "type": "string",
                                "enum": ["parameter", "dynamic_token", "static_value"],
                            },
                            "name": {"type": "string"},
                            "observed_value": {"type": "string"},
                            "requires_dynamic_resolution": {"type": "boolean"},
                            "values_to_scan_for": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Values to search for when resolving this variable",
                            },
                        },
                        "required": ["type", "name", "observed_value", "requires_dynamic_resolution"],
                    },
                    "description": "List of extracted variables",
                },
            },
            "required": ["transaction_id", "variables"],
        },
    },
    {
        "name": "record_resolved_variable",
        "description": (
            "Record the resolved source for a dynamic variable. Specify where the value originates:\n"
            "- storage: From cookie, localStorage, or sessionStorage\n"
            "- window_property: From JavaScript window object\n"
            "- transaction: From a prior transaction's response body\n"
            "- hardcode: Cannot be resolved, hardcode the observed value\n\n"
            "If source_type is 'transaction', the source transaction will be auto-added to the queue.\n\n"
            "IMPORTANT: If scan_for_value found the value in BOTH storage AND a prior transaction,\n"
            "always provide alternative_transaction_source even when using storage as the primary source.\n"
            "This ensures the routine fetches fresh values via network rather than relying on session storage."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "variable_name": {
                    "type": "string",
                    "description": "Name of the variable being resolved",
                },
                "transaction_id": {
                    "type": "string",
                    "description": "The transaction this variable belongs to",
                },
                "source_type": {
                    "type": "string",
                    "enum": ["storage", "window_property", "transaction", "hardcode"],
                    "description": "Type of source where the value originates",
                },
                "storage_source": {
                    "type": "object",
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": ["cookie", "localStorage", "sessionStorage"],
                        },
                        "dot_path": {"type": "string"},
                    },
                    "description": "For storage sources: type and dot path",
                },
                "window_property_source": {
                    "type": "object",
                    "properties": {
                        "dot_path": {"type": "string"},
                    },
                    "description": "For window property sources: the dot path",
                },
                "transaction_source": {
                    "type": "object",
                    "properties": {
                        "transaction_id": {"type": "string"},
                        "dot_path": {"type": "string"},
                    },
                    "description": "For transaction sources: source transaction ID and path in response",
                },
                "alternative_transaction_source": {
                    "type": "object",
                    "properties": {
                        "transaction_id": {"type": "string"},
                        "dot_path": {"type": "string"},
                    },
                    "description": "When value is found in both storage AND a prior transaction, provide the transaction source here. The transaction will be added to the queue to ensure fresh value retrieval.",
                },
            },
            "required": ["variable_name", "transaction_id", "source_type"],
        },
    },

    # === Validation Tools ===
    {
        "name": "execute_routine",
        "description": (
            "Execute the constructed routine to validate it works correctly. "
            "Use this AFTER construct_routine succeeds to test the routine with real parameters. "
            "Returns detailed diagnostics on any failures:\n"
            "- failed_placeholders: List of placeholders that couldn't be resolved (e.g., storage keys not found)\n"
            "- operation_errors: List of (operation_type, error_message) for operations that failed\n"
            "- warnings: Non-fatal issues encountered\n\n"
            "If execution fails due to placeholder resolution, re-examine where those values should come from "
            "using scan_for_value, update the routine, and try again."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "parameters": {
                    "type": "object",
                    "description": "Parameters to pass to the routine (key-value pairs matching routine parameter definitions)",
                    "additionalProperties": True,
                },
            },
            "required": ["parameters"],
        },
    },

    # === Construction Tools ===
    {
        "name": "construct_routine",
        "description": (
            "Construct and finalize the routine from all processed transactions and resolved variables. "
            "Call this after all transactions in the queue have been processed. "
            "On success, the routine is automatically converted to production format.\n\n"
            "PLACEHOLDER SYNTAX:\n"
            "- Parameters: {{param_name}} (no prefix)\n"
            "- Storage: {{cookie:name}}, {{localStorage:key}}, {{sessionStorage:path.to.value}}\n"
            "- Window: {{windowProperty:obj.key}}\n\n"
            "JSON QUOTE RULES:\n"
            "- String values: \"key\": \\\"{{param}}\\\" (escaped quotes)\n"
            "- Numbers/booleans: \"key\": \"{{param}}\" (regular quotes, stripped at runtime)\n\n"
            "OPERATION STRUCTURE:\n"
            "1. First: navigate operation to target page\n"
            "2. Middle: sleep (2-3s), then fetch operations\n"
            "3. Last: return operation with sessionStorage key"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name for the routine (snake_case preferred)",
                },
                "description": {
                    "type": "string",
                    "description": "Description of what the routine does",
                },
                "parameters": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "description": {"type": "string"},
                            "type": {
                                "type": "string",
                                "enum": ["string", "integer", "number", "boolean"],
                            },
                            "required": {"type": "boolean"},
                        },
                        "required": ["name", "description", "type", "required"],
                    },
                    "description": "User-facing parameters for the routine",
                },
                "operations": {
                    "type": "array",
                    "items": {
                        "oneOf": [
                            {
                                "type": "object",
                                "description": "Navigate to a URL",
                                "properties": {
                                    "type": {"type": "string", "enum": ["navigate"]},
                                    "url": {"type": "string", "description": "URL to navigate to"},
                                },
                                "required": ["type", "url"],
                            },
                            {
                                "type": "object",
                                "description": "Sleep for a duration",
                                "properties": {
                                    "type": {"type": "string", "enum": ["sleep"]},
                                    "timeout_seconds": {"type": "number", "description": "Seconds to sleep (typically 2-3)"},
                                },
                                "required": ["type", "timeout_seconds"],
                            },
                            {
                                "type": "object",
                                "description": "Fetch from an endpoint and store result",
                                "properties": {
                                    "type": {"type": "string", "enum": ["fetch"]},
                                    "endpoint": {
                                        "type": "object",
                                        "properties": {
                                            "url": {"type": "string"},
                                            "method": {"type": "string", "enum": ["GET", "POST", "PUT", "DELETE", "PATCH"]},
                                            "headers": {"type": "string", "description": "JSON string of headers"},
                                            "body": {"type": "string", "description": "JSON string of request body"},
                                            "credentials": {"type": "string", "enum": ["same-origin", "include", "omit"], "default": "same-origin"},
                                        },
                                        "required": ["url", "method", "headers", "body"],
                                    },
                                    "session_storage_key": {"type": "string", "description": "Key to store response in sessionStorage"},
                                },
                                "required": ["type", "endpoint", "session_storage_key"],
                            },
                            {
                                "type": "object",
                                "description": "Return value from sessionStorage",
                                "properties": {
                                    "type": {"type": "string", "enum": ["return"]},
                                    "session_storage_key": {"type": "string", "description": "Key to return from sessionStorage (must match last fetch's key)"},
                                },
                                "required": ["type", "session_storage_key"],
                            },
                        ],
                    },
                    "description": "List of operations in execution order. Structure: 1) navigate to page, 2) sleep 2-3s, 3) fetch operations, 4) return",
                },
            },
            "required": ["name", "description", "parameters", "operations"],
        },
    },
]


def get_tool_by_name(name: str) -> dict[str, Any] | None:
    """Get a tool definition by name."""
    for tool in TOOL_DEFINITIONS:
        if tool["name"] == name:
            return tool
    return None


def get_all_tool_names() -> list[str]:
    """Get a list of all tool names."""
    return [tool["name"] for tool in TOOL_DEFINITIONS]
