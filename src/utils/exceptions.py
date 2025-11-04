"""
src/utils/exceptions.py

Custom exceptions for the project.
"""

class ApiKeyNotFoundError(Exception):
    """
    Raised when an API key is not found in the environment variables.
    """


class LLMStructuredOutputError(Exception):
    """
    Exception raised when LLM structured output parsing fails.
    """


class TransactionIdentificationFailedError(Exception):
    """
    Exception raised when the agent fails to identify a network transaction
    that corresponds to the user's requested task after exhausting all attempts.
    """
