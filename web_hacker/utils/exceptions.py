"""
web_hacker/utils/exceptions.py

Custom exceptions for the project.
"""

class UnsupportedFileFormat(Exception):
    """
    Raised when encountering an unsupported file type for some opertation.
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


class BrowserConnectionError(Exception):
    """
    Exception raised when unable to connect to the browser or create a browser tab.
    """


class RoutineExecutionError(Exception):
    """
    Exception raised when routine execution fails.
    """


class WebHackerError(Exception):
    """
    Base exception for all Web Hacker errors.
    """
