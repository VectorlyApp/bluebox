"""
bluebox/utils/exceptions.py

Custom exceptions for bluebox-sdk.

Contains:
- ApiKeyNotFoundError: Missing API key
- BrowserConnectionError, ChromiumConnectionError: CDP connection failures
- NavigationError, NavigationBlockedError: Page navigation failures
- RoutineExecutionError: Routine run failures
- HTTPClientError, HTTPServerError: HTTP request failures
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


class ChromiumConnectionError(Exception):
    """
    Exception raised when there is an error connecting to the Chromium service.
    """


class NoSessionIdError(Exception):
    """
    Exception raised when a method requires a session ID but none is available.
    """


class HTTPClientError(Exception):
    """
    Exception raised when an HTTP request returns a 4xx client error status code.
    """


class HTTPServerError(Exception):
    """
    Exception raised when an HTTP request returns a 5xx server error status code.
    """


class NavigationError(Exception):
    """
    Raised when a page navigation appears to have failed or produced an invalid page.
    """


class NavigationBlockedError(NavigationError):
    """
    Raised when a page navigation appears to have been blocked by anti-bot or related mechanisms.
    """


class RoutineExecutionError(Exception):
    """
    Exception raised when routine execution fails.
    """


class BlueboxError(Exception):
    """
    Base exception for all Bluebox errors.
    """


class UnknownToolError(Exception):
    """
    Raised when attempting to execute a tool that does not exist.
    """
