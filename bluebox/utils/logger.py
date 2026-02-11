"""
bluebox/utils/logger.py

Centralized logging configuration for the project.
Provides a default logger and factory function for creating additional loggers.
All loggers are configured to work with ECS/CloudWatch logging.
"""

import io
import logging
import os
import sys

from textual.logging import TextualHandler

from bluebox.config import Config


# When True, _configure_logger uses the TUI-safe handler instead of StreamHandler.
_tui_mode: bool = False

# Original stderr, saved before redirect so file logging still works.
_original_stderr: io.TextIOWrapper | None = None


# Private functions _______________________________________________________________________________

def _create_handler() -> logging.StreamHandler:
    """
    Create and configure a StreamHandler for stdout/stderr logging.
    This ensures logs are captured by ECS and sent to CloudWatch.
    Returns:
        logging.StreamHandler: Configured handler
    """
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt=Config.LOG_FORMAT,
        datefmt=Config.LOG_DATE_FORMAT
    )
    handler.setFormatter(fmt=formatter)
    return handler


def _create_tui_handler() -> logging.Handler:
    """Create a TextualHandler that routes logs through Textual instead of stderr."""
    from textual.logging import TextualHandler
    handler = TextualHandler()
    handler.setFormatter(logging.Formatter(
        fmt=Config.LOG_FORMAT,
        datefmt=Config.LOG_DATE_FORMAT
    ))
    return handler


def _strip_stream_handlers(logger: logging.Logger) -> None:
    """Remove all StreamHandlers (except FileHandlers) from a logger."""
    logger.handlers = [
        h for h in logger.handlers
        if not isinstance(h, logging.StreamHandler) or isinstance(h, logging.FileHandler)
    ]


def _configure_logger(logger: logging.Logger) -> logging.Logger:
    """
    Configure a logger with proper settings for ECS/CloudWatch.
    Args:
        logger (logging.Logger): The logger to configure
    Returns:
        logging.Logger: The configured logger
    """
    # set log level
    logger.setLevel(Config.LOG_LEVEL)

    if _tui_mode:
        # Strip any existing StreamHandlers that would write to stderr/stdout
        _strip_stream_handlers(logger)
        if not logger.handlers:
            logger.addHandler(_create_tui_handler())
        logger.propagate = False
        return logger

    # prevent duplicate handlers
    if not logger.handlers:
        handler = _create_handler()
        logger.addHandler(handler)

    # force handler format consistency even if caplog interferes
    for handler in logger.handlers:
        handler.setFormatter(
            fmt=logging.Formatter(
                fmt=Config.LOG_FORMAT,
                datefmt=Config.LOG_DATE_FORMAT
            )
        )

    # prevent propagation to avoid duplicate logs
    logger.propagate = False
    return logger


# Exports _________________________________________________________________________________________

def get_logger(name: str) -> logging.Logger:
    """
    Get a configured logger by name.
    Args:
        name (str): Logger name.
    Returns:
        logging.Logger: Configured logger instance
    """
    logger_name = name
    logger = logging.getLogger(name=logger_name)
    return _configure_logger(logger)


def enable_tui_logging(log_file: str | None = None, quiet: bool = False) -> None:
    """
    Switch all logging to TUI-safe mode and redirect stderr to prevent
    any rogue writes from corrupting the Textual display.

    This does three things:
    1. Replaces StreamHandlers on ALL existing loggers with TextualHandler
    2. Redirects sys.stderr to devnull (or a log file) so any print() or
       direct stderr writes from third-party libs can't corrupt the TUI
    3. Sets _tui_mode so future get_logger() calls also use TextualHandler

    Call this once before launching a Textual app.

    Args:
        log_file: Optional path to also write logs to a file.
        quiet: If True, suppress all logging entirely.
    """
    global _tui_mode, _original_stderr  # noqa: PLW0603
    _tui_mode = True

    # Save original stderr before we redirect it
    _original_stderr = sys.stderr

    if quiet:
        logging.getLogger().setLevel(logging.CRITICAL + 1)
        logging.getLogger("bluebox").setLevel(logging.CRITICAL + 1)
        # Still redirect stderr even in quiet mode
        sys.stderr = open(os.devnull, mode="w")  # noqa: SIM115
        return

    tui_handler = TextualHandler()
    file_handler: logging.FileHandler | None = None
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(
            fmt="[%(asctime)s] %(levelname)s:%(name)s:%(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))

    # Patch EVERY existing logger — not just root and bluebox.
    # This catches httpx, openai, httpcore, urllib3, etc. that already
    # created loggers with StreamHandlers before TUI mode was enabled.
    for name in list(logging.Logger.manager.loggerDict):
        existing = logging.getLogger(name)
        _strip_stream_handlers(existing)
        existing.propagate = True  # let root's TextualHandler catch everything

    # Patch root logger
    root = logging.getLogger()
    _strip_stream_handlers(root)
    root.addHandler(tui_handler)
    if file_handler:
        root.addHandler(file_handler)

    # Bluebox namespace — explicit handler + no propagation to avoid dupes
    bluebox_logger = logging.getLogger("bluebox")
    _strip_stream_handlers(bluebox_logger)
    bluebox_logger.addHandler(tui_handler)
    if file_handler:
        bluebox_logger.addHandler(file_handler)
    bluebox_logger.propagate = False

    # Redirect stderr so any rogue writes (print(..., file=sys.stderr),
    # warnings.warn, C extensions, etc.) go to the log file or devnull
    # instead of corrupting the Textual display.
    if log_file:
        sys.stderr = open(log_file, mode="a")  # noqa: SIM115
    else:
        sys.stderr = open(os.devnull, mode="w")  # noqa: SIM115
