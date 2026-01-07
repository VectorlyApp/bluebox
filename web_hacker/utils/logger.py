"""
web_hacker/utils/logger.py

Centralized logging configuration for the project.
Provides a default logger and factory function for creating additional loggers.
All loggers are configured to work with ECS/CloudWatch logging.
"""

import logging

from web_hacker.config import Config


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
