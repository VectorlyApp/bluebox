"""
web_hacker/config.py

Environment variable configuration.

Contains:
- Config: Centralized settings from environment variables
- OPENAI_API_KEY, LOG_LEVEL, DEFAULT_MODEL, etc.
"""

import logging
import os
from typing import Any

from dotenv import load_dotenv

load_dotenv()

# configure httpx logger to suppress verbose HTTP logs
logging.getLogger("httpx").setLevel(logging.WARNING)


class Config():
    """
    Centralized configuration for environment variables.
    """

    # logging configuration
    LOG_LEVEL: int = logging.getLevelNamesMapping().get(
        os.getenv("LOG_LEVEL", "INFO").upper(),
        logging.INFO
    )
    LOG_DATE_FORMAT: str = os.getenv("LOG_DATE_FORMAT", "%Y-%m-%d %H:%M:%S")
    LOG_FORMAT: str = os.getenv("LOG_FORMAT", "[%(asctime)s] %(levelname)s:%(name)s:%(message)s")

    # API keys
    OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")

    @classmethod
    def as_dict(cls) -> dict[str, Any]:
        """
        Return a dictionary of all UPPERCASE class attributes and their values.
        Return:
            dict[str, str | None]: A dictionary of all UPPERCASE class attributes and their values.
        """
        return {
            key: getattr(cls, key)
            for key in dir(cls)
            if key.isupper()
        }
