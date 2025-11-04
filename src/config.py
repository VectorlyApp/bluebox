"""
src/config.py

Centralized environment variable configuration.
"""

import os
from typing import Any

from dotenv import load_dotenv

load_dotenv()


class Config():
    """
    Centralized configuration for environment variables.
    """

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
