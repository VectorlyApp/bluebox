"""
bluebox/utils/llm_serialization.py

Utilities for controlling what data gets sent to LLMs from tool results.

The LLMExclude marker lets you annotate Pydantic model fields that should be
stripped before a tool result is serialized for the LLM — e.g. large blobs,
internal IDs, or raw data the model doesn't need.

Usage on models::

    from typing import Annotated
    from pydantic import BaseModel
    from bluebox.utils.llm_serialization import LLMExclude

    class NetworkTransaction(BaseModel):
        url: str
        method: str
        response_body: Annotated[str, LLMExclude()]   # stripped before LLM sees it

Tool handlers can return these models (or dicts containing them) directly —
the agent infrastructure calls strip_llm_excluded() automatically.
"""

from __future__ import annotations

import functools
import json
from enum import StrEnum
from typing import Any, NamedTuple

from pydantic import BaseModel


class SerializedContentType(StrEnum):
    """
    Content type of a serialized tool result.

    Attributes:
        JSON: Successfully serialized as JSON.
        TEXT: Fell back to ``str()`` representation.
    """
    JSON = "json"
    TEXT = "text"


class SerializedToolResult(NamedTuple):
    """
    Result of serializing a tool return value for the LLM.

    Attributes:
        serialized: The serialized string (JSON or plain text).
        content_type: How the value was serialized.
    """
    serialized: str
    content_type: SerializedContentType


class LLMExclude:
    """
    Marker: exclude this field from LLM tool results.

    Attach via ``Annotated``::

        name: str                                    # included
        raw_blob: Annotated[bytes, LLMExclude()]     # excluded
    """
    pass


def serialize_tool_result(tool_result: Any) -> SerializedToolResult:
    """
    Serialize a tool result to a JSON or plain-text string for the LLM.

    Attempts JSON serialization first (using ``default=str`` for non-serializable
    types). Falls back to ``str()`` if JSON encoding fails.

    Args:
        tool_result: The value returned by a tool handler (typically a dict).

    Returns:
        A :class:`SerializedToolResult` (also unpacks as a two-tuple).
    """
    try:
        return SerializedToolResult(
            serialized=json.dumps(
                tool_result,
                ensure_ascii=False,
                default=str,
                indent=2
            ),
            content_type=SerializedContentType.JSON,
        )
    except (TypeError, ValueError):
        return SerializedToolResult(
            serialized=str(tool_result),
            content_type=SerializedContentType.TEXT
        )


@functools.lru_cache(maxsize=256)
def _excluded_fields(model_cls: type[BaseModel]) -> frozenset[str]:
    """
    Return the set of field names annotated with LLMExclude for a model class.

    Scans ``model_cls.model_fields`` and checks each field's ``metadata`` list
    for an ``LLMExclude`` instance (attached via ``Annotated[Type, LLMExclude()]``).

    Results are cached per class via ``lru_cache``. Safe because Pydantic field
    definitions are fixed at class creation time.

    Args:
        model_cls: A Pydantic BaseModel subclass to inspect.

    Returns:
        Frozen set of field names that should be excluded from LLM serialization.
        Empty frozenset if the model has no LLMExclude annotations.
    """
    return frozenset(
        name
        for name, info in model_cls.model_fields.items()
        if any(isinstance(m, LLMExclude) for m in info.metadata)
    )


def strip_llm_excluded(obj: Any) -> Any:
    """
    Recursively strip LLMExclude-annotated fields from Pydantic models.

    Walks the object tree and converts any ``BaseModel`` instance into a dict
    with LLMExclude-annotated fields removed. Non-BaseModel values pass through
    unchanged (just an ``isinstance`` check).

    Supported containers (recursed into):
        - ``BaseModel``: fields filtered, remaining values recursed
        - ``dict``: values recursed, keys preserved
        - ``list`` / ``tuple``: elements recursed, container type preserved

    Args:
        obj: Any object — typically a tool handler's return value. Can be a
            BaseModel, dict, list, tuple, or primitive.

    Returns:
        A plain-dict / list / tuple / primitive copy with all LLMExclude fields
        removed from any BaseModel instances found at any nesting depth.
    """
    if isinstance(obj, BaseModel):
        cls = type(obj)
        excluded = _excluded_fields(cls)
        result = {
            name: strip_llm_excluded(value)
            for name in cls.model_fields
            if name not in excluded
            for value in (getattr(obj, name),)  # bind to local for clarity
        }
        # include @computed_field properties (not in model_fields)
        for name in cls.model_computed_fields:
            if name not in excluded:
                result[name] = strip_llm_excluded(getattr(obj, name))
        return result
    if isinstance(obj, dict):
        return {k: strip_llm_excluded(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [strip_llm_excluded(item) for item in obj]
    if isinstance(obj, tuple):
        return tuple(strip_llm_excluded(item) for item in obj)
    return obj
