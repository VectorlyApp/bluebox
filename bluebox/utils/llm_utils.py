"""
bluebox/utils/llm_utils.py

LLM API utilities for structured outputs.

Contains:
- Functions for calling OpenAI with structured Pydantic outputs
- Response parsing and validation helpers
"""

import json
from functools import wraps
from typing import Any, Callable, Type

from openai import OpenAI
from openai.types.responses import Response
from pydantic import BaseModel
from toon import encode

from bluebox.config import Config
from bluebox.utils.exceptions import LLMStructuredOutputError
from bluebox.utils.logger import get_logger

logger = get_logger(__name__)


def token_optimized(func: Callable[..., dict[str, Any]]) -> Callable[..., str]:
    """Decorator that encodes dict outputs with toon for token efficiency.

    Use this decorator on tool functions that return dict outputs where
    structural integrity is not critical. The output will be encoded
    using toon for reduced token usage.

    Args:
        func: A function that returns dict[str, Any]

    Returns:
        A wrapped function that returns the toon-encoded string
    """
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> str:
        result = func(*args, **kwargs)
        return encode(result)
    return wrapper


def manual_llm_parse_text_to_model(
    text: str,
    pydantic_model: Type[BaseModel],
    client: OpenAI,
    context: str | None = None,
    llm_model: str = "gpt-5-mini",
    n_tries: int = 3,
) -> BaseModel:
    """
    Manual LLM parse text to model (without using structured output).
    Args:
        text (str): The text to parse.
        context (str): The context to use for the parsing (stringified message history between user and assistant).
        pydantic_model (Type[BaseModel]): The pydantic model to parse the text to.
        client (OpenAI): The OpenAI client to use.
        llm_model (str): The LLM model to use.
        n_tries (int): The number of tries to parse the text.
    Returns:
        BaseModel: The parsed pydantic model.
    """
    # define system prompt
    SYSTEM_PROMPT = f"""
    You are a helpful assistant that extracts information and structures it into a JSON object.
    You must output ONLY the valid JSON object that matches the provided schema.
    Do not include any explanations, markdown formatting, or code blocks.
    """

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}"},
        {"role": "user", "content": f"Text to parse: {text}"},
        {"role": "user", "content": f"Target Model Schema (TOON format):\n{encode(pydantic_model.model_json_schema())}"},
        {"role": "user", "content": "Extract the data and return a JSON object that validates against the schema above."}
    ]

    for current_try in range(n_tries):
        
        try:
            response = client.chat.completions.create(
                model=llm_model,
                messages=messages,
                response_format={"type": "json_object"},
            )
            
            response_content = response.choices[0].message.content
            messages.append({"role": "assistant", "content": response_content})
            
            # Basic cleanup to ensure we just get the JSON
            clean_content = response_content.strip()
            if clean_content.startswith("```json"):
                clean_content = clean_content[7:]
            if clean_content.startswith("```"):
                clean_content = clean_content[3:]
            if clean_content.endswith("```"):
                clean_content = clean_content[:-3]
            clean_content = clean_content.strip()

            parsed_model = pydantic_model(**json.loads(clean_content))
            return parsed_model

        except Exception as e:
            logger.warning(f"Try {current_try + 1} failed with error: {e}")
            messages.append(
                {"role": "user", "content": f"Previous attempt failed with error: {e}. Please try again and ensure the JSON matches the schema exactly."}
            )

    logger.error(f"Failed to parse text to model after {n_tries} tries")
    raise LLMStructuredOutputError(f"Failed to parse text to model after {n_tries} tries")


def collect_text_from_response(resp: Response) -> str:
    """
    Collect the text from the response.
    Args:
        resp (Response): The response to collect the text from.
    Returns:
        str: The collected text.
    """
    raw_text = getattr(resp, "output_text", None)
    if raw_text:
        return raw_text

    chunks = []
    for item in getattr(resp, "output", []) or []:
        if getattr(item, "type", None) == "message":
            content = getattr(item, "content", "")
            if isinstance(content, str):
                chunks.append(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") in ("output_text", "text"):
                        chunks.append(part.get("text", ""))
        if getattr(item, "type", None) == "output_text":
            chunks.append(getattr(item, "text", ""))
    return "\n".join(chunks).strip()
