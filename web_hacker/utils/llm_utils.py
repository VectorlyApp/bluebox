"""
web_hacker/utils/llm_utils.py

Utility functions for LLM API calls.
"""

import json
from typing import Type, Callable, Optional, Any

from openai import OpenAI
from openai.types.responses import Response
from pydantic import BaseModel

from web_hacker.config import Config
from web_hacker.utils.exceptions import LLMStructuredOutputError
from web_hacker.utils.logger import get_logger

logger = get_logger(__name__)


# Model context window limits based on OpenAI documentation
MODEL_CONTEXT_LIMITS = {
    # GPT-5 series (400k context)
    "gpt-5": 400000,
    "gpt-5-mini": 400000,
    "gpt-5-nano": 400000,
    "gpt-5.1": 400000,
    "gpt-5.1": 400000,
    # GPT-4.1 series (1M+ context)
    "gpt-4.1": 1047576,
    "gpt-4.1-mini": 1047576,
    "gpt-4.1-nano": 1047576,
    # GPT-4o series (128k context)
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
    # Legacy models
    "gpt-4-turbo": 128000,
    "gpt-4": 8192,
    "gpt-3.5-turbo": 16385,
}


class ContextManagedOpenAI:
    """
    Wrapper around OpenAI client that provides context-managed responses.

    Automatically detects when context window is approaching limits and triggers
    a summarization callback to reset the conversation context.

    Usage:
        def on_context_limit():
            # Summarize conversation and reset
            pass

        managed_client = ContextManagedOpenAI(
            client=openai_client,
            on_context_limit_reached=on_context_limit
        )

        # Use like normal OpenAI client
        response = managed_client.responses.parse(...)
    """

    class _ResponsesWrapper:
        """Inner class that wraps client.responses with context management."""

        def __init__(
            self,
            client: OpenAI,
            on_context_limit_reached: Callable[[], None],
            context_window_threshold: float,
            default_model: str,
        ):
            self._client = client
            self._on_context_limit_reached = on_context_limit_reached
            self._context_window_threshold = context_window_threshold
            self._default_model = default_model

        def _get_context_limit(self, model: str) -> int:
            """Get context window limit for a model."""
            return MODEL_CONTEXT_LIMITS.get(model, 400000)

        def _is_context_length_error(self, error: Exception) -> bool:
            """Check if an exception is a context length exceeded error."""
            error_str = str(error).lower()
            return 'context_length_exceeded' in error_str or 'context window' in error_str

        def _check_context_usage(self, response, model: str) -> bool:
            """Check if context usage has exceeded threshold."""
            if not hasattr(response, 'usage') or response.usage is None:
                return False

            usage = response.usage
            input_tokens = getattr(usage, 'input_tokens', 0)
            context_limit = self._get_context_limit(model)

            usage_ratio = input_tokens / context_limit
            logger.debug(f"Context usage: {input_tokens}/{context_limit} tokens ({usage_ratio:.1%})")

            if usage_ratio >= self._context_window_threshold:
                logger.warning(f"Context window usage at {usage_ratio:.1%} - triggering summarization")
                self._on_context_limit_reached()
                return True

            return False

        def parse(self, **kwargs) -> Any:
            """Wrapper around client.responses.parse with context management."""
            model = kwargs.get('model', self._default_model)
            max_retries = 2

            for attempt in range(max_retries):
                try:
                    response = self._client.responses.parse(**kwargs)
                    self._check_context_usage(response, model)
                    return response

                except Exception as e:
                    if attempt < max_retries - 1 and self._is_context_length_error(e):
                        logger.warning(f"Context length exceeded on attempt {attempt + 1}, summarizing and retrying...")
                        self._on_context_limit_reached()
                        kwargs['previous_response_id'] = None
                        continue
                    raise

        def create(self, **kwargs) -> Any:
            """Wrapper around client.responses.create with context management."""
            model = kwargs.get('model', self._default_model)
            max_retries = 2

            for attempt in range(max_retries):
                try:
                    response = self._client.responses.create(**kwargs)
                    self._check_context_usage(response, model)
                    return response

                except Exception as e:
                    if attempt < max_retries - 1 and self._is_context_length_error(e):
                        logger.warning(f"Context length exceeded on attempt {attempt + 1}, summarizing and retrying...")
                        self._on_context_limit_reached()
                        kwargs['previous_response_id'] = None
                        continue
                    raise

        def __getattr__(self, name: str) -> Any:
            """Delegate unknown attributes to the original client.responses."""
            return getattr(self._client.responses, name)

    def __init__(
        self,
        client: OpenAI,
        on_context_limit_reached: Callable[[], None],
        context_window_threshold: float = 0.9,
        default_model: str = "gpt-5.1",
    ):
        self._client = client
        self.responses = self._ResponsesWrapper(
            client=client,
            on_context_limit_reached=on_context_limit_reached,
            context_window_threshold=context_window_threshold,
            default_model=default_model,
        )

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to the underlying client."""
        return getattr(self._client, name)


def manual_llm_parse_text_to_model(
    text: str,
    context: str,
    pydantic_model: Type[BaseModel],
    client: OpenAI,
    llm_model: str = "gpt-5-nano",
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
        {"role": "user", "content": f"Context: {context}"},
        {"role": "user", "content": f"Text to parse: {text}"},
        {"role": "user", "content": f"Target Model JSON Schema: {json.dumps(pydantic_model.model_json_schema())}"},
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
