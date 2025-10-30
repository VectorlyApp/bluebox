from pydantic import BaseModel
from openai import OpenAI
from typing import Type
import json


def llm_parse_text_to_model(
    text: str,
    context: str,
    pydantic_model: Type[BaseModel],
    client: OpenAI,
    llm_model: str = "gpt-5-nano",
    n_tries: int = 3
) -> BaseModel:
    """
    Call to cheap LLM model to parse outtext to a pydantic model.
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
    
    
    SYSTEM_PROMPT = f"""
    You are a helpful assistant that parses text to a pydantic model.
    You must conform to the provided pydantic model schema.
    """
    
    message = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context: {context}"},
        {"role": "user", "content": f"Text to parse: {text}"}
    ]
    
    for current_try in range(n_tries):
        try:
            completion = client.chat.completions.parse(
                model=llm_model,
                messages=message, 
                response_format=pydantic_model,
            )
            return completion.choices[0].message.parsed
        
        except Exception as e:
            print(f"Try {current_try + 1} failed with error: {e}")
            message.append({"role": "user", "content": f"Previous attempt failed with error: {e}. Please try again."})
                
            
    raise Exception(f"Failed to parse text to model after {n_tries} tries")


def manual_llm_parse_text_to_model(
    text: str,
    context: str,
    pydantic_model: Type[BaseModel],
    client: OpenAI,
    llm_model: str = "gpt-5-nano",
    n_tries: int = 3,
) -> BaseModel:
    """
    Manual LLM parse text to model. (without using structured output)
    """
    
    SYSTEM_PROMPT = f"""
    You are a helpful assistant that parses text to a pydantic model.
    You must conform to the provided pydantic model schema.
    """
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context: {context}"},
        {"role": "user", "content": f"Text to parse: {text}"},
        {"role": "user", "content": f"Model JSON schema: {pydantic_model.model_json_schema()}"},
        {"role": "user", "content": f"Please respond in the following format above. Your response must be a valid JSON object."}
    ]
    
    for current_try in range(n_tries):
        try:
            response = client.chat.completions.create(
                model=llm_model,
                messages=messages,
            )
            messages.append({"role": "assistant", "content": response.choices[0].message.content})
            text = response.choices[0].message.content
            
            parsed_model = pydantic_model(**json.loads(text))
            
            return parsed_model
        
        except Exception as e:
            print(f"Try {current_try + 1} failed with error: {e}")
            messages.append({"role": "user", "content": f"Previous attempt failed with error: {e}. Please try again."})
            
    raise Exception(f"Failed to parse text to model after {n_tries} tries")

    

def collect_text_from_response(resp) -> str:
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