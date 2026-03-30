"""
utils/llm_client.py
Instantiates and exposes the shared LLM client used by all agents.
"""

import os

from pydantic import SecretStr

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from typing import Union

def get_llm(
    model_type: str = "openai",
    model_name: str = "gpt-4o-mini"
) -> Union[ChatOpenAI, ChatGoogleGenerativeAI, ChatAnthropic]:
    """
    Build and return a ChatAnthropic client.

    Reads ANTHROPIC_API_KEY from the environment.
    Uses claude-3-5-haiku for low latency and cost efficiency.

    Parameters
    ----------
    model_type : str
        The type of LLM to instantiate. Supported values: "openai", "google", "anthropic".
    model_name : str
        The specific model name to use for the chosen LLM type. Ignored for "anthropic" as it defaults to "claude-3-5-haiku".

    Returns
    -------
    ChatAnthropic
        Configured LLM client.

    Raises
    ------
    EnvironmentError
        If ANTHROPIC_API_KEY is not set.
    """
    
    if model_type == 'openai':
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY environment variable is not set.")
        
        model_class = ChatOpenAI
        model_params = {"model": model_name}
    
    elif model_type == 'google':
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError("GOOGLE_API_KEY environment variable is not set.")
        
        model_class = ChatGoogleGenerativeAI
        model_params = {"model": model_name}
    
    elif model_type == 'anthropic':
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError("ANTHROPIC_API_KEY environment variable is not set.")
        
        model_class = ChatAnthropic
        model_params = {
            "model_name": model_name,
            "timeout": 30.0,  # seconds
            "stop": ["\n\n"]
        }
    
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    api_key_secret = SecretStr(api_key)

    llm = model_class(
        temperature=0,
        api_key=api_key_secret,
        **model_params # type: ignore
    )

    return llm
