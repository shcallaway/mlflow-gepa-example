"""
Configuration for OpenAI API client.
"""

import os
from openai import OpenAI


def get_openai_client(api_key: str = None, **kwargs):
    """
    Get an OpenAI API client instance.

    Args:
        api_key: Optional API key (defaults to OPENAI_API_KEY environment variable)
        **kwargs: Additional arguments for the OpenAI client

    Returns:
        OpenAI client instance
    """
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError(
            "OpenAI API key not found. Set OPENAI_API_KEY environment variable "
            "or pass api_key parameter."
        )

    return OpenAI(api_key=api_key, **kwargs)


def get_default_model():
    """
    Get the default model name.

    Returns:
        Default model string for OpenAI API
    """
    return "gpt-4o-mini"


# Model configurations
MODEL_CONFIGS = {
    "default": "gpt-4o-mini",
    "reflection": "gpt-4o",  # Used for GEPA instruction generation
}
