"""
Configuration for OpenAI API client.
"""

import os
import time
from typing import Callable, TypeVar
from openai import OpenAI, RateLimitError, APIError

T = TypeVar('T')


# Singleton client instance
_client_instance = None


def get_openai_client(api_key: str = None, **kwargs):
    """
    Get an OpenAI API client instance (singleton pattern).

    Args:
        api_key: Optional API key (defaults to OPENAI_API_KEY environment variable)
        **kwargs: Additional arguments for the OpenAI client

    Returns:
        OpenAI client instance
    """
    global _client_instance

    # Return existing instance if available
    if _client_instance is not None:
        return _client_instance

    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError(
            "OpenAI API key not found. Set OPENAI_API_KEY environment variable "
            "or pass api_key parameter."
        )

    _client_instance = OpenAI(api_key=api_key, **kwargs)
    return _client_instance


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


def with_retry(func: Callable[..., T], max_retries: int = 3, initial_delay: float = 1.0) -> T:
    """
    Execute a function with exponential backoff retry logic.

    Args:
        func: Function to execute
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds (doubles on each retry)

    Returns:
        Result from the function

    Raises:
        The last exception encountered if all retries fail
    """
    delay = initial_delay
    last_exception = None

    for attempt in range(max_retries):
        try:
            return func()
        except RateLimitError as e:
            last_exception = e
            if attempt < max_retries - 1:
                print(f"Rate limit hit. Retrying in {delay}s... (attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
                delay *= 2  # Exponential backoff
            else:
                raise
        except APIError as e:
            last_exception = e
            if attempt < max_retries - 1:
                print(f"API error: {e}. Retrying in {delay}s... (attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
                delay *= 2
            else:
                raise
        except Exception as e:
            # Don't retry on other exceptions (e.g., validation errors)
            raise

    # Should never reach here, but for type safety
    raise last_exception if last_exception else RuntimeError("Retry failed")
