"""Model definitions for all tasks."""

from .sentiment import predict as sentiment_predict, PROMPT as SENTIMENT_PROMPT
from .qa import predict as qa_predict, PROMPT as QA_PROMPT
from .math import predict as math_predict, PROMPT as MATH_REACT_PROMPT

__all__ = [
    "sentiment_predict",
    "SENTIMENT_PROMPT",
    "qa_predict",
    "QA_PROMPT",
    "math_predict",
    "MATH_REACT_PROMPT",
]
