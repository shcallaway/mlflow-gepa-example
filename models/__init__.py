"""Model definitions for all tasks."""

from .sentiment import sentiment_predict, SENTIMENT_PROMPT
from .qa import qa_predict, QA_PROMPT
from .math import math_predict, MATH_REACT_PROMPT

__all__ = [
    "sentiment_predict",
    "SENTIMENT_PROMPT",
    "qa_predict",
    "QA_PROMPT",
    "math_predict",
    "MATH_REACT_PROMPT",
]
