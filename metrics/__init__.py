"""Evaluation metrics for all tasks."""

from .sentiment import accuracy as sentiment_accuracy, metric as sentiment_metric, scorer_fn as sentiment_scorer
from .qa import accuracy as qa_accuracy, metric as qa_metric, scorer_fn as qa_scorer
from .math import accuracy as math_accuracy, metric as math_metric, scorer_fn as math_scorer

__all__ = [
    "sentiment_accuracy",
    "sentiment_metric",
    "sentiment_scorer",
    "qa_accuracy",
    "qa_metric",
    "qa_scorer",
    "math_accuracy",
    "math_metric",
    "math_scorer",
]
