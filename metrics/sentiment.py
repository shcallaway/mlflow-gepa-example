"""Sentiment classification metrics."""

from mlflow.metrics.genai import EvaluationExample, make_genai_metric
from typing import Dict


def sentiment_accuracy(gold: Dict, pred: str) -> bool:
    """
    Check if predicted sentiment matches expected sentiment.

    Args:
        gold: Dictionary with format {"inputs": {...}, "expectations": {"sentiment": ...}}
        pred: Predicted sentiment string

    Returns:
        True if sentiments match (case-insensitive), False otherwise
    """
    expected = gold["expectations"]["sentiment"].lower()
    predicted = pred.lower()
    return expected == predicted


def sentiment_scorer(predictions: str, targets: Dict) -> float:
    """
    MLflow scorer for sentiment classification.

    Args:
        predictions: Model prediction (sentiment string)
        targets: Dictionary with expectations

    Returns:
        1.0 if correct, 0.0 if incorrect
    """
    expected = targets.get("sentiment", "").lower()
    predicted = predictions.lower()
    return 1.0 if expected == predicted else 0.0


# Create MLflow metric using make_genai_metric
sentiment_metric = make_genai_metric(
    name="sentiment_accuracy",
    definition=(
        "Sentiment accuracy measures whether the predicted sentiment "
        "(positive or negative) matches the expected sentiment."
    ),
    grading_prompt=(
        "Compare the predicted sentiment with the expected sentiment. "
        "Return 1 if they match (case-insensitive), 0 otherwise."
    ),
    examples=[
        EvaluationExample(
            input="This movie was great!",
            output="positive",
            score=1,
            justification="Predicted 'positive' matches expected 'positive'"
        ),
        EvaluationExample(
            input="Terrible experience.",
            output="positive",
            score=0,
            justification="Predicted 'positive' but expected 'negative'"
        ),
    ],
    version="v1",
    model="openai:/gpt-4o-mini",
    grading_context_columns=[],
    parameters={"temperature": 0.0},
    aggregations=["mean", "variance", "p90"],
    greater_is_better=True,
)
