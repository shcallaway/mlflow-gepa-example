"""Sentiment classification metrics."""

from mlflow.genai.judges import make_judge
from typing import Dict, Literal


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


# Create MLflow metric using make_judge
sentiment_metric = make_judge(
    name="sentiment_accuracy",
    instructions=(
        "Evaluate whether the predicted sentiment matches the expected sentiment.\n\n"
        "Text: {{ inputs }}\n"
        "Predicted Sentiment: {{ outputs }}\n"
        "Expected Sentiment: {{ expectations }}\n\n"
        "Compare the predicted sentiment with the expected sentiment (case-insensitive).\n"
        "Return 'correct' if they match, 'incorrect' otherwise."
    ),
    feedback_value_type=Literal["correct", "incorrect"],
    model="openai:/gpt-4o-mini",
)
