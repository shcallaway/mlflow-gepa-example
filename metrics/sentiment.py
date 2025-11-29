"""Sentiment classification metrics."""

from mlflow.genai.judges import make_judge, CategoricalRating
from mlflow.genai import scorer
from mlflow.entities import Feedback
from typing import Dict, Literal, Any


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


@scorer
def sentiment_scorer(outputs: str, expectations: Dict[str, Any]) -> Feedback:
    """
    MLflow scorer for sentiment classification (for GEPA optimization).

    Args:
        outputs: Model prediction (sentiment string)
        expectations: Dictionary with expected values

    Returns:
        Feedback with categorical rating
    """
    expected = str(expectations.get("sentiment", "")).lower().strip()
    predicted = str(outputs).lower().strip()

    return Feedback(
        name="sentiment_accuracy",
        value=CategoricalRating.YES if expected == predicted else CategoricalRating.NO
    )


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
