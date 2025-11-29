"""Math word problem metrics."""

from mlflow.metrics.genai import EvaluationExample, make_genai_metric
from typing import Dict


def math_accuracy(gold: Dict, pred: str) -> bool:
    """
    Check if predicted answer matches expected answer numerically.

    Handles both integer and decimal answers with tolerance for floating point.

    Args:
        gold: Dictionary with format {"inputs": {...}, "expectations": {"answer": ...}}
        pred: Predicted answer string

    Returns:
        True if answers match numerically, False otherwise
    """
    try:
        # Extract and clean the answers
        expected = str(gold["expectations"]["answer"]).strip()
        predicted = str(pred).strip()

        # Try exact string match first (fastest)
        if expected.lower() == predicted.lower():
            return True

        # Try numerical comparison (handles "12.0" vs "12", etc.)
        expected_num = float(expected)
        predicted_num = float(predicted)

        # Use small tolerance for floating point comparison
        return abs(expected_num - predicted_num) < 1e-6

    except (ValueError, AttributeError, KeyError):
        # If conversion fails, fall back to string comparison
        return str(gold.get("expectations", {}).get("answer", "")).lower().strip() == str(pred).lower().strip()


def math_scorer(predictions: str, targets: Dict) -> float:
    """
    MLflow scorer for math word problems.

    Args:
        predictions: Model prediction (numeric answer string)
        targets: Dictionary with expectations

    Returns:
        1.0 if correct, 0.0 if incorrect
    """
    try:
        expected = str(targets.get("answer", "")).strip()
        predicted = str(predictions).strip()

        # Try exact string match
        if expected.lower() == predicted.lower():
            return 1.0

        # Try numerical comparison
        expected_num = float(expected)
        predicted_num = float(predicted)

        # Use small tolerance for floating point comparison
        return 1.0 if abs(expected_num - predicted_num) < 1e-6 else 0.0

    except (ValueError, AttributeError):
        # Fall back to string comparison
        return 1.0 if str(targets.get("answer", "")).lower().strip() == str(predictions).lower().strip() else 0.0


# Create MLflow metric using make_genai_metric
math_metric = make_genai_metric(
    name="math_accuracy",
    definition=(
        "Math accuracy measures whether the predicted numeric answer "
        "matches the expected answer, with tolerance for floating point differences."
    ),
    grading_prompt=(
        "Compare the predicted numeric answer with the expected answer. "
        "Return 1 if they match (allowing for minor floating point differences), 0 otherwise."
    ),
    examples=[
        EvaluationExample(
            input="What is 5 + 3?",
            output="8",
            score=1,
            justification="Predicted '8' matches expected '8'"
        ),
        EvaluationExample(
            input="What is 100 / 4?",
            output="25.0",
            score=1,
            justification="Predicted '25.0' matches expected '25' numerically"
        ),
        EvaluationExample(
            input="What is 10 * 5?",
            output="45",
            score=0,
            justification="Predicted '45' but expected '50'"
        ),
    ],
    version="v1",
    model="openai:/gpt-4o-mini",
    grading_context_columns=[],
    parameters={"temperature": 0.0},
    aggregations=["mean", "variance", "p90"],
    greater_is_better=True,
)
