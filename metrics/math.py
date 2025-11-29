"""Math word problem metrics."""

from mlflow.genai.judges import make_judge
from typing import Dict, Literal


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


# Create MLflow metric using make_judge
math_metric = make_judge(
    name="math_accuracy",
    instructions=(
        "Evaluate whether the predicted numeric answer matches the expected answer.\n\n"
        "Problem: {{ inputs }}\n"
        "Predicted Answer: {{ outputs }}\n"
        "Expected Answer: {{ expectations }}\n\n"
        "Compare the predicted answer with the expected answer numerically.\n"
        "Consider them matching if they are equal when converted to numbers, "
        "allowing for minor floating point differences (e.g., '25' and '25.0' are the same).\n"
        "Return 'correct' if they match, 'incorrect' otherwise."
    ),
    feedback_value_type=Literal["correct", "incorrect"],
    model="openai:/gpt-4o-mini",
)
