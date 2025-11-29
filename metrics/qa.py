"""Question answering metrics."""

from mlflow.genai.judges import make_judge
from typing import Dict, Literal


def qa_accuracy(gold: Dict, pred: str) -> bool:
    """
    Check if predicted answer matches expected answer.
    Uses case-insensitive exact match.

    Args:
        gold: Dictionary with format {"inputs": {...}, "expectations": {"answer": ...}}
        pred: Predicted answer string

    Returns:
        True if answers match (case-insensitive), False otherwise
    """
    expected = str(gold["expectations"]["answer"]).lower().strip()
    predicted = str(pred).lower().strip()
    return expected == predicted


def qa_scorer(predictions: str, targets: Dict) -> float:
    """
    MLflow scorer for question answering.

    Args:
        predictions: Model prediction (answer string)
        targets: Dictionary with expectations

    Returns:
        1.0 if correct, 0.0 if incorrect
    """
    expected = str(targets.get("answer", "")).lower().strip()
    predicted = str(predictions).lower().strip()
    return 1.0 if expected == predicted else 0.0


# Create MLflow metric using make_judge
qa_metric = make_judge(
    name="qa_accuracy",
    instructions=(
        "Evaluate whether the predicted answer matches the expected answer.\n\n"
        "Question: {{ inputs }}\n"
        "Predicted Answer: {{ outputs }}\n"
        "Expected Answer: {{ expectations }}\n\n"
        "Compare the predicted answer with the expected answer using case-insensitive exact match.\n"
        "Return 'correct' if they match exactly, 'incorrect' otherwise."
    ),
    feedback_value_type=Literal["correct", "incorrect"],
    model="openai:/gpt-4o-mini",
)
