"""Question answering metrics."""

from mlflow.metrics.genai import EvaluationExample, make_genai_metric
from typing import Dict


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


# Create MLflow metric using make_genai_metric
qa_metric = make_genai_metric(
    name="qa_accuracy",
    definition=(
        "QA accuracy measures whether the predicted answer "
        "matches the expected answer using case-insensitive exact match."
    ),
    grading_prompt=(
        "Compare the predicted answer with the expected answer. "
        "Return 1 if they match exactly (case-insensitive), 0 otherwise."
    ),
    examples=[
        EvaluationExample(
            input="What is the capital of France?",
            output="Paris",
            score=1,
            justification="Predicted 'Paris' matches expected 'Paris'"
        ),
        EvaluationExample(
            input="What is the capital of France?",
            output="London",
            score=0,
            justification="Predicted 'London' but expected 'Paris'"
        ),
    ],
    version="v1",
    model="openai:/gpt-4o-mini",
    grading_context_columns=[],
    parameters={"temperature": 0.0},
    aggregations=["mean", "variance", "p90"],
    greater_is_better=True,
)
