"""Question answering metrics."""

from mlflow.genai.judges import make_judge, CategoricalRating
from mlflow.genai import scorer
from mlflow.entities import Feedback
from typing import Dict, Literal, Any


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


@scorer
def qa_scorer(outputs: str, expectations: Dict[str, Any]) -> Feedback:
    """
    MLflow scorer for question answering (for GEPA optimization).

    Args:
        outputs: Model prediction (answer string)
        expectations: Dictionary with expected values

    Returns:
        Feedback with categorical rating
    """
    expected = str(expectations.get("answer", "")).lower().strip()
    predicted = str(outputs).lower().strip()

    return Feedback(
        name="qa_accuracy",
        value=CategoricalRating.YES if expected == predicted else CategoricalRating.NO
    )


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
