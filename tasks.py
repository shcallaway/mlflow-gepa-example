"""Task registry and configuration for MLflow GEPA examples."""

from datasets import get_sentiment_data, get_qa_data, get_math_data
from models import sentiment_predict, qa_predict, math_predict
from models import SENTIMENT_PROMPT, QA_PROMPT, MATH_REACT_PROMPT
from metrics import sentiment_accuracy, qa_accuracy, math_accuracy
from metrics import sentiment_metric, qa_metric, math_metric
from metrics import sentiment_scorer, qa_scorer, math_scorer


# Task Configuration Registry
TASKS = {
    "sentiment": {
        "name": "Sentiment Classification",
        "get_data": get_sentiment_data,
        "predict_fn": sentiment_predict,
        "prompt_template": SENTIMENT_PROMPT,
        "prompt_name": "sentiment_classifier",
        "metric": sentiment_metric,
        "scorer": sentiment_scorer,
        "accuracy_fn": sentiment_accuracy,
        "gepa_max_calls": 100,
        "input_fields": ["text"],
        "output_field": "sentiment",
    },
    "qa": {
        "name": "Question Answering",
        "get_data": get_qa_data,
        "predict_fn": qa_predict,
        "prompt_template": QA_PROMPT,
        "prompt_name": "qa_model",
        "metric": qa_metric,
        "scorer": qa_scorer,
        "accuracy_fn": qa_accuracy,
        "gepa_max_calls": 120,
        "input_fields": ["question", "context"],
        "output_field": "answer",
    },
    "math": {
        "name": "Math Word Problems (ReAct)",
        "get_data": get_math_data,
        "predict_fn": math_predict,
        "prompt_template": MATH_REACT_PROMPT,
        "prompt_name": "math_react_solver",
        "metric": math_metric,
        "scorer": math_scorer,
        "accuracy_fn": math_accuracy,
        "gepa_max_calls": 80,
        "input_fields": ["problem"],
        "output_field": "answer",
    },
}
