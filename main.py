"""
MLflow GEPA Tutorial - Multi-Task Examples
==========================================

Demonstrates prompt optimization using MLflow GEPA on multiple tasks:
- Sentiment Classification: Classify text as positive/negative
- Question Answering: Answer questions from context
- Math Word Problems: Solve math problems using ReAct with calculator tool

Usage:
    python main.py --task sentiment
    python main.py --task qa
    python main.py --task math

    Optional flags:
    --skip-optimization: Skip GEPA optimization and just run baseline evaluation
"""

import argparse
import os
from typing import Dict, List, Callable
import warnings
import logging

# Suppress MLflow and Alembic database logs BEFORE importing mlflow
# Disable propagation for these loggers to prevent their messages from showing
for logger_name in ['mlflow', 'alembic', 'mlflow.store.db.utils', 'alembic.runtime.migration']:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.ERROR)
    logger.propagate = False

# Suppress MLflow integration warnings
os.environ['MLFLOW_GENAI_EVAL_SKIP_TRACE_VALIDATION'] = 'True'
warnings.filterwarnings('ignore', message='.*prompts were not used during evaluation.*')

# Note: MLflow GEPA requires mlflow>=3.5.0
try:
    import mlflow
    import mlflow.genai
    from mlflow.genai.optimize import optimize_prompts
    from mlflow.genai.optimize.optimizers import GepaPromptOptimizer
    from mlflow.models import ModelSignature
    from mlflow.types.schema import Schema, ColSpec
    MLFLOW_AVAILABLE = True
except ImportError:
    print("Warning: MLflow not installed. Install with: pip install mlflow>=3.5.0")
    MLFLOW_AVAILABLE = False

from tasks import TASKS
from config import get_default_model


def create_predict_wrapper(task_config: Dict, prompt_uri: str = None) -> Callable:
    """
    Create a wrapper function for the predict_fn that matches MLflow's expected interface.

    MLflow GEPA expects: predict_fn(inputs: Dict) -> str
    Our functions expect individual parameters like predict_fn(text="...")

    Args:
        task_config: Task configuration from TASKS registry
        prompt_uri: Optional MLflow prompt URI (for optimized prompts)

    Returns:
        Wrapped prediction function with MLflow tracing
    """
    predict_fn = task_config["predict_fn"]
    input_fields = task_config["input_fields"]

    @mlflow.trace
    def wrapper(**kwargs) -> str:
        """Wrapper that accepts inputs as keyword arguments."""
        # Extract just the input fields needed for this task
        filtered_kwargs = {field: kwargs[field] for field in input_fields if field in kwargs}
        return predict_fn(**filtered_kwargs)

    return wrapper


def run_baseline_evaluation(task_config: Dict, data: List[Dict]) -> float:
    """
    Evaluate baseline model (without optimization).

    Args:
        task_config: Task configuration
        data: List of examples in format {"inputs": {...}, "expectations": {...}}

    Returns:
        Accuracy score
    """
    print("Testing BASELINE (unoptimized) model...")
    print("-" * 60)

    predict_fn = task_config["predict_fn"]
    accuracy_fn = task_config["accuracy_fn"]
    input_fields = task_config["input_fields"]
    output_field = task_config["output_field"]

    correct = 0
    for example in data:
        # Extract inputs and call predict function
        inputs = example["inputs"]
        kwargs = {field: inputs[field] for field in input_fields}
        prediction = predict_fn(**kwargs)

        # Check accuracy
        is_correct = accuracy_fn(example, prediction)
        correct += is_correct

        # Print result
        print_example_result(example, prediction, is_correct, task_config)

    score = correct / len(data)
    print(f"\nBaseline Accuracy: {score:.1%} ({correct}/{len(data)})")
    print()
    return score


def run_gepa_optimization(task_config: Dict, train_data: List[Dict], val_data: List[Dict]):
    """
    Run MLflow GEPA optimization.

    Args:
        task_config: Task configuration
        train_data: Training data
        val_data: Validation data

    Returns:
        Optimized prompt URI (or None if optimization fails)
    """
    if not MLFLOW_AVAILABLE:
        print("MLflow not available. Skipping optimization.")
        return None

    print("Running MLflow GEPA optimization...")
    print("-" * 60)
    print(f"Task: {task_config['name']}")
    print(f"Max GEPA calls: {task_config['gepa_max_calls']}")
    print(f"Training examples: {len(train_data)}")
    print("This may take several minutes as GEPA evolves the prompts...")
    print()

    try:
        # Register the initial prompt in MLflow
        prompt_name = task_config["prompt_name"]
        prompt_template = task_config["prompt_template"]

        print(f"Registering initial prompt: {prompt_name}")
        base_prompt = mlflow.genai.register_prompt(
            name=prompt_name,
            template=prompt_template
        )
        print(f"Registered prompt URI: {base_prompt.uri}")
        print()

        # Create predict wrapper for MLflow
        predict_wrapper = create_predict_wrapper(task_config, base_prompt.uri)

        # Create GEPA optimizer
        print("Creating GEPA optimizer...")
        optimizer = GepaPromptOptimizer(
            reflection_model=f"openai:/{get_default_model()}",
            max_metric_calls=task_config["gepa_max_calls"]
        )

        # Run optimization
        print("Starting optimization (this will take several minutes)...")
        result = optimize_prompts(
            predict_fn=predict_wrapper,
            train_data=train_data,
            prompt_uris=[base_prompt.uri],
            optimizer=optimizer,
            scorers=[task_config["scorer"]],
            enable_tracking=True
        )

        if result and result.optimized_prompts:
            optimized_prompt_uri = result.optimized_prompts[0].uri
            print()
            print(f"✓ Optimization complete!")
            print(f"Optimized prompt URI: {optimized_prompt_uri}")
            return optimized_prompt_uri
        else:
            print("\n⚠ Optimization completed but no optimized prompt returned")
            return None

    except Exception as e:
        print(f"\n✗ Error during optimization: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        print("\nContinuing with baseline evaluation only.")
        return None


def evaluate_optimized_prompt(task_config: Dict, optimized_prompt_uri: str, data: List[Dict]) -> float:
    """
    Evaluate the optimized prompt on validation data.

    Args:
        task_config: Task configuration
        optimized_prompt_uri: MLflow prompt URI for optimized prompt
        data: Validation data

    Returns:
        Accuracy score
    """
    print("\nTesting OPTIMIZED model...")
    print("-" * 60)

    predict_wrapper = create_predict_wrapper(task_config, optimized_prompt_uri)
    accuracy_fn = task_config["accuracy_fn"]

    correct = 0
    for example in data:
        # Call predict function with inputs
        prediction = predict_wrapper(example["inputs"])

        # Check accuracy
        is_correct = accuracy_fn(example, prediction)
        correct += is_correct

        # Print result
        print_example_result(example, prediction, is_correct, task_config)

    score = correct / len(data)
    print(f"\nOptimized Accuracy: {score:.1%} ({correct}/{len(data)})")
    print()
    return score


def print_comparison(baseline_score: float, optimized_score: float):
    """Print comparison of baseline vs optimized results."""
    print("=" * 60)
    print("RESULTS COMPARISON")
    print("=" * 60)
    print(f"Baseline Accuracy:  {baseline_score:.1%}")
    print(f"Optimized Accuracy: {optimized_score:.1%}")

    improvement = optimized_score - baseline_score
    if improvement > 0:
        print(f"Improvement:        +{improvement:.1%} ✓")
    elif improvement < 0:
        print(f"Change:             {improvement:.1%} ✗")
    else:
        print(f"Change:             {improvement:.1%} (no change)")

    print("=" * 60)
    print()


def print_example_result(example: Dict, prediction: str, is_correct: bool, task_config: Dict):
    """Print example result with task-specific formatting."""
    check = '✓' if is_correct else '✗'
    inputs = example["inputs"]
    expectations = example["expectations"]
    output_field = task_config["output_field"]

    if task_config["input_fields"] == ["text"]:
        # Sentiment task
        print(f"Text: {inputs['text'][:50]}...")
        print(f"Expected: {expectations['sentiment']} | Predicted: {prediction} | {check}")
    elif task_config["input_fields"] == ["problem"]:
        # Math task
        print(f"Problem: {inputs['problem']}")
        print(f"Expected: {expectations['answer']} | Predicted: {prediction} | {check}")
    else:
        # QA task
        print(f"Q: {inputs['question']}")
        print(f"Context: {inputs['context'][:60]}...")
        print(f"Expected: {expectations['answer']} | Predicted: {prediction} | {check}")
    print()


def demo_baseline_model(task_config: Dict):
    """Demo model on new examples (task-specific)."""
    print("Demo: Testing the model on new examples...")
    print("-" * 60)

    predict_fn = task_config["predict_fn"]

    if task_config["input_fields"] == ["text"]:
        # Sentiment examples
        test_texts = [
            "This is the best thing ever!",
            "I'm very disappointed with this.",
        ]
        for text in test_texts:
            result = predict_fn(text=text)
            print(f"Text: {text}")
            print(f"Sentiment: {result}")
            print()
    elif task_config["input_fields"] == ["problem"]:
        # Math examples
        test_problems = [
            "If 3 pizzas cost $45 total, how much does one pizza cost?",
            "A train travels 60 miles per hour for 3.5 hours. How far does it go?",
        ]
        for problem in test_problems:
            result = predict_fn(problem=problem)
            print(f"Problem: {problem}")
            print(f"Answer: {result}")
            print()
    else:
        # QA examples
        test_examples = [
            {"question": "What is the largest ocean?", "context": "The Pacific Ocean is the largest ocean on Earth."},
            {"question": "When was the internet invented?", "context": "The internet was developed in the 1960s-1970s."},
        ]
        for ex in test_examples:
            result = predict_fn(question=ex["question"], context=ex["context"])
            print(f"Q: {ex['question']}")
            print(f"Answer: {result}")
            print()


def main():
    """Main workflow with task selection."""
    parser = argparse.ArgumentParser(
        description="MLflow GEPA Multi-Task Examples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--task",
        choices=list(TASKS.keys()),
        default="sentiment",
        help="Task to run (default: sentiment)"
    )
    parser.add_argument(
        "--skip-optimization",
        action="store_true",
        help="Skip GEPA optimization and just run baseline evaluation"
    )
    args = parser.parse_args()

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Set it with: export OPENAI_API_KEY='your-key-here'")
        return

    # Get task configuration
    task_config = TASKS[args.task]

    # Set up MLflow experiment tracking
    if MLFLOW_AVAILABLE:
        # Suppress MLflow/Alembic logs one more time right before tracking URI setup
        for logger_name in ['mlflow', 'alembic', 'mlflow.store.db.utils', 'alembic.runtime.migration']:
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.ERROR)
            logger.propagate = False
            # Remove all handlers to prevent any output
            logger.handlers = []

        # Use SQLite backend instead of deprecated filesystem backend
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        experiment_name = f"GEPA-{task_config['name'].replace(' ', '-')}"
        mlflow.set_experiment(experiment_name)
        mlflow.openai.autolog()
        print(f"MLflow experiment: {experiment_name}")
        print()

    print("=" * 60)
    print(f"MLflow GEPA: {task_config['name']}")
    print("=" * 60)
    print()

    # Load data
    train_data, dev_data = task_config["get_data"]()
    print(f"Loaded {len(train_data)} training examples, {len(dev_data)} dev examples")
    print()

    # Baseline evaluation
    baseline_score = run_baseline_evaluation(task_config, dev_data)

    # Optional: GEPA optimization
    optimized_prompt_uri = None
    if not args.skip_optimization:
        optimized_prompt_uri = run_gepa_optimization(task_config, train_data, dev_data)

        # If optimization succeeded, evaluate optimized prompt
        if optimized_prompt_uri:
            optimized_score = evaluate_optimized_prompt(task_config, optimized_prompt_uri, dev_data)
            print_comparison(baseline_score, optimized_score)

    # Demo on new examples
    demo_baseline_model(task_config)


if __name__ == "__main__":
    main()
