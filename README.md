# MLflow GEPA Project

A modular example demonstrating **GEPA (Generative Evolutionary Prompt Adaptation)** for prompt optimization using direct OpenAI API calls and MLflow integration.

> **✅ Status**: Fully functional! This project demonstrates complete MLflow GEPA integration with automatic prompt optimization. Run baseline evaluation or full GEPA optimization across multiple NLP tasks (sentiment classification, QA, math word problems).

## Quick Start

```bash
# 1. Create and activate virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your OpenAI API key
export OPENAI_API_KEY='your-api-key-here'

# 4. Run with GEPA optimization (may take several minutes)
python main.py --task sentiment

# Or run baseline evaluation only (faster)
python main.py --task sentiment --skip-optimization
```

## Project Structure

```
mlflow-gepa-example/
├── config.py              # OpenAI client configuration
├── datasets/              # Dataset definitions (per-task organization)
│   ├── __init__.py
│   ├── sentiment.py       # Sentiment classification data
│   ├── qa.py              # Question answering data
│   └── math.py            # Math word problems data
├── models/                # Prompt templates and predict functions (per-task)
│   ├── __init__.py
│   ├── sentiment.py       # Sentiment prediction with prompts
│   ├── qa.py              # QA prediction with prompts
│   └── math.py            # Math prediction with ReAct loop
├── metrics/               # Evaluation metrics (per-task)
│   ├── __init__.py
│   ├── sentiment.py       # Sentiment accuracy + MLflow metric
│   ├── qa.py              # QA accuracy + MLflow metric
│   └── math.py            # Math accuracy + MLflow metric
├── tasks.py               # Task registry (glues everything together)
├── main.py                # Main workflow orchestration
└── requirements.txt       # Project dependencies
```

## What This Project Demonstrates

This project uses direct OpenAI API calls with prompt templates for **multiple NLP tasks**:

### Sentiment Classification
- Classify text as positive or negative
- Single-input task demonstrating basic prompt engineering
- Uses GPT-4o-mini with structured prompts

### Question Answering
- Answer questions based on context
- Multi-input task (question + context)
- Demonstrates prompt composition

### Math Word Problems
- Solve math problems using manual ReAct loop
- Implements thought-action-observation pattern
- Includes calculator tool simulation

### Complete GEPA Workflow

1. **Baseline Evaluation** - Test initial prompts with direct OpenAI API calls
2. **GEPA Optimization** - Evolutionary prompt improvement using MLflow GEPA
3. **Optimized Evaluation** - Test optimized prompts and compare to baseline
4. **Results Comparison** - Automatic reporting of accuracy improvements
5. **Demo** - Test on new examples

The per-task file organization makes it easy to:
- Understand what code belongs to which task
- Add new tasks without touching existing ones
- Experiment with different prompts and models
- Transition to MLflow GEPA when ready

## Prerequisites

- Python 3.9 or higher
- OpenAI API key
  - Get one at: https://platform.openai.com/api-keys
- MLflow 3.5.0 or higher (for future GEPA integration)

## Setup

1. **Create a virtual environment (recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set your API key (required):**
   ```bash
   export OPENAI_API_KEY='your-api-key-here'
   ```

   **Note:** The script will not work without an API key set.

## Usage

**Note:** Make sure you've activated your virtual environment and set your API key before running!

```bash
source venv/bin/activate  # Activate virtual environment
export OPENAI_API_KEY='your-api-key-here'  # Set API key
```

### Run Sentiment Classification (Default)

```bash
# With GEPA optimization (takes a few minutes)
python main.py --task sentiment

# Baseline only (faster)
python main.py --task sentiment --skip-optimization
```

### Run Question Answering

```bash
# With GEPA optimization
python main.py --task qa

# Baseline only
python main.py --task qa --skip-optimization
```

### Run Math Word Problems

```bash
# With GEPA optimization
python main.py --task math

# Baseline only
python main.py --task math --skip-optimization
```

The optimization process will:
1. Evaluate baseline prompt
2. Run GEPA evolutionary optimization (several minutes)
3. Evaluate optimized prompt
4. Show comparison and improvement

## Features

### Fully Implemented ✅

- ✅ Direct OpenAI API calls with prompt templates
- ✅ Baseline evaluation
- ✅ Task registry architecture
- ✅ MLflow GEPA optimizer integration
- ✅ Automatic prompt evolution and optimization
- ✅ Baseline vs optimized comparison
- ✅ MLflow experiment tracking
- ✅ Error handling with retry logic
- ✅ Singleton OpenAI client pattern
- ✅ Custom MLflow scorers for each task

### Known Limitations

- Small datasets (for demo purposes - expand for production use)
- Math ReAct limited to 5 iterations (configurable)
- No automated tests yet

## Adding New Tasks

The per-task file organization makes adding new tasks straightforward. **IMPORTANT**: Follow the standardized naming convention where all task files export the same named members without task-specific prefixes.

### Standardized Export Convention

Each task exports consistent names:
- **`models/{task}.py`**: `predict`, `PROMPT`
- **`metrics/{task}.py`**: `accuracy`, `scorer_fn`, `metric`
- **`datasets/{task}.py`**: `get_data`

The `__init__.py` files import these with task-specific aliases (e.g., `predict as sentiment_predict`) for use in `tasks.py`. This ensures consistency across all tasks while maintaining clean imports.

### Creating a New Task

Each task needs 3 files:

### 1. Add Your Dataset

Create `datasets/your_task.py`:

```python
"""Your task dataset."""

from typing import List, Dict, Tuple

TRAIN_DATA = [
    ("input 1", "output 1"),
    ("input 2", "output 2"),
    # ...
]

DEV_DATA = [
    ("input 1", "output 1"),
    # ...
]

def get_data() -> Tuple[List[Dict], List[Dict]]:
    """Get your task train and dev datasets."""
    train = []
    for input_val, output_val in TRAIN_DATA:
        train.append({
            "inputs": {"input": input_val},
            "expectations": {"output": output_val}
        })

    dev = []
    for input_val, output_val in DEV_DATA:
        dev.append({
            "inputs": {"input": input_val},
            "expectations": {"output": output_val}
        })

    return train, dev
```

Update `datasets/__init__.py`:
```python
from .your_task import get_data as get_your_task_data

__all__ = [..., "get_your_task_data"]
```

### 2. Define Your Model

Create `models/your_task.py` (use standardized names `PROMPT` and `predict`):

```python
"""Your task models."""

from config import get_openai_client, get_default_model, with_retry

# Define your prompt template (named PROMPT, not YOUR_TASK_PROMPT)
PROMPT = """You are an expert at [task description].

Input: {input}

Provide your answer:"""

def predict(input: str) -> str:
    """
    Predict output for the given input.

    Args:
        input: Input text

    Returns:
        Predicted output string
    """
    client = get_openai_client()
    model = get_default_model()

    prompt = PROMPT.format(input=input)

    def make_api_call():
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return response.choices[0].message.content.strip()

    return with_retry(make_api_call)
```

Update `models/__init__.py` (import with alias):
```python
from .your_task import predict as your_task_predict, PROMPT as YOUR_TASK_PROMPT

__all__ = [..., "your_task_predict", "YOUR_TASK_PROMPT"]
```

### 3. Add Evaluation Metric

Create `metrics/your_task.py` (use standardized names `accuracy`, `scorer_fn`, `metric`):

```python
"""Your task metrics."""

from typing import Dict, Literal, Any
from mlflow.genai.judges import make_judge, CategoricalRating
from mlflow.genai import scorer
from mlflow.entities import Feedback

def accuracy(gold: Dict, pred: str) -> bool:
    """
    Check if prediction is correct.

    Args:
        gold: Dictionary with format {"inputs": {...}, "expectations": {...}}
        pred: Predicted output string

    Returns:
        True if correct, False otherwise
    """
    expected = str(gold["expectations"]["output"]).lower().strip()
    predicted = str(pred).lower().strip()
    return expected == predicted

@scorer
def scorer_fn(outputs: str, expectations: Dict[str, Any]) -> Feedback:
    """
    MLflow scorer for GEPA optimization.

    Args:
        outputs: Model prediction
        expectations: Expected values

    Returns:
        Feedback with categorical rating
    """
    expected = str(expectations.get("output", "")).lower().strip()
    predicted = str(outputs).lower().strip()

    return Feedback(
        name="your_task_accuracy",
        value=CategoricalRating.YES if expected == predicted else CategoricalRating.NO
    )

# Create MLflow metric using make_judge
metric = make_judge(
    name="your_task_accuracy",
    instructions=(
        "Evaluate whether the predicted output matches the expected output.\n\n"
        "Input: {{ inputs }}\n"
        "Predicted: {{ outputs }}\n"
        "Expected: {{ expectations }}\n\n"
        "Return 'correct' if they match, 'incorrect' otherwise."
    ),
    feedback_value_type=Literal["correct", "incorrect"],
    model="openai:/gpt-4o-mini",
)
```

Update `metrics/__init__.py` (import with aliases):
```python
from .your_task import accuracy as your_task_accuracy, scorer_fn as your_task_scorer, metric as your_task_metric

__all__ = [..., "your_task_accuracy", "your_task_scorer", "your_task_metric"]
```

### 4. Register in tasks.py

Add to the `TASKS` dictionary in `tasks.py`:

```python
from models import your_task_predict, YOUR_TASK_PROMPT
from metrics import your_task_accuracy, your_task_metric, your_task_scorer
from datasets import get_your_task_data

TASKS = {
    # ... existing tasks ...
    "your_task": {
        "name": "Your Task Name",
        "get_data": get_your_task_data,
        "predict_fn": your_task_predict,
        "prompt_template": YOUR_TASK_PROMPT,
        "prompt_name": "your_task_prompt",
        "metric": your_task_metric,
        "scorer": your_task_scorer,
        "accuracy_fn": your_task_accuracy,
        "gepa_max_calls": 20,  # Max optimization iterations
        "input_fields": ["input"],
        "output_field": "output",
    },
}
```

Then run:
```bash
python main.py --task your_task --skip-optimization
```

### Why Standardized Names?

This naming convention provides:
- **Consistency**: All task files export the same member names
- **Clarity**: Reading any task file shows familiar exports (`predict`, `accuracy`, `metric`)
- **Maintainability**: New tasks follow the same pattern
- **Clean code**: Individual task files are self-contained with standard interfaces

## Architecture Overview

### Task Registry Pattern

The entire application is built around a **task registry pattern** (`tasks.py`). Each task is a self-contained unit with:
- Dataset loader function (returns dicts with "inputs" and "expectations")
- Predict function (takes keyword args, returns string)
- Accuracy evaluation function
- MLflow metric definition
- GEPA optimization parameters
- Input/output field definitions

This registry drives the generic workflow in `main.py`, allowing new tasks to be added without modifying the core evaluation logic.

### Data Format

All datasets use a consistent dictionary format:
```python
{
    "inputs": {
        "field1": "value1",
        "field2": "value2",
        # ... task-specific input fields
    },
    "expectations": {
        "output_field": "expected_value"
    }
}
```

This replaces the previous `dspy.Example` format and is compatible with MLflow evaluation APIs.

### Model Interface

All models expose a `predict()` function:
```python
def predict(**kwargs) -> str:
    """
    Args:
        **kwargs: Task-specific input fields

    Returns:
        Prediction as a string
    """
```

This simple interface makes it easy to swap models or integrate with MLflow.

## Module Reference

### `config.py`
- `get_openai_client()`: Get OpenAI API client instance
- `get_default_model()`: Get default model name (gpt-4o-mini)
- `MODEL_CONFIGS`: Model configuration dictionary

### `datasets/`
Each task has its own dataset file:
- `sentiment.py`: Sentiment classification data and loader
- `qa.py`: Question answering data and loader
- `math.py`: Math word problems data and loader
- Returns format: `(train_data, dev_data)` as lists of dicts

### `models/`
Each task has its own model file with prompt templates:
- `sentiment.py`: Sentiment classification prompt and predict function
- `qa.py`: QA prompt and predict function
- `math.py`: Math ReAct loop implementation with calculator tool
- Each exposes a `predict(**kwargs) -> str` function

### `metrics/`
Each task has its own metrics file:
- `sentiment.py`: `sentiment_accuracy()` function and `sentiment_metric` MLflow metric
- `qa.py`: `qa_accuracy()` function and `qa_metric` MLflow metric
- `math.py`: `math_accuracy()` function and `math_metric` MLflow metric

### `tasks.py`
- Task configuration registry (`TASKS` dictionary)
- Imports and organizes all task components
- Defines prompt templates and GEPA parameters

### `main.py`
- Generic evaluation functions that work with all tasks
- Command-line interface for task selection
- MLflow GEPA workflow (placeholder implementation)
- Baseline evaluation fully functional

## Expected Output

Running any task with `--skip-optimization` will show:

1. Baseline model performance on dev set
2. Example-by-example predictions and correctness
3. Overall accuracy score
4. Demo predictions on new examples

Running without `--skip-optimization` will additionally show:
- MLflow GEPA placeholder message
- Instructions for completing GEPA integration

## Troubleshooting

### `ModuleNotFoundError: No module named 'mlflow'`
Make sure you've installed dependencies and activated your virtual environment:
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### `command not found: python`
Use `python3` instead:
```bash
python3 main.py --task sentiment --skip-optimization
```

### API Key Errors
Ensure your API key is set as an environment variable:
```bash
# Check if it's set
echo $OPENAI_API_KEY

# Set it if needed
export OPENAI_API_KEY='your-api-key-here'
```

### "Error: OPENAI_API_KEY environment variable not set"
The script checks for the API key before running. Make sure to export it:
```bash
export OPENAI_API_KEY='sk-...'
```

### Rate Limit and Quota Errors

If you encounter `RateLimitError` or quota exceeded errors:

**Error: "You exceeded your current quota"**
This means you've hit your OpenAI billing/usage cap:
1. Check your usage at https://platform.openai.com/usage
2. Verify you have credits or add more at https://platform.openai.com/settings/organization/billing
3. Create a new API key if needed at https://platform.openai.com/api-keys

**Error: "Rate limit exceeded"**
You're making too many requests per minute. Solutions:

1. **Retry logic is automatic** - The code includes exponential backoff retry logic

2. **Reduce API call volume** if retries aren't enough:
   - Use fewer training examples
   - Reduce ReAct iterations in math task (edit `models/math.py`)
   - Reduce `gepa_max_calls` in `tasks.py`

3. **Adjust retry settings** in `config.py`:
   - Increase `initial_delay` parameter in `with_retry()` function

**Understanding API Call Volume:**
- Each example makes 1 API call
- ReAct tasks (math) make multiple calls per example (up to `max_iters`)
- When GEPA is implemented, it will test multiple prompt variations

### MLflow GEPA Taking Too Long

GEPA optimization involves evolutionary prompt improvement which can take several minutes. To speed up:
- Use `--skip-optimization` for baseline-only evaluation
- Reduce `gepa_max_calls` in `tasks.py` (trades thoroughness for speed)
- Use smaller training datasets

## Migration Notes

This project was migrated from DSPy to direct OpenAI API calls with MLflow integration. Key changes:

### Breaking Changes from DSPy Version
- No longer uses `dspy.Module` or `dspy.Signature`
- Data format changed from `dspy.Example` to plain dictionaries
- Models return strings instead of structured objects
- GEPA optimization uses MLflow instead of DSPy

### What Was Preserved
- Task registry architecture
- Per-task file organization
- Three example tasks (sentiment, qa, math)
- Evaluation workflow structure

### What's Different
- Direct OpenAI API calls instead of DSPy abstraction
- Manual prompt templates instead of DSPy signatures
- Manual ReAct loop for math task (previously `dspy.ReAct`)
- MLflow metrics instead of DSPy metrics
- Simplified model interface

If you need the original DSPy version, check the commit history before `89c3878`.

## Learn More

- [MLflow GenAI Documentation](https://mlflow.org/docs/latest/genai/)
- [MLflow Prompt Engineering](https://mlflow.org/docs/latest/genai/prompt-registry/)
- [OpenAI API Documentation](https://platform.openai.com/docs/api-reference)
- [GEPA Paper/Concept](https://dspy.ai/tutorials/gepa_aime/) (Original DSPy implementation)

## Contributing

This is a working demonstration of MLflow GEPA. Feel free to:
- Add new tasks and datasets
- Experiment with different prompts and models
- Extend evaluation metrics
- Improve optimization parameters
- Add tests and benchmarks
- Share your improvements!

## Roadmap

- [x] Complete MLflow GEPA optimizer integration
- [x] Add error handling for API calls with retry logic
- [x] Implement singleton OpenAI client pattern
- [x] Add MLflow experiment tracking
- [x] Add baseline vs optimized comparison
- [ ] Add unit tests for accuracy functions
- [ ] Add integration tests for GEPA workflow
- [ ] Add more example tasks (summarization, translation, etc.)
- [ ] Expand datasets with more examples
- [ ] Add logging and debugging utilities
