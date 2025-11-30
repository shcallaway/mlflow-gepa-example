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

**Three NLP Tasks:**
- **Sentiment Classification** - Classify text as positive/negative with structured prompts
- **Question Answering** - Answer questions from context (multi-input task)
- **Math Word Problems** - Solve math with ReAct loop and calculator tool

**Complete GEPA Workflow:**
1. Baseline evaluation with direct OpenAI API calls
2. Evolutionary prompt optimization using MLflow GEPA
3. Optimized evaluation and automatic comparison
4. Interactive demo on new examples

**Benefits:**
- Per-task file organization for easy extension
- Add new tasks without modifying core logic
- Experiment with different prompts and models

## Prerequisites

- Python 3.9 or higher
- OpenAI API key ([get one here](https://platform.openai.com/api-keys))
- MLflow 3.5.0+ (installed via requirements.txt)

## Usage

**Note:** Activate your virtual environment and set your API key before running (see Quick Start above).

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

Adding a new task requires creating 3 files following the standardized naming convention. **See `datasets/sentiment.py`, `models/sentiment.py`, and `metrics/sentiment.py` as reference examples.**

### Standardized Export Convention

Each task exports consistent names (without task-specific prefixes):
- **`models/{task}.py`**: `PROMPT`, `predict`
- **`metrics/{task}.py`**: `accuracy`, `scorer_fn`, `metric`
- **`datasets/{task}.py`**: `get_data`

The `__init__.py` files import with task-specific aliases (e.g., `predict as sentiment_predict`).

### Steps to Add a Task

**1. Create `datasets/your_task.py`:**
- Define `get_data()` returning `(train_data, dev_data)`
- Format: `{"inputs": {...}, "expectations": {...}}`
- Update `datasets/__init__.py` with import alias

**2. Create `models/your_task.py`:**
- Define `PROMPT` template with format placeholders
- Define `predict(**kwargs) -> str` using OpenAI API
- Use `with_retry()` for error handling
- Update `models/__init__.py` with import aliases

**3. Create `metrics/your_task.py`:**
- Define `accuracy(gold: Dict, pred: str) -> bool`
- Define `scorer_fn` with `@scorer` decorator for GEPA
- Create MLflow `metric` using `make_judge()`
- Update `metrics/__init__.py` with import aliases

**4. Register in `tasks.py`:**
```python
TASKS = {
    "your_task": {
        "name": "Your Task Name",
        "get_data": get_your_task_data,
        "predict_fn": your_task_predict,
        "prompt_template": YOUR_TASK_PROMPT,
        "prompt_name": "your_task_prompt",
        "metric": your_task_metric,
        "scorer": your_task_scorer,
        "accuracy_fn": your_task_accuracy,
        "gepa_max_calls": 20,
        "input_fields": ["input"],  # Adjust to your fields
        "output_field": "output",
    },
}
```

**5. Test:**
```bash
python main.py --task your_task --skip-optimization
```

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
- Complete MLflow GEPA workflow with optimization
- Baseline and optimized evaluation with comparison

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
- GEPA optimization tests multiple prompt variations (controlled by `gepa_max_calls`)

### MLflow GEPA Taking Too Long

GEPA optimization involves evolutionary prompt improvement which can take several minutes. To speed up:
- Use `--skip-optimization` for baseline-only evaluation
- Reduce `gepa_max_calls` in `tasks.py` (trades thoroughness for speed)
- Use smaller training datasets

## Migration from DSPy

This project was migrated from DSPy to direct OpenAI API calls + MLflow. Key changes: `dspy.Module` → `predict()` functions, `dspy.Example` → plain dicts, DSPy GEPA → MLflow GEPA. The task registry architecture was preserved. See commit `89c3878` for the original DSPy version.

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
