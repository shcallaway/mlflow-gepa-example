# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a modular prompt optimization project demonstrating the **GEPA (Generative Evolutionary Prompt Adaptation)** concept for automatic prompt optimization across multiple NLP tasks. The project uses direct OpenAI API calls with a task registry architecture for multi-task support.

**Current Status**: The project has been migrated from DSPy to MLflow GEPA. Baseline evaluation is fully functional, but GEPA optimization is a placeholder implementation awaiting completion.

## Common Commands

### Setup and Installation
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set API key (required)
export OPENAI_API_KEY='your-api-key-here'
```

### Running Tasks
```bash
# Run sentiment classification (default) - baseline only
python main.py --task sentiment --skip-optimization

# Run question answering - baseline only
python main.py --task qa --skip-optimization

# Run math word problems - baseline only
python main.py --task math --skip-optimization

# Run with GEPA placeholder (shows intended workflow)
python main.py --task sentiment

# Use python3 if python command not available
python3 main.py --task sentiment --skip-optimization
```

## Architecture

### Core Pattern: Task Registry System

The entire application is built around a **task registry pattern** (`tasks.py`). Each task is a self-contained unit with:
- Dataset loader function (returns dicts with "inputs" and "expectations")
- Predict function (direct OpenAI API call with prompt template)
- Accuracy evaluation function
- MLflow metric definition
- GEPA optimization parameters
- Input/output field definitions

This registry drives the generic workflow in `main.py`, allowing new tasks to be added without modifying the core evaluation logic.

### Key Architectural Components

**1. Task Configuration (`tasks.py`)**
- Central registry (`TASKS` dict) maps task names to configurations
- Each task config specifies: data loader, predict function, accuracy function, MLflow metric, GEPA params, and field names
- Main workflow uses this registry to run tasks generically

**2. Generic Workflow (`main.py`)**
- Single workflow handles all tasks using dynamic field access
- Uses `task_config["input_fields"]` to extract inputs from examples
- Formats output based on task type (checks `input_fields` to distinguish tasks)
- GEPA optimization is currently a placeholder (see `main.py:119-165`)

**3. Modular Organization**
```
datasets/    # Per-task data loaders (sentiment.py, qa.py, math.py)
models/      # Per-task prompt templates + predict functions (sentiment.py, qa.py, math.py)
metrics/     # Per-task evaluation functions + MLflow metrics (sentiment.py, qa.py, math.py)
config.py    # OpenAI client configuration
tasks.py     # Task registry (glues everything together)
main.py      # Generic workflow orchestration
```

**4. Direct OpenAI API Integration**
- All models use direct `client.chat.completions.create()` calls
- Prompts are string templates formatted with task inputs
- Models return raw string predictions
- MLflow metrics defined using `make_judge()` from MLflow 3.4+

**5. Data Format**
- All datasets return lists of dictionaries:
  ```python
  {
      "inputs": {"field1": "value1", "field2": "value2"},
      "expectations": {"output": "expected_value"}
  }
  ```
- This format replaced the previous `dspy.Example` objects
- Compatible with MLflow evaluation APIs

### GEPA Optimization Flow (Placeholder)

**Note**: The GEPA optimizer is not currently implemented. The intended flow would be:

1. Baseline model evaluated using direct API calls with initial prompts
2. MLflow GEPA optimizer would take: metric function, train data, val data
3. Reflection LM would generate prompt variations during optimization
4. Optimizer would compile optimized prompt using trainset and valset
5. Optimized prompt would be evaluated on valset for comparison

**To complete**: See `main.py:119-165` for placeholder code and MLflow documentation.

### Adding New Tasks

To add a task, create 3 files:

1. `datasets/your_task.py` - Define data and `get_data()` loader
2. `models/your_task.py` - Define prompt template and `predict()` function
3. `metrics/your_task.py` - Define `accuracy()` and MLflow metric

Then register in `tasks.py`:
```python
from models import your_task_predict
from metrics import your_task_accuracy, your_task_metric
from datasets import get_your_task_data

TASKS = {
    "your_task": {
        "name": "Your Task Name",
        "get_data": get_your_task_data,
        "predict_fn": your_task_predict,
        "accuracy_fn": your_task_accuracy,
        "metric": your_task_metric,
        "gepa_max_calls": 20,
        "input_fields": ["field1", "field2"],
        "output_field": "output",
        "prompt_name": "your_task_prompt",
        "prompt_template": "Your prompt template here",
    },
}
```

Update `__init__.py` files in each directory to export your new functions/classes.

## Important Implementation Details

### Data Format
- Dataset loaders return lists of dictionaries
- Format: `{"inputs": {...}, "expectations": {...}}`
- Use dictionary access like `example["inputs"]["field"]` to get values
- The `inputs` dict contains all input fields
- The `expectations` dict contains the expected output(s)

### Predict Function Interface
- Signature: `predict(**kwargs) -> str`
- Takes task-specific keyword arguments (e.g., `text`, `question`, `context`)
- Returns a string prediction
- Should create OpenAI client and make API call
- Example:
  ```python
  def predict(text: str) -> str:
      client = get_openai_client()
      model = get_default_model()
      prompt = PROMPT_TEMPLATE.format(text=text)
      response = client.chat.completions.create(
          model=model,
          messages=[{"role": "user", "content": prompt}],
          temperature=0.0,
      )
      return response.choices[0].message.content.strip()
  ```

### Accuracy Functions
- Signature: `accuracy(gold: Dict, pred: str) -> bool`
- Must compare `gold["expectations"][field]` with `pred` (string)
- Return `True` for correct, `False` for incorrect
- Example:
  ```python
  def accuracy(gold: Dict, pred: str) -> bool:
      expected = str(gold["expectations"]["sentiment"]).lower().strip()
      predicted = str(pred).lower().strip()
      return expected == predicted
  ```

### MLflow Metrics
- Created using `make_judge()` from `mlflow.genai.judges` (MLflow 3.4+)
- Each judge includes: name, instructions (with template variables), feedback type, and model
- Uses template syntax: `{{ inputs }}`, `{{ outputs }}`, `{{ expectations }}`
- Metrics are defined alongside accuracy functions in `metrics/*.py`
- Used for future GEPA integration and MLflow tracking

### OpenAI Client Configuration
- `config.py` provides `get_openai_client()` for client instantiation
- `get_default_model()` returns the default model name ("gpt-4o-mini")
- API keys read from `OPENAI_API_KEY` environment variable
- **Note**: Current implementation creates a new client on each call (should be refactored to singleton)

### Math Task - ReAct Implementation
- `models/math.py` implements a manual ReAct (Reasoning + Acting) loop
- Parses thoughts and actions from LLM responses
- Simulates calculator tool for arithmetic operations
- Replaces the previous `dspy.ReAct` implementation
- Max iterations set to prevent infinite loops

## Dependencies

- `mlflow>=3.5.0` - MLflow with GenAI features (GEPA support when implemented)
- `openai>=1.0.0` - OpenAI API client

Requires Python 3.9+.

## Known Limitations and TODOs

### Current Limitations
1. **GEPA Optimization Not Implemented**: The optimizer in `main.py` is a placeholder
2. **No Error Handling**: API calls lack try-except blocks and retry logic
3. **Client Creation**: OpenAI client is recreated on every prediction (inefficient)
4. **No Tests**: Migration lacks tests comparing to previous DSPy implementation
5. **Manual ReAct**: Math task's ReAct loop could have edge cases

### Priority TODOs
1. Complete MLflow GEPA optimizer integration
2. Add error handling and retry logic to all predict functions
3. Implement singleton pattern for OpenAI client
4. Add unit tests for accuracy functions
5. Add integration tests for end-to-end workflows
6. Document GEPA optimization workflow once implemented

## Migration from DSPy

This project was migrated from DSPy framework to direct OpenAI API calls. Key changes:

### What Changed
- **Framework**: DSPy → Direct OpenAI API calls + MLflow
- **Data Format**: `dspy.Example` objects → Plain dictionaries
- **Models**: `dspy.Module` classes → `predict()` functions with prompts
- **Predictions**: Structured objects → Raw strings
- **Metrics**: DSPy metrics → Custom accuracy functions + MLflow metrics
- **GEPA**: DSPy GEPA optimizer → MLflow GEPA (placeholder)

### What Stayed the Same
- Task registry architecture pattern
- Per-task file organization
- Three example tasks (sentiment, qa, math)
- Generic workflow in `main.py`

### Breaking Changes
- Cannot use `dspy.Example.with_inputs()`
- Cannot use `prediction.field` (predictions are strings)
- Cannot import `dspy` modules
- Different data loader return format
- Different metric function signatures

### Why the Migration
- More direct control over prompts
- Easier to understand and debug
- Simpler dependencies
- Preparation for MLflow GEPA integration

## Troubleshooting

### Common Issues

**API Key Not Set**
- Ensure `OPENAI_API_KEY` is exported: `export OPENAI_API_KEY='sk-...'`

**MLflow GEPA Not Working**
- This is expected. GEPA optimization is a placeholder. Use `--skip-optimization` flag.

**Rate Limits**
- Add retry logic to predict functions (not currently implemented)
- Reduce number of training examples
- Use smaller datasets for testing

**Module Not Found**
- Activate virtual environment: `source venv/bin/activate`
- Install dependencies: `pip install -r requirements.txt`

## Code Style and Conventions

- Use type hints for function parameters and return values
- Document functions with docstrings
- Follow the task registry pattern for new tasks
- Keep task code isolated in per-task files
- Use the standard data format: `{"inputs": {...}, "expectations": {...}}`
- Predict functions should return strings
- Accuracy functions should take `(gold: Dict, pred: str)` and return `bool`
