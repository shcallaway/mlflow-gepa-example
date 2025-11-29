"""Math word problem solver using ReAct pattern."""

import re
from config import get_openai_client, get_default_model


def calculate(expression: str) -> str:
    """
    Evaluate a mathematical expression safely.

    Supports basic arithmetic operations: +, -, *, /, (), and numbers.

    Args:
        expression: A mathematical expression to evaluate (e.g., "25 * 4 + 10")

    Returns:
        A string containing the result or error message

    Examples:
        calculate("10 + 5") returns "15"
        calculate("(100 - 50) / 2") returns "25.0"
        calculate("100 / 8") returns "12.5"
    """
    try:
        # Sanitize input - only allow numbers, operators, parentheses, spaces
        allowed_chars = set('0123456789+-*/.()\t\n ')
        if not all(c in allowed_chars for c in expression):
            return "Error: Invalid characters in expression. Use only numbers and +, -, *, /, (, )"

        # Evaluate the expression safely
        result = eval(expression, {"__builtins__": {}}, {})

        # Return as string (handle both int and float results)
        if isinstance(result, float) and result.is_integer():
            return str(int(result))
        return str(result)

    except ZeroDivisionError:
        return "Error: Division by zero"
    except SyntaxError:
        return "Error: Invalid mathematical syntax"
    except Exception as e:
        return f"Error: {str(e)}"


# ReAct prompt template for math word problems
MATH_REACT_PROMPT = """Solve this math word problem step by step using the ReAct (Reasoning and Acting) approach.

Problem: {problem}

You can use the calculate(expression) tool to evaluate mathematical expressions.

Use this exact format:
Thought: [Your reasoning about what to do next]
Action: calculate(expression) [Only if you need to compute something]
Observation: [The result will be provided here]
... (You can repeat Thought/Action/Observation if needed)
Thought: [Final reasoning leading to the answer]
Answer: [Just the numeric result, nothing else]

Available tool:
- calculate(expression): Evaluates a mathematical expression and returns the numeric result

Begin solving!

{history}"""


# Regex patterns for parsing
ACTION_PATTERN = r'Action:\s*calculate\((.*?)\)'
ANSWER_PATTERN = r'Answer:\s*([0-9.]+)'


def extract_calculation(text: str) -> str:
    """
    Extract the calculation expression from an Action line.

    Args:
        text: The LLM response containing "Action: calculate(...)"

    Returns:
        The expression to calculate, or empty string if not found
    """
    match = re.search(ACTION_PATTERN, text, re.IGNORECASE)
    return match.group(1).strip() if match else ""


def extract_answer(text: str) -> str:
    """
    Extract the final answer from the response.

    Args:
        text: The LLM response containing "Answer: X"

    Returns:
        The extracted numeric answer, or the stripped text as fallback
    """
    match = re.search(ANSWER_PATTERN, text, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Fallback: look for "Answer:" and take the next line or token
    answer_idx = text.lower().find("answer:")
    if answer_idx != -1:
        after_answer = text[answer_idx + 7:].strip()
        # Take first token/line
        first_token = after_answer.split()[0] if after_answer.split() else after_answer
        # Remove non-numeric characters
        numeric = re.sub(r'[^0-9.]', '', first_token)
        if numeric:
            return numeric

    # Last resort: return stripped text
    return text.strip()


def math_predict(problem: str) -> str:
    """
    Solve a math word problem using ReAct pattern.

    Args:
        problem: The math word problem to solve

    Returns:
        The predicted numeric answer
    """
    client = get_openai_client()
    model = get_default_model()

    conversation = []
    max_iterations = 2

    for i in range(max_iterations):
        # Format the prompt with current conversation history
        prompt = MATH_REACT_PROMPT.format(
            problem=problem,
            history="\n".join(conversation)
        )

        # Call OpenAI API
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500
        )

        text = response.choices[0].message.content

        # Check if there's a tool call (Action: calculate(...))
        if "calculate(" in text.lower():
            expr = extract_calculation(text)
            if expr:
                # Execute the calculation
                result = calculate(expr)
                # Add to conversation history
                conversation.append(text)
                conversation.append(f"Observation: {result}")
                continue

        # Check for final answer
        if "answer:" in text.lower():
            return extract_answer(text)

    # If we exit the loop without finding an answer, try to extract anyway
    return extract_answer(text)
