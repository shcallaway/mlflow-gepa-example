"""Question answering model using OpenAI API."""

from config import get_openai_client, get_default_model


# Prompt template for question answering
QA_PROMPT = """Answer the question based on the provided context. Provide a concise and accurate answer.

Context: {context}

Question: {question}

Let's think step by step:
1. Identify the relevant information in the context
2. Extract the key facts that answer the question
3. Formulate a clear and concise answer

Answer:"""


def qa_predict(question: str, context: str) -> str:
    """
    Answer a question based on the provided context.

    Args:
        question: The question to answer
        context: The context passage containing the answer

    Returns:
        The predicted answer
    """
    client = get_openai_client()
    model = get_default_model()

    # Format the prompt
    prompt = QA_PROMPT.format(question=question, context=context)

    # Call OpenAI API
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=200
    )

    # Extract and return the answer
    answer = response.choices[0].message.content.strip()
    return answer
