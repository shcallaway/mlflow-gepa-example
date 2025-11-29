"""Sentiment classification model using OpenAI API."""

from config import get_openai_client, get_default_model


# Prompt template for sentiment classification
SENTIMENT_PROMPT = """Classify the sentiment of the following text as either 'positive' or 'negative'.

Text: {text}

Let's think step by step:
1. Analyze the tone and emotional language used in the text
2. Identify positive or negative indicators (words, phrases, context)
3. Make the final classification based on the overall sentiment

Sentiment:"""


def parse_sentiment(text: str) -> str:
    """
    Parse the sentiment from the LLM response.

    Args:
        text: The raw LLM response

    Returns:
        Either 'positive' or 'negative'
    """
    text_lower = text.lower().strip()

    # Look for sentiment keywords
    if "positive" in text_lower:
        return "positive"
    elif "negative" in text_lower:
        return "negative"

    # Fallback: return first line stripped
    return text.strip().split('\n')[0].strip()


def sentiment_predict(text: str) -> str:
    """
    Predict the sentiment of the given text.

    Args:
        text: The text to classify

    Returns:
        Predicted sentiment: 'positive' or 'negative'
    """
    client = get_openai_client()
    model = get_default_model()

    # Format the prompt
    prompt = SENTIMENT_PROMPT.format(text=text)

    # Call OpenAI API
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=150
    )

    # Parse and return the result
    raw_output = response.choices[0].message.content
    return parse_sentiment(raw_output)
