from typing import List

from llm import get_chat_model


def extract_atomic_facts(text: str) -> List[str]:
    """
    Uses LLM to extract and reformulate atomic facts from the input text.

    Args:
        text (str): Input model-generated text.

    Returns:
        List[str]: List of atomic fact strings.
    """
    chat_model = get_chat_model()  # Get the ChatOpenAI instance
    prompt = f"""
You are a helpful assistant. Given the following text, extract and reformulate the atomic factual statements. 
Each statement should be self-contained and concise. Output one fact per line.

TEXT:
\"\"\"{text}\"\"\"
    """

    response = chat_model.invoke(prompt)  # Use the chat model to get the response

    print(response.content)

    facts = [line.strip() for line in response.content.split("\n") if line.strip()]
    return facts
