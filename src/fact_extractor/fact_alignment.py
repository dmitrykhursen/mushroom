from typing import List, Dict
from llm import get_chat_model

def align_facts_to_text(facts: List[str], original_text: str) -> List[Dict]:
    """
    Uses LLM to find start-end character spans for each fact in the original text.

    Args:
        facts (List[str]): List of atomic facts.
        original_text (str): Original model output text.

    Returns:
        List[Dict]: Each dict has 'fact', 'start', 'end' indices.
    """
    facts_text = "\n".join(f"- {fact}" for fact in facts)
    
    prompt = f"""
Given the original text and a list of extracted atomic facts, find for each fact the start and end character index
inside the original text where the fact appears or is most closely paraphrased.

Output JSON list like:
[
  {{ "fact": "fact text", "start": START_INDEX, "end": END_INDEX }},
  ...
]

OR if the fact is hallucinated (no good match), you can set "start": -1, "end": -1.

ORIGINAL TEXT:
\"\"\"{original_text}\"\"\"

FACTS:
{facts_text}
    """

    chat_model = get_chat_model()  # Get the ChatOpenAI instance
    response = chat_model(prompt)  # Use the instance to get the response
    
    spans = eval(response)  # You might want to use json.loads() if response is clean JSON
    return spans


def align_fact_to_text(fact: str, original_text: str) -> Dict:
    """
    Uses LLM to find start-end character spans for a single fact in the original text.

    Args:
        fact (str): An atomic fact.
        original_text (str): Original model output text.

    Returns:
        Dict: A dict with 'start' and 'end' indices.
    """
    prompt = f"""
Given the original text and an extracted atomic fact, find the start and end character index
inside the original text where the fact appears or is most closely paraphrased.

Output JSON like:
{{ "start": START_INDEX, "end": END_INDEX }}

OR if the fact is hallucinated (no good match), you can set "start": -1, "end": -1.

ORIGINAL TEXT:
\"\"\"{original_text}\"\"\"

FACT:
- {fact}
    """

    chat_model = get_chat_model()  # Get the ChatOpenAI instance
    response = chat_model.invoke(prompt)  # Use the instance to get the response
    
    span = eval(response.content)  # You might want to use json.loads() if response is clean JSON
    return {"start": span["start"], "end": span["end"]}