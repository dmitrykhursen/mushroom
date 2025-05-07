from pyexpat import model
from typing import Dict, List, Tuple

import json

from .llm import get_chat_model


def align_facts_to_text(facts: List[str], model_input: str, model_output_text: str) -> List[Dict]:
    """
    Uses LLM to find start-end character spans for each fact in the original text.

    Args:
        facts (List[str]): List of atomic facts.
        original_text (str): Original model output text.

    Returns:
        List[Dict]: Each dict has 'fact', 'start', 'end' indices.
    """
    facts_text = "\n".join(f"- {fact}" for fact in facts)

    system_prompt = prompt = """
    You are given:
    - model_input: the user’s original question or instruction  
    - model_output_text: the raw text the model produced  
    - facts: facts extracted from the model_output text

    Your job is to locate a key word in the model_output_text, for each fact. This key word should express the fact and distinct its meaning from other facts.
    - Try to find such words, which would most likely be marked as untrue, if the original fact was untrue.
    - Two facts cannot have the same keyword

    Find the start and end character index inside the model_output_text for each of the located words.
    - START and END indexes, should be integers

    Output JSON list like:
    [
      {{ "fact": "fact text", "word": key_word", "start": START, "end": END}},
      ...
    ]

    Example1:
    model_input:"Is the Arts and Humanities Citation Index still maintained?"
    model_output_text:" Yes, the A&HCI is still being maintained by the University of Chicago. " {
    facts: [
        { "id": "F1", "text": "The A&HCI is still being maintained." },
        { "id": "F2", "text": "The University of Chicago is maintaining the A\&HCI." },
        { "id": "F3", "text": "A&HCI stands for Arts and Humanities Citation Index." }
    ]
    }
    Output1:
    [
      { "fact": "The A&HCI is still being maintained.", "word": "Yes", "start": 1, "end": 3 },
      { "fact": "The University of Chicago is maintaining the A&HCI.", "word": "University of Chicago", "start": 50, "end": 68 },
      { "fact": "A&HCI stands for Arts and Humanities Citation Index.", "word": "A&HCI", "start": 10, "end": 15 }
    ]
    """

    user_prompt = f"""
model_input: "{model_input}"
model_output_text"{model_output_text}"
facts: {facts_text}

FACTS:
{facts_text}
"""

    chat = get_chat_model()
    response = chat.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ])
    # Parse the JSON output
    try:
        result = json.loads(response.content)
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON from LLM: {response.content}")

    for i, entry in enumerate(result):
        if not isinstance(entry, dict):
            raise ValueError(f"Entry {i} not a dict: {entry}")
        for key in ("fact","word","start","end"):
            if key not in entry:
                raise ValueError(f"Entry {i} missing {key}")

    # simply return the LLM’s list of span‐dicts
    return result

def align_fact_to_text(fact: str, original_text: str) -> Dict:
    """
    Uses LLM to find start-end character spans for a single fact in the original text.

    Args:
        fact (str): An atomic fact.
        original_text (str): Original model output text.

    Returns:
        Dict: A dict with 'start' and 'end' indices.
    """

    prompt = """
    You are given:
    - model_input: the user’s original question or instruction  
    - model_output_text: the raw text the model produced  
    - facts: facts extracted from the model_output text

    Your job is to locate a key word or continuous span of key words in the model_output_text, for each fact. 
    - This key word should express the fact and distinct its meaning from other facts.
    - Try to find such words, which would most likely be marked as untrue, if the original fact was untrue.

    Find the start and end character index inside the model_output_text for each of the located words.
    - START and END indexes, should be integers

    Output JSON list like:
    [
      {{ "fact": "fact text", "word": "key word", "start": START, "end": END}},
      ...
    ]

    Example1:
    model_input:"Is the Arts and Humanities Citation Index still maintained?"
    model_output_text:" Yes, the A&HCI is still being maintained by the University of Chicago. " {
    facts: [
        { "id": "F1", "text": "The A&HCI is still being maintained." },
        { "id": "F2", "text": "The University of Chicago is maintaining the A\&HCI." },
        { "id": "F3", "text": "A&HCI stands for Arts and Humanities Citation Index." }
    ]
    }
    Output1:
    [
      { "fact": "The A&HCI is still being maintained.", "word": "Yes", "start": 1, "end": 3 },
      { "fact": "The University of Chicago is maintaining the A&HCI.", "word": "University of Chicago", "start": 50, "end": 68 },
      { "fact": "A&HCI stands for Arts and Humanities Citation Index.", "word": "A&HCI", "start": 10, "end": 15 }
    ]
    """

    chat_model = get_chat_model()  # Get the ChatOpenAI instance
    response = chat_model.invoke(prompt)  # Use the instance to get the response

    span = eval(
        response.content
    )  # You might want to use json.loads() if response is clean JSON
    return {"start": span["start"], "end": span["end"]}
