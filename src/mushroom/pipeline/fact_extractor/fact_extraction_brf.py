import json

from typing import List, Dict

from .llm import get_chat_model

import re
from colorama import Fore, Style

def extract_atomic_facts(model_output_text: str) -> List[Dict]:
    """
    
    """
    facts = get_facts_with_objects(model_output_text)

    fact_claims_with_spans = []

    for fact in facts:
        
        # get all number of occurrences of the fact-object in the text
        matches = [m.start() for m in re.finditer(re.escape(fact["Object"]), model_output_text)]
        # if there are no matches, skip this fact
        
        if len(matches) == 0:
            continue
        
        # if there is exactly one match of the Object in the original model_output, use it
        elif len(matches) == 1:
            start, end = matches[0], matches[0] + len(fact["Object"])
        
        # if there are multiple matches, use the LLM to determine which one to use
        else:
            occurence_index = determine_index_occurence(model_output_text, fact["Reformulation"], fact["Object"], len(matches)) - 1
            start, end = matches[occurence_index], matches[occurence_index] + len(fact["Object"])

        fact_claims_with_spans.append({
            "fact": fact["Reformulation"],
            "start": start,
            "end": end,
        })

        print('\n')
        print(Fore.BLUE + f"Fact: {fact['Reformulation']!r}" + Style.RESET_ALL)
        print(Fore.BLUE + f" Object: {fact['Object']!r}" + Style.RESET_ALL)
        print(Fore.BLUE + f"   Span chars [{start}:{end}]" + Style.RESET_ALL)
    
    return fact_claims_with_spans


def get_facts_with_objects(model_output_text: str):
    """
    Uses LLM to extract and reformulate atomic facts from the input text.

    Args:
        text (str): Input model-generated text.

    Returns:
        List[str]: List of atomic fact strings.
    """
    chat_model = get_chat_model()  # Get the ChatOpenAI instance
    system_prompt = """
You are given: 
- text: original text  

Your job is to split the text into Binary Relational Facts, each consisting of Predicate, Subject and Object.  
Togethher with these three elements, you will also provide a reformulation of the fact.

For each fact, produce:  
  - Predicate : the action or state of being
  - Subject : the entity that performs the action or is in the state
  - Object : the entity that is affected by the action or state
  - Reformulation : a reformulation of the fact 

Return exactly this JSON structure:
{
  "facts": [
    { "Predicate": "...", "Subject": "...", "Object": "...", "Reformulation": "..."},
    { "Predicate": "...", "Subject": "...", "Object": "...", "Reformulation": "..."},
    …
  ]
}

# Example:
text: "Petra van Stoveren won a silver medal in the 2008 Summer Olympics in Beijing, China."

Facts:
json
{
  "facts": [
  {
    "Predicate": "won",
    "Subject": "Petra van Stoveren",
    "Object": "silver medal",
    "Reformulation": "Petra van Stoveren won a silver medal."
  },
  {
    "Predicate": "won medal in event",
    "Subject": "Petra van Stoveren",
    "Object": "2008 Summer Olympics",
    "Reformulation": "Petra van Stoveren won a medal in the 2008 Summer Olympics."
  },
  {
    "Predicate": "won medal in location",
    "Subject": "Petra van Stoveren",
    "Object": "Beijing, China",
    "Reformulation": "Petra van Stoveren won a medal in Beijing, China."
  },
  {
    "Predicate": "held in",
    "Subject": "2008 Summer Olympics",
    "Object": "Beijing, China",
    "Reformulation": "The 2008 Summer Olympics were held in Beijing, China."
  }
]
}
    """

    user_prompt = f"""
        text: {model_output_text}
     """

    response = chat_model.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
            ])  # Use the chat model to get the response

    # Parse the JSON output
    try:
        result = json.loads(response.content)
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON from LLM: {response.content}")

    # Validate structure
    if "facts" not in result or not isinstance(result["facts"], list):
        raise ValueError(f"JSON missing 'facts' list: {result}")

    # Return the list of atomic facts
    return result["facts"]


def determine_index_occurence(model_output_text: str, fact: str, fact_object: str, number_of_occurences: int) -> int:
    """
    Determines the index of the occurrence of a fact in the model output text.

    Args:
        model_output_text (str): The original model output text.
        fact (str): The fact derived from the model output text.
        fact_object (str): The Object of the fact.
        number_of_occurences (int): The number of occurrences of the fact string in the text.

    Returns:
        int: The index of the occurrence to be used.
    """
    chat_model = get_chat_model()  # Get the ChatOpenAI instance
    system_prompt = f"""
You are given:
- model_output_text: the original model output text
- fact: a fact from the text
- fact_object: the object of the fact
- number_of_occurences: the number of occurrences of the fact string in the text
Your job is to determine the index of the occurrence of the Object in the model output text which relates to the fact given
There are exactly {number_of_occurences} occurrences of the fact in the text. Retun number from 1 to {number_of_occurences}.
You can use the whole fact to help you find the index of the words in the text.
Return the index of the occurrence to be used, as an integer.
Do not return any other text or explanation.
Return a single integer, like:
1


# Example1:
model_output_text: "Petra van Stoveren won a silver medal in the 2008 Summer Olympics in Beijing, China. Jacob van Stoveren won a silver medal in the 2012 Summer Olympics in London, England."
fact: "Petra van Stoveren won a silver medal."
fact_object: "silver medal"
number_of_occurences: 2
Return:
1

# Example2:
model_output_text: "Petra van Stoveren won a silver medal in the 2008 Summer Olympics in Beijing, China. Jacob van Stoveren won a silver medal in the 2012 Summer Olympics in London, England."
fact: "Jacob van Stoveren won a silver medal."
fact_object: "silver medal"
number_of_occurences: 2
Return:
2
    """

    user_prompt = f"""
        text: {model_output_text}
        fact: {fact}
        fact_object: {fact_object}
     """

    response = chat_model.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
            ])  # Use the chat model to get the response


    print('-__----------------------------------------------------------------------------------------------')
    print(response.content)

    return eval(response.content)
    

