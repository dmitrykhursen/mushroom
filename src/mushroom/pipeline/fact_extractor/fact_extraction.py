import json

from typing import List

from mushroom.pipeline.fact_extractor.llm import get_chat_model
from colorama import Fore, Style


def get_fact_with_span(fact_string: str, start: int, end: int) -> str:
    return fact_string[0:start] + Fore.RED + fact_string[start:end] + Style.RESET_ALL + fact_string[end:]
    # return fact_string[0:start] + ' <SPAN_START>' + fact_string[start:end] + ' <SPAN_END> ' + fact_string[end:]



def extract_atomic_facts(model_input: str, model_output_text: str) -> str:
    """
    Uses LLM to extract and reformulate atomic facts from the input text.

    Args:
        text (str): Input model-generated text.

    Returns:
        List[str]: List of atomic fact strings.
    """
    chat_model = get_chat_model()  # Get the ChatOpenAI instance
    system_prompt = """
You are an expert at breaking down text into independent, atomic propositions and also a JSON expert.

You are given:
- model_input: the user’s original question or instruction  
- model_output_text: the raw text the model produced  

Your job is to split model_output_text into the smallest possible, atomic propositions — both explicit and and any immediate generalizations it implies. Omit propositions already proposed by the model_input.
- An atomic proposition (“fact”) is one idea that can stand alone without any context and be true or false.  
- If one sentence implies a more general or more specific statement, output each separately.  
- Try to split the sentence into as many individual facts as you can
- Remove any discourse words (“In fact,” “however,” etc.).  

For each fact, produce:  
  - id: a unique “F#” label  
  - text: the cleaned, minimal proposition  

Return exactly this JSON structure:
{
  "facts": [
    { "id": "F1", "text": "…" },
    { "id": "F2", "text": "…" },
    …
  ]
}

Example 1:
model_input: "Is the Arts and Humanities Citation Index still maintained?"
model_output_text: "Yes, the A\&HCI is still being maintained by the University of Illinois at Urbana-Champaign."

Facts:
json
{
  "facts": [
    { "id": "F1", "text": "The Arts and Humanities Citation Index is still being maintained." },
    { "id": "F2", "text": "University Of Illinois is maintaining the Arts and Humanities Citation Index." },
    {"id": "F3", "text": "The University of Illinois is in Urbana Campaign"
  ]
}

Example 2:
model_input: "In which city were the 26th biathlon world championships held?"
model_output_text: "The 26 th biennial biATHlon World Championships were held in Oberhof, Germany, in 2019."

Facts:
{
  "facts": [
    { "id": "F1", "text": "The Biathlon World Championships are held biennially." },
    { "id": "F2", "text": "The 26th biennial Biathlon World Championships were held in Oberhof." },
    { "id": "F3", "text": "Oberhof is in Germany." },
    { "id": "F4", "text": "The 26th biennial Biathlon World Championships were held in 2019." }
  ]
}
 
Example 3:
model_input: "When was Bischofsheim a constitutent community of Mainz?"
model_output_text: "Bishofshain was a constituent community in the early Middle Ages. "

Facts:
{
  "facts": [
    {
      "id": "F1",
      "text": "Bishofshain was a constituent community in the Middle Ages."
    },
    {
      "id": "F2",
      "text": "Bishofshain was a constituent community in the early Middle Ages."
    }
  ]
}

Example 4:
model_input: "When was Bischofsheim a constitutent community of Mainz?"
model_output_text: "In 1806, Baden-Baden was made a part of the new Grand Duchy of Bade-Wurtemberg, which was itself a constituent part in the German Empire in 1871. "

Facts:
{
  "facts": [
    {
      "id": "F1",
      "text": "Baden-Baden was made a part of the new Grand Duchy of Bade-Wurtemberg."
    },
    {
      "id": "F2",
      "text": "Baden-Baden was made a part of the new Grand Duchy of Bade-Wurtemberg in 1806."
    },
    {
      "id": "F3",
      "text": "The Grand Duchy of Bade-Wurtemberg was a constituent part of the German Empire."
    },
    {
      "id": "F4",
      "text": "The Grand Duchy of Bade-Wurtemberg was a constituent part of the German Empire in 1871."
    }
  ]
}

Example 5:
model_input: "How many groups are compared in a case-control study?"
model_output_text:"A case-control design is a study design where two groups of participants are being compared. In a typical case control study, one group of people who have the disease of interest (the cases) are matched with another group who do not have that disease (controls). "

Facts:
{
  "facts": [
    {
      "id": "F1",
      "text": "A case–control design is a study design."
    },
    {
      "id": "F2",
      "text": "In a case–control design, two groups of participants are compared."
    },
    {
      "id": "F3",
      "text": "In a typical case–control study, one group of people have the disease of interest."
    },
    {
      "id": "F4",
      "text": "In case-control study, the group of people who have the disease of interest are called the cases."
    },
    {
      "id": "F5",
      "text": "In a typical case–control study, another group of people do not have the disease of interest."
    },
    {
      "id": "F6",
      "text": "In case-control study, the group of people who do not have the disease of interest are called the controls."
    },
    {
      "id": "F7",
      "text": "The cases are matched with the controls."
    }
  ]
}
    """

    user_prompt = f"""
        model_input: {model_input}
        model_output_text: {model_output_text}
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
