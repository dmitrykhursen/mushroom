# from pyexpat import model
# from typing import Dict, List, Tuple

# import json

# #from .llm import get_chat_model


# def align_facts_to_text(facts: List[str], model_input: str, model_output_text: str) -> List[Dict]:
#     """
#     Uses LLM to find start-end character spans for each fact in the original text.

#     Args:
#         facts (List[str]): List of atomic facts.
#         original_text (str): Original model output text.

#     Returns:
#         List[Dict]: Each dict has 'fact', 'start', 'end' indices.
#     """
#     facts_text = "\n".join(f"- {fact}" for fact in facts)

#     system_prompt = prompt = """
#     You are given:
#     - model_input: the user’s original question or instruction  
#     - model_output_text: the raw text the model produced  
#     - facts: facts extracted from the model_output text

#     Your job is to locate a **keyword or a phrase** in the model_output_text for each fact. This word or phrase should **uniquely express the fact's meaning**, and be specific enough to distinguish it from other facts.

#     **Guidelines:**
#     - The chosen word or phrase should be the most direct and informative piece of evidence for the fact.
#     - Prefer proper nouns, dates, places, numbers, or key terms that would most likely be marked incorrect if the fact was untrue.
#     - You must **not** select character fragments or mid-word substrings. Always select **whole words or named entities**.
#     - Do **not** include surrounding punctuation (e.g., don't include periods or commas unless they are part of the word, such as "U.S.").
#     - Always output the **start and end character indices** of the word or phrase in the model_output_text (zero-indexed, end-exclusive).

#     Output JSON list like:
#     [
#       {{ "fact": "fact text", "word": key_word", "start": START, "end": END}},
#       ...
#     ]

#     ### Examples

#     Example1:
#     model_input:"Is the Arts and Humanities Citation Index still maintained?"
#     model_output_text:" Yes, the A&HCI is still being maintained by the University of Chicago. " {
#     facts: [
#         { "id": "F1", "text": "The A&HCI is still being maintained." },
#         { "id": "F2", "text": "The University of Chicago is maintaining the A\&HCI." },
#         { "id": "F3", "text": "A&HCI stands for Arts and Humanities Citation Index." }
#     ]
#     }
#     Output1:
#     [
#       { "fact": "The A&HCI is still being maintained.", "word": "Yes", "start": 1, "end": 4 },
#       { "fact": "The University of Chicago is maintaining the A&HCI.", "word": "University of Chicago", "start": 49, "end": 70 },
#       { "fact": "A&HCI stands for Arts and Humanities Citation Index.", "word": "A&HCI", "start": 10, "end": 15 }
#     ]

#     Example2:
#     model_input: "Who is the CEO of OpenAI and where is it headquartered?"
#     model_output_text: "Sam Altman is the CEO of OpenAI, which is based in San Francisco."
#     {
#     facts: [
#         { "id": "F1", "text": "Sam Altman is the CEO of OpenAI." },
#         { "id": "F2", "text": "OpenAI is based in San Francisco." }
#     ]
#     }
#     Output2:
#     [
#         { "fact": "Sam Altman is the CEO of OpenAI.", "word": "Sam Altman", "start": 0, "end": 10 },
#         { "fact": "OpenAI is based in San Francisco.", "word": "San Francisco", "start": 51, "end": 68 }
#     ]

#     Example3:
#     model_input: "What is the capital of France and its population?"
#     model_output_text: "Paris is the capital of France and it has a population of over 2 million."
#     {
#     facts: [
#         { "id": "F1", "text": "Paris is the capital of France." },
#         { "id": "F2", "text": "France has a population of over 2 million." }
#     ]
#     }
#     Output3:
#     [
#         { "fact": "Paris is the capital of France.", "word": "Paris", "start": 0, "end": 5 },
#         { "fact": "France has a population of over 2 million.", "word": "2 million", "start": 63, "end": 72 }
#     ]

#     Example4:
#     model_input: "When did the Berlin Wall fall and what was its significance?"
#     model_output_text: "The Berlin Wall fell in 1989. This event symbolized the end of the Cold War."
#     {
#     facts: [
#         { "id": "F1", "text": "The Berlin Wall fell in 1989." },
#         { "id": "F2", "text": "The fall of the Berlin Wall symbolized the end of the Cold War." }
#     ]
#     }
#     Output4:
#     [
#         { "fact": "The Berlin Wall fell in 1989.", "word": "1989", "start": 26, "end": 30 },
#         { "fact": "The fall of the Berlin Wall symbolized the end of the Cold War.", "word": "Cold War", "start": 78, "end": 86 }
#     ]

#     Example5:
#     model_input: "Tell me about the Mars rover Perseverance and its mission."
#     model_output_text: "NASA's Perseverance rover landed on Mars in February 2021. Its mission is to search for signs of ancient life."
#     {
#     facts: [
#         { "id": "F1", "text": "Perseverance landed on Mars in February 2021." },
#         { "id": "F2", "text": "Perseverance's mission is to search for signs of ancient life." }
#     ]
#     }
#     Output5:
#     [
#         { "fact": "Perseverance landed on Mars in February 2021.", "word": "February 2021", "start": 49, "end": 63 },
#         { "fact": "Perseverance's mission is to search for signs of ancient life.", "word": "ancient life", "start": 104, "end": 116 }
#     ]

#     Example6:
#     model_input: "What are the colors of the flag of Italy?"
#     model_output_text: "The flag of Italy features three vertical stripes in green, white, and red."
#     {
#     facts: [
#         { "id": "F1", "text": "The flag of Italy has a green stripe." },
#         { "id": "F2", "text": "The flag of Italy has a white stripe." },
#         { "id": "F3", "text": "The flag of Italy has a red stripe." }
#     ]
#     }
#     Output6:
#     [
#         { "fact": "The flag of Italy has a green stripe.", "word": "green", "start": 54, "end": 59 },
#         { "fact": "The flag of Italy has a white stripe.", "word": "white", "start": 61, "end": 66 },
#         { "fact": "The flag of Italy has a red stripe.", "word": "red", "start": 72, "end": 75 }
#     ]
#     """

#     user_prompt = f"""
# model_input: "{model_input}"
# model_output_text"{model_output_text}"
# facts: {facts_text}

# FACTS:
# {facts_text}
# """

#     chat = get_chat_model()
#     response = chat.invoke([
#         {"role": "system", "content": system_prompt},
#         {"role": "user", "content": user_prompt}
#     ])
#     # Parse the JSON output
#     try:
#         result = json.loads(response.content)
#     except json.JSONDecodeError:
#         raise ValueError(f"Invalid JSON from LLM: {response.content}")

#     for i, entry in enumerate(result):
#         if not isinstance(entry, dict):
#             raise ValueError(f"Entry {i} not a dict: {entry}")
#         for key in ("fact","word","start","end"):
#             if key not in entry:
#                 raise ValueError(f"Entry {i} missing {key}")

#     # simply return the LLM’s list of span‐dicts
#     return result

# def align_fact_to_text(fact: str, original_text: str) -> Dict:
#     """
#     Uses LLM to find start-end character spans for a single fact in the original text.

#     Args:
#         fact (str): An atomic fact.
#         original_text (str): Original model output text.

#     Returns:
#         Dict: A dict with 'start' and 'end' indices.
#     """

#     prompt = """
#     You are given:
#     - model_input: the user’s original question or instruction  
#     - model_output_text: the raw text the model produced  
#     - facts: facts extracted from the model_output text

#     Your job is to locate a key word or continuous span of key words in the model_output_text, for each fact. 
#     - This key word should express the fact and distinct its meaning from other facts.
#     - Try to find such words, which would most likely be marked as untrue, if the original fact was untrue.

#     Find the start and end character index inside the model_output_text for each of the located words.
#     - START and END indexes, should be integers

#     Output JSON list like:
#     [
#       {{ "fact": "fact text", "word": "key word", "start": START, "end": END}},
#       ...
#     ]

#     Example1:
#     model_input:"Is the Arts and Humanities Citation Index still maintained?"
#     model_output_text:" Yes, the A&HCI is still being maintained by the University of Chicago. " {
#     facts: [
#         { "id": "F1", "text": "The A&HCI is still being maintained." },
#         { "id": "F2", "text": "The University of Chicago is maintaining the A\&HCI." },
#         { "id": "F3", "text": "A&HCI stands for Arts and Humanities Citation Index." }
#     ]
#     }
#     Output1:
#     [
#       { "fact": "The A&HCI is still being maintained.", "word": "Yes", "start": 1, "end": 3 },
#       { "fact": "The University of Chicago is maintaining the A&HCI.", "word": "University of Chicago", "start": 50, "end": 68 },
#       { "fact": "A&HCI stands for Arts and Humanities Citation Index.", "word": "A&HCI", "start": 10, "end": 15 }
#     ]
#     """

#     chat_model = get_chat_model()  # Get the ChatOpenAI instance
#     response = chat_model.invoke(prompt)  # Use the instance to get the response

#     span = eval(
#         response.content
#     )  # You might want to use json.loads() if response is clean JSON
#     return {"start": span["start"], "end": span["end"]}
