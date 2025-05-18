import json
import re
import sys
from typing import List, Dict
from colorama import Fore, Style

from .llm import get_chat_model
from .prompts import FACT_EXTRACTION, INDEXING

from mushroom.pipeline.interface import Entry

MAX_TRIES = 3
REQUIRED_KEYS = {"Predicate", "Subject", "Object", "Reformulation"}


class FactExtraction:
    def __init__(self):
        self.chat_model = get_chat_model()
        self.system_prompts = {
            "extract": FACT_EXTRACTION,
            "index": INDEXING
        }

    def run(self, entries: List[Entry]) -> List[Entry]:
        """
            Apply fact extraction to a list of Entries, storing results in `entry.fact_spans`.

            Parameters
            ----------
            entries : List[Entry]

            Returns
            -------
            List[Entry]
        """
        for entry in entries:
            entry.fact_spans = self.extract_atomic_facts(entry.model_output_text)
        return entries

    def extract_atomic_facts(self, model_output_text: str) -> List[Dict]:
        """
            Extract reformulated binary facts from text and locate each Object’s character span.

            Args:
                model_output_text (str): Raw model output containing sentences.

            Returns:
                List[Dict]: Each dict has:
                    - "fact": reformulated fact string
                    - "start": start index of Object in the text
                    - "end": end index of Object in the text
        """

        facts = self.get_facts_with_objects(model_output_text)

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
                occurence_index = self.determine_index_occurence(model_output_text, fact["Reformulation"], fact["Object"], len(matches)) - 1
                start, end = matches[occurence_index], matches[occurence_index] + len(fact["Object"])

            fact_claims_with_spans.append({
                "fact": fact["Reformulation"],
                "start": start,
                "end": end,
            })
  
        return fact_claims_with_spans


    def get_facts_with_objects(self, model_output_text: str):
        """
        Uses LLM to extract and reformulate atomic facts from the input text.

        Args:
            text (str): Input model-generated text.

        Returns:
            List[str]: List of atomic fact strings.
        """
        system_prompt = self.system_prompts["extract"]

        additional_input = ""
        for attempt in range(0, MAX_TRIES):
            missing = None
            user_prompt = f"""
                {additional_input}
                text: {model_output_text}
             """

            response = self.chat_model.invoke(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                    ])  # Use the chat model to get the response

            # Parse the JSON output
            try:
                result = json.loads(response.content)
            except json.JSONDecodeError:
                print(f"Fact extraction [Attempt {attempt}]: JSON parse error, retrying…", file=sys.stderr)
                additional_input = "!!! Your output was not a valid JSON format. Make sure you are outputting valid JSON."
                continue

            # Validate 
            facts = result.get("facts")
            for fact in facts:
                missing = REQUIRED_KEYS - fact.keys()
                if missing:
                    break

            if missing:
                additional_input = f"Each fact needs to have the following keys: {(' ').join(REQUIRED_KEYS)}"
                print(f"Fact extraction [Attempt {attempt}]: JSON parse error (facts do not contain all keys), retrying…", file=sys.stderr)
                continue

            return facts
            

        print(f"Fact extraction: All atempts at producing JSON failed, I am giving up", file=sys.stderr)
        return []



    def determine_index_occurence(self, model_output_text: str, fact: str, fact_object: str, number_of_occurences: int) -> int:
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
        system_prompt = self.system_prompts["index"]
        additional_input = ""
        for attempt in range(0, MAX_TRIES):

            user_prompt = f"""
                {additional_input}
                text: {model_output_text}
                fact: {fact}
                fact_object: {fact_object}
                number_of_occurencies: {number_of_occurences}
             """

            response = self.chat_model.invoke(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                    ])  # Use the chat model to get the response

            try: 
                response = int(response.content)
            except ValueError:
                print(f"Index occurence [Attempt {attempt}]: Failed to produce int, retrying...", file=sys.stderr)
                additional_input = "!!!!!! You did not produce a number. Make sure you ar eproducing a singloe number !!!!!"
                continue

            if response < 1 or response > number_of_occurences:
                print(f"Index occurence [Attempt {attempt}]: Failed to produce int in valid range, retrying...", file=sys.stderr)
                additional_input = "!!!!!! The number was not within the specified range` !!!!!"
            else:
                return response

        return 1
    

