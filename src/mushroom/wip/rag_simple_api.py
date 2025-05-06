import json
import logging
import re

import requests
from mistralai import Mistral


class WikiApi:
    def __init__(self, lang_code):
        self._base_url = (
            f"https://api.wikimedia.org/core/v1/wikipedia/{lang_code}/search/page"
        )

    def search(self, search_string):
        params = {
            "q": search_string,
            "limit": 1,
        }
        api_results = requests.get(self._base_url, params).json()["pages"]
        results = []
        for api_result in api_results:
            cleaned_extract = re.sub(
                r"<span .*?>(.*?)</span>", r"\1", api_result["excerpt"]
            )
            results.append(cleaned_extract)
        return results


class FactSupportExtractor:
    _logger = logging.getLogger("FactSupportExtractor")

    def __init__(
        self,
        mistral_api_key=None,
        wiki_api=WikiApi("en"),
        mistral_model="mistral-large-latest",
    ):
        self._mistral = (
            Mistral(api_key=mistral_api_key) if mistral_api_key is not None else None
        )
        self._mistral_model_name = mistral_model
        self._wiki_api = wiki_api
        if not self._mistral:
            self._logger.info(
                "Mistral is not available (no api key provided?). Facts will be used as is for querying Wiki"
            )

    def find_support(self, facts: list[str]) -> list[list[str]]:
        """
        Givent the fact, tries to find relevant supporting info from Wikipedia
        """
        assert isinstance(facts, list)
        prompt = f"""
                Given this list of facts: '{facts}'
                Formulate a search query for each of them which can be used to find relevant info. Output only the search strings.
                Try to form queries exactly as they can be found in the Wikipedia TITLE.
                You must output list of the same size as the list of facts I provided.
                Each element of the output list is a list of queries relevant to the current fact.
                Output as few queries per fact as possible.
                
                Use these as examples:
                Facts: ['Prague is the capital of the Czech Republic', 'Xiong Ai was a Chinese warlord'],
                Your response: [['Czech Republic'], ['Xiong Ai']];
                Output json list of lists.
                """
        if self._mistral is not None:
            ai_response = self._mistral.chat.complete(
                model=self._mistral_model_name,
                messages=[{"role": "user", "content": prompt}],
                response_format={
                    "type": "json_object",
                },
            )
            queries = json.loads(ai_response.choices[0].message.content)
        else:
            queries = [[fact] for fact in facts]
        fact_supports = []
        for i in range(len(facts)):
            current_fact_supports = []
            for query in queries[i]:
                results = self._wiki_api.search(queries[i])
                current_fact_supports.extend(results)
            fact_supports.append(current_fact_supports)
            self._logger.debug(
                f"For fact '{facts[i]}' the queries are '{queries[i]}' and search results are '{current_fact_supports}'"
            )
        return fact_supports


if __name__ == "__main__":
    logging.config.dict(level=logging.DEBUG)
    # Put your mistral api key here
    api_key = None
    support_finder = FactSupportExtractor(api_key)
    found_support = support_finder.find_support(
        ["Xiong Ai was a Chinese warlord", "all insects and arachnids"]
    )
    print(found_support)
