#%%
import re
import wikipedia
import faiss
import asyncio
from sentence_transformers import SentenceTransformer
from typing import List, Optional
from mushroom.pipeline.interface import Entry
from mushroom.config import settings
from pydantic_settings import BaseSettings
from copy import deepcopy
from openai import OpenAI, AsyncOpenAI
import aiohttp
from concurrent.futures import ThreadPoolExecutor

prompt_get_wiki_title = """
Given this question: '{question}'
Formulate a search query to wikipedia which will find pages relevant to this question.
The query should be a word or phrase which likely be contained in the relevant page's title.
Make the query as clean as possible e.g. just one relevant named entity only which is the question's subject.
OUTPUT THE SEARCH QUERY ONLY, ABSOLUTELY NOTHING BESIDES THE QUERY, NO OTHER NOTES!
Examples:
Question: "What is the capital of the Czech Republic?" You answer: "Czech Republic"
Question "When Albert Einstein was born?" You answer: "Albert Einstein"
"""

class WikiApi:
    def __init__(self, lang_code):
        self._base_url = f"https://api.wikimedia.org/core/v1/wikipedia/{lang_code}/search/page"

    async def search(self, search_string):
        params = {
            'q': search_string,
            'limit': 1,
        }
        async with aiohttp.ClientSession() as session:
            async with session.get(self._base_url, params=params) as response:
                response.raise_for_status()
                json_data = await response.json()
                api_results = json_data.get('pages', [])
                if not api_results:
                    return None
                return api_results[0]['title']

class Retrieval:
    def __init__(self, config: Optional[BaseSettings] = None):
        self.config = config if config is not None else settings
        self.model = SentenceTransformer(self.config.retrieval.embed_model_name)
        self.client = AsyncOpenAI(
            base_url=self.config.retrieval.api_base_url,
            api_key=self.config.retrieval.api_key,
        )
        self.wiki_api = WikiApi(lang_code=self.config.retrieval.wiki_lang_code)

    def fetch_wikipedia_content(self, title):
        try:
            page = wikipedia.page(title, auto_suggest=False)
            return page.content
        except wikipedia.exceptions.DisambiguationError as e:
            print(f"[DISAMBIGUATION] {title} -> {e.options[0]}")
            try:
                return wikipedia.page(e.options[0], auto_suggest=False).content
            except:
                return ""
        except wikipedia.exceptions.PageError:
            print(f"[NOT FOUND] {title}")
            return ""
        except Exception as e:
            print(f"[ERROR] {title}: {e}")
            return ""

    def split_into_sentences(self, text):
        return [s for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s]

    def sliding_windows(self, sentences, window_size):
        return [" ".join(sentences[i:i+window_size]) for i in range(len(sentences)-window_size+1)]

    async def get_page_title(self, entry: Entry) -> str:
        _prompt = prompt_get_wiki_title.format(question=entry.model_input)
        messages: list = [{
            "role": "user", "content": _prompt
        }]
        completion = await self.client.beta.chat.completions.parse(
            model=self.config.retrieval.llm_model_name,
            messages=messages,
            max_tokens=self.config.retrieval.max_tokens,
            temperature=self.config.retrieval.temperature,
            top_p=self.config.retrieval.top_p,
        )
        wiki_query = completion.choices[0].message.content
        title = await self.wiki_api.search(wiki_query)
        return title
        
    async def process_entry(self, entry: Entry) -> Entry:
        title = await self.get_page_title(entry)
        if not title:
            return entry

        content = self.fetch_wikipedia_content(title)
        sentences = self.split_into_sentences(content)
        chunks = self.sliding_windows(sentences, 1)

        if not chunks:
            entry.retrieval_output = {
                "retrieved": [],
                "wiki_content": content
            }
            return entry

        embeddings = self.model.encode(chunks, convert_to_numpy=True)
        faiss.normalize_L2(embeddings)
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        fact_spans = getattr(entry, "fact_spans", [])
        retrieved = []
        for fact_entry in fact_spans:
            fact_text = getattr(fact_entry, "fact", "")
            if not fact_text:
                continue
            query = self.model.encode([fact_text], convert_to_numpy=True)
            faiss.normalize_L2(query)
            distances, indices = index.search(query, self.config.retrieval.top_k)

            top_chunks = []
            for score, idx in zip(distances[0], indices[0]):
                if idx >= 0 and idx < len(chunks):
                    top_chunks.append({"chunk": chunks[idx], "score": round(float(score), 3)})

            retrieved.append({
                "fact": fact_text,
                "top_3": top_chunks
            })

        entry.retrieval_output = {
            "retrieved": retrieved,
            "wiki_content": content
        }
        return entry

    async def run(self, dataset: List[Entry]) -> List[Entry]:
        dataset = deepcopy(dataset)
        tasks = [self.process_entry(entry) for entry in dataset]
        processed = await asyncio.gather(*tasks)
        return processed

    def __call__(self, *args, **kwargs):
        try:
            loop = asyncio.get_running_loop()
            return loop.create_task(self.run(*args, **kwargs))
        except RuntimeError:
            return asyncio.run(self.run(*args, **kwargs))
#%%
# Example usage (if run as script)
if __name__ == "__main__":
    #%%
    from mushroom.pipeline.data_connector.data_connector import read_dataset
    from pathlib import Path

    project_root = Path(settings.project_root)
    file_path = project_root / "facts_retrieval_ready/entries_with_facts_and_retrieval.json"
    output_path = project_root / "outputs"
    output_path.mkdir(parents=True, exist_ok=True)
    output_file_path = output_path / "retrieval_predictions.jsonl"

    dataset = read_dataset(file_path)
    
    
    #%%
    retrieval_step = Retrieval()
    predictions = retrieval_step(dataset[:10])
    predictions