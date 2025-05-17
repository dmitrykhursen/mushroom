#%%
import json
import re
from urllib.parse import unquote, urlparse
import wikipedia
import numpy as np
# import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Optional
from mushroom.pipeline.interface import Entry
from mushroom.config import settings
from pydantic_settings import BaseSettings
from copy import deepcopy

class Retrieval:
    def __init__(self, config: Optional[BaseSettings] = None):
        self.config = config if config is not None else settings
        # self.model = SentenceTransformer(self.config.model_name)

    def normalize_url_and_extract_title(self, url):
        decoded = unquote(url)
        parsed = urlparse(decoded)
        title = parsed.path.split("/wiki/")[-1].replace("_", " ")
        return title

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
    # def get_url_from_api(self, ):
        
    def process_entry(self, entry: Entry) -> Entry:
        url = getattr(entry, "wikipedia_url", None)
        if not url:
            return entry

        title = self.normalize_url_and_extract_title(url)
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
            distances, indices = index.search(query, self.config.top_k)

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
    
    def run(self, dataset: List[Entry]) -> List[Entry]:
        dataset = deepcopy(dataset)
        for entry in dataset:
            entry = self.process_entry(entry)
        return dataset

    def __call__(self, *args, **kwargs) -> List[Entry]:
        return self.run(*args, **kwargs)
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
    predictions = retrieval_step(dataset)