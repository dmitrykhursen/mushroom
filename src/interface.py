import datasets
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any


@dataclass
class Entry:
    id: str
    lang: str
    model_input: str
    model_output: str
    model_id: str
    soft_labels: List[Dict[str, Any]]
    hard_labels: List[List[int]]
    model_output_logits: List[float]
    model_output_tokens: List[str]
    wikipedia_url: str
    annotations: Dict[str, List[List[int]]]

class AtomicFact:
    fact_phrase: str
    indices: Tuple[int, int]
    

@dataclass
class EntryAtomicFacts(Entry):
    atomic_facts: List[AtomicFact]

@dataclass
class RetrievalInfo(Entry):
    wiki_info: List[List[str]]
    llm_prompt: str


    

    
    
    
    