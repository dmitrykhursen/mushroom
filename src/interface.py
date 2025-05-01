import datasets
from dataclasses import dataclass
from typing import List, Dict, Any


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


    

    
    
    
    