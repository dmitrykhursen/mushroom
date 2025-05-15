
import dataclasses
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import dacite


@dataclass
class AtomicFact:
    fact: Optional[str] = None
    start: Optional[int] = None
    end: Optional[int] = None


@dataclass
class RetrievalOutput:
    retrieved: Optional[List[Dict]] = None
    wiki_content: Optional[str] = None


@dataclass
class SpanLabelingOutput:
    soft_predictions: Optional[List[Dict[str, Any]]] = None
    hard_predictions: Optional[List[List[int]]] = None


@dataclass
class Entry:
    model_input: str
    model_output_text: str
    soft_labels: Optional[List[Dict[str, Any]]] = (
        None  # {"start": int, "end": int, "prob": float}
    )
    hard_labels: Optional[List[List[int]]] = None  # [[start, end], [start, end]]

    id: Optional[str] = None
    lang: Optional[str] = None
    model_id: Optional[str] = None
    model_output_logits: Optional[List[float]] = None
    model_output_tokens: Optional[List[str]] = None
    wikipedia_url: Optional[str] = None
    annotations: Optional[Dict[str, List[List[int]]]] = None
    text_len: Optional[int] = None

    fact_spans: Optional[List[AtomicFact]] = dataclasses.field(default_factory=list)

    retrieval_output: Optional[RetrievalOutput] = dataclasses.field(
        default_factory=RetrievalOutput
    )

    span_labeling_output: Optional[SpanLabelingOutput] = dataclasses.field(
        default_factory=SpanLabelingOutput
    )

    def __post_init__(self):
        if self.text_len is None:
            self.text_len = len(self.model_output_text)

    def to_dict(self):
        return dataclasses.asdict(self)

    @staticmethod
    def from_dict(data: Dict[str, Any]):
        return dacite.from_dict(data_class=Entry, data=data)