from pydantic import BaseModel, Field
from typing import Optional, List, Dict


class FactExtractionReturn(BaseModel):
    facts: List[Dict[str, str]] = Field(
        description="List of extracted facts, each containing Predicate, Subject, Object, and Reformulation.",
        default_factory=list
    )
    
class FactExtractionIndexReturn(BaseModel):
    index: int = Field(
        description="Index of the occurrence of the Object in the model output text.",
    )