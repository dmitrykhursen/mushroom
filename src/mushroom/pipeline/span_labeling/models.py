from pydantic import BaseModel, Field
from enum import Enum
from typing import Optional

class HallucinationDetectionEnum(str, Enum):
    HALLUCINATED = "HALLUCINATED"
    PARTIALLY_HALLUCINATED = "PARTIALLY HALLUCINATED"
    SUPPORTED = "SUPPORTED"


class HallucinationDetectionReturn(BaseModel):
    label: HallucinationDetectionEnum = Field(default=HallucinationDetectionEnum.SUPPORTED,
        description="The label for the hallucination detection task.")
    reason: str = Field(
        default="",
        description="The reason for the label. This is not used in the final output, but can be useful for debugging.")

class HallucinationDetectionSpan(BaseModel):
    start: Optional[int] = Field(
        description="The start index of the hallucinated span in the model output.")
    end: Optional[int] = Field(
        description="The end index of the hallucinated span in the model output.")
    text: Optional[str] = Field(
        description="The text of the hallucinated span in the model output.")
    reason: Optional[str] = Field(
        description="The reason for the label.")
    
    
class HallucinationDetectionSpansReturn(BaseModel):
    spans: Optional[list[HallucinationDetectionSpan]] = Field(default_factory=list,
        description="The spans of the hallucinated text in the model output.")