#%%
from dataclasses import dataclass
from typing import List, Dict, Any
from pathlib import Path
from enum import Enum

from mushroom.pipeline.span_labeling.data_connector import Entry
class FileType(Enum):
    JSONL = "jsonl"
    CSV = "csv"
    TSV = "tsv"
    PARQUET = "parquet"
    TXT = "txt"
    

def get_filetype(file_path: str) -> str:
    file_type = Path(file_path).suffix[1:].lower()



if __name__ == "__main__":
    file_path = "/home/dan/Things/uni/llm_class/mushroom/src/pipeline/span_labeling/example.jsonl"
    print(get_filetype(file_path))






