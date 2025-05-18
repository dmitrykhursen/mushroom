#%%
import dataclasses
import json
from typing import List, Dict, Any
from pathlib import Path

from mushroom.pipeline.interface import Entry

def read_json(file_path: str) -> List[Dict[str, Any]]:
    with open(file_path, "r") as f:
        data = json.load(f)
    return data

def read_jsonl(file_path: str) -> List[Dict[str, Any]]:
    with open(file_path, "r") as f:
        lines = f.readlines()
        data = [json.loads(line) for line in lines]
    return data

def read_dataset(file_path: str) -> List[Entry]:
    file_path = Path(file_path)
    if file_path.suffix == ".jsonl":
        data = read_jsonl(file_path)
    elif file_path.suffix == ".json":
        data = read_json(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")
    dataset = [Entry.from_dict(entry) for entry in data]
    return dataset

def write_dataset(filepath: str, dataset: List[Entry], comment=None) -> str:
    filepath = Path(filepath)
    if comment is not None:
        filepath_extension = Path(filepath).suffix
        filepath_name = Path(filepath).stem
        filepath_directory = Path(filepath).parent
        filepath = filepath_directory / f"{filepath_name}_{comment}{filepath_extension}"
        
        
    
    with open(filepath, "w") as f:
        for entry in dataset:
            f.write(json.dumps(dataclasses.asdict(entry)) + "\n")
            
    return filepath.as_posix()