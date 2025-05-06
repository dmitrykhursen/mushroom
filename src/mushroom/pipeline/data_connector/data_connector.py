#%%
import dataclasses
import json
from typing import List

from mushroom.pipeline.interface import Entry


def read_dataset(file_path: str) -> List[Entry]:
    with open(file_path, "r") as f:
        lines = f.readlines()
        data = [json.loads(line) for line in lines]
        
    dataset = [Entry.from_dict(entry) for entry in data]
    return dataset

def write_dataset(filepath: str, dataset: List[Entry]) -> None:
    with open(filepath, "w") as f:
        for entry in dataset:
            f.write(json.dumps(dataclasses.asdict(entry)) + "\n")