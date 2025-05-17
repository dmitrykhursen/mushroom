#%%
from typing import Any, List, Optional, Union
import asyncio

from mushroom.config import settings
from mushroom.pipeline.data_connector import read_dataset, write_dataset
from mushroom.pipeline.interface import Entry
from mushroom.pipeline.span_labeling.evaluate_span_labeling import \
    evaluate_span_labeling
from mushroom.pipeline.span_labeling.span_labeling import SpanLabeling


class Pipeline:
    def __init__(self, config = None):
        self.config = config if config is not None else settings
        self.span_labeling = SpanLabeling(config=self.config)
        
        
        self.steps = [
            self.span_labeling,
        ] # maybe we should use our steps like this?
    
    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)
    
    def run(self, dataset: Union[List[Entry], str], output_path: Optional[str] = None) -> List[Entry]:
        if not isinstance(dataset, List):
            dataset = read_dataset(dataset)
            
        
        dataset = self.span_labeling(dataset)
        
        
        if output_path:
            write_dataset(output_path, dataset)
        
        return dataset
    
    def evaluate(self, processed_dataset: Union[List[Entry], str]) -> List[Any]:
        ious, cors = evaluate_span_labeling(processed_dataset)
        
        print(f"IOUs: {ious.mean()}")
        print(f"CORs: {cors.mean()}")
        
        return {
            "ious": ious,
            "cors": cors
        }
#%%
            
if __name__ == "__main__":
    
    #%%
    from pathlib import Path
    project_root = Path(settings.project_root)
    
    file_path = project_root / "facts_retrieval_ready/entries_with_facts_and_retrieval.json"
    output_path = project_root / "outputs"
    
    output_path.mkdir(parents=True, exist_ok=True)
    output_file_path = output_path / "predictions.jsonl"
    
    dataset = read_dataset(file_path)
    
    for entry in dataset:
        for retrieved_i in entry.retrieval_output.retrieved:
            retrieved_i["chunks"] = retrieved_i["top_3"]
            del retrieved_i["top_3"]
    #%%
    dataset = dataset[:2]
    
    #%%
    pipeline = Pipeline()
    predictions = pipeline.run(dataset, output_file_path)
    results = pipeline.evaluate(output_file_path)

#%%
        