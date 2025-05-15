#%%
from typing import Any, List, Optional, Union

from mushroom.config import settings
from mushroom.pipeline.data_connector import read_dataset, write_dataset
from mushroom.pipeline.interface import Entry
from mushroom.pipeline.span_labeling.evaluate_span_labeling import \
    evaluate_span_labeling
from mushroom.pipeline.span_labeling.span_labeling import \
    span_labeling_baseline


class Pipeline:
    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)
    
    def run(self, dataset: Union[List[Entry], str], output_path: Optional[str] = None) -> List[Entry]:
        if not isinstance(dataset, List):
            dataset = read_dataset(dataset)
        
        processed_dataset = span_labeling_baseline(dataset)
        
        
        if output_path:
            write_dataset(output_path, processed_dataset)
        
        return processed_dataset
    
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
    pipeline = Pipeline()
    
    from pathlib import Path
    project_root = Path(settings.project_root)
    
    file_path = project_root / "data/facts_retrieval_ready/entries_with_facts_and_retrieval.json"
    output_path = project_root / "outputs"
    
    output_path.mkdir(parents=True, exist_ok=True)
    output_file_path = output_path / "predictions.jsonl"
    
    dataset = read_dataset(file_path)
    
    
    predictions = pipeline.run(dataset, output_file_path)
    results = pipeline.evaluate(output_file_path)

#%%
        