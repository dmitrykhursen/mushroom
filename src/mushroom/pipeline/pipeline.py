#%%
from typing import Any, List, Optional, Union
import asyncio

from mushroom.config import settings
from mushroom.pipeline.data_connector import read_dataset, write_dataset
from mushroom.pipeline.interface import Entry
from mushroom.pipeline.span_labeling.evaluate_span_labeling import \
    evaluate_span_labeling

from mushroom.pipeline.fact_extractor.fact_extraction_brf import FactExtraction
from mushroom.pipeline.span_labeling.span_labeling import SpanLabeling, SpanLabelingBaselineAll
from mushroom.pipeline.retrieval.retrieval import Retrieval


class Pipeline:
    def __init__(self, config = None):
        self.config = config if config is not None else settings
        self.fact_extractor = FactExtraction(config=self.config)
        self.retrieval = Retrieval(config=self.config)
        self.span_labeling = SpanLabeling(config=self.config)
        self.span_labeling_baseline = SpanLabelingBaselineAll(config=self.config)
        
        
        self.steps = [
            # ("fact_extraction", self.fact_extractor),
            # ("retrieval", self.retrieval),
            # ("span_labeling_baseline", self.span_labeling_baseline),
            # ("retrieval", self.retrieval),
            ("span_labeling", self.span_labeling),
            
        ] # maybe we should use our steps like this?
    
    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)
    
    def run(self, dataset: Union[List[Entry], str], output_path: Optional[str] = None) -> List[Entry]:
        if not isinstance(dataset, List):
            dataset = read_dataset(dataset)
            
        
        # dataset = self.span_labeling(dataset)
        
        for step_name, step in self.steps:
            dataset = step(dataset)
            print(f"Step: {step_name} completed")
        
            if output_path:
                write_dataset(output_path, dataset, comment=step_name)
        
        return dataset
    
    def evaluate(self, processed_dataset: Union[List[Entry], str]) -> List[Any]:
        if isinstance(processed_dataset, str):
            print(f"Loading dataset from {processed_dataset}")
        
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
    
    # file_path = project_root / "facts_retrieval_ready/entries_with_facts_and_retrieval.json"
    # file_path = project_root / "data/test_unlabeled/v1/mushroom.en-tst.v1.jsonl"
    file_path = project_root / "data/extra/splits/val/v2/mushroom.en-val.v2.extra.jsonl"
    output_path = project_root / "outputs_sample_what"
    
    # import json
    # with open(file_path, "r") as f:
    #     dataset = json.load(f)
        
        
    
    output_path.mkdir(parents=True, exist_ok=True)
    output_file_path = output_path / "predictions.jsonl"
    
    dataset = read_dataset(file_path)
    
    #%%
    dataset = dataset[:10]
    #%%
    
    pipeline = Pipeline()
    predictions = pipeline.run(dataset, output_file_path)
    results = pipeline.evaluate(predictions)
    

# %%
