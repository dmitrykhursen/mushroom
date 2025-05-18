#%%
from typing import Any, List, Optional, Union
import asyncio

from mushroom.config import settings
from mushroom.pipeline.data_connector import read_dataset, write_dataset
from mushroom.pipeline.interface import Entry
from mushroom.pipeline.span_labeling.evaluate_span_labeling import \
    evaluate_span_labeling

from mushroom.pipeline.fact_extractor.fact_extraction_brf import FactExtraction
from mushroom.pipeline.span_labeling.span_labeling import SpanLabeling, SpanLabelingBaselineAll, SpanLabelingBaselineAllWords
from mushroom.pipeline.retrieval.retrieval import Retrieval
from copy import deepcopy


class Pipeline:
    def __init__(self, config = None, steps = None):
        self.config = config if config is not None else settings
        
        self.registry = {
            "fact_extraction": FactExtraction,
            "retrieval": Retrieval,
            "span_labeling": SpanLabeling,
            "span_labeling_baseline": SpanLabelingBaselineAll,
            "span_labeling_baseline_all_words": SpanLabelingBaselineAllWords
        }
        
        
        if steps is None:
            self.steps = [
                ("fact_extraction", self.registry["fact_extraction"](config=self.config)),
                ("retrieval", self.registry["retrieval"](config=self.config)),
                ("span_labeling", self.registry["span_labeling"](config=self.config)),
            
            ]
        else:
            self.steps = [
                (step_name, self.registry[step_name](config=self.config))
                for step_name in steps
            ]
            
    
    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)
    
    def run(self, dataset: Union[List[Entry], str], output_path: Optional[str] = None) -> List[Entry]:
        if not isinstance(dataset, List):
            dataset = read_dataset(dataset)
        else:
            dataset = deepcopy(dataset)
            
        
        # for entry in dataset:
            
        #     entry_list = [entry]
        #     for step_name, step in self.steps:
        #         entry_list = step(entry_list)
        #         if output_path:
        #             write_dataset(output_path, entry_list, comment=step_name, mode="a")
        
        for step_name, step in self.steps:
            dataset = step(dataset)
            if output_path:
                write_dataset(output_path, dataset, comment=step_name, mode="w")
            
        return dataset
    
    def evaluate(self, processed_dataset: Union[List[Entry], str]) -> List[Any]:
        if isinstance(processed_dataset, str):
            print(f"Loading dataset from {processed_dataset}")
        
        ious, cors = evaluate_span_labeling(processed_dataset)
        
        print(f"IOUs: {ious.mean():.4f}")
        print(f"CORs: {cors.mean():.4f}")
        
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
    # file_path = project_root / "data/extra/splits/val/v2/mushroom.en-val.v2.extra.jsonl"
    # file_path = project_root / "outputs/mushroom.en-tst.v1.extra_test_baseline/predictions_retrieval.jsonl"
    # dataset = read_dataset(file_path)
    
    # pipeline = Pipeline(steps=["span_labeling"])
    
    # predictions = pipeline(dataset[4:5])
    
    
    # pipeline.evaluate(predictions)
    
    #%%
    print(f"* English validation set")
    print(f"1. Nothing baseline")
    file_path = project_root / "data/extra/splits/val/v2/mushroom.en-val.v2.extra.jsonl"
    dataset = read_dataset(file_path)
    Pipeline.evaluate(None, dataset);
    
    print(f"2. All facts baseline")
    file_path = project_root / "outputs/mushroom.en-val.v2.extra_val_baseline/predictions_span_labeling_baseline.jsonl"
    dataset = read_dataset(file_path)
    Pipeline.evaluate(None, dataset);
    
    
    print(f"3. LLM fact-checking baseline")
    
    file_path = project_root / "outputs/mushroom.en-val.v2.extra_val_main/predictions_span_labeling.jsonl"
    dataset = read_dataset(file_path)
    Pipeline.evaluate(None, dataset);
    #%%
    
    print(f"* English test set")
    print(f"1. Nothing baseline")
    file_path = project_root / "data/extra/splits/test_labeled/v1/mushroom.en-tst.v1.extra.jsonl"
    dataset = read_dataset(file_path)
    Pipeline.evaluate(None, dataset);
    
    print(f"2. All facts baseline")
    file_path = project_root / "outputs/mushroom.en-tst.v1.extra_test_baseline/predictions_span_labeling_baseline.jsonl"
    dataset = read_dataset(file_path)
    Pipeline.evaluate(None, dataset)
    
    
    print(f"3. LLM fact-checking baseline")
    file_path = project_root / "outputs/mushroom.en-tst.v1.extra_test_main/predictions_span_labeling.jsonl"
    dataset = read_dataset(file_path)
    Pipeline.evaluate(None, dataset);

# %%
