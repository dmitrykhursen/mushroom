from typing import Any, List, Optional, Union
import json

from colorama import Fore, Style

from mushroom.config import settings
from mushroom.pipeline.data_connector import read_dataset, write_dataset
from mushroom.pipeline.fact_extractor import fact_alignment, fact_extraction, fact_extraction_brf
from mushroom.pipeline.interface import Entry
from mushroom.pipeline.span_labeling.evaluate_span_labeling import \
    evaluate_span_labeling
from mushroom.pipeline.span_labeling.span_labeling import \
    span_labeling_baseline

def get_fact_with_span(fact_string: str, start: int, end: int) -> str:
    return fact_string[0:start] + Fore.RED + fact_string[start:end] + Style.RESET_ALL + fact_string[end:]
    # return fact_string[0:start] + ' <SPAN_START>' + fact_string[start:end] + ' <SPAN_END> ' + fact_string[end:]

class Pipeline:
    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)
    
    def run(self, dataset: Union[List[Entry], str], output_path: Optional[str] = None) -> List[Entry]:
        if not isinstance(dataset, List):
            dataset = read_dataset(dataset)
        
        extracted_facts = []
        debug_i = 0
        for entry in dataset:
            if debug_i > 3:
              break
            # entry.atomic_facts = fact_extraction.extract_atomic_facts(entry.model_input, entry.model_output_text)
            # entry.fact_spans = fact_alignment.align_facts_to_text(entry.atomic_facts, entry.model_input, entry.model_output_text)
            entry.fact_spans = fact_extraction_brf.extract_atomic_facts(entry.model_output_text)
            debug_i += 1

           # print('\n')
            #print(entry.model_output_text)
            #print("\nFact spans:")
            #for span in entry.fact_spans:
            #    fact  = span["fact"]
            #    start = span["start"]
            #    end   = span["end"]
            #    # (optionally: word = span["word"])
            #    snippet = entry.model_output_text[start:end]
            #    print(f"Fact: {fact!r}")
            #    print(f" Span chars [{start}:{end}]")
            #    print(f"  Snippet: {snippet!r}")
            #    print(f"   Fact with span: " + get_fact_with_span(entry.model_output_text, start, end))

        entries_as_dict = [entry.to_dict() for entry in dataset]
        with open("entries.json", "w", encoding="utf-8") as f:
            json.dump(entries_as_dict, f, ensure_ascii=False, indent=2)

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
        
            
if __name__ == "__main__":
    pipeline = Pipeline()
    
    from pathlib import Path
    project_root = Path(settings.project_root)
    
    file_path = project_root / "data/splits/val/v2/mushroom.en-val.v2.extra.jsonl"
    output_path = project_root / "outputs"
    
    output_path.mkdir(parents=True, exist_ok=True)
    output_file_path = output_path / "predictions.jsonl"
    
    predictions = pipeline.run(file_path, output_file_path)
    results = pipeline.evaluate(output_file_path)


#%%
        