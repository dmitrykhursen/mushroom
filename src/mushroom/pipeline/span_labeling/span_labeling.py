#%%
import re
from typing import Callable, List

from mushroom.pipeline.interface import Entry
from mushroom.pipeline.span_labeling.evaluate_span_labeling import \
    evaluate_span_labeling
from mushroom.config import settings
from pprint import pprint



prompt_classify = '''
You are a fact-checking assistant. You are given:

- A factual question.
- A generated answer from a language model.
- A relevant Wikipedia article.

Your task is:
1. Read the claim and the Wikipedia information carefully.  
2. Decide which single label fits best:
   - **SUPPORTS** - all essential aspects of the claim are directly verified by the information.  
   - **REFUTES** - the information clearly contradicts the claim.  
   - **NOT ENOUGH INFO** - the information is insufficient to determine support or refutation.
3. **Output only one word** — one of the three labels above.  
   Do **not** explain your reasoning, do **not** output anything else.

Return your result in JSON format with "label" and "reasons"
### User
Question:
{question}

Generated Answer:
{fact}

Wikipedia Information:
{wiki}
'''


class SpanLabeling:
    def __init__(self):
        pass
    
    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)
    
    def run(self, dataset: List[Entry]) -> List[Entry]:
        
        for entry in dataset:
            entry = self.process_entry(entry)
            
        return dataset
    
    def process_entry(self, entry: Entry) -> Entry:
        facts = entry.fact_spans
        retrieval_outputs = entry.retrieval_output
        
        wiki_text = retrieval_outputs.wiki_content
        model_output_text = entry.model_output_text
        
        # matches = [(m.start(), m.end()) for m in re.finditer(r'\S+', entry.model_output_text)]
        
        matches = entry.hard_labels
        
        for i, fact in enumerate(facts):
            
            print(fact, model_output_text[fact.start:fact.end], "|||", fact.start, fact.end)
        
        # import numpy as np
        for start, end in matches:
            
            # if [start, end] not in entry.hard_labels and [start, end] not in entry.soft_labels:
            #     continue
            
            
            
            # print(
            #     [start, end],
            #     entry.hard_labels,
            # )
            
            entry.span_labeling_output.hard_predictions.append([start, end])
            entry.span_labeling_output.soft_predictions.append(
                {
                    "start": start,
                    "end": end,
                    "prob": 1.0,
                }
            )
            
            print(f">{model_output_text[start:end]}<")
            
        
        
        # for fact_i, retrieval_i in zip(facts, retrieval_outputs.retrieved):
        #     assert fact_i.fact == retrieval_i["fact"]
            
        #     start, end, fact = fact_i.start, fact_i.end, fact_i.fact
        #     chunks = retrieval_i["chunks"]
            
            
            
            
        #     entry.span_labeling_output.hard_predictions.append([start, end])
        #     entry.span_labeling_output.soft_predictions.append(
        #         {
        #             "start": start,
        #             "end": end,
        #             "prob": 0.1
        #         }
        #     )
                
            
        return entry
    
            
            
        
        

    
#%%
if __name__ == "__main__":
    #%%
    from mushroom.pipeline.data_connector.data_connector import read_dataset
    from pathlib import Path
    
    project_root = Path(settings.project_root)
    
    file_path = project_root / "data/facts_retrieval_ready/entries_with_facts_and_retrieval.json"
    output_path = project_root / "outputs"
    
    output_path.mkdir(parents=True, exist_ok=True)
    output_file_path = output_path / "predictions.jsonl"
    
    dataset = read_dataset(file_path)
    
    for entry in dataset:
        for retrieved_i in entry.retrieval_output.retrieved:
            retrieved_i["chunks"] = retrieved_i["top_3"]
            del retrieved_i["top_3"]
            
    
    
    #%% 
    span_labeling_step = SpanLabeling()
    predictions = span_labeling_step(dataset)
    
    
    ious, cors = evaluate_span_labeling(predictions)
    
    print("IOUs:", ious.mean())
    print("CORs:", cors.mean())
    
#%%