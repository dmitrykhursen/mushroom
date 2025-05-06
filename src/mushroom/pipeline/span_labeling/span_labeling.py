#%%
import re
from typing import Callable, List

from mushroom.pipeline.interface import Entry
from mushroom.pipeline.span_labeling.evaluate_span_labeling import \
    evaluate_span_labeling
from mushroom.config import settings


def baseline_entry_all_spans(entry: Entry) -> Entry:
    prediction = entry

    matches = [(m.start(), m.end()) for m in re.finditer(r'\S+', entry.model_output_text)]
    
    hard_predictions = []
    soft_predictions = []
    for start, end in matches:
        soft_predictions.append({
            "start": start,
            "end": end,
            "prob": 1.0,
            "label": None
        })
        hard_predictions.append([start, end])
        
        
    prediction.span_labeling_output.soft_predictions = soft_predictions
    prediction.span_labeling_output.hard_predictions = hard_predictions 
    
    return prediction


def baseline_entry_empty(entry: Entry) -> Entry:
    prediction = entry
    
    prediction.soft_predictions = []
    prediction.hard_predictions = []
    
    return prediction
    

def span_labeling_baseline(dataset: List[Entry], entry_function: Callable[[List[Entry]], List[Entry]] = None) -> List[Entry]:
    # Initialize an empty list to store the predictions
    predictions = []
    
    if entry_function is None:
        entry_function = baseline_entry_all_spans
    
    
    # Iterate over each entry in the dataset
    for entry in dataset:
        # Generate a prediction for the entry
        prediction = entry_function(entry)
        
        # Append the prediction to the list
        predictions.append(prediction)
    
    return predictions

if __name__ == "__main__":
    from mushroom.pipeline.data_connector.data_connector import read_dataset
   

    from pathlib import Path
    project_root = Path(settings.project_root)
    
    file_path = project_root / "data/extra/splits/val/v2/mushroom.en-val.v2.extra.jsonl"
    dataset = read_dataset(file_path)
    predictions = span_labeling_baseline(dataset)
    
    ious, cors = evaluate_span_labeling(predictions)
    
    print("IOUs:", ious.mean())
    print("CORs:", cors.mean())
    
#%%