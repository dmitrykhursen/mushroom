from typing import List, Union, Tuple

import numpy as np

from mushroom.participant_kit.scorer import main as scorer
from mushroom.pipeline.data_connector import read_dataset
from mushroom.pipeline.interface import Entry


def evaluate_span_labeling(dataset: Union[List[Entry], str]) -> Tuple[np.ndarray, np.ndarray]:
    if not isinstance(dataset, List):
        dataset = read_dataset(dataset)
            
    labels = [
        {
            "id": entry.id,
            "soft_labels": entry.soft_labels,
            "hard_labels": entry.hard_labels,
            "text_len": entry.text_len,
        }
        for entry in dataset
        if entry.hard_labels is not None or entry.soft_labels is not None
    ]
    if not labels:
        raise ValueError("The provided dataset has no labels.")
    
    predictions = [
        {
            "id": entry.id,
            "soft_labels": entry.span_labeling_output.soft_predictions,
            "hard_labels": entry.span_labeling_output.hard_predictions,
            "text_len": entry.text_len,
        }
        for entry in dataset
        if entry.span_labeling_output is not None
        and entry.span_labeling_output.hard_predictions is not None
        and entry.span_labeling_output.soft_predictions is not None
    ]
    if not predictions:
        raise ValueError("The provided dataset has no predictions.")
    
    
    ious, cors = scorer(labels, predictions)
    return ious, cors
    
    