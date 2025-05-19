import seaborn as sns
import pandas as pd
from getpass import getpass
import re  
from datasets import load_dataset
import json
import zipfile
from recommendation_research_data.clanky_entities.entity_db import (
    get_entities as get_entities_clanky)

from recommendation_research_llm_handler.llm_client import LLMHandler
import getpass
from recommendation_content_classifier.utils import (
    plot_confusion_matrix)

import matplotlib.pyplot as plt
import numpy as np

import json, pathlib, itertools
from typing import List, Dict, Any, Iterable
# import openai 

def plot_confusion_matrix(y_true, y_pred,
                          gt_order=("CORRECT", "INCORRECT"),
                          pred_order=("SUPPORTS", "REFUTES", "NOT ENOUGH INFO")):
    cm_df = pd.crosstab(
        pd.Categorical(y_true, categories=gt_order),
        pd.Categorical(y_pred, categories=pred_order),
        dropna=False
    )

    plt.figure(figsize=(6, 5))
    ax = sns.heatmap(cm_df,
                     annot=True,     
                     fmt="d",
                     cmap="Blues",          
                     linewidths=.5,         
                     linecolor="white",
                     cbar=True)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    plt.tight_layout()
    plt.show()
    
    
# DATASET 3 -----------------------------------------------------------------------------
def read_json(path):
    text = open(path, 'r', encoding='utf-8').read().lstrip()
    data = json.loads(text)
    if not isinstance(data, list):
        raise ValueError("Top-level JSON is not a list!")
    return data

def extract_facts(sample):
    return [f["fact"] for f in sample["fact_spans"]]


def evidence_for_fact(sample, fact_text):
    ret = sample.get("retrieval_output", {}).get("retrieved", [])
    for block in ret:
        if block["fact"].strip().lower() == fact_text.strip().lower():
            return "\n".join([c["chunk"] for c in block["top_3"]])
    return sample.get("retrieval_output", {}).get("wiki_content", "")

def evidence_sentences(sample, claim: str):
    retrieved = sample.get("retrieval_output", {}).get("retrieved", [])
    for block in retrieved:
        if block["fact"].strip().lower() == claim.strip().lower():
            return [c["chunk"] for c in block["top_3"]]

    wiki_text = sample.get("retrieval_output", {}).get("wiki_content", "")
    return re.split(r"\.\s+", wiki_text)[:3] if wiki_text else []

def build_df(records):
    rows=[]
    for sample in records:
        wiki_url  = sample.get("retrieval_output", "").get("wiki_content", "")
        question = sample.get("model_input","")
        hard_spans = [tuple(span) for span in sample.get("hard_labels",[])]
        for fs in sample.get("fact_spans", []):
            fact = fs["fact"]
            span = (fs["start"], fs["end"])
            label = "INCORRECT" if any(spans_overlap(span, hs) for hs in hard_spans) else "CORRECT"
            rows.append(dict(
                fact=fact,
                wiki=wiki_url,
                sentences=evidence_sentences(sample, fact),
                question=question,
                label=label
            ))
    return pd.DataFrame(rows)


def spans_overlap(a, b):
    # a, b are (start, end) half-open
    return not (a[1] <= b[0] or b[1] <= a[0])

# DATASET 2 -----------------------------------------------------------

def add_wikipedia_text(dataset, wiki_dict):
    def enrich(example):
        page = example['evidence_wiki_url']
        example['wiki_info'] = wiki_dict.get(page, None)
        return example

    return dataset.map(enrich)

def filter_missing_wiki(example):
    return example['wiki_info'] is not None

def load_all_wiki_pages(folder_path):
    wiki_dict = {}
    
    for filename in os.listdir(folder_path):
        if filename.endswith('.jsonl'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        wiki_dict[entry['id']] = entry['text']
                    except json.JSONDecodeError:
                        continue 

    return wiki_dict

def load_wiki_dict_from_zip(zip_path):
    wiki_dict = {}
    with zipfile.ZipFile(zip_path, 'r') as zipf:
        for filename in zipf.namelist():
            # Check that file is in the wiki-pages/ directory and ends with .jsonl
            if filename.startswith('wiki-pages/') and filename.endswith('.jsonl'):
                with zipf.open(filename) as f:
                    for line in f:
                        try:
                            entry = json.loads(line.decode('utf-8'))
                            wiki_dict[entry['id']] = entry['text']
                        except json.JSONDecodeError:
                            continue  
    return wiki_dict