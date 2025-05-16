#%%
import re
from typing import Callable, List

from mushroom.pipeline.interface import Entry
from mushroom.pipeline.span_labeling.evaluate_span_labeling import \
    evaluate_span_labeling
from mushroom.config import settings
from pprint import pprint
import openai
from openai import OpenAI, AsyncOpenAI
from pydantic import BaseModel, Field
from enum import Enum

class HallucinationDetectionEnum(str, Enum):
    HALLUCINATED = "HALLUCINATED"
    PARTIALLY_HALLUCINATED = "PARTIALLY HALLUCINATED"
    SUPPORTED = "SUPPORTED"


class HallucinationDetectionReturn(BaseModel):
    label: HallucinationDetectionEnum = Field(
        description="The label for the hallucination detection task.")
    reason: str = Field(
        description="The reason for the label. This is not used in the final output, but can be useful for debugging.")
    
    

prompt_classify = '''
You are a fact-checking assistant. You are given:

- A factual question.
- A generated answer from a language model.
- A relevant Wikipedia article.

Your task is:
1. Read the claim and the Wikipedia information carefully.  
2. Decide which single label fits best:
    - **HALLUCINATED**: The claim is not supported by the Wikipedia information.
    - **PARTIALLY HALLUCINATED**: The claim is partially supported by the Wikipedia information.
    - **SUPPORTED**: The claim is fully supported by the Wikipedia information.
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
        
        self.client = AsyncOpenAI(
            base_url="http://localhost:8000/v1",
            api_key="token-abc123",
        )

    
    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)
    
    async def run(self, dataset: List[Entry]) -> List[Entry]:
        
        for entry in dataset:
            entry = await self.process_entry(entry)
            
        return dataset
    
    async def call_llm(self, question: str, fact: str, wiki: str) -> str:
        completion = await self.client.completions.create(
        model="google/gemma-3-4b-it",
        prompt=prompt_classify.format(
            question=question,
            fact=fact,
            wiki=wiki,
        ),
        # messages=[
        #     {"role": "user", "content": prompt_classify.format(
        #         question=question,
        #         fact=fact,
        #         wiki=wiki,
        #     )},
        # ],
        max_tokens=1024,
        temperature=0.7,
        top_p=0.9,
        seed=1,
        # extra_headers={
        #     "response_format": str(HallucinationDetectionReturn.model_json_schema()),
        # }
           extra_body={
            "guided_json": HallucinationDetectionReturn.model_json_schema()
        },
        # response_format=HallucinationDetectionReturn,
        )

        response_text = completion.choices[0].text
        try:
            parsed_output = HallucinationDetectionReturn.model_validate_json(response_text)
            print("Parsed output:", parsed_output)
        except Exception as e:
            print("Error parsing output:", e)
            print("Response text:", response_text)
            parsed_output = HallucinationDetectionReturn(
                label=HallucinationDetectionEnum.SUPPORTED,
                reason="Parsing error"
            )
        return parsed_output
        
        


    
    async def process_entry(self, entry: Entry) -> Entry:
        facts = entry.fact_spans
        retrieval_outputs = entry.retrieval_output
        
        wiki_text = retrieval_outputs.wiki_content
        model_output_text = entry.model_output_text
        
        # matches = [(m.start(), m.end()) for m in re.finditer(r'\S+', entry.model_output_text)]
        
        # matches = entry.hard_labels
        
        for fact_i, retrieval_i in zip(facts, retrieval_outputs.retrieved):
            assert fact_i.fact == retrieval_i["fact"]
            
            start, end, fact = fact_i.start, fact_i.end, fact_i.fact
            wiki_chunk = retrieval_i["chunks"][0]["chunk"] if retrieval_i["chunks"] else wiki_text
            
            response = await self.call_llm(
                question=entry.model_input,
                fact=fact,
                wiki=wiki_chunk,
            )
            if response.label == HallucinationDetectionEnum.HALLUCINATED:
                entry.span_labeling_output.hard_predictions.append([start, end])
                entry.span_labeling_output.soft_predictions.append(
                    {
                        "start": start,
                        "end": end,
                        "prob": 1.0
                    }
                )
            elif response.label == HallucinationDetectionEnum.PARTIALLY_HALLUCINATED:
                entry.span_labeling_output.hard_predictions.append([start, end])
                entry.span_labeling_output.soft_predictions.append(
                    {
                        "start": start,
                        "end": end,
                        "prob": 0.5
                    }
                )
            elif response.label == HallucinationDetectionEnum.SUPPORTED:
                pass
            else:
                raise ValueError(f"Unknown label: {response.label}")
            
        #     print(fact, model_output_text[fact.start:fact.end], "|||", fact.start, fact.end)
        
        # # import numpy as np
        # for start, end in matches:
            
        #     # if [start, end] not in entry.hard_labels and [start, end] not in entry.soft_labels:
        #     #     continue
            
            
            
        #     # print(
        #     #     [start, end],
        #     #     entry.hard_labels,
        #     # )
            
        #     entry.span_labeling_output.hard_predictions.append([start, end])
        #     entry.span_labeling_output.soft_predictions.append(
        #         {
        #             "start": start,
        #             "end": end,
        #             "prob": 1.0,
        #         }
        #     )
            
        #     print(f">{model_output_text[start:end]}<")
            
        
        
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
    predictions = await span_labeling_step.run(dataset)
    
    #%%
    for entry in predictions:
        print(entry.span_labeling_output.hard_predictions)
        print(entry.span_labeling_output.soft_predictions)
        print()
    #%%
    
    ious, cors = evaluate_span_labeling(predictions)
    
    print("IOUs:", ious.mean())
    print("CORs:", cors.mean())
    
#%%