#%%
import re
from typing import List

from pydantic_settings import BaseSettings
import asyncio

from mushroom.pipeline.interface import Entry
from mushroom.pipeline.span_labeling.evaluate_span_labeling import \
    evaluate_span_labeling
    
from mushroom.pipeline.span_labeling.models import (
    HallucinationDetectionReturn,
    HallucinationDetectionEnum,
    HallucinationDetectionSpansReturn,
)

from mushroom.pipeline.span_labeling.prompts import (
    prompt_atomic,
    prompt_spans,
)
from tqdm.asyncio import tqdm_asyncio

from mushroom.config import settings
from pprint import pprint
import openai
from openai import OpenAI, AsyncOpenAI
from pydantic import BaseModel, Field
from enum import Enum
from copy import deepcopy
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

class SpanLabeling:
    def __init__(self, config: BaseSettings = None):
        self.config = config if config is not None else settings
        
        self.client = AsyncOpenAI(
            base_url=self.config.span_labeling.api_base_url,
            api_key=self.config.span_labeling.api_key,
        )

    
    def __call__(self, *args, **kwargs):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.run(*args, **kwargs))
        else:
            def run_async_in_thread(coro):
                result_container = {}
                def runner():
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    result_container['result'] = loop.run_until_complete(coro)
                    loop.close()
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(runner)
                    future.result()
                return result_container['result']
            
            return run_async_in_thread(self.run(*args, **kwargs))
            
    async def async_call(self, *args, **kwargs):
        return await self.run(*args, **kwargs)
    
    async def run(self, dataset: List[Entry]) -> List[Entry]:
        dataset_copy = deepcopy(dataset)
        
        tasks = []
        for i, entry in enumerate(dataset_copy):
            tasks.append(self.process_entry(entry))
            
        processed_entries = await tqdm_asyncio.gather(*tasks, desc="Span Labeling")
        
        return processed_entries
    
    async def call_llm(self, messages: list, pydantic_model = None):
        try:
            completion = await self.client.beta.chat.completions.parse(
                model=self.config.span_labeling.model_name,
                messages=messages,
                max_tokens=self.config.span_labeling.max_tokens,
                temperature=self.config.span_labeling.temperature,
                top_p=self.config.span_labeling.top_p,
                response_format=pydantic_model,
                seed=self.config.span_labeling.seed,
                    
            )
        except Exception as e:
            print("Error in LLM call:", e)
            parsed_output = pydantic_model()
            return parsed_output
        
        try:
            parsed_output = completion.choices[0].message.parsed
            # print("Parsed output:", parsed_output)
        except Exception as e:
            print("Error parsing output:", e)
            print("Response text:", completion.choices[0].message.content)
            parsed_output = pydantic_model()
            
        return parsed_output
        
    async def process_entry(self, entry: Entry) -> Entry:
        facts = entry.fact_spans
        retrieval_outputs = entry.retrieval_output
        
        wiki_text = retrieval_outputs.wiki_content
        model_output_text = entry.model_output_text
        
        tasks = []
        for fact_i, retrieval_i in zip(facts, retrieval_outputs.retrieved):
            assert fact_i.fact == retrieval_i["fact"]

            start, end, fact = fact_i.start, fact_i.end, fact_i.fact
            
            wiki_chunks = "\n".join([chunk["chunk"] for chunk in retrieval_i["chunks"]])
            _prompt = prompt_atomic.format(
                question=entry.model_input,
                fact=fact,
                wiki=wiki_chunks,
                json_schema=HallucinationDetectionReturn.model_json_schema(),
            )
            
            
            messages = self.build_messages(_prompt)
            tasks.append(self.call_llm(messages, HallucinationDetectionReturn))
        responses = await tqdm_asyncio.gather(*tasks, desc="Span Labeling facts in one entry")
        
        for response in responses:
        # response = await self.call_llm(messages, HallucinationDetectionReturn)
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
                        "prob": 1.0
                    }
                )
            
        return entry
    
    def build_messages(self, prompt: str) -> list:
        return [
            {
                "role": "user", "content": prompt
            }
        ]
    
        

class SpanLabelingSpans(SpanLabeling):   
    
    async def process_entry(self, entry: Entry) -> Entry:
        facts = entry.fact_spans
        retrieval_outputs = entry.retrieval_output
        
        
        wiki_text = retrieval_outputs.wiki_content
        
        
        model_output_text = entry.model_output_text
        modified_model_output_text = self.add_positions(model_output_text)
        
        _prompt = prompt_spans.format(
            question=entry.model_input,
            llm_output=modified_model_output_text,
            wiki=wiki_text,
            json_schema=HallucinationDetectionSpansReturn.model_json_schema(),
        )
        
        
        messages = self.build_messages(_prompt)
        response = await self.call_llm(messages, HallucinationDetectionSpansReturn)
        
        for span in response.spans:
            start = span.start
            end = span.start + len(span.text)
            
            entry.span_labeling_output.hard_predictions.append([start, end])
            entry.span_labeling_output.soft_predictions.append(
                {
                    "start": start,
                    "end": end,
                    "prob": 1.0
                }
            )
        return entry
    
    
    def add_positions(self, text):
        out = ""
    
        for i in range(len(text)):
            if ((i > 0 and text[i - 1].isspace()) or i == 0) and not text[i].isspace():
                out += f"|>{i}<|"
            
            out += text[i]
        return out
    
    
        
class SpanLabelingBaselineAll(SpanLabeling):
    async def process_entry(self, entry: Entry) -> Entry:
        facts = entry.fact_spans
        
        retrieval_outputs = entry.retrieval_output
        
        wiki_text = retrieval_outputs.wiki_content
        model_output_text = entry.model_output_text
        
        for fact_i, retrieval_i in zip(facts, retrieval_outputs.retrieved):
            assert fact_i.fact == retrieval_i["fact"]
            
            start, end, fact = fact_i.start, fact_i.end, fact_i.fact
            
            if [start,end] in entry.span_labeling_output.hard_predictions:
                continue
            
            entry.span_labeling_output.hard_predictions.append([start, end])
            entry.span_labeling_output.soft_predictions.append(
                {
                    "start": start,
                    "end": end,
                    "prob": 1.0
                }
            )
            
        return entry
    
            

    
#%%
if __name__ == "__main__":
    #%%
    from mushroom.pipeline.data_connector.data_connector import read_dataset
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
    # import asyncio
    span_labeling_step = SpanLabeling()
    
     
    # predictions = await span_labeling_step.async_call(dataset)
    predictions = span_labeling_step(dataset)
    #%%    
    ious, cors = evaluate_span_labeling(predictions)
    
    print("IOUs:", ious.mean())
    print("CORs:", cors.mean())
    
    
    #%%
    
    
    #%%
    settings


