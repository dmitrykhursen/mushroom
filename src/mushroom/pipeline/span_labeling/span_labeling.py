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
from copy import deepcopy
from typing import Optional

class HallucinationDetectionEnum(str, Enum):
    HALLUCINATED = "HALLUCINATED"
    PARTIALLY_HALLUCINATED = "PARTIALLY HALLUCINATED"
    SUPPORTED = "SUPPORTED"


class HallucinationDetectionReturn(BaseModel):
    label: HallucinationDetectionEnum = Field(
        description="The label for the hallucination detection task.")
    reason: str = Field(
        description="The reason for the label. This is not used in the final output, but can be useful for debugging.")

class HallucinationDetectionSpan(BaseModel):
    start: Optional[int] = Field(
        description="The start index of the hallucinated span in the model output.")
    end: Optional[int] = Field(
        description="The end index of the hallucinated span in the model output.")
    text: Optional[str] = Field(
        description="The text of the hallucinated span in the model output.")
    reason: Optional[str] = Field(
        description="The reason for the label.")
    
    
    
class HallucinationDetectionSpansReturn(BaseModel):
    spans: Optional[List[HallucinationDetectionSpan]] = Field(
        description="The spans of the hallucinated text in the model output.")
    

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

prompt_spans = """
Instructions:
You are given an *input prompt*, the *output from an LLM (large language model)*, and a reference *ground truth* or *context* (such as a passage, database, or knowledge source). Your task is to analyze the LLM's output and identify any hallucinations—statements or facts that are not supported by, or directly contradict, the reference information.

- A **hallucination** is any information, claim, or detail present in the LLM's output that is provably factually incorrect.
- Do not mark typos or orphan symbols as hallucinations.
- Lack of specificity in the LLM's output does not count as a hallucination if it does not contradict the context.
- Irrelevant or off-topic information is not considered a hallucination if it does not contradict the context.
- Concentrate on identifying hallucinations that are explicitly factually incorrect or explicitly contradicted by the reference context.
- Mark each hallucinated span or sentence, and briefly justify why it is hallucinated, citing any contradictions.
- For instance, if the LLM stated the year incorrectly, point to the year.
- Do not include grammatical errors or stylistic issues in your analysis.

Hallucination Analysis:
- List all hallucinated spans.
- For each, explain why it is considered a hallucination, referencing the context.
- the correct indices u can use are marked with "|>*index*<|". If the span is longer, use the end index of the token that is not in the span.

Now, analyze the following:
(Input prompt, LLM output, and reference context will be provided below.)

Input Prompt:
{question}

LLM Output:
{llm_output}

Reference Context:
{wiki}


Format your response as a JSON object:
{json_schema}

"""


class SpanLabeling:
    def __init__(self):
        
        self.client = AsyncOpenAI(
            # base_url="http://localhost:8000/v1",
            # api_key="token-abc123",
              base_url='http://localhost:11434/v1', # ollama server endpoint
            api_key='ollama', # placeholder
        )

    
    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)
    
    async def run(self, dataset: List[Entry]) -> List[Entry]:
        dataset = deepcopy(dataset)
        
        for entry in dataset:
            entry = await self.process_entry(entry)
            
        return dataset
    
    async def call_llm(self, question: str, fact: str, wiki: str) -> str:
        completion = await self.client.completions.create(
        model="gemma-3-27b-it",
        prompt=prompt_classify.format(
            question=question,
            fact=fact,
            wiki=wiki,
        ),
        max_tokens=1024,
        temperature=0.7,
        top_p=0.9,
        seed=1,
           extra_body={
            "guided_json": HallucinationDetectionReturn.model_json_schema(),
            # "guided_decoding_backend": "outlines"
        },
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

class SpanLabelingSpans(SpanLabeling):
    def __init__(self):
        super().__init__()
        
    
       
    async def call_llm(self, question: str, llm_output: str, wiki: str) -> HallucinationDetectionSpansReturn:
        
        # add indices to llm_output for more robust result
        # matches = [(m.start(), m.end()) for m in re.finditer(r'\S+', llm_output)]
        # llm_output = " ".join([f"{i} {llm_output[i]}" for i in range(len(llm_output))])
        
        matches = [(m.start(), m.end()) for m in re.finditer(r'\S+', llm_output)]
        llm_output_modified = " ".join([f"|>{s}<|{llm_output[s:e]}" for s,e in matches])
        
        
        
        
        llm_output_modified = add_positions(llm_output)
        
        
        try: 
            completion = await self.client.beta.chat.completions.parse(
            model="gemma3:27b-it-qat",
            messages=[{"role": "user", "content": 
            
            
            prompt_spans.format(
                question=question,
                llm_output=llm_output_modified,
                wiki=wiki,
                json_schema=HallucinationDetectionSpansReturn.model_json_schema(),
            ),
            
            
            
                }],
        
            max_tokens=1024,
            temperature=0.7,
            top_p=0.9,
            response_format=HallucinationDetectionSpansReturn,
            )
        except Exception as e:
            print("Error in LLM call:", e)
            return HallucinationDetectionSpansReturn(
                spans=[],
            )
        
        
        
        # completion = await self.client.completions.create(
        # model="google/gemma-3-27b-it",
        # prompt=
        # prompt_spans.format(
        #     question=question,
        #     llm_output=llm_output_modified,
        #     wiki=wiki,
        #     json_schema=HallucinationDetectionSpansReturn.model_json_schema(),
        # ),
        # max_tokens=1024,
        # temperature=0.7,
        # top_p=0.9,
        # seed=1,
        # extra_body={
        #     "guided_json": HallucinationDetectionSpansReturn.model_json_schema(),
        #     # "guided_decoding_backend": "outlines"
        # },
        # )

        # response_text = completion.choices[0].text

        try:
            # parsed_output = HallucinationDetectionSpansReturn.model_validate_json(response_text)
            parsed_output = completion.choices[0].message.parsed
            print("Parsed output:", parsed_output)
        except Exception as e:
            print("Error parsing output:", e)
            print("Response text:", completion.choices[0].message.content)
            parsed_output = HallucinationDetectionSpansReturn(
                spans=[],
            )
        return parsed_output
        
    
    async def process_entry(self, entry: Entry) -> Entry:
        facts = entry.fact_spans
        retrieval_outputs = entry.retrieval_output
        
        wiki_text = retrieval_outputs.wiki_content
        model_output_text = entry.model_output_text
        
        span_output = await self.call_llm(
            question=entry.model_input,
            llm_output=model_output_text,
            wiki=wiki_text,)
        
        for span in span_output.spans:
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
            # elif span.label == HallucinationDetectionEnum.PARTIALLY_HALLUCINATED:
            #     entry.span_labeling_output.hard_predictions.append([span.start, span.end])
            #     entry.span_labeling_output.soft_predictions.append(
            #         {
            #             "start": span.start,
            #             "end": span.end,
            #             "prob": 0.5
            #         }
            #     )
            # elif span.label == HallucinationDetectionEnum.SUPPORTED:
            #     pass
            # else:
            #     raise ValueError(f"Unknown label: {span.label}")
        # print("Span output:", span_output)
        
        return entry
        
        
            

class SpanLabelingBaselineAll(SpanLabeling):
    def __init__(self):
        super().__init__()
        
    
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
    
            
def add_positions(text):
    out = ""
    
    for i in range(len(text)):
        if ((i > 0 and text[i - 1].isspace()) or i == 0) and not text[i].isspace():
            out += f"|>{i}<|"
            
        out += text[i]
    return out
        

    
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
    span_labeling_step = SpanLabeling()
    predictions = await span_labeling_step.run(dataset)
    
    #%%
    for entry in predictions:
        print(entry.span_labeling_output.hard_predictions)
        print(entry.span_labeling_output.soft_predictions)
        print()
        # entry.span_labeling_output.hard_predictions = list(set(entry.span_labeling_output.hard_predictions))
        # entry.span_labeling_output.soft_predictions = list(set(entry.span_labeling_output.soft_predictions))
    #%%
    ious, cors = evaluate_span_labeling(predictions)
    
    print("IOUs:", ious.mean())
    print("CORs:", cors.mean())
    
    
    #%%
       
    span_labeling_baseline_step = SpanLabelingBaselineAll()
    predictions_baseline = await span_labeling_baseline_step.run(dataset)
    
    ious, cors = evaluate_span_labeling(predictions_baseline)
    
    print("IOUs:", ious.mean())
    print("CORs:", cors.mean())
    
    #%%
    span_labeling_span = SpanLabelingSpans()
    predictions_span = await span_labeling_span.run(dataset)
    
    
#     #%%
    
    

    
#     #%%
    
#     for entry in predictions_span:
#         # print(entry.span_labeling_output.hard_predictions)
#         # for start, end in entry.span_labeling_output.hard_predictions:
#         #     print(entry.model_output_text[start:end])
        
#         for i, span in enumerate(entry.span_labeling_output.hard_predictions):
#             print(i, span)
#             start, end = span
            
#             # entry.span_labeling_output.hard_predictions[i] = [start, end]
#             entry.span_labeling_output.hard_predictions[i] = [start, min(end, len(entry.model_output_text))]
            
            
#             print(entry.retrieval_output.wiki_content) 
#         for span in entry.span_labeling_output.soft_predictions:
#             start, end = span["start"], span["end"]
            
#             span["end"] = min(end, len(entry.model_output_text))
            
        
#         print(entry.span_labeling_output.soft_predictions)
#         print()
#         # entry.span_labeling_output.hard_predictions = list(set(entry.span_labeling_output.hard_predictions))
#         # entry.span_labeling_output.soft_predictions = list(set(entry.span_labeling_output.soft_predictions))
        
        
#     #%%
    
    
    
#     ious, cors = evaluate_span_labeling(predictions_span)
#     print("IOUs:", ious.mean())
#     print("CORs:", cors.mean())
    
#     #%%
#     client = OpenAI(
#         base_url='http://localhost:11434/v1', # ollama server endpoint
#         api_key='ollama', # placeholder
#     )
    
#     client.beta.completions.parse(
#         model="gemma3:27b-it-qat",
#         prompt="aaaa",
#         response_format=HallucinationDetectionSpansReturn,
#     )
#     #%%
#     text ='''
# I have two pets.
# A cat named Luna who is 5 years old and loves playing with yarn. She has grey fur.
# I also have a 2 year old black cat named Loki who loves tennis balls.
#             '''
            
#     llm_output_modified = add_positions(text)
    
    
#     inp = """
    
#     just list the correct spans in the llm output. The correct indices are marked with "|>*index*<|".
    
#     {llm_output}
    
    
#     """
    
#     inp = inp.format(llm_output=llm_output_modified)
#     print(inp)
#     #%%
    
#     #%%
#     a = client.beta.chat.completions.parse(
#         model="gemma3:27b-it-qat",
#         messages=[{"role": "user", "content": inp}],
        
        
        
#         response_format=HallucinationDetectionSpansReturn,
        
#     )
    
#     #%%
    
#     #%%
#     for span in a.choices[0].message.parsed.spans:
#         # print(span.reason)
#         print(span.text)
#         # print(text[span.start:span.end])
#         # print(text[span.start:span.end])
#         print(text[span.start:span.start + len(span.text)])
#         print(span.start, span.end)
        
#         print(span.text == text[span.start:span.start + len(span.text)])
        
#     #%%
#     text[0]
#     #%%
#     llm_output = "The capital; of France is Paris."
    
#     # add indices to llm_output for more robust result
#     matches = [(m.start(), m.end()) for m in re.finditer(r'\S+', llm_output)]
#     llm_output_modified = " ".join([f"|>{s}<|{llm_output[s:e]}" for s,e in matches])
    
#     llm_output_modified
#     #
    
    
    
# #%%
# s = """
# Format your response as a JSON object:
# {{
#     "spans": [
#         {{
#             "start": <start_index>,
#             "end": <end_index>,
#             "text": "<text_of_the_hallucinated_span>",
#             "reason": "<reason_for_hallucination>",
#             "label": "<hallucination_label>",
#         }},
#         ...
#     ]      
# }}

# Now, analyze the following:
# (Input prompt, LLM output, and reference context will be provided below.)

# Input Prompt:
# {question}

# LLM Output:
# {llm_output}

# Reference Context:
# {wiki}
# """


# s.format(question="What is the capital of France?", llm_output="The capital; of France is Paris.", wiki="The capital of France is Paris.")
# # %%
# HallucinationDetectionSpansReturn.model_json_schema()