
prompt_atomic = '''
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

Return your result in a JSON format with the following schema:
{json_schema}

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
