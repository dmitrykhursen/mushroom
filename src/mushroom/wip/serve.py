import openai



from openai import OpenAI
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="token-abc123",
)

# client.models.llm_engine.abort_request(list(map(lambda x: str(x), list(range(100000)))))
# print(client.models.llm_engine.get_num_unfinished_requests())
# import vllm
completion = client.chat.completions.create(
  model="google/gemma-3-4b-it",
  messages=[
    {"role": "user", "content": "Hello! Tell me a joke."},
  ],
  max_tokens=1024,
  temperature=5.7,
  top_p=0.9,
  seed=1,
#   min_tokens=10,
)

print(completion.choices[0].message)

