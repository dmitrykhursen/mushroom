from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import LlamaForCausalLM, LlamaTokenizer
import ast

def retrieve_passages(query, k):
    query_embedding = embedder.encode([query], convert_to_numpy=True).astype(np.float32)
    distances, indices = index.search(query_embedding, k)
    
    retrieved_passages = []
    for i, idx in enumerate(indices[0]):
        retrieved_passages.append({
            "passage_id": text_corpus_df.iloc[idx]['id'],
            "passage_text": text_corpus_df.iloc[idx]['passage'],
            "similarity_score": distances[0][i]
        })
    return retrieved_passages


def evaluate_retrieval(question_idx, k=3):
    # Get the question from the dataset
    user_query = qs_df.iloc[question_idx]['question']
    answer = qs_df.iloc[question_idx]['answer']
    relevant_passage_ids = ast.literal_eval(qs_df.iloc[question_idx]['relevant_passage_ids'])  # safely convert to list
    
    # Retrieve passages based on the query
    retrieved = retrieve_passages(user_query, k=k)
    
    print("= " * 50)
    print(f"Question: {user_query}")
    print(f"Answer: {answer}")
    print(f"Relevant Passages (IDs): {relevant_passage_ids}")
    
    for r in retrieved:
        # Check if the retrieved passage is relevant based on its ID
        is_relevant = r['passage_id'] in relevant_passage_ids
        print(f"\nRetrieved Passage ID: {r['passage_id']}")
        print(f"Similarity Score: {r['similarity_score']:.4f}")
        print(f"Passage: {r['passage_text'][:200]}...")  # Print the beginning of the passage text
        print(f"Is this passage relevant? {'Yes' if is_relevant else 'No'}")
        print("-" * 50)
    
    return user_query, retrieved


# Define the classification prompt
def classify_query_with_llama(query, retrieved_passages):
    # Format the context to be passed to the Llama model
    context = "\n\n".join([f"Passage ID: {r['passage_id']}\n{r['passage_text']}" for r in retrieved_passages])
    
    # Construct the prompt
    prompt = (
        f"Context from Wikipedia:\n{context}\n\n"
        f"Question: \"{query}\"\n"
        "Is this statement True (supported) or False (not supported/hallucinated)? "
        "Answer 'True' or 'False' and provide a brief explanation."
    )

    # Tokenize and generate a response
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs['input_ids'], max_length=512)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return response.strip()


qs_dataset = load_dataset("rag-datasets/rag-mini-bioasq", "question-answer-passages")
text_corpus_dataset = load_dataset("rag-datasets/rag-mini-bioasq", "text-corpus")

print(f'{qs_dataset=}')
print(f'{text_corpus_dataset=}')

# Convert both to pandas DataFrames
qs_df = qs_dataset['test'].to_pandas()
text_corpus_df = text_corpus_dataset['passages'].to_pandas()

# Print them
print(qs_df.head())
print()
print(qs_df.iloc[0])
print("- - -")
print(text_corpus_df.head())


# Load a pre-trained sentence-transformer model
embedder = SentenceTransformer("all-MiniLM-L6-v2")  # You can also choose other models

print(f"Before encoding passage")
# Create embeddings for the passages
passage_embeddings = embedder.encode(text_corpus_df['passage'].head(1000).tolist(), convert_to_numpy=True)
print(f'{passage_embeddings=}')
# exit()


# Create FAISS index for fast retrieval
dimension = passage_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(passage_embeddings.astype(np.float32))

# Display index info
print(f"FAISS index created with {index.ntotal} passages.")


query, retrieved = evaluate_retrieval(question_idx=0, k=10)

# Load Llama model and tokenizer
model_name = "meta-llama/Llama-2-7b-hf"  # You can change this to any available Llama model
model = LlamaForCausalLM.from_pretrained(model_name)
tokenizer = LlamaTokenizer.from_pretrained(model_name)

print("*-*"*50)
# Print Llama's classification result
llama_response = classify_query_with_llama(query, retrieved)
print("Llama's Classification Output:")
print(llama_response)

# TODO: use HF token