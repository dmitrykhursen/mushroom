import torch
import faiss
import numpy as np
import ast
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
from transformers import LlamaForCausalLM, LlamaTokenizer
from sentence_transformers import CrossEncoder

# Select device (GPU if available)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# CONFIG: choose similarity metric: 'l2' or 'cosine'
SIMILARITY_METRIC = 'cosine'
#SIMILARITY_METRIC = 'l2'

# Load datasets
qs_dataset = load_dataset("rag-datasets/rag-mini-bioasq", "question-answer-passages")
text_corpus_dataset = load_dataset("rag-datasets/rag-mini-bioasq", "text-corpus")
qs_df = qs_dataset['test'].to_pandas()
text_corpus_df = text_corpus_dataset['passages'].to_pandas()

# Load embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2", device=device)
print("Encoding passages...")

# Create embeddings
#passages = text_corpus_df['passage'].head(10000).tolist()
passages = text_corpus_df['passage'].tolist()
passage_embeddings = embedder.encode(passages, convert_to_tensor=True, device=device)

# Normalize if using cosine
if SIMILARITY_METRIC == 'cosine':
    passage_embeddings = torch.nn.functional.normalize(passage_embeddings, p=2, dim=1)

# Convert to numpy (for FAISS)
passage_embeddings_np = passage_embeddings.cpu().numpy().astype(np.float32)

# Create FAISS index
dimension = passage_embeddings_np.shape[1]
if SIMILARITY_METRIC == 'l2':
    index = faiss.IndexFlatL2(dimension)
elif SIMILARITY_METRIC == 'cosine':
    index = faiss.IndexFlatIP(dimension)
else:
    raise ValueError("Unsupported similarity metric")

index.add(passage_embeddings_np)
print(f"FAISS index created with {index.ntotal} passages using {SIMILARITY_METRIC} similarity.")

def retrieve_passages(query, k):
    query_embedding = embedder.encode([query], convert_to_tensor=True, device=device)
    if SIMILARITY_METRIC == 'cosine':
        query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=1)

    query_embedding_np = query_embedding.cpu().numpy().astype(np.float32)
    distances, indices = index.search(query_embedding_np, k)

    retrieved_passages = []
    for i, idx in enumerate(indices[0]):
        retrieved_passages.append({
            "passage_id": text_corpus_df.iloc[idx]['id'],
            "passage_text": text_corpus_df.iloc[idx]['passage'],
            "similarity_score": distances[0][i]
        })
    return retrieved_passages

def evaluate_retrieval(question_idx, k=3):
    user_query = qs_df.iloc[question_idx]['question']
    answer = qs_df.iloc[question_idx]['answer']
    relevant_passage_ids = ast.literal_eval(qs_df.iloc[question_idx]['relevant_passage_ids'])

    retrieved = retrieve_passages(user_query, k=k)

    print("=" * 100)
    print(f"Question: {user_query}")
    print(f"Answer: {answer}")
    print(f"Relevant Passages (IDs): {relevant_passage_ids}")

    for r in retrieved:
        is_relevant = r['passage_id'] in relevant_passage_ids
        print(f"\nRetrieved Passage ID: {r['passage_id']}")
        print(f"Similarity Score: {r['similarity_score']:.4f}")
        print(f"Passage: {r['passage_text'][:200]}...")
        print(f"Is this passage relevant? {'Yes' if is_relevant else 'No'}")
        print("-" * 50)

    return user_query, retrieved

# Main evaluation
query, rag_retrieved = evaluate_retrieval(question_idx=0, k=20)


# Load cross-encoder reranker
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=device)
# Prepare input pairs (query, passage)
rerank_inputs = [(query, r['passage_text']) for r in rag_retrieved]
# Get relevance scores from cross-encoder
rerank_scores = reranker.predict(rerank_inputs)
# Attach scores and rerank
for i, r in enumerate(rag_retrieved):
    r['rerank_score'] = rerank_scores[i]

# Sort by rerank score (descending)
cross_encoder_retrieved = sorted(rag_retrieved, key=lambda x: x['rerank_score'], reverse=True)
# Print top 5 reranked passages
for i, r in enumerate(cross_encoder_retrieved[:5]):
    print(f"\nReranked #{i+1}")
    print(f"Passage ID: {r['passage_id']}")
    print(f"Cross-Encoder Score: {r['rerank_score']:.4f}")
    print(f"Passage: {r['passage_text'][:200]}...")


# Load QA pipeline (you can try different models like PubMedBERT if needed)
qa_model = "deepset/roberta-base-squad2"
# qa_model = "ktrapeznikov/biobert_v1.1_pubmed_squad_v2"
qa_model="allenai/biomed_roberta_base"

qa_pipeline = pipeline("question-answering", model=qa_model, device=0 if device == "cuda" else -1)
# Apply QA to top reranked passage
top_k = 5
qa_results = []
for i in range(top_k):
    context = rag_retrieved[i]['passage_text']

    # context = cross_encoder_retrieved[i]['passage_text']
    result = qa_pipeline(question=query, context=context)
    result["passage_id"] = retrieved[i]['passage_id']
    result["context_snippet"] = context[:150]
    qa_results.append(result)

# Sort by score
qa_results.sort(key=lambda x: x['score'], reverse=True)

# Print top answer
print("\n=== QA VALIDATION TOP ANSWER ===")
top_ans = qa_results[0]
print(f"Passage ID: {top_ans['passage_id']}")
print(f"Answer: \"{top_ans['answer']}\" (Score: {top_ans['score']:.4f})")
print(f"Context: {top_ans['context_snippet']}...")

# OPTIONAL: Llama classification stub (currently commented out)
# model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
# tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
# def classify_query_with_llama(...): ...
