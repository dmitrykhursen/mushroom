import json
import re
from urllib.parse import unquote, urlparse
import wikipedia
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# -------------------- CONFIG --------------------
MODEL_NAME = 'all-MiniLM-L6-v2'  # you can swap with other models later
TOP_K = 3  # top retrieved passages
OUTPUT_SUFFIX = '_new.json'

# -------------------- UTILS --------------------
def normalize_url_and_extract_title(url):
    decoded = unquote(url)
    parsed = urlparse(decoded)
    title = parsed.path.split("/wiki/")[-1].replace("_", " ")
    return title

def fetch_wikipedia_content(title):
    try:
        page = wikipedia.page(title, auto_suggest=False)
        return page.content
    except wikipedia.exceptions.DisambiguationError as e:
        print(f"[DISAMBIGUATION] {title} -> {e.options[0]}")
        try:
            return wikipedia.page(e.options[0], auto_suggest=False).content
        except:
            return ""
    except wikipedia.exceptions.PageError:
        print(f"[NOT FOUND] {title}")
        return ""
    except Exception as e:
        print(f"[ERROR] {title}: {e}")
        return ""

def split_into_sentences(text):
    return [s for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s]

def sliding_windows(sentences, window_size):
    return [" ".join(sentences[i:i+window_size]) for i in range(len(sentences)-window_size+1)]

# -------------------- LOAD DATA --------------------
with open("entries_with_facts.json", "r", encoding="utf-8") as f:
    data = json.load(f)

model = SentenceTransformer(MODEL_NAME)

for item in data:
    url = item.get("wikipedia_url")
    if not url:
        continue

    title = normalize_url_and_extract_title(url)
    content = fetch_wikipedia_content(title)
    # item["wiki_content"] = content

    # Chunking
    sentences = split_into_sentences(content)
    chunks = []
    # for w in [1, 2, 3]:
    #     chunks.extend(sliding_windows(sentences, w))
    
    chunks.extend(sliding_windows(sentences, 1))
    
    # item["chunks"] = chunks

    # Embedding chunks
    if not chunks:
        continue
    embeddings = model.encode(chunks, convert_to_numpy=True)
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]

    # FAISS index
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    # Fact retrieval
    fact_spans = item.get("fact_spans", [])
    if isinstance(fact_spans, str):
        fact_spans = json.loads(fact_spans)

    retrieved = []
    for fact_entry in fact_spans:
        fact_text = fact_entry.get("fact", "")
        if not fact_text:
            continue
        query = model.encode([fact_text], convert_to_numpy=True)
        faiss.normalize_L2(query)
        distances, indices = index.search(query, TOP_K)

        top_chunks = []
        for score, idx in zip(distances[0], indices[0]):
            if idx >= 0 and idx < len(chunks):
                top_chunks.append({"chunk": chunks[idx], "score": round(float(score), 3)})

        print(f"\nFact: \"{fact_text}\"")
        for i, entry in enumerate(top_chunks):
            print(f" Top {i+1}: (score={entry['score']:.3f}) {entry['chunk'][:100]}...")

        retrieved.append({
            "fact": fact_text,
            "top_3": top_chunks
        })

    item["retrieval_output"] = {
        "retrieved": retrieved,
        "wiki_content": content
    }

    # print(f"{item=}")
    # break

# -------------------- SAVE OUTPUT --------------------
output_file = "entries_with_facts" + OUTPUT_SUFFIX
with open(output_file, "w", encoding="utf-8") as f_out:
    json.dump(data, f_out, indent=2, ensure_ascii=False)

print(f"\n✅ Saved enriched data to {output_file}")
