import json
import pandas as pd
import wikipedia

def get_wikipedia_content(url):
    """
    Extract full Wikipedia page content from the given URL.
    """
    try:
        page_title = url.split("/wiki/")[-1].replace("_", " ")
        page = wikipedia.page(title=page_title, auto_suggest=False)
        return page.content
    except wikipedia.exceptions.PageError:
        print(f"[ERROR] Page not found for URL: {url}")
        return None
    except wikipedia.exceptions.DisambiguationError as e:
        print(f"[WARNING] Disambiguation error for {url}. Options: {e.options}")
        return None
    except Exception as e:
        print(f"[ERROR] Unexpected error for URL {url}: {e}")
        return None


# Load the JSON file
with open("entries_with_facts.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Normalize records
rows = []
for item in data:
    row = {
        "id": item.get("id"),
        "lang": item.get("lang"),
        "model_id": item.get("model_id"),
        "model_input": item.get("model_input"),
        "model_output_text": item.get("model_output_text"),
        "wikipedia_url": item.get("wikipedia_url"),
        "text_len": item.get("text_len"),
        
        # Serialize lists/dicts to strings for DataFrame compatibility
        "model_output_tokens": json.dumps(item.get("model_output_tokens", [])),
        "model_output_logits": json.dumps(item.get("model_output_logits", [])),
        "soft_labels": json.dumps(item.get("soft_labels", [])),
        "hard_labels": json.dumps(item.get("hard_labels", [])),
        "annotations": json.dumps(item.get("annotations", {})),
        "fact_spans": json.dumps(item.get("fact_spans", [])),
        "atomic_facts": json.dumps(item.get("atomic_facts", [])),
        "retrieval_output": json.dumps(item.get("retrieval_output", {})),
        "span_labeling_output": json.dumps(item.get("span_labeling_output", {})),
    }
    rows.append(row)

# Create DataFrame
df = pd.DataFrame(rows)

# Display or save
# import ace_tools as tools; tools.display_dataframe_to_user(name="RAG Fact-Checking Dataset", dataframe=df)
print(df.head())  # or df.to_string() for full output
# Print size (rows, columns)
print(f"\n📊 DataFrame shape: {df.shape[0]} rows × {df.shape[1]} columns")

print(df.iloc[0])  # Print the first row

# Parse the JSON string in the first row of the 'fact_spans' column
fact_spans_row = json.loads(df.loc[0, "fact_spans"])

# Print the full content nicely
print("🧠 fact_spans (first row):")
for fact in fact_spans_row:
    print(f" - Fact: {fact['fact']}")
    print(f"   Span: ({fact['start']}, {fact['end']})")

# Get URL from first row
first_url = df.loc[0, "wikipedia_url"]

# Fetch content
# wiki_content = get_wikipedia_content(first_url)

my_example_url = "https://en.wikipedia.org/wiki/Bundesautobahn_73"
my_example_url = "https://en.wikipedia.org/wiki/Ukraine"
wiki_content = get_wikipedia_content(my_example_url)


# Print full content
print(f"📝 Wikipedia Content for: {first_url}\n{'-'*60}")

print(wiki_content if wiki_content else "[No content retrieved]")
# print(repr(wiki_content) if wiki_content else "[No content retrieved]")

# TODO: actually wiki api only terurns text content, it doesnt' return tables or images. Should I take the full html content and then parse it?