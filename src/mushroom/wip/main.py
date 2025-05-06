import fact_extractor
from data_loader import load_model_outputs

# load the data from the raw data directory
texts = load_model_outputs("data/raw")

# get the atomic facts from the texts
facts = []
for text in texts:
    textFacts = fact_extractor.TextFacts(text)
    textFacts.process()
    facts.append(textFacts.get_facts())

print(facts)
