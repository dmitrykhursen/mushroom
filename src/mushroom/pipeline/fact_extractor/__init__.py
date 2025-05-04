from collections import namedtuple
from . import fact_alignment as alignment
from . import fact_extraction as extraction

Fact = namedtuple('Fact', ['atomic_fact', 'text_indices'])

class TextFacts:
    def __init__(self, text):
        self.text = text

    def process(self):
        atomic_facts = extraction.extract_atomic_facts(self.text)

        self.facts = [Fact(atomic_fact, alignment.align_fact_to_text(atomic_fact, self.text)) 
                        for atomic_fact in atomic_facts]
        
    def get_facts(self):
        return self.facts
