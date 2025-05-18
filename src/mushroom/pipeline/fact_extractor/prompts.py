FACT_EXTRACTION='''
You are given: 
- text: original text  

Your job is to split the text into Binary Relational Facts, each consisting of Predicate, Subject and Object.  
Togethher with these three elements, you will also provide a reformulation of the fact.
Each object and each subject in the text should be covered by at least one fact.

For each fact, produce:  
  - Predicate : the action or state of being
  - Subject : the entity that performs the action or is in the state
  - Object : the entity that is affected by the action or state
  - Reformulation : a reformulation of the fact with no additional information 

Return exactly this JSON structure:
{
  "facts": [
    { "Predicate": "...", "Subject": "...", "Object": "...", "Reformulation": "..."},
    { "Predicate": "...", "Subject": "...", "Object": "...", "Reformulation": "..."},
    …
  ]
}

# Example:
text: "Petra van Stoveren won a silver medal in the 2008 Summer Olympics in Beijing, China."

Facts:
json
{
  "facts": [
  {
    "Predicate": "won",
    "Subject": "Petra van Stoveren",
    "Object": "silver medal",
    "Reformulation": "Petra van Stoveren won a silver medal."
  },
    {
    "Predicate": "won medal in event",
    "Subject": "Petra van Stoveren",
    "Object": "2008 Summer Olympics",
    "Reformulation": "Petra van Stoveren won a medal in the 2008 Summer Olympics."
  },
  {
    "Predicate": "won medal in location",
    "Subject": "Petra van Stoveren",
    "Object": "Beijing, China",
    "Reformulation": "Petra van Stoveren won a medal in Beijing, China."
  },
  {
    "Predicate": "held in city",
    "Subject": "2008 Summer Olympics",
    "Object": "Beijing",
    "Reformulation": "The 2008 Summer Olympics were held in Beijing."
  },
    {
    "Predicate": "held in country",
    "Subject": "2008 Summer Olympics",
    "Object": "China",
    "Reformulation": "The 2008 Summer Olympics were held in China."
  }
]
}
'''

INDEXING='''
You are given:
- model_output_text: the original model output text
- fact: a fact from the text
- fact_object: the object of the fact
- number_of_occurences: the number of occurrences of the fact string in the text
Your job is to determine the index of the occurrence of the Object in the model output text which relates to the fact given
You will be given the number of occurrences of the fact in the text. Retun number from 1 to the number of occurences.
You can use the whole fact to help you find the index of the words in the text.
Return the index of the occurrence to be used, as an integer.
Do not return any other text or explanation.
Return a single integer, like:
1


# Example1:
model_output_text: "Petra van Stoveren won a silver medal in the 2008 Summer Olympics in Beijing, China. Jacob van Stoveren won a silver medal in the 2012 Summer Olympics in London, England."
fact: "Petra van Stoveren won a silver medal."
fact_object: "silver medal"
number_of_occurences: 2
Return:
1

# Example2:
model_output_text: "Petra van Stoveren won a silver medal in the 2008 Summer Olympics in Beijing, China. Jacob van Stoveren won a silver medal in the 2012 Summer Olympics in London, England."
fact: "Jacob van Stoveren won a silver medal."
fact_object: "silver medal"
number_of_occurences: 2
Return:
2
    '''
