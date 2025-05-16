from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
import torch
from huggingface_hub import login

login("")


model_name = "google/gemma-2-2b"  # or any model you like

def get_opensource_chat_model():
    bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                
    bnb_4bit_quant_type="nf4",        
    bnb_4bit_compute_dtype=torch.bfloat16, 
    bnb_4bit_use_double_quant=True     
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config = bnb_config,     
        device_map="auto",        
        torch_dtype=torch.float16 
    )

    gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
        max_new_tokens=1000,
    )

    return gen