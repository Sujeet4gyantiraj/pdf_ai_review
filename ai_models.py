import torch
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM

logger = logging.getLogger(__name__)

model_name = "mistralai/Mistral-7B-Instruct-v0.2"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

device = "cuda" if torch.cuda.is_available() else "cpu"


# -------- Level 1: Chunk Summary --------
def summarize_chunk(chunk_text):

    prompt = f"""
You are an expert document analyst.

Summarize the following content clearly and concisely.

CONTENT:
{chunk_text}
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=3500
    ).to(device)

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=500,
            do_sample=False,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
        )

    result = tokenizer.decode(output[0], skip_special_tokens=True)
    result = result.replace(prompt, "").strip()

    torch.cuda.empty_cache()
    return result


# -------- Level 2: Final Structured Summary --------
def generate_final_summary(all_summaries):

    combined_text = "\n\n".join(all_summaries)

    prompt = f"""
You are a professional AI document summarizer.

Using the below summaries, generate:

1. Overview
2. Detailed Summary
3. Key Highlights

CONTENT:
{combined_text}
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=3500
    ).to(device)

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=700,
            do_sample=False,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
        )

    result = tokenizer.decode(output[0], skip_special_tokens=True)
    result = result.replace(prompt, "").strip()

    torch.cuda.empty_cache()
    return result