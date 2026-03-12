# from transformers import AutoTokenizer, AutoModelForCausalLM
# import torch

# # Load GPU model (Mistral 7B)
# model_name = "mistralai/Mistral-7B-Instruct-v0.2"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     torch_dtype=torch.float16,
#     device_map="auto"
# )

# torch.backends.cudnn.benchmark = True
# torch.backends.cuda.matmul.allow_tf32 = True

# def generate_analysis(text):
#     """Generate advanced, professional 4-section PDF analysis using LLM"""

    


#     prompt = f"""
#     You are a professional document analyst capable of reviewing contracts, reports, policies, financial documents, academic papers, and general business PDFs.

#     Analyze the document and return ONLY valid JSON.

#     STRICT RULES:
#     - Do NOT explain anything.
#     - Do NOT add extra text.
#     - Output must be valid JSON only.

#     Return EXACTLY this format:

#     {{
#     "document_type": "Type of document (e.g., Contract, Report, Invoice, Policy, Research Paper, Other)",
#     "summary": "Concise executive summary",
#     "key_highlights": [
#     "Important point 1",
#     "Important point 2",
#     "Important point 3",
#     "Important point 4"
#     ],
#     "risk_analysis": [
#     "Risk or concern 1 (if applicable)",
#     "Risk or concern 2",
#     "Risk or concern 3",
#     "Risk or concern 4"
#     ]
#     }}

#     Document Content:
#     ----------------
#     {text}
#     ----------------

#     JSON:
#     """


#     inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
#     output = model.generate(
#         **inputs,
#         max_new_tokens=500,   # increased for detailed JSON
#         temperature=0.3,       # low for deterministic output
#         do_sample=False
#     )

# #    result = tokenizer.decode(output[0], skip_special_tokens=True)
#     result = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
#     return result


import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM

# Model name
model_name = "mistralai/Mistral-7B-Instruct-v0.2"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load model to GPU
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Performance optimization
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True


def generate_analysis(text: str) -> str:
    """
    Generate overview, summary and highlights
    from given PDF text chunk.
    """

    prompt = f"""
You are a professional document analyst capable of reviewing business documents, contracts, reports, policies, and academic PDFs.

Analyze the document and return ONLY valid JSON.

STRICT RULES:
- Do NOT explain anything.
- Do NOT add extra text.
- Output must be valid JSON only.

Return EXACTLY this format:

{{
"overview": "Short high-level description of what this document is about",
"summary": "Clear executive summary explaining purpose and key content",
"highlights": [
  "Important point 1",
  "Important point 2",
  "Important point 3",
  "Important point 4"
]
}}

Document Content:
----------------
{text}
----------------

JSON:
"""

    # Tokenize safely (prevent overflow)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=6000
    ).to("cuda")

    # Generate response
    output = model.generate(
        **inputs,
        max_new_tokens=500,
        temperature=0.2,          # Low randomness for stable JSON
        do_sample=False,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id
    )

    # Extract only generated part
    result = tokenizer.decode(
        output[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )

    return result