from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load GPU model (Mistral 7B)
model_name = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

def generate_analysis(text):
    """Generate advanced, professional 4-section PDF analysis using LLM"""

    


    prompt = f"""
You are a professional PDF reviewer and document analyst.

Analyze the document and return the result ONLY as valid JSON.

DO NOT explain anything.
DO NOT repeat the instructions.
DO NOT write text outside JSON.

Return EXACTLY this JSON format:

{{
"summary": "short summary",
"positive_points": ["point1","point2","point3","point4"],
"negative_points": ["point1","point2","point3","point4"],
"how_to_avoid": ["tip1","tip2"]
}}

Document:
{text}

JSON:
    """


    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    output = model.generate(
        **inputs,
        max_new_tokens=500,   # increased for detailed JSON
        temperature=0.3,       # low for deterministic output
        do_sample=False
    )

#    result = tokenizer.decode(output[0], skip_special_tokens=True)
    result = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return result
