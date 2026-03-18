import torch
import asyncio
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

model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"

# Prevent concurrent GPU overload
semaphore = asyncio.Semaphore(1)


def _run_inference(prompt: str) -> str:

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=4096
    ).to(device)

    try:
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=400,
                do_sample=False,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id
            )

        result = tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        raise RuntimeError("GPU out of memory.")

    finally:
        del inputs
        if device == "cuda":
            torch.cuda.empty_cache()

    return result


async def generate_analysis(text: str) -> str:

    prompt = f"""
You are an expert document analyst.

Return ONLY this JSON format:
{{
  "overview": "",
  "summary": "",
  "highlights": []
}}

Document:
{text}
"""

    async with semaphore:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _run_inference, prompt)