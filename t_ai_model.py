import torch
import asyncio
import logging

logger = logging.getLogger(__name__)

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
MAX_INPUT_TOKENS = 6000
MAX_NEW_TOKENS = 600

# ---------------------------------------------------------------------------
# Lazy model loading                                                (Fix #7)
# The model is NOT loaded at import time.  Call load_model() once from the
# FastAPI lifespan hook so tests and imports don't consume 14 GB of VRAM.
# ---------------------------------------------------------------------------
_tokenizer = None
_model = None
_device = None


def load_model() -> None:
    """
    Load the tokenizer and model into memory.
    Call this once at application startup (e.g. from a FastAPI lifespan event).
    Safe to call multiple times — subsequent calls are no-ops.
    """
    global _tokenizer, _model, _device

    if _model is not None:
        return  # already loaded

    from transformers import AutoTokenizer, AutoModelForCausalLM

    # Decide device BEFORE loading the model so everything is consistent (Fix #8)
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading {MODEL_NAME} onto device: {_device}")

    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if _device == "cuda":
        # Single-device CUDA path — no device_map="auto" so inputs and model
        # always live on the same device.   (Fix #8)
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
        ).to(_device)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
    else:
        # CPU path — use float32 (float16 is unsupported on CPU)
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32,
        )

    _model.eval()
    logger.info("Model loaded successfully.")


def _run_inference(prompt: str) -> str:
    """
    Synchronous inference — runs inside a thread-pool executor so it does
    not block the async event loop.
    """
    if _model is None or _tokenizer is None:
        raise RuntimeError("Model is not loaded. Call load_model() at startup.")

    # --- Tokenize once with truncation (Fix #9 — was tokenized twice) ---
    inputs = _tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_TOKENS,
    )

    input_length = inputs["input_ids"].shape[1]  # length AFTER any truncation

    if input_length == MAX_INPUT_TOKENS:
        logger.warning(
            f"Input was truncated to {MAX_INPUT_TOKENS} tokens. "
            "Consider reducing chunk size."
        )

    inputs = inputs.to(_device)

    try:
        with torch.no_grad():
            output = _model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                repetition_penalty=1.15,
                pad_token_id=_tokenizer.eos_token_id,
            )
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        raise RuntimeError(
            "GPU out of memory — reduce MAX_INPUT_TOKENS or chunk size."
        )
    finally:
        del inputs
        if _device == "cuda":
            torch.cuda.empty_cache()

    # Decode only the newly generated tokens.
    # Use `input_length` (the TRUNCATED length) so the slice is always correct.
    # (Fix #10 — previously used the pre-truncation length from a second encode call)
    new_tokens = output[0][input_length:]
    result = _tokenizer.decode(new_tokens, skip_special_tokens=True)

    return result


async def generate_analysis(text: str) -> str:
    """
    Build a prompt from `text` and run inference asynchronously.
    The blocking inference call is offloaded to a thread-pool executor so
    the FastAPI event loop stays free.
    """
    prompt = f"""<s>[INST]
You are an expert document analyst with deep knowledge across legal, medical,
financial, technical, academic, and business domains.

Your task is to read any type of document and produce a structured JSON briefing
that is accurate, professional, and immediately useful to the reader.

EXAMPLE OUTPUT (follow this structure exactly):
{{
  "overview": "This is a [document type] about [main subject], intended for [audience/purpose].",
  "summary": "4-6 sentence executive summary covering the document purpose, key content,
               important findings or obligations, and conclusions or outcomes.",
  "highlights": [
    "Specific important fact, figure, date, obligation, finding, or condition from the document.",
    "Specific important fact, figure, date, obligation, finding, or condition from the document.",
    "Specific important fact, figure, date, obligation, finding, or condition from the document.",
    "Specific important fact, figure, date, obligation, finding, or condition from the document."
  ]
}}

FIELD RULES:

"overview" — 1-2 sentences only.
  - Identify what TYPE of document this is.
    Examples of types: employment contract, research paper, medical report,
    user manual, financial statement, privacy policy, invoice, academic thesis,
    insurance policy, government notice, product specification, meeting minutes.
  - State the subject, purpose, and intended audience or parties.

"summary" — 4-6 sentences in professional, neutral tone.
  - For CONTRACTS/LEGAL: cover parties, obligations, terms, penalties, duration.
  - For MEDICAL/HEALTH: cover diagnosis, findings, recommendations, medications, follow-up.
  - For FINANCIAL: cover revenue, expenses, profit/loss, forecasts, risks.
  - For TECHNICAL/MANUAL: cover product purpose, key features, requirements, warnings.
  - For RESEARCH/ACADEMIC: cover objective, methodology, key findings, conclusions.
  - For REPORTS/BUSINESS: cover context, analysis, recommendations, outcomes.
  - If document type is unclear, summarize the most important information a reader needs.

"highlights" — 4 to 6 items.
  - Each must be ONE complete, specific, factual sentence.
  - Always include real values from the document: numbers, dates, names,
    percentages, durations, prices, dosages, deadlines, versions, scores, or clauses.
  - Do NOT write vague statements like "The document contains important information."
  - Do NOT write opinions or analysis — only facts extracted directly from the document.
  - Prioritize: critical obligations, risks, key figures, deadlines, warnings, or outcomes.

STRICT OUTPUT RULES:
  - Output ONLY the JSON object.
  - Do NOT add any explanation, greeting, or commentary.
  - Do NOT use markdown, code blocks, or backticks.
  - Do NOT repeat these instructions in your output.
  - All string values must be properly escaped.
  - If a section has insufficient information, write "Not enough information available."

Document:
----------------
{text}
----------------
[/INST]
"""

    # Use get_running_loop() — get_event_loop() is deprecated in Python 3.10+ (Fix #12)
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, _run_inference, prompt)
    return result

async def run_llm(
    text: str,
    system_prompt: str,
    max_input_tokens: int = MAX_INPUT_TOKENS
) -> str:
    """
    Generic reusable LLM runner.
    Accepts:
    - text (document content)
    - system_prompt (custom instruction per use-case)

    Returns:
    - model output (string)
    """

    prompt = f"""<s>[INST]
{system_prompt}

Document:
----------------
{text[:4000]}
----------------
[/INST]
"""

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, _run_inference, prompt)

    return result

