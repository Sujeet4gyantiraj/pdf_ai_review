import asyncio
import time
import torch
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.schema import Document

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model config — tune MAX_CHUNKS / CHUNK_SIZE to your GPU VRAM:
#   24 GB → MAX_CHUNKS=20  |  16 GB → MAX_CHUNKS=12  |  8 GB → MAX_CHUNKS=6
# ---------------------------------------------------------------------------
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
MAX_CHUNKS = 12
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

torch.backends.cudnn.benchmark        = True
torch.backends.cuda.matmul.allow_tf32 = True

# ---------------------------------------------------------------------------
# Load model once at import time
# ---------------------------------------------------------------------------
logger.info(f"[ai_model] Loading tokenizer: {MODEL_NAME}")
t0         = time.perf_counter()
_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
logger.info(f"[ai_model] Tokenizer loaded ({time.perf_counter() - t0:.2f}s)")

# ---------------------------------------------------------------------------
# Safe model loading
#
# Root cause of "Tensor on device cuda:0 is not on the expected device meta":
#   device_map="auto" splits layers across GPU + meta/CPU when VRAM is tight.
#   Inputs land on cuda:0 but some weights stay on meta → RuntimeError.
#
# Fix strategy (in priority order):
#   1. If enough VRAM → load fully onto cuda:0, no offloading at all.
#   2. If VRAM is tight → load in 8-bit (bitsandbytes) so the whole model
#      stays on one real device. Requires: pip install bitsandbytes
#   3. CPU fallback → float32, slow but always works.
# ---------------------------------------------------------------------------

def _vram_gb() -> float:
    """Return total VRAM in GB for device 0, or 0.0 if no GPU."""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.get_device_properties(0).total_memory / 1024 ** 3


def _load_model():
    vram = _vram_gb()
    logger.info(f"[ai_model] Detected VRAM: {vram:.1f} GB")

    # Mistral-7B in float16 needs ~14 GB. Use 8-bit if tighter.
    if DEVICE == "cuda" and vram >= 14.0:
        logger.info("[ai_model] Loading in float16 fully on cuda:0 (no offloading)")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map={"": 0},        # force ALL layers onto cuda:0
        )

    elif DEVICE == "cuda" and vram >= 6.0:
        logger.info(
            f"[ai_model] VRAM {vram:.1f} GB < 14 GB — "
            "loading in 8-bit via bitsandbytes to keep all layers on one device"
        )
        try:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                quantization_config=bnb_config,
                device_map={"": 0},    # all layers on cuda:0
            )
        except ImportError:
            logger.warning(
                "[ai_model] bitsandbytes not installed — "
                "falling back to float16 with device_map={'':0}. "
                "Install with: pip install bitsandbytes"
            )
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.float16,
                device_map={"": 0},
            )

    else:
        logger.warning(
            f"[ai_model] No usable GPU (VRAM={vram:.1f} GB) — "
            "loading on CPU in float32. Inference will be slow."
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32,
            device_map="cpu",
        )

    # Confirm no layers ended up on meta device
    meta_params = [
        name for name, p in model.named_parameters()
        if p.device.type == "meta"
    ]
    if meta_params:
        logger.error(
            f"[ai_model] {len(meta_params)} parameter(s) still on meta device "
            f"after loading — first few: {meta_params[:3]}"
        )
    else:
        logger.info("[ai_model] All parameters on real devices (no meta offloading)")

    model.eval()
    return model


logger.info(f"[ai_model] Loading model (device='{DEVICE}') ...")
t0     = time.perf_counter()
_model = _load_model()
logger.info(f"[ai_model] Model ready ({time.perf_counter() - t0:.2f}s)")

if DEVICE == "cuda":
    allocated = torch.cuda.memory_allocated() / 1024 ** 3
    reserved  = torch.cuda.memory_reserved()  / 1024 ** 3
    logger.info(
        f"[ai_model] GPU memory after model load — "
        f"allocated: {allocated:.2f} GB, reserved: {reserved:.2f} GB"
    )


# ---------------------------------------------------------------------------
# Prompt builders — plain f-strings, no LangChain overhead
# ---------------------------------------------------------------------------

def _build_map_prompt(text: str) -> str:
    """Map prompt — applied to every individual chunk."""
    return f"""<s>[INST]
You are an expert document analyst with deep knowledge across medical,
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
    Examples of types: research paper, medical report,
    user manual, financial statement, privacy policy, invoice, academic thesis,
    insurance policy, government notice, product specification, meeting minutes.
  - State the subject, purpose, and intended audience or parties.

"summary" — 4-6 sentences in professional, neutral tone.
  - For MEDICAL/HEALTH: cover diagnosis, findings, recommendations, medications, follow-up.
  - For FINANCIAL: cover revenue, expenses, profit/loss, forecasts, risks.
  - For TECHNICAL/MANUAL: cover product purpose, key features, requirements, warnings.
  - For RESEARCH/ACADEMIC: cover objective, methodology, key findings, conclusions.
  - For REPORTS/BUSINESS: cover context, analysis, recommendations, outcomes.
  - If document type is unclear, summarize the most important information a reader needs.

"highlights" — 4 to 6 items.
  - Each must be ONE complete, specific, factual sentence.
  - Always include real values from the document: numbers, dates, names,
    percentages, durations, prices, dosages, deadlines, versions, or scores.
  - Do NOT write vague statements like "The document contains important information."
  - Do NOT write opinions or analysis — only facts extracted directly from the document.
  - Prioritize: critical risks, key figures, deadlines, warnings, or outcomes.

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


def _build_reduce_prompt(partial_results: list[dict]) -> str:
    """
    Reduce prompt — called once after all map results are collected.
    Formats the partial JSON dicts into a readable block for the model.
    """
    parts = []
    for i, r in enumerate(partial_results, 1):
        overview   = r.get("overview",   "")
        summary    = r.get("summary",    "")
        highlights = "\n".join(f"  - {h}" for h in r.get("highlights", []))
        parts.append(
            f"[Part {i}]\n"
            f"Overview: {overview}\n"
            f"Summary: {summary}\n"
            f"Highlights:\n{highlights}"
        )

    combined = "\n\n".join(parts)

    return f"""<s>[INST]
You are an expert document analyst. You have been given partial analyses of
different sections of the same large document. Synthesize them into one single,
coherent JSON briefing for the entire document.

Partial analyses:
----------------
{combined}
----------------

Produce a final unified JSON in exactly this format:
{{
  "overview": "1-2 sentence description of what the entire document is and its purpose.",
  "summary": "4-6 sentence coherent executive summary of the entire document — write as one flowing paragraph, not a list of parts.",
  "highlights": [
    "Most important specific fact from the entire document.",
    "Second most important specific fact.",
    "Third most important specific fact.",
    "Fourth most important specific fact.",
    "Fifth most important specific fact."
  ]
}}

STRICT OUTPUT RULES:
  - Output ONLY the JSON object.
  - Do NOT mention Part 1, Part 2, chunk numbers, or section references in the output.
  - Do NOT add any explanation, greeting, or commentary.
  - Do NOT use markdown, code blocks, or backticks.
  - All string values must be properly escaped.
[/INST]
"""


# ---------------------------------------------------------------------------
# Core inference — direct model.generate(), no wrappers
# ---------------------------------------------------------------------------

def _run_inference(prompt: str, max_new_tokens: int = 600, label: str = "") -> str:
    """
    Synchronous direct inference using model.generate().
    Always called via run_in_executor — never on the async event loop directly.

    label — optional tag shown in logs to identify map vs reduce calls.

    Device is derived from the first model parameter so it always matches
    where the model was actually loaded — avoids the meta-device mismatch
    that occurs when device_map="auto" splits layers across devices.
    """
    tag = f"[{label}] " if label else ""

    # Resolve the real device from the model itself (not the global DEVICE string)
    model_device = next(_model.parameters()).device
    logger.debug(f"{tag}_run_inference: model device = {model_device}")

    # Measure token count before truncation
    encoded     = _tokenizer(prompt, return_tensors="pt", truncation=False)
    token_count = encoded["input_ids"].shape[1]
    logger.debug(f"{tag}_run_inference: prompt is {token_count} tokens")

    if token_count > 6000:
        logger.warning(
            f"{tag}_run_inference: {token_count} tokens exceeds 6000 — truncating. "
            "Consider lowering CHUNK_SIZE in pdf_utils.py."
        )

    inputs = _tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=6000,
    ).to(model_device)

    if model_device.type == "cuda":
        free_gb = (
            torch.cuda.get_device_properties(model_device).total_memory
            - torch.cuda.memory_allocated(model_device)
        ) / 1024 ** 3
        logger.debug(f"{tag}_run_inference: GPU free before generate: {free_gb:.2f} GB")

    t_gen = time.perf_counter()
    try:
        with torch.inference_mode():
            output = _model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.15,
                pad_token_id=_tokenizer.eos_token_id,
            )
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        logger.error(
            f"{tag}_run_inference: CUDA OutOfMemoryError — "
            "lower MAX_CHUNKS or CHUNK_SIZE and retry"
        )
        raise RuntimeError("GPU out of memory — lower MAX_CHUNKS or CHUNK_SIZE and retry.")
    except Exception as e:
        logger.exception(f"{tag}_run_inference: unexpected error during model.generate(): {e}")
        raise
    finally:
        del inputs
        if model_device.type == "cuda":
            torch.cuda.empty_cache()

    elapsed     = time.perf_counter() - t_gen
    new_tokens  = output.shape[1] - encoded["input_ids"].shape[1]
    tok_per_sec = new_tokens / elapsed if elapsed > 0 else 0
    logger.info(
        f"{tag}_run_inference: generated {new_tokens} tokens "
        f"in {elapsed:.2f}s ({tok_per_sec:.1f} tok/s)"
    )

    # Decode only the newly generated tokens
    return _tokenizer.decode(
        output[0][encoded["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )


# ---------------------------------------------------------------------------
# Map-reduce pipeline
# ---------------------------------------------------------------------------

async def generate_analysis(chunks: list[Document]) -> dict:
    """
    Manual map-reduce pipeline using direct model.generate():

    MAP  — run _run_inference on each chunk independently in a thread pool.
           Each result is immediately parsed into a dict by extract_json so
           the reduce step receives structured data, not raw strings.

    REDUCE — if there are multiple chunks, build a single reduce prompt from
             all map dicts and run one final inference call to produce a
             coherent unified output.

    Single-chunk PDFs skip the reduce step entirely (zero extra inference cost).

    Returns a parsed dict {overview, summary, highlights}.
    """
    from s_main import extract_json  # avoid circular import

    logger.info(f"[generate_analysis] Called with {len(chunks)} chunk(s)")

    if not chunks:
        logger.warning("[generate_analysis] No chunks received — returning empty result")
        return {"overview": "", "summary": "", "highlights": []}

    # Enforce chunk limit to protect GPU memory
    if len(chunks) > MAX_CHUNKS:
        logger.warning(
            f"[generate_analysis] {len(chunks)} chunks exceeds MAX_CHUNKS={MAX_CHUNKS} — "
            f"truncating. Raise MAX_CHUNKS or lower CHUNK_SIZE to cover more content."
        )
        chunks = chunks[:MAX_CHUNKS]

    loop        = asyncio.get_event_loop()
    t_pipeline  = time.perf_counter()

    # ------------------------------------------------------------------
    # MAP phase
    # ------------------------------------------------------------------
    logger.info(f"[generate_analysis] ── MAP phase: {len(chunks)} chunk(s) ──")
    map_results = []

    for i, chunk in enumerate(chunks):
        chunk_label = f"map {i + 1}/{len(chunks)}"
        page_ref    = chunk.metadata.get("page", "?")
        logger.info(
            f"[generate_analysis] [{chunk_label}] "
            f"page={page_ref}, chars={len(chunk.page_content)}"
        )

        t_chunk = time.perf_counter()
        try:
            prompt = _build_map_prompt(chunk.page_content)
            raw    = await loop.run_in_executor(
                None,
                lambda p=prompt, lbl=chunk_label: _run_inference(p, label=lbl)
            )
        except Exception as e:
            logger.error(
                f"[generate_analysis] [{chunk_label}] inference failed: {e} — "
                "inserting empty result and continuing"
            )
            map_results.append({"overview": "", "summary": "", "highlights": []})
            continue

        parsed = extract_json(raw)
        map_results.append(parsed)
        logger.info(
            f"[generate_analysis] [{chunk_label}] done "
            f"({time.perf_counter() - t_chunk:.2f}s) — "
            f"overview={'yes' if parsed.get('overview') else 'empty'}, "
            f"highlights={len(parsed.get('highlights', []))}"
        )

    logger.info(
        f"[generate_analysis] MAP phase complete — "
        f"{len(map_results)} result(s) collected"
    )

    # ------------------------------------------------------------------
    # REDUCE phase
    # ------------------------------------------------------------------
    if len(map_results) == 1:
        logger.info("[generate_analysis] Single chunk — skipping reduce phase")
        result = map_results[0]
    else:
        logger.info(
            f"[generate_analysis] ── REDUCE phase: synthesizing {len(map_results)} results ──"
        )
        t_reduce      = time.perf_counter()
        reduce_prompt = _build_reduce_prompt(map_results)

        try:
            raw_reduced = await loop.run_in_executor(
                None,
                lambda: _run_inference(reduce_prompt, max_new_tokens=800, label="reduce")
            )
        except Exception as e:
            logger.error(
                f"[generate_analysis] Reduce inference failed: {e} — "
                "falling back to first map result"
            )
            result = map_results[0]
        else:
            result = extract_json(raw_reduced)
            logger.info(
                f"[generate_analysis] REDUCE phase complete "
                f"({time.perf_counter() - t_reduce:.2f}s)"
            )

    total = time.perf_counter() - t_pipeline
    logger.info(
        f"[generate_analysis] Pipeline complete — "
        f"total: {total:.2f}s, "
        f"highlights: {len(result.get('highlights', []))}"
    )
    return result