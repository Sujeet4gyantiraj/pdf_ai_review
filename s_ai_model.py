import asyncio
import os
import time
import torch
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.schema import Document

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model config
# Tune MAX_CHUNKS to how many inference calls you want per request.
# Tune CHUNK_SIZE in pdf_utils.py to control tokens per call.
#   39 GB total / ~8 GB used by PaddleOCR → ~31 GB free
#   Mistral-7B float16 = ~14 GB → ~17 GB left for KV cache and activations
#   Safe to run MAX_CHUNKS=30 with CHUNK_SIZE=12000
# ---------------------------------------------------------------------------
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

# ---------------------------------------------------------------------------
# Sizing strategy for your server (39 GB GPU, ~8 GB used by PaddleOCR):
#
# Finance Bill 2026: 232 pages, ~440K chars total
#
# CHUNK_SIZE=40000  → 440K ÷ 40000 = ~11 chunks  (full doc coverage)
# MAX_CHUNKS=12     → covers all 11 chunks + buffer
# MAX_INPUT_TOKENS  → 40000 chars ÷ 3.5 chars/token ≈ 11400 tokens
#                     prompt template ≈ 400 tokens → need ~11800 total
#                     Mistral context = 32768 tokens (v0.2) → fits easily
#
# At 7.7 tok/s with 8-bit:
#   Map:    11 chunks × ~500 tokens output × (1/7.7) ≈ 11 × 65s = ~12 min
#   Reduce: 2 batches + 1 final × ~60s = ~3 min
#   Total:  ~15 min for the full 232-page document
# ---------------------------------------------------------------------------
CHUNK_SIZE    = 40000   # large chunks → few calls → full doc coverage
MAX_CHUNKS    = 12      # 11 chunks for 232 pages + 1 buffer
MAX_INPUT_TOKENS      = 14000  # 40000 chars ÷ 3.5 + 400 prompt tokens
MAX_NEW_TOKENS_MAP    = 600    # enough for clean JSON with all fields
MAX_NEW_TOKENS_REDUCE = 700    # slightly more room for synthesis

# Reduce batch size — feed N map results per reduce call
# 12 results ÷ 6 per batch = 2 batches → 1 final = 3 reduce calls total
REDUCE_BATCH_SIZE = 6

# Speed: set before any CUDA allocation
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

torch.backends.cudnn.benchmark        = True
torch.backends.cuda.matmul.allow_tf32 = True

# ---------------------------------------------------------------------------
# Load model once at import time
# ---------------------------------------------------------------------------
logger.info(f"[ai_model] Loading tokenizer: {MODEL_NAME}")
t0         = time.perf_counter()
_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

# Left-padding is required for batched generation with decoder-only models.
# For single-sequence inference it has no effect but sets us up for future batching.
_tokenizer.padding_side = "left"
if _tokenizer.pad_token is None:
    _tokenizer.pad_token = _tokenizer.eos_token

logger.info(f"[ai_model] Tokenizer loaded ({time.perf_counter() - t0:.2f}s)")


def _vram_free_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    props     = torch.cuda.get_device_properties(0)
    allocated = torch.cuda.memory_allocated(0)
    reserved  = torch.cuda.memory_reserved(0)
    return (props.total_memory - reserved - allocated) / 1024 ** 3


def _vram_total_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.get_device_properties(0).total_memory / 1024 ** 3


def _load_model():
    total_vram = _vram_total_gb()
    free_vram  = _vram_free_gb()
    logger.info(
        f"[ai_model] VRAM — total: {total_vram:.1f} GB, "
        f"free: {free_vram:.1f} GB "
        f"(other processes: {total_vram - free_vram:.1f} GB)"
    )

    # ── Attention implementation ──────────────────────────────────────────
    attn_impl = "eager"
    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
        logger.info("[ai_model] Flash Attention 2 available — using FA2 kernel")
    except ImportError:
        logger.info(
            "[ai_model] flash-attn not installed — using standard attention. "
            "Install for 2-4x inference speedup: pip install flash-attn --no-build-isolation"
        )

    common_kwargs = dict(attn_implementation=attn_impl)

    # ── Loading strategy ──────────────────────────────────────────────────
    # Why we use total VRAM (not free VRAM) to decide the loading strategy:
    #
    # PaddleOCR-VL is loaded at import time and holds ~22 GB of the 39 GB GPU.
    # That makes free_vram appear to be only ~8 GB, which would trigger 8-bit
    # loading — but 8-bit is 2-3x SLOWER than float16 due to dequantization
    # overhead on every matrix multiply.
    #
    # Key insight: PaddleOCR and the LLM never run at the same time.
    # PaddleOCR finishes all OCR pages during extraction (Step 3), then the
    # LLM runs during inference (Step 5). They share the GPU sequentially,
    # not concurrently.
    #
    # Therefore: use total VRAM to decide dtype.
    # If total >= 39 GB (your server) → float16 is safe even with PaddleOCR loaded
    # because inference never overlaps with OCR.
    # The ~22 GB PaddleOCR uses is irrelevant during LLM forward passes.
    #
    # float16 speed vs 8-bit on your GPU:
    #   8-bit  : ~7-8 tok/s   (dequant overhead on every matmul)
    #   float16: ~18-25 tok/s (native tensor cores, no dequant)
    # That alone cuts inference time by 2-3x.

    if DEVICE == "cuda" and total_vram >= 35.0:
        # Large GPU (A100 40GB, A100 80GB, H100, etc.)
        # Load float16 unconditionally — fast, full precision
        logger.info(
            f"[ai_model] Large GPU detected ({total_vram:.1f} GB total) — "
            "loading float16 on cuda:0 for maximum speed"
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map={"": 0},
            **common_kwargs,
        )

    elif DEVICE == "cuda" and free_vram >= 14.0:
        # Enough free VRAM right now for float16
        logger.info(f"[ai_model] {free_vram:.1f} GB free ≥ 14 GB — loading float16 on cuda:0")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map={"": 0},
            **common_kwargs,
        )

    elif DEVICE == "cuda" and free_vram >= 7.0:
        logger.info(f"[ai_model] {free_vram:.1f} GB free — loading 8-bit via bitsandbytes")
        try:
            from transformers import BitsAndBytesConfig
            bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                quantization_config=bnb_cfg,
                device_map={"": 0},
                **common_kwargs,
            )
        except ImportError:
            logger.warning("[ai_model] bitsandbytes not installed — trying float16")
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.float16,
                device_map={"": 0},
                **common_kwargs,
            )

    elif DEVICE == "cuda" and free_vram >= 4.0:
        logger.info(f"[ai_model] {free_vram:.1f} GB free — loading 4-bit via bitsandbytes")
        try:
            from transformers import BitsAndBytesConfig
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                quantization_config=bnb_cfg,
                device_map={"": 0},
                **common_kwargs,
            )
        except ImportError:
            logger.error("[ai_model] bitsandbytes not installed — cannot load 4-bit")
            raise RuntimeError(
                f"Only {free_vram:.1f} GB free — install: pip install bitsandbytes"
            )

    else:
        logger.warning(f"[ai_model] Only {free_vram:.1f} GB free — falling back to CPU float32")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32,
            device_map="cpu",
        )

   

    meta_params = [n for n, p in model.named_parameters() if p.device.type == "meta"]
    if meta_params:
        logger.error(f"[ai_model] {len(meta_params)} param(s) on meta — first: {meta_params[:3]}")
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
        f"[ai_model] GPU memory after load — "
        f"allocated: {allocated:.2f} GB, reserved: {reserved:.2f} GB"
    )


# ---------------------------------------------------------------------------
# Prompt builders
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
    Examples: research paper, medical report, user manual, financial statement,
    privacy policy, invoice, academic thesis, insurance policy, government notice,
    product specification, meeting minutes.
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
  - Always include real values: numbers, dates, names, percentages, durations,
    prices, dosages, deadlines, versions, or scores.
  - Do NOT write vague statements.
  - Do NOT write opinions — only facts extracted directly from the document.
  - Prioritize: critical risks, key figures, deadlines, warnings, or outcomes.

STRICT OUTPUT RULES:
  - Output ONLY the JSON object. No explanation. No markdown. No backticks.
  - All strings must be properly escaped.
  - Keep each field concise — overview max 2 sentences, summary max 4 sentences,
    highlights exactly 4 items of max 20 words each.
  - If insufficient information, write "Not enough information available."

Document:
----------------
{text}
----------------
[/INST]
"""


def _build_reduce_prompt(partial_results: list[dict]) -> str:
    """
    Compact reduce prompt — uses only overview + highlights from each map result,
    NOT the full summary. This keeps the prompt small enough to fit in context
    even with many map results.

    Token budget estimate:
      Header + instructions ≈ 200 tokens
      Per result: overview(~30) + 4 highlights(~80) ≈ 110 tokens
      5 results × 110 = 550 tokens → well within limit
    """
    parts = []
    for i, r in enumerate(partial_results, 1):
        overview   = r.get("overview", "").strip()
        highlights = r.get("highlights", [])
        # Limit to 4 highlights per chunk to keep prompt compact
        hl_text = "\n".join(f"  - {h}" for h in highlights[:4])
        if overview or hl_text:
            parts.append(
                f"[Section {i}]\n"
                f"Overview: {overview}\n"
                f"Key facts:\n{hl_text}"
            )

    combined = "\n\n".join(parts)

    return f"""<s>[INST]
You are an expert document analyst. Synthesize these section analyses into one
single coherent JSON briefing for the entire document.

Section analyses:
----------------
{combined}
----------------

Produce a final unified JSON:
{{
  "overview": "1-2 sentence description of the entire document and its purpose.",
  "summary": "4-6 sentence executive summary as one flowing paragraph.",
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
  - Do NOT mention Section 1, Section 2, or part numbers in the output.
  - Do NOT add explanation, greeting, or commentary.
  - Do NOT use markdown, code blocks, or backticks.
  - All string values must be properly escaped.
[/INST]
"""


def _batched_reduce(results: list[dict], extract_json_fn) -> dict:
    """
    Two-level reduce for large numbers of map results.

    The problem: 30 map results × ~110 tokens each = ~3300 tokens of content
    plus the prompt template itself pushes past the model's context window.

    The fix: process results in batches of REDUCE_BATCH_SIZE, producing one
    intermediate result per batch, then do a final reduce over those.

    Example with 30 results and REDUCE_BATCH_SIZE=5:
      Level 1: 6 batches × 5 results → 6 intermediate results  (6 inference calls)
      Level 2: 1 final reduce of 6 intermediates               (1 inference call)
      Total: 7 reduce calls instead of 1 overflowing call

    This keeps every prompt well within the token limit.
    """
    if len(results) <= REDUCE_BATCH_SIZE:
        # Small enough — single reduce
        prompt = _build_reduce_prompt(results)
        raw    = _run_inference(prompt, MAX_NEW_TOKENS_REDUCE, label="reduce")
        return extract_json_fn(raw)

    # Level 1 — batch reduce
    logger.info(
        f"[_batched_reduce] {len(results)} results → "
        f"batching into groups of {REDUCE_BATCH_SIZE}"
    )
    intermediates = []
    for i in range(0, len(results), REDUCE_BATCH_SIZE):
        batch     = results[i: i + REDUCE_BATCH_SIZE]
        batch_lbl = f"reduce-batch {i // REDUCE_BATCH_SIZE + 1}"
        logger.info(f"[_batched_reduce] {batch_lbl}: {len(batch)} results")
        prompt = _build_reduce_prompt(batch)
        raw    = _run_inference(prompt, MAX_NEW_TOKENS_REDUCE, label=batch_lbl)
        intermediates.append(extract_json_fn(raw))

    # Level 2 — final reduce over intermediates
    logger.info(f"[_batched_reduce] Final reduce over {len(intermediates)} intermediate(s)")
    prompt = _build_reduce_prompt(intermediates)
    raw    = _run_inference(prompt, MAX_NEW_TOKENS_REDUCE, label="reduce-final")
    return extract_json_fn(raw)


# ---------------------------------------------------------------------------
# Core inference — direct model.generate()
# ---------------------------------------------------------------------------

def _run_inference(prompt: str, max_new_tokens: int = MAX_NEW_TOKENS_MAP, label: str = "") -> str:
    """
    Synchronous direct inference — always called via run_in_executor.

    Speed optimisations applied here:
    - torch.inference_mode()  : disables gradient tracking (~10% faster, less memory)
    - use_cache=True          : KV-cache reuse across decoding steps (default but explicit)
    - model device resolved once from parameters (avoids repeated meta-device bug)
    - empty_cache() only when needed (not on every call unconditionally)
    """
    tag          = f"[{label}] " if label else ""
    model_device = next(_model.parameters()).device
    logger.debug(f"{tag}_run_inference: device={model_device}")

    # ── Speed optimisation 2: count tokens once, reuse encoding ──────────
    # Tokenize once without truncation to measure length, then again with
    # truncation only if needed — avoids double tokenization in the common case.
    encoded     = _tokenizer(prompt, return_tensors="pt", truncation=False)
    token_count = encoded["input_ids"].shape[1]
    logger.info(f"{tag}_run_inference: {token_count} input tokens, max_new={max_new_tokens}")

    if token_count > MAX_INPUT_TOKENS:
        logger.warning(
            f"{tag}_run_inference: {token_count} tokens > {MAX_INPUT_TOKENS} — truncating. "
            "Lower CHUNK_SIZE in pdf_utils.py to avoid this."
        )
        inputs = _tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_INPUT_TOKENS,
        ).to(model_device)
    else:
        # Reuse the already-encoded tensor — move to device directly
        inputs = {k: v.to(model_device) for k, v in encoded.items()}

    if model_device.type == "cuda":
        free_gb = (
            torch.cuda.get_device_properties(model_device).total_memory
            - torch.cuda.memory_allocated(model_device)
        ) / 1024 ** 3
        logger.debug(f"{tag}_run_inference: GPU free before generate: {free_gb:.2f} GB")

    t_gen = time.perf_counter()
    try:
        # ── Speed optimisation 3: inference_mode + use_cache ─────────────
        with torch.inference_mode():
            output = _model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.15,
                use_cache=True,             # explicit KV-cache reuse
                pad_token_id=_tokenizer.eos_token_id,
            )
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        logger.error(f"{tag}_run_inference: CUDA OOM — lower MAX_CHUNKS or CHUNK_SIZE")
        raise RuntimeError("GPU out of memory — lower MAX_CHUNKS or CHUNK_SIZE and retry.")
    except Exception as e:
        logger.exception(f"{tag}_run_inference: model.generate() failed: {e}")
        raise
    finally:
        del inputs
        # ── Speed optimisation 4: skip empty_cache unless actually low ───
        # empty_cache() is slow (~50ms). Only call it when free VRAM < 2 GB.
        if model_device.type == "cuda":
            free_after = (
                torch.cuda.get_device_properties(model_device).total_memory
                - torch.cuda.memory_allocated(model_device)
            ) / 1024 ** 3
            if free_after < 2.0:
                torch.cuda.empty_cache()
                logger.debug(f"{tag}_run_inference: cache cleared (low VRAM: {free_after:.2f} GB)")

    elapsed    = time.perf_counter() - t_gen
    new_tokens = output.shape[1] - token_count
    logger.info(
        f"{tag}_run_inference: {new_tokens} tokens in {elapsed:.2f}s "
        f"({new_tokens / elapsed:.1f} tok/s)"
    )

    return _tokenizer.decode(output[0][token_count:], skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Map-reduce pipeline
# ---------------------------------------------------------------------------

async def generate_analysis(chunks: list[Document]) -> dict:
    """
    Map-reduce pipeline with all speed optimisations:

    MAP  — each chunk is sent to _run_inference via run_in_executor so the
           async event loop stays free. The GPU runs one forward pass at a
           time (serialised by the GIL + CUDA stream), so there's no benefit
           to firing multiple executor tasks concurrently — they'd queue anyway.

    REDUCE — one final inference call synthesizes all map outputs.
             Skipped entirely for single-chunk PDFs.
    """
    from s_main import extract_json  # avoid circular import

    logger.info(f"[generate_analysis] Called with {len(chunks)} chunk(s)")

    if not chunks:
        logger.warning("[generate_analysis] No chunks — returning empty result")
        return {"overview": "", "summary": "", "highlights": []}

    if len(chunks) > MAX_CHUNKS:
        logger.warning(
            f"[generate_analysis] {len(chunks)} chunks exceeds expected MAX_CHUNKS={MAX_CHUNKS}. "
            f"Processing ALL {len(chunks)} chunks — this may take longer than usual. "
            f"Consider raising MAX_CHUNKS or CHUNK_SIZE if this happens regularly."
        )
        # Do NOT truncate — process every chunk so no content is silently dropped

    loop       = asyncio.get_event_loop()
    t_pipeline = time.perf_counter()

    # ── MAP phase ─────────────────────────────────────────────────────────
    logger.info(f"[generate_analysis] ── MAP phase: {len(chunks)} chunk(s) ──")
    map_results = []

    for i, chunk in enumerate(chunks):
        lbl      = f"map {i+1}/{len(chunks)}"
        page_ref = chunk.metadata.get("page", "?")
        logger.info(f"[generate_analysis] [{lbl}] page={page_ref}, chars={len(chunk.page_content)}")

        t_chunk = time.perf_counter()
        try:
            prompt = _build_map_prompt(chunk.page_content)
            raw    = await loop.run_in_executor(
                None,
                lambda p=prompt, l=lbl: _run_inference(p, MAX_NEW_TOKENS_MAP, l)
            )
        except Exception as e:
            logger.error(f"[generate_analysis] [{lbl}] failed: {e} — inserting empty result")
            map_results.append({"overview": "", "summary": "", "highlights": []})
            continue

        parsed = extract_json(raw)
        map_results.append(parsed)
        logger.info(
            f"[generate_analysis] [{lbl}] done ({time.perf_counter() - t_chunk:.2f}s) — "
            f"overview={'yes' if parsed.get('overview') else 'empty'}, "
            f"highlights={len(parsed.get('highlights', []))}"
        )

    logger.info(f"[generate_analysis] MAP complete — {len(map_results)} result(s)")

    # ── REDUCE phase ──────────────────────────────────────────────────────
    if len(map_results) == 1:
        logger.info("[generate_analysis] Single chunk — skipping reduce")
        result = map_results[0]
    else:
        logger.info(
            f"[generate_analysis] ── REDUCE phase: "
            f"{len(map_results)} results, batch_size={REDUCE_BATCH_SIZE} ──"
        )
        t_reduce = time.perf_counter()
        try:
            # Run batched reduce in thread pool (blocking GPU calls)
            result = await loop.run_in_executor(
                None,
                lambda: _batched_reduce(map_results, extract_json)
            )
            logger.info(
                f"[generate_analysis] REDUCE done "
                f"({time.perf_counter() - t_reduce:.2f}s)"
            )
        except Exception as e:
            logger.error(
                f"[generate_analysis] Reduce failed: {e} — "
                "merging map results directly"
            )
            # Fallback: collect all highlights and use first non-empty overview/summary
            result = {
                "overview":   next((r["overview"]  for r in map_results if r.get("overview")),  ""),
                "summary":    next((r["summary"]   for r in map_results if r.get("summary")),   ""),
                "highlights": list({
                    h for r in map_results for h in r.get("highlights", []) if h
                })[:6],
            }

    logger.info(
        f"[generate_analysis] Pipeline complete — "
        f"total: {time.perf_counter() - t_pipeline:.2f}s, "
        f"highlights: {len(result.get('highlights', []))}"
    )
    return result

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

def _vram_free_gb() -> float:
    """
    Return FREE (available) VRAM in GB for device 0.
    Using free VRAM — not total — is critical when other processes
    (e.g. PaddleOCR-VL) already hold GPU memory at startup.
    """
    if not torch.cuda.is_available():
        return 0.0
    props     = torch.cuda.get_device_properties(0)
    allocated = torch.cuda.memory_allocated(0)
    reserved  = torch.cuda.memory_reserved(0)
    # Free = total - already reserved by PyTorch + any other process usage
    free = props.total_memory - reserved - allocated
    return free / 1024 ** 3


def _vram_total_gb() -> float:
    """Return total VRAM in GB for device 0."""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.get_device_properties(0).total_memory / 1024 ** 3


def _load_model():
    total_vram = _vram_total_gb()
    free_vram  = _vram_free_gb()
    logger.info(
        f"[ai_model] VRAM — total: {total_vram:.1f} GB, "
        f"free: {free_vram:.1f} GB "
        f"(in use by other processes: {total_vram - free_vram:.1f} GB)"
    )

    # Set expandable segments to reduce fragmentation before any allocation
    import os
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # Decision is based on FREE VRAM, not total.
    # Mistral-7B float16  ≈ 14 GB
    # Mistral-7B 8-bit    ≈  7 GB
    # Mistral-7B 4-bit    ≈  4 GB

    if DEVICE == "cuda" and free_vram >= 14.0:
        logger.info(
            f"[ai_model] {free_vram:.1f} GB free ≥ 14 GB — "
            "loading float16 fully on cuda:0"
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map={"": 0},
        )

    elif DEVICE == "cuda" and free_vram >= 7.0:
        logger.info(
            f"[ai_model] {free_vram:.1f} GB free — "
            "loading in 8-bit via bitsandbytes (≈7 GB needed)"
        )
        try:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                quantization_config=bnb_config,
                device_map={"": 0},
            )
        except ImportError:
            logger.warning(
                "[ai_model] bitsandbytes not installed — "
                "trying float16 anyway. Install: pip install bitsandbytes"
            )
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.float16,
                device_map={"": 0},
            )

    elif DEVICE == "cuda" and free_vram >= 4.0:
        logger.info(
            f"[ai_model] {free_vram:.1f} GB free — "
            "loading in 4-bit via bitsandbytes (≈4 GB needed)"
        )
        try:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                quantization_config=bnb_config,
                device_map={"": 0},
            )
        except ImportError:
            logger.error(
                "[ai_model] bitsandbytes not installed — cannot load in 4-bit. "
                "Install: pip install bitsandbytes  then restart."
            )
            raise RuntimeError(
                f"Only {free_vram:.1f} GB VRAM free — install bitsandbytes "
                "for 4-bit loading: pip install bitsandbytes"
            )

    else:
        logger.warning(
            f"[ai_model] Only {free_vram:.1f} GB VRAM free — "
            "falling back to CPU float32. Inference will be very slow."
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