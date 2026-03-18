import asyncio
import os
import time
import torch
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.schema import Document

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

# Your server: 39 GB GPU, ~22 GB used by PaddleOCR at startup
# LLM inference runs after OCR finishes — they never overlap
# Float16 Mistral-7B needs ~14 GB → safe on 39 GB total
CHUNK_SIZE            = 40000
MAX_CHUNKS            = 12
MAX_INPUT_TOKENS      = 14000
MAX_NEW_TOKENS_MAP    = 400   # JSON output is ~200-350 tokens — 400 is safe ceiling
MAX_NEW_TOKENS_REDUCE = 500
REDUCE_BATCH_SIZE     = 6

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

torch.backends.cudnn.benchmark        = True
torch.backends.cuda.matmul.allow_tf32 = True

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------
logger.info(f"[ai_model] Loading tokenizer: {MODEL_NAME}")
t0         = time.perf_counter()
_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
_tokenizer.padding_side = "left"
if _tokenizer.pad_token is None:
    _tokenizer.pad_token = _tokenizer.eos_token
logger.info(f"[ai_model] Tokenizer loaded ({time.perf_counter() - t0:.2f}s)")


# ---------------------------------------------------------------------------
# VRAM helpers
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def _load_model():
    total_vram = _vram_total_gb()
    free_vram  = _vram_free_gb()
    logger.info(
        f"[ai_model] VRAM total={total_vram:.1f} GB, "
        f"free={free_vram:.1f} GB, "
        f"other={total_vram - free_vram:.1f} GB"
    )

    # Flash Attention 2 — 2-4x faster, lower memory for long sequences
    attn_impl = "eager"
    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
        logger.info("[ai_model] Flash Attention 2 — enabled")
    except ImportError:
        logger.info(
            "[ai_model] Flash Attention 2 not installed — "
            "install for 2-4x speedup: pip install flash-attn --no-build-isolation"
        )

    common_kwargs = dict(attn_implementation=attn_impl)

    # Use total VRAM for dtype decision — PaddleOCR and LLM never run concurrently
    if DEVICE == "cuda" and total_vram >= 35.0:
        logger.info(
            f"[ai_model] Large GPU ({total_vram:.1f} GB total) → "
            "float16 on cuda:0 (fastest)"
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map={"": 0},
            **common_kwargs,
        )

    elif DEVICE == "cuda" and free_vram >= 14.0:
        logger.info(f"[ai_model] {free_vram:.1f} GB free → float16 on cuda:0")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map={"": 0},
            **common_kwargs,
        )

    elif DEVICE == "cuda" and free_vram >= 7.0:
        logger.info(f"[ai_model] {free_vram:.1f} GB free → 8-bit quantization")
        try:
            from transformers import BitsAndBytesConfig
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                quantization_config=BitsAndBytesConfig(load_in_8bit=True),
                device_map={"": 0},
                **common_kwargs,
            )
        except ImportError:
            logger.warning("[ai_model] bitsandbytes not installed → float16 fallback")
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.float16,
                device_map={"": 0},
                **common_kwargs,
            )

    elif DEVICE == "cuda" and free_vram >= 4.0:
        logger.info(f"[ai_model] {free_vram:.1f} GB free → 4-bit quantization")
        try:
            from transformers import BitsAndBytesConfig
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                ),
                device_map={"": 0},
                **common_kwargs,
            )
        except ImportError:
            logger.error("[ai_model] bitsandbytes not installed — cannot load 4-bit")
            raise RuntimeError("Install bitsandbytes: pip install bitsandbytes")

    else:
        logger.warning(f"[ai_model] {free_vram:.1f} GB free → CPU float32 (slow)")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32,
            device_map="cpu",
        )

    # Verify no meta device
    meta_params = [n for n, p in model.named_parameters() if p.device.type == "meta"]
    if meta_params:
        logger.error(f"[ai_model] {len(meta_params)} param(s) on meta: {meta_params[:3]}")
    else:
        logger.info("[ai_model] All parameters on real devices")

    model.eval()

    # torch.compile — fuses GPU kernels, 10-30% faster on repeated calls
    # Requires PyTorch 2.0+ and takes ~30s to compile on first call (cached after)
    try:
        model = torch.compile(model, mode="reduce-overhead")
        logger.info("[ai_model] torch.compile enabled (reduce-overhead mode)")
    except Exception as e:
        logger.info(f"[ai_model] torch.compile not available ({e}) — skipping")

    return model


logger.info(f"[ai_model] Loading model (device='{DEVICE}') ...")
t0     = time.perf_counter()
_model = _load_model()
logger.info(f"[ai_model] Model ready ({time.perf_counter() - t0:.2f}s)")

if DEVICE == "cuda":
    allocated = torch.cuda.memory_allocated() / 1024 ** 3
    reserved  = torch.cuda.memory_reserved()  / 1024 ** 3
    logger.info(
        f"[ai_model] GPU after load — "
        f"allocated={allocated:.2f} GB, reserved={reserved:.2f} GB"
    )

# Pre-compute EOS token id list for stopping criteria
_EOS_TOKEN_IDS = [_tokenizer.eos_token_id]


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

# Pre-built static prompt prefix and suffix — only the document text changes
# between calls. Computing the static parts once at module level avoids
# rebuilding the same ~400-token string on every inference call.
_MAP_PROMPT_PREFIX = """\
<s>[INST]
You are an expert document analyst across medical, financial, technical, \
academic, and business domains.

Produce a structured JSON briefing from the document below.

OUTPUT FORMAT (follow exactly):
{
  "overview": "1-2 sentence document type and purpose.",
  "summary": "3-4 sentence executive summary.",
  "highlights": [
    "Specific fact with real value (number/date/name/%).",
    "Specific fact with real value.",
    "Specific fact with real value.",
    "Specific fact with real value."
  ]
}

RULES:
- Output ONLY the JSON. No markdown, no backticks, no commentary.
- overview: max 2 sentences.
- summary: max 4 sentences.
- highlights: exactly 4 items, max 20 words each, facts only.
- Escape all strings properly.

Document:
----------------
"""
_MAP_PROMPT_SUFFIX = "\n----------------\n[/INST]\n"


def _build_map_prompt(text: str) -> str:
    """Fast map prompt — prefix and suffix are pre-built constants."""
    return _MAP_PROMPT_PREFIX + text + _MAP_PROMPT_SUFFIX


_REDUCE_PROMPT_PREFIX = """\
<s>[INST]
You are an expert document analyst. Synthesize these section analyses into \
one final JSON briefing for the entire document.

OUTPUT FORMAT (follow exactly):
{
  "overview": "1-2 sentence description of the entire document.",
  "summary": "4-5 sentence coherent executive summary as one paragraph.",
  "highlights": [
    "Most important fact from the entire document.",
    "Second most important fact.",
    "Third most important fact.",
    "Fourth most important fact.",
    "Fifth most important fact."
  ]
}

RULES:
- Output ONLY the JSON. No markdown. No section numbers in the output.
- Escape all strings properly.

Section analyses:
----------------
"""
_REDUCE_PROMPT_SUFFIX = "\n----------------\n[/INST]\n"


def _build_reduce_prompt(partial_results: list[dict]) -> str:
    """Compact reduce prompt using only overview + highlights per result."""
    parts = []
    for i, r in enumerate(partial_results, 1):
        overview   = r.get("overview", "").strip()
        highlights = r.get("highlights", [])
        hl_text    = "\n".join(f"  - {h}" for h in highlights[:4])
        if overview or hl_text:
            parts.append(
                f"[Section {i}]\n"
                f"Overview: {overview}\n"
                f"Key facts:\n{hl_text}"
            )
    return _REDUCE_PROMPT_PREFIX + "\n\n".join(parts) + _REDUCE_PROMPT_SUFFIX


# ---------------------------------------------------------------------------
# Batched reduce
# ---------------------------------------------------------------------------
def _batched_reduce(results: list[dict], extract_json_fn) -> dict:
    """
    Two-level reduce: batch map results → intermediate results → final result.
    Keeps every reduce prompt well within the token limit.
    """
    if len(results) <= REDUCE_BATCH_SIZE:
        raw = _run_inference(
            _build_reduce_prompt(results), MAX_NEW_TOKENS_REDUCE, label="reduce"
        )
        return extract_json_fn(raw)

    logger.info(
        f"[_batched_reduce] {len(results)} results → "
        f"batching into groups of {REDUCE_BATCH_SIZE}"
    )
    intermediates = []
    for i in range(0, len(results), REDUCE_BATCH_SIZE):
        batch = results[i: i + REDUCE_BATCH_SIZE]
        lbl   = f"reduce-batch-{i // REDUCE_BATCH_SIZE + 1}"
        logger.info(f"[_batched_reduce] {lbl}: {len(batch)} results")
        raw = _run_inference(_build_reduce_prompt(batch), MAX_NEW_TOKENS_REDUCE, label=lbl)
        intermediates.append(extract_json_fn(raw))

    logger.info(f"[_batched_reduce] Final reduce: {len(intermediates)} intermediates")
    raw = _run_inference(
        _build_reduce_prompt(intermediates), MAX_NEW_TOKENS_REDUCE, label="reduce-final"
    )
    return extract_json_fn(raw)


# ---------------------------------------------------------------------------
# Core inference
# ---------------------------------------------------------------------------
def _run_inference(prompt: str, max_new_tokens: int = MAX_NEW_TOKENS_MAP, label: str = "") -> str:
    """
    Direct model.generate() — always called via run_in_executor.

    Speed strategies applied:
    1. torch.inference_mode()     — no gradient tracking, ~10% faster
    2. use_cache=True             — KV-cache reuse during decoding
    3. Token reuse                — tokenize once, move to device
    4. eos_token_id stopping      — stop as soon as JSON closes (})
    5. Selective empty_cache()    — only when VRAM < 2 GB
    6. torch.compile (at load)    — fused kernels, 10-30% faster
    7. Float16 (not 8-bit)        — 2-3x faster on large GPU
    8. Flash Attention 2          — 2-4x faster for long sequences
    """
    tag          = f"[{label}] " if label else ""
    model_device = next(_model.parameters()).device

    encoded     = _tokenizer(prompt, return_tensors="pt", truncation=False)
    token_count = encoded["input_ids"].shape[1]
    logger.info(f"{tag}_run_inference: {token_count} tokens in, max_new={max_new_tokens}")

    if token_count > MAX_INPUT_TOKENS:
        logger.warning(
            f"{tag}_run_inference: {token_count} > {MAX_INPUT_TOKENS} — truncating"
        )
        inputs = _tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=MAX_INPUT_TOKENS
        ).to(model_device)
    else:
        inputs = {k: v.to(model_device) for k, v in encoded.items()}

    t_gen = time.perf_counter()
    try:
        with torch.inference_mode():
            output = _model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.1,
                use_cache=True,
                # Stop on EOS — model stops as soon as it closes the JSON
                # instead of generating padding tokens up to max_new_tokens
                eos_token_id=_EOS_TOKEN_IDS,
                pad_token_id=_tokenizer.eos_token_id,
            )
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        logger.error(f"{tag}_run_inference: CUDA OOM")
        raise RuntimeError("GPU OOM — lower CHUNK_SIZE and retry.")
    except Exception as e:
        logger.exception(f"{tag}_run_inference: generate failed: {e}")
        raise
    finally:
        del inputs
        if model_device.type == "cuda":
            free_after = (
                torch.cuda.get_device_properties(model_device).total_memory
                - torch.cuda.memory_allocated(model_device)
            ) / 1024 ** 3
            if free_after < 2.0:
                torch.cuda.empty_cache()

    elapsed    = time.perf_counter() - t_gen
    new_tokens = output.shape[1] - token_count
    logger.info(
        f"{tag}_run_inference: {new_tokens} tokens out in "
        f"{elapsed:.2f}s ({new_tokens/elapsed:.1f} tok/s)"
    )

    return _tokenizer.decode(output[0][token_count:], skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Map-reduce pipeline
# ---------------------------------------------------------------------------
async def generate_analysis(chunks: list[Document]) -> dict:
    """
    Map-reduce inference pipeline.
    MAP:    one inference call per chunk (serialised — GPU handles one at a time)
    REDUCE: batched synthesis of all map results into one final JSON
    """
    from s_main import extract_json  # avoid circular import

    logger.info(f"[generate_analysis] {len(chunks)} chunk(s)")

    if not chunks:
        return {"overview": "", "summary": "", "highlights": []}

    if len(chunks) > MAX_CHUNKS:
        logger.warning(
            f"[generate_analysis] {len(chunks)} chunks > expected {MAX_CHUNKS} — "
            "processing all (no truncation)"
        )

    loop       = asyncio.get_event_loop()
    t_pipeline = time.perf_counter()

    # ── MAP ──────────────────────────────────────────────────────────────
    logger.info(f"[generate_analysis] MAP: {len(chunks)} chunk(s)")
    map_results = []

    for i, chunk in enumerate(chunks):
        lbl = f"map {i+1}/{len(chunks)}"
        logger.info(
            f"[generate_analysis] [{lbl}] "
            f"page={chunk.metadata.get('page','?')}, "
            f"chars={len(chunk.page_content)}"
        )
        t_chunk = time.perf_counter()
        try:
            prompt = _build_map_prompt(chunk.page_content)
            raw    = await loop.run_in_executor(
                None,
                lambda p=prompt, l=lbl: _run_inference(p, MAX_NEW_TOKENS_MAP, l)
            )
        except Exception as e:
            logger.error(f"[generate_analysis] [{lbl}] failed: {e} — empty result")
            map_results.append({"overview": "", "summary": "", "highlights": []})
            continue

        parsed = extract_json(raw)
        map_results.append(parsed)
        logger.info(
            f"[generate_analysis] [{lbl}] done ({time.perf_counter()-t_chunk:.2f}s) "
            f"overview={'yes' if parsed.get('overview') else 'empty'} "
            f"highlights={len(parsed.get('highlights',[]))}"
        )

    logger.info(f"[generate_analysis] MAP complete — {len(map_results)} result(s)")

    # ── REDUCE ───────────────────────────────────────────────────────────
    if len(map_results) == 1:
        logger.info("[generate_analysis] Single chunk — skipping reduce")
        result = map_results[0]
    else:
        logger.info(
            f"[generate_analysis] REDUCE: {len(map_results)} results, "
            f"batch_size={REDUCE_BATCH_SIZE}"
        )
        t_reduce = time.perf_counter()
        try:
            result = await loop.run_in_executor(
                None,
                lambda: _batched_reduce(map_results, extract_json)
            )
            logger.info(
                f"[generate_analysis] REDUCE done ({time.perf_counter()-t_reduce:.2f}s)"
            )
        except Exception as e:
            logger.error(f"[generate_analysis] Reduce failed: {e} — merging directly")
            result = {
                "overview":   next((r["overview"] for r in map_results if r.get("overview")), ""),
                "summary":    next((r["summary"]  for r in map_results if r.get("summary")),  ""),
                "highlights": list({
                    h for r in map_results for h in r.get("highlights", []) if h
                })[:6],
            }

    logger.info(
        f"[generate_analysis] done — total={time.perf_counter()-t_pipeline:.2f}s "
        f"highlights={len(result.get('highlights',[]))}"
    )
    return result