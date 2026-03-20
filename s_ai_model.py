import asyncio
import os
import time
import torch
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

TOKEN_CHUNK_SIZE      = 5500
TOKEN_CHUNK_OVERLAP   = 50
MAX_INPUT_TOKENS      = 6500
MAX_NEW_TOKENS_MAP    = 600
MAX_NEW_TOKENS_REDUCE = 600
REDUCE_BATCH_SIZE     = 10

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
    p = torch.cuda.get_device_properties(0)
    return (p.total_memory - torch.cuda.memory_allocated(0) - torch.cuda.memory_reserved(0)) / 1024**3


def _vram_total_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.get_device_properties(0).total_memory / 1024**3


# ---------------------------------------------------------------------------
# Token-accurate chunking
# ---------------------------------------------------------------------------
def split_by_tokens(text: str) -> list[str]:
    if not text:
        return []

    t0       = time.perf_counter()
    all_ids  = _tokenizer.encode(text, add_special_tokens=False)
    n_tokens = len(all_ids)
    logger.info(
        f"[split_by_tokens] {len(text):,} chars → {n_tokens:,} tokens "
        f"({len(text)/n_tokens:.2f} chars/token) in {time.perf_counter()-t0:.3f}s"
    )

    chunks = []
    start  = 0
    while start < n_tokens:
        end       = min(start + TOKEN_CHUNK_SIZE, n_tokens)
        chunk_ids = all_ids[start:end]
        chunks.append(_tokenizer.decode(chunk_ids, skip_special_tokens=True))
        start    += TOKEN_CHUNK_SIZE - TOKEN_CHUNK_OVERLAP

    logger.info(f"[split_by_tokens] → {len(chunks)} chunk(s) of ≤{TOKEN_CHUNK_SIZE} tokens")
    return chunks


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def _load_model():
    total_vram = _vram_total_gb()
    free_vram  = _vram_free_gb()
    logger.info(f"[ai_model] VRAM total={total_vram:.1f} GB, free={free_vram:.1f} GB")

    attn_impl = "eager"
    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
        logger.info("[ai_model] Flash Attention 2 enabled")
    except ImportError:
        logger.info("[ai_model] flash-attn not installed — install for 2-4x speedup: pip install flash-attn --no-build-isolation")

    kw = dict(attn_implementation=attn_impl)

    if DEVICE == "cuda" and total_vram >= 35.0:
        logger.info(f"[ai_model] Large GPU ({total_vram:.1f} GB) → float16")
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map={"": 0}, **kw)
    elif DEVICE == "cuda" and free_vram >= 14.0:
        logger.info(f"[ai_model] {free_vram:.1f} GB free → float16")
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map={"": 0}, **kw)
    elif DEVICE == "cuda" and free_vram >= 7.0:
        logger.info(f"[ai_model] {free_vram:.1f} GB free → 8-bit")
        try:
            from transformers import BitsAndBytesConfig
            model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=BitsAndBytesConfig(load_in_8bit=True), device_map={"": 0}, **kw)
        except ImportError:
            model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map={"": 0}, **kw)
    elif DEVICE == "cuda" and free_vram >= 4.0:
        logger.info(f"[ai_model] {free_vram:.1f} GB free → 4-bit")
        try:
            from transformers import BitsAndBytesConfig
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True),
                device_map={"": 0}, **kw,
            )
        except ImportError:
            raise RuntimeError("Install bitsandbytes: pip install bitsandbytes")
    else:
        logger.warning("[ai_model] Falling back to CPU float32")
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32, device_map="cpu")

    meta = [n for n, p in model.named_parameters() if p.device.type == "meta"]
    if meta:
        logger.error(f"[ai_model] {len(meta)} params on meta: {meta[:3]}")
    else:
        logger.info("[ai_model] All parameters on real devices")

    model.eval()

    # FIX: mode="reduce-overhead" uses CUDA graphs which require fixed input
    # shapes — our token lengths vary per chunk causing recompile on every call.
    # mode="default" is safe with variable-length inputs.
    try:
        model = torch.compile(model, mode="default")
        logger.info("[ai_model] torch.compile enabled (mode=default)")
    except Exception as e:
        logger.info(f"[ai_model] torch.compile skipped ({e})")
    return model


logger.info("[ai_model] Loading model ...")
t0     = time.perf_counter()
_model = _load_model()
logger.info(f"[ai_model] Model ready ({time.perf_counter() - t0:.2f}s)")

if DEVICE == "cuda":
    logger.info(
        f"[ai_model] GPU: allocated={torch.cuda.memory_allocated()/1024**3:.2f} GB, "
        f"reserved={torch.cuda.memory_reserved()/1024**3:.2f} GB"
    )

_EOS_TOKEN_IDS = [_tokenizer.eos_token_id]


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
_MAP_PREFIX = (
    "<s>[INST]\nAnalyse this document excerpt. Output ONLY valid JSON:\n"
    '{"overview":"1-2 sentence type and purpose.",'
    '"summary":"3-4 sentence executive summary.",'
    '"highlights":["specific fact with number/date/name.","fact.","fact.","fact."]}\n'
    "No markdown. No commentary. Properly escape all strings.\n\nDocument:\n---\n"
)
_MAP_SUFFIX = "\n---\n[/INST]\n"

_REDUCE_PREFIX = (
    "<s>[INST]\nSynthesize these section analyses into ONE final JSON:\n"
    '{"overview":"1-2 sentences covering entire document.",'
    '"summary":"4-5 sentence coherent paragraph.",'
    '"highlights":["most important fact.","fact.","fact.","fact.","fact."]}\n'
    "No markdown. No section numbers in output. Properly escape all strings.\n\nSections:\n---\n"
)
_REDUCE_SUFFIX = "\n---\n[/INST]\n"


def _build_map_prompt(text: str) -> str:
    return _MAP_PREFIX + text + _MAP_SUFFIX


def _build_reduce_prompt(results: list[dict]) -> str:
    parts = []
    for i, r in enumerate(results, 1):
        overview = r.get("overview", "").strip()
        hl       = "\n".join(f"- {h}" for h in r.get("highlights", [])[:4] if h)
        if overview or hl:
            parts.append(f"[{i}] {overview}\n{hl}")
    return _REDUCE_PREFIX + "\n\n".join(parts) + _REDUCE_SUFFIX


# ---------------------------------------------------------------------------
# Core inference
# ---------------------------------------------------------------------------
def _run_inference(prompt: str, max_new_tokens: int = MAX_NEW_TOKENS_MAP, label: str = "") -> str:
    tag          = f"[{label}] " if label else ""
    model_device = next(_model.parameters()).device

    encoded     = _tokenizer(prompt, return_tensors="pt", truncation=False)
    token_count = encoded["input_ids"].shape[1]
    logger.info(f"{tag}tokens_in={token_count} max_new={max_new_tokens}")

    current_limit   = min(token_count, MAX_INPUT_TOKENS)
    max_oom_retries = 3

    for attempt in range(max_oom_retries + 1):
        if token_count > current_limit:
            logger.warning(f"{tag}truncating {token_count}→{current_limit}" + (f" (retry {attempt})" if attempt else ""))
            inputs        = _tokenizer(prompt, return_tensors="pt", truncation=True, max_length=current_limit).to(model_device)
            actual_tokens = current_limit
        else:
            inputs        = {k: v.to(model_device) for k, v in encoded.items()}
            actual_tokens = token_count

        t0 = time.perf_counter()
        try:
            with torch.inference_mode():
                output = _model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    repetition_penalty=1.1,
                    use_cache=True,
                    eos_token_id=_EOS_TOKEN_IDS,
                    pad_token_id=_tokenizer.eos_token_id,
                )
            elapsed    = time.perf_counter() - t0
            new_tokens = output.shape[1] - actual_tokens
            logger.info(f"{tag}{new_tokens} tokens_out in {elapsed:.2f}s ({new_tokens/elapsed:.1f} tok/s)" + (" [OOM-recovered]" if attempt else ""))
            del inputs
            return _tokenizer.decode(output[0][actual_tokens:], skip_special_tokens=True)

        except torch.cuda.OutOfMemoryError:
            del inputs
            torch.cuda.empty_cache()
            time.sleep(0.5)
            if attempt < max_oom_retries:
                current_limit = current_limit // 2
                logger.warning(f"{tag}OOM → retrying with {current_limit} tokens (attempt {attempt+2}/{max_oom_retries+1})")
            else:
                logger.error(f"{tag}OOM after {max_oom_retries} retries — giving up")
                raise RuntimeError("GPU OOM after retries. Restart server to free memory.")

        except Exception as e:
            del inputs
            logger.exception(f"{tag}generate failed: {e}")
            raise

        finally:
            if model_device.type == "cuda":
                free = (torch.cuda.get_device_properties(model_device).total_memory - torch.cuda.memory_allocated(model_device)) / 1024**3
                if free < 2.0:
                    torch.cuda.empty_cache()

    raise RuntimeError("_run_inference: all attempts exhausted")


# ---------------------------------------------------------------------------
# Batched reduce
# ---------------------------------------------------------------------------
def _batched_reduce(results: list[dict], extract_json_fn) -> dict:
    valid   = [r for r in results if r.get("overview") or r.get("highlights")]
    skipped = len(results) - len(valid)
    if skipped:
        logger.info(f"[_batched_reduce] Filtered {skipped} empty result(s)")
    if not valid:
        logger.warning("[_batched_reduce] All map results empty — returning empty")
        return {"overview": "", "summary": "", "highlights": []}

    if len(valid) <= REDUCE_BATCH_SIZE:
        return extract_json_fn(_run_inference(_build_reduce_prompt(valid), MAX_NEW_TOKENS_REDUCE, "reduce"))

    logger.info(f"[_batched_reduce] {len(valid)} results → batches of {REDUCE_BATCH_SIZE}")
    intermediates = []
    for i in range(0, len(valid), REDUCE_BATCH_SIZE):
        batch = valid[i: i + REDUCE_BATCH_SIZE]
        lbl   = f"reduce-batch-{i // REDUCE_BATCH_SIZE + 1}"
        logger.info(f"[_batched_reduce] {lbl}: {len(batch)} results")
        raw = _run_inference(_build_reduce_prompt(batch), MAX_NEW_TOKENS_REDUCE, lbl)
        intermediates.append(extract_json_fn(raw))

    logger.info(f"[_batched_reduce] final reduce: {len(intermediates)} intermediates")
    return extract_json_fn(_run_inference(_build_reduce_prompt(intermediates), MAX_NEW_TOKENS_REDUCE, "reduce-final"))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
async def generate_analysis(merged_text: str) -> dict:
    from s_main import extract_json

    _EMPTY_CHUNK = {"overview": "", "summary": "", "highlights": []}

    if not merged_text or not merged_text.strip():
        return dict(_EMPTY_CHUNK)

    t0     = time.perf_counter()
    chunks = split_by_tokens(merged_text)
    logger.info(f"[generate_analysis] {len(chunks)} token-accurate chunk(s) in {time.perf_counter()-t0:.3f}s")

    # FIX: use get_running_loop() — get_event_loop() is deprecated in Python
    # 3.10 and raises RuntimeError in Python 3.12 inside a running event loop.
    loop        = asyncio.get_running_loop()
    t_pipeline  = time.perf_counter()
    map_results = []

    # ── MAP ──────────────────────────────────────────────────────────────
    logger.info(f"[generate_analysis] MAP: {len(chunks)} chunk(s)")
    for i, chunk_text in enumerate(chunks):
        lbl = f"map {i+1}/{len(chunks)}"
        logger.info(f"[generate_analysis] [{lbl}] chars={len(chunk_text):,}")
        t_chunk = time.perf_counter()
        try:
            prompt = _build_map_prompt(chunk_text)
            raw    = await loop.run_in_executor(None, lambda p=prompt, l=lbl: _run_inference(p, MAX_NEW_TOKENS_MAP, l))
        except Exception as e:
            logger.error(f"[generate_analysis] [{lbl}] inference failed: {e} — inserting empty")
            map_results.append(dict(_EMPTY_CHUNK))
            continue

        parsed = extract_json(raw)
        if not isinstance(parsed, dict):
            logger.error(
                f"[generate_analysis] [{lbl}] extract_json returned {type(parsed).__name__} "
                f"instead of dict — inserting empty. raw[:200]={raw[:200]!r}"
            )
            parsed = dict(_EMPTY_CHUNK)

        map_results.append(parsed)
        logger.info(
            f"[generate_analysis] [{lbl}] done ({time.perf_counter()-t_chunk:.2f}s) "
            f"overview={'yes' if parsed.get('overview') else 'empty'} "
            f"highlights={len(parsed.get('highlights', []))}"
        )

    valid_count = sum(1 for r in map_results if r.get("overview") or r.get("highlights"))
    logger.info(f"[generate_analysis] MAP complete — {len(map_results)} total, {valid_count} with content")

    # ── REDUCE ───────────────────────────────────────────────────────────
    if len(map_results) == 1:
        result = map_results[0]
    else:
        logger.info(f"[generate_analysis] REDUCE: {len(map_results)} results, batch_size={REDUCE_BATCH_SIZE}")
        t_reduce = time.perf_counter()
        try:
            result = await loop.run_in_executor(None, lambda: _batched_reduce(map_results, extract_json))
            logger.info(f"[generate_analysis] REDUCE done ({time.perf_counter()-t_reduce:.2f}s)")
        except Exception as e:
            logger.error(f"[generate_analysis] REDUCE failed: {e} — direct merge fallback")
            result = {
                "overview":   next((r["overview"] for r in map_results if r.get("overview")), ""),
                "summary":    next((r["summary"]  for r in map_results if r.get("summary")),  ""),
                "highlights": list({h for r in map_results for h in r.get("highlights", []) if h})[:6],
            }

    if not isinstance(result, dict):
        logger.error(f"[generate_analysis] Final result is {type(result).__name__} — returning empty")
        result = dict(_EMPTY_CHUNK)

    logger.info(f"[generate_analysis] total={time.perf_counter()-t_pipeline:.2f}s highlights={len(result.get('highlights', []))}")
    return result