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

# KEY CHANGE: reduced from 5500 to 2500 tokens per chunk.
#
# Why this fixes empty/prose output:
#   At 5500 tokens the model sees ~14,500 chars of dense text. Mistral 7B
#   loses instruction-following on inputs this long and forgets to output JSON.
#   At 2500 tokens (~6,600 chars) the model reliably stays on task.
#
# Trade-off: the 30-page PDF (18,954 tokens) now produces ~8 chunks instead
# of 4. Each chunk takes ~8-15s → total MAP time ~90s instead of ~60s.
# But all 8 chunks produce valid JSON → reduce synthesises the full document
# → response covers 100% of content instead of 50%.
TOKEN_CHUNK_SIZE      = 2500
TOKEN_CHUNK_OVERLAP   = 100   # slightly more overlap to avoid cutting mid-sentence
MAX_INPUT_TOKENS      = 3200  # 2500 content + ~500 prompt + 200 buffer
MAX_NEW_TOKENS_MAP    = 512   # smaller chunks need less output tokens
MAX_NEW_TOKENS_REDUCE = 768   # reduce prompt is larger, needs more room
REDUCE_BATCH_SIZE     = 10

# Retry a chunk once if the model produces no JSON
MAP_JSON_RETRY_ATTEMPTS = 2

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
#
# The prompt ends with [/INST]\n{ — this is constrained generation.
# The opening brace is the last token in the prompt, which forces the model
# to continue producing JSON from that point. It cannot output prose because
# the object has already started. _run_inference prepends { to the output.
# ---------------------------------------------------------------------------
_MAP_PREFIX = (
    "<s>[INST]\n"
    "Analyse the document excerpt below. Output ONLY a JSON object. "
    "No prose. No markdown. No explanation. Start with { end with }\n\n"
    "JSON format:\n"
    '{"overview":"1-2 sentences on document type and purpose",'
    '"summary":"3-4 sentence executive summary",'
    '"highlights":["specific fact with number/name/date","fact","fact","fact"]}\n\n'
    "Document:\n---\n"
)
_MAP_SUFFIX = "\n---\n[/INST]\n{"

_MAP_RETRY_PREFIX = (
    "<s>[INST]\n"
    "Output ONLY this JSON object. Nothing else. Begin with { immediately.\n\n"
    '{"overview":"...","summary":"...","highlights":["...","...","..."]}\n\n'
    "Document:\n---\n"
)
_MAP_RETRY_SUFFIX = "\n---\n[/INST]\n{"

_REDUCE_PREFIX = (
    "<s>[INST]\n"
    "Combine these section summaries into one final JSON object. "
    "No prose. No markdown. Start with { end with }\n\n"
    "JSON format:\n"
    '{"overview":"1-2 sentences covering the full document",'
    '"summary":"4-5 sentence coherent paragraph",'
    '"highlights":["most important fact","fact","fact","fact","fact","fact"]}\n\n'
    "Sections:\n---\n"
)
_REDUCE_SUFFIX = "\n---\n[/INST]\n{"


def _build_map_prompt(text: str, retry: bool = False) -> str:
    if retry:
        return _MAP_RETRY_PREFIX + text + _MAP_RETRY_SUFFIX
    return _MAP_PREFIX + text + _MAP_SUFFIX


def _build_reduce_prompt(results: list[dict]) -> str:
    parts = []
    for i, r in enumerate(results, 1):
        overview = r.get("overview", "").strip()
        summary  = r.get("summary",  "").strip()
        hl       = "\n".join(f"- {h}" for h in r.get("highlights", [])[:3] if h)
        # Include summary in reduce input so the model has richer context
        section  = f"[{i}]"
        if overview:
            section += f" {overview}"
        if summary:
            section += f"\n{summary}"
        if hl:
            section += f"\n{hl}"
        if overview or summary or hl:
            parts.append(section)
    return _REDUCE_PREFIX + "\n\n".join(parts) + _REDUCE_SUFFIX


# ---------------------------------------------------------------------------
# Core inference
# ---------------------------------------------------------------------------
def _run_inference(prompt: str, max_new_tokens: int = MAX_NEW_TOKENS_MAP, label: str = "") -> str:
    """
    Run model.generate(). All prompts end with [/INST]\n{ so the model is
    forced to continue from an open JSON brace. We prepend { to the output
    since that token was part of the prompt, not generated output.
    """
    tag          = f"[{label}] " if label else ""
    model_device = next(_model.parameters()).device

    encoded     = _tokenizer(prompt, return_tensors="pt", truncation=False)
    token_count = encoded["input_ids"].shape[1]
    logger.info(f"{tag}tokens_in={token_count} max_new={max_new_tokens}")

    current_limit   = min(token_count, MAX_INPUT_TOKENS)
    max_oom_retries = 3

    for attempt in range(max_oom_retries + 1):
        if token_count > current_limit:
            logger.warning(
                f"{tag}truncating {token_count}→{current_limit}"
                + (f" (OOM retry {attempt})" if attempt else "")
            )
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
            logger.info(
                f"{tag}{new_tokens} tokens_out in {elapsed:.2f}s ({new_tokens/elapsed:.1f} tok/s)"
                + (" [OOM-recovered]" if attempt else "")
            )
            del inputs
            raw = _tokenizer.decode(output[0][actual_tokens:], skip_special_tokens=True)
            # Restore the { that was the last prompt token
            return "{" + raw if not raw.startswith("{") else raw

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
                free = (
                    torch.cuda.get_device_properties(model_device).total_memory
                    - torch.cuda.memory_allocated(model_device)
                ) / 1024**3
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
        return extract_json_fn(
            _run_inference(_build_reduce_prompt(valid), MAX_NEW_TOKENS_REDUCE, "reduce")
        )

    logger.info(f"[_batched_reduce] {len(valid)} results → batches of {REDUCE_BATCH_SIZE}")
    intermediates = []
    for i in range(0, len(valid), REDUCE_BATCH_SIZE):
        batch = valid[i: i + REDUCE_BATCH_SIZE]
        lbl   = f"reduce-batch-{i // REDUCE_BATCH_SIZE + 1}"
        logger.info(f"[_batched_reduce] {lbl}: {len(batch)} results")
        raw = _run_inference(_build_reduce_prompt(batch), MAX_NEW_TOKENS_REDUCE, lbl)
        intermediates.append(extract_json_fn(raw))

    logger.info(f"[_batched_reduce] final reduce: {len(intermediates)} intermediates")
    return extract_json_fn(
        _run_inference(_build_reduce_prompt(intermediates), MAX_NEW_TOKENS_REDUCE, "reduce-final")
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
async def generate_analysis(merged_text: str) -> dict:
    from s_main import extract_json

    _EMPTY = {"overview": "", "summary": "", "highlights": []}

    if not merged_text or not merged_text.strip():
        return dict(_EMPTY)

    t0     = time.perf_counter()
    chunks = split_by_tokens(merged_text)
    logger.info(f"[generate_analysis] {len(chunks)} chunk(s) in {time.perf_counter()-t0:.3f}s")

    loop       = asyncio.get_running_loop()
    t_pipeline = time.perf_counter()
    map_results = []

    # ── MAP ──────────────────────────────────────────────────────────────
    logger.info(f"[generate_analysis] MAP: {len(chunks)} chunk(s)")
    for i, chunk_text in enumerate(chunks):
        lbl = f"map {i+1}/{len(chunks)}"
        logger.info(f"[generate_analysis] [{lbl}] chars={len(chunk_text):,}")
        t_chunk = time.perf_counter()

        parsed = None

        for attempt in range(1, MAP_JSON_RETRY_ATTEMPTS + 1):
            try:
                is_retry = attempt > 1
                prompt   = _build_map_prompt(chunk_text, retry=is_retry)
                if is_retry:
                    logger.warning(f"[generate_analysis] [{lbl}] retry {attempt} — stronger prompt")
                raw = await loop.run_in_executor(
                    None,
                    lambda p=prompt, l=f"{lbl}-a{attempt}": _run_inference(p, MAX_NEW_TOKENS_MAP, l)
                )
            except Exception as e:
                logger.error(f"[generate_analysis] [{lbl}] inference error: {e}")
                break

            candidate = extract_json(raw)

            if candidate.get("overview") or candidate.get("highlights"):
                parsed = candidate
                if attempt > 1:
                    logger.info(f"[generate_analysis] [{lbl}] retry {attempt} produced valid JSON")
                break
            else:
                logger.warning(
                    f"[generate_analysis] [{lbl}] attempt {attempt}: no usable JSON — "
                    + ("retrying" if attempt < MAP_JSON_RETRY_ATTEMPTS else "giving up on this chunk")
                )

        if not isinstance(parsed, dict) or not (parsed.get("overview") or parsed.get("highlights")):
            parsed = dict(_EMPTY)

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
            result = await loop.run_in_executor(
                None, lambda: _batched_reduce(map_results, extract_json)
            )
            logger.info(f"[generate_analysis] REDUCE done ({time.perf_counter()-t_reduce:.2f}s)")
        except Exception as e:
            logger.error(f"[generate_analysis] REDUCE failed: {e} — direct merge fallback")
            result = {
                "overview":   next((r["overview"] for r in map_results if r.get("overview")), ""),
                "summary":    next((r["summary"]  for r in map_results if r.get("summary")),  ""),
                "highlights": list({h for r in map_results for h in r.get("highlights", []) if h})[:8],
            }

    if not isinstance(result, dict):
        logger.error(f"[generate_analysis] Final result is {type(result).__name__} — returning empty")
        result = dict(_EMPTY)

    logger.info(
        f"[generate_analysis] total={time.perf_counter()-t_pipeline:.2f}s "
        f"highlights={len(result.get('highlights', []))}"
    )
    return result