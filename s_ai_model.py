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

TOKEN_CHUNK_SIZE      = 2500   # small enough for reliable JSON output from Mistral 7B
TOKEN_CHUNK_OVERLAP   = 100
MAX_INPUT_TOKENS      = 3200   # 2500 content + ~500 prompt + 200 buffer
MAX_NEW_TOKENS_MAP    = 512    # per-chunk extraction
MAX_NEW_TOKENS_SYNTH  = 1024   # synthesis — no length cap, needs room for large docs

# Retry a chunk once if the model produces no usable JSON
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



# Token-accurate chunking
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



# Model loading
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
# All prompts end with [/INST]\n{ — constrained generation.
# The opening brace is the last prompt token, forcing the model to continue
# producing JSON. It cannot output prose because the object is already open.
# _run_inference prepends { to the decoded output.
# ---------------------------------------------------------------------------

# MAP: extract facts from one chunk.
# No sentence count constraints — length is driven by how much content the chunk has.
_MAP_PREFIX = (
    "<s>[INST]\n"
    "Analyse the document excerpt below. Output ONLY a JSON object. "
    "No prose. No markdown. No explanation. Start with { end with }\n\n"
    "JSON format:\n"
    '{"overview":"what this document is — type, subject, and purpose",'
    '"summary":"cover all key points in this excerpt — as long or short as the content requires",'
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

# SYNTHESIS: write ONE clean overview + summary from all chunk summaries.
# Highlights are NOT included here — they are merged in Python, not by the LLM.
# No sentence count constraints — length matches what the document actually needs.
_SYNTH_PREFIX = (
    "<s>[INST]\n"
    "You are given summaries of consecutive sections of a single document. "
    "Write a JSON object with two fields only:\n"
    "- overview: describe what the entire document is — its type, subject, and main purpose. "
    "Write as much or as little as the document warrants.\n"
    "- summary: cover ALL major topics across the entire document. "
    "Do NOT repeat the same topic. Work through the document from start to end. "
    "Be specific — include key subjects, people, figures, decisions, and conclusions. "
    "Write as much as needed to accurately represent the full document — do not pad, do not cut short.\n\n"
    "No prose outside the JSON. No markdown. Start with { end with }\n\n"
    'JSON format: {"overview":"...","summary":"..."}\n\n'
    "Section summaries (in document order):\n---\n"
)
_SYNTH_SUFFIX = "\n---\n[/INST]\n{"


def _build_map_prompt(text: str, retry: bool = False) -> str:
    if retry:
        return _MAP_RETRY_PREFIX + text + _MAP_RETRY_SUFFIX
    return _MAP_PREFIX + text + _MAP_SUFFIX


def _build_synth_prompt(results: list[dict]) -> str:
    """
    Build synthesis prompt from all chunk overviews + summaries.
    Kept compact: only text fields, no highlights (those are merged in Python).
    """
    parts = []
    for i, r in enumerate(results, 1):
        overview = r.get("overview", "").strip()
        summary  = r.get("summary",  "").strip()
        if overview or summary:
            lines = [f"[Section {i}]"]
            if overview:
                lines.append(f"Type: {overview}")
            if summary:
                lines.append(f"Content: {summary}")
            parts.append("\n".join(lines))
    return _SYNTH_PREFIX + "\n\n".join(parts) + _SYNTH_SUFFIX



# Highlights merge — pure Python, no LLM, preserves every distinct fact
def _merge_highlights(results: list[dict]) -> list[str]:
    """
    Collect all highlights from all chunks and deduplicate by normalised text.

    Normalisation: lowercase + collapse whitespace.
    Keeps the original casing/wording of the FIRST occurrence of each fact.
    Highlights from overlapping chunk boundaries that are near-identical
    (same words, different capitalisation) are deduplicated.

    No facts are dropped — this is a union, not a selection.
    """
    seen:       set  = set()
    highlights: list = []

    for r in results:
        for h in r.get("highlights", []):
            h = h.strip()
            if not h:
                continue
            key = " ".join(h.lower().split())  # normalise for comparison only
            if key not in seen:
                seen.add(key)
                highlights.append(h)

    logger.info(f"[_merge_highlights] {len(highlights)} distinct highlight(s) preserved")
    return highlights



# Core inference
def _run_inference(prompt: str, max_new_tokens: int = MAX_NEW_TOKENS_MAP, label: str = "") -> str:
    """
    Run model.generate(). Prompt ends with { so model continues from open JSON.
    Prepends { to decoded output since it was a prompt token, not generated.
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



# Public API
async def generate_analysis(merged_text: str) -> dict:
    from s_main import extract_json

    _EMPTY = {"overview": "", "summary": "", "highlights": []}

    if not merged_text or not merged_text.strip():
        return dict(_EMPTY)

    t0     = time.perf_counter()
    chunks = split_by_tokens(merged_text)
    logger.info(f"[generate_analysis] {len(chunks)} chunk(s) in {time.perf_counter()-t0:.3f}s")

    loop        = asyncio.get_running_loop()
    t_pipeline  = time.perf_counter()
    map_results = []

    # ── MAP: extract facts from each chunk ───────────────────────────────
    logger.info(f"[generate_analysis] MAP: {len(chunks)} chunk(s)")
    for i, chunk_text in enumerate(chunks):
        lbl = f"map {i+1}/{len(chunks)}"
        logger.info(f"[generate_analysis] [{lbl}] chars={len(chunk_text):,}")
        t_chunk = time.perf_counter()

        parsed = None

        for attempt in range(1, MAP_JSON_RETRY_ATTEMPTS + 1):
            try:
                prompt = _build_map_prompt(chunk_text, retry=(attempt > 1))
                if attempt > 1:
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

    # ── SINGLE CHUNK: return directly, no synthesis needed ───────────────
    if len(map_results) == 1:
        result = map_results[0]
        logger.info("[generate_analysis] single chunk — returning directly")

    else:
        # ── HIGHLIGHTS: merge all in Python — no LLM, no data loss ───────
        #
        # Every distinct fact from every chunk is kept.
        # Deduplication is by normalised text (lowercase + collapsed whitespace).
        # The LLM is NOT used for highlights — it would drop facts.
        all_highlights = _merge_highlights(map_results)

        # ── OVERVIEW + SUMMARY: one LLM synthesis call ────────────────────
        #
        # The LLM reads all chunk overviews+summaries and writes:
        #   - overview: 2-3 sentences on what the whole document is
        #   - summary:  5-7 sentences covering ALL major topics, no repetition
        #
        # This is what the LLM is good at — synthesising readable prose from
        # structured inputs. It is NOT used for highlights (too lossy).
        logger.info(f"[generate_analysis] SYNTHESIS: overview+summary from {valid_count} chunk(s)")
        t_synth = time.perf_counter()

        synth_result = {"overview": "", "summary": ""}
        try:
            synth_prompt = _build_synth_prompt(
                [r for r in map_results if r.get("overview") or r.get("summary")]
            )
            raw_synth = await loop.run_in_executor(
                None,
                lambda: _run_inference(synth_prompt, MAX_NEW_TOKENS_SYNTH, "synthesis")
            )
            parsed_synth = extract_json(raw_synth)

            synth_result["overview"] = parsed_synth.get("overview", "").strip()
            synth_result["summary"]  = parsed_synth.get("summary",  "").strip()

            logger.info(
                f"[generate_analysis] SYNTHESIS done ({time.perf_counter()-t_synth:.2f}s) "
                f"overview={'yes' if synth_result['overview'] else 'empty'} "
                f"summary={'yes' if synth_result['summary'] else 'empty'}"
            )
        except Exception as e:
            logger.error(f"[generate_analysis] SYNTHESIS failed: {e} — using first chunk overview")
            synth_result["overview"] = next((r["overview"] for r in map_results if r.get("overview")), "")
            synth_result["summary"]  = next((r["summary"]  for r in map_results if r.get("summary")),  "")

        # Fallback: if synthesis produced empty overview, use first chunk's
        if not synth_result["overview"]:
            synth_result["overview"] = next((r["overview"] for r in map_results if r.get("overview")), "")
        if not synth_result["summary"]:
            synth_result["summary"]  = next((r["summary"]  for r in map_results if r.get("summary")),  "")

        result = {
            "overview":   synth_result["overview"],
            "summary":    synth_result["summary"],
            "highlights": all_highlights,
        }

    if not isinstance(result, dict):
        logger.error(f"[generate_analysis] result is {type(result).__name__} — returning empty")
        result = dict(_EMPTY)

    logger.info(
        f"[generate_analysis] total={time.perf_counter()-t_pipeline:.2f}s "
        f"highlights={len(result.get('highlights', []))}"
    )
    return result


# ---------------------------------------------------------------------------
# generate_analysis_stream — async generator for the SSE streaming endpoint
#
# Same logic as generate_analysis but yields progress events so the client
# sees results as each chunk finishes rather than waiting for the full pipeline.
#
# Yielded events (event_type, payload):
#
#   ("chunk_start",    {"chunk": N, "total": N})
#   ("chunk_done",     {"chunk": N, "total": N, "overview": str,
#                       "new_highlights": [...], "all_highlights_so_far": [...]})
#   ("synthesis_start", {})
#   ("synthesis_done", {"overview": str, "summary": str})
#   ("done",           {})
# ---------------------------------------------------------------------------

async def generate_analysis_stream(merged_text: str):
    """
    Streaming version of generate_analysis.

    Yields (event_type, payload) tuples as each stage of the pipeline
    completes. The caller (s_main.py /analyze/stream) converts these
    into SSE messages and sends them to the client.

    Highlights are yielded incrementally — after each chunk completes,
    new highlights are yielded immediately so the client can render them
    without waiting for the full pipeline.
    """
    from s_main import extract_json

    _EMPTY = {"overview": "", "summary": "", "highlights": []}

    if not merged_text or not merged_text.strip():
        yield ("done", {})
        return

    t0     = time.perf_counter()
    chunks = split_by_tokens(merged_text)
    logger.info(f"[generate_analysis_stream] {len(chunks)} chunk(s)")

    loop             = asyncio.get_running_loop()
    map_results      = []
    all_highlights:  list = []   # running deduped list — grows as chunks finish
    seen_keys:       set  = set()

    # ── MAP: extract from each chunk, yield progress after each ──────────
    for i, chunk_text in enumerate(chunks):
        lbl = f"map {i+1}/{len(chunks)}"
        yield ("chunk_start", {"chunk": i + 1, "total": len(chunks)})

        parsed = None
        for attempt in range(1, MAP_JSON_RETRY_ATTEMPTS + 1):
            try:
                prompt = _build_map_prompt(chunk_text, retry=(attempt > 1))
                if attempt > 1:
                    logger.warning(f"[stream] [{lbl}] retry {attempt}")
                raw = await loop.run_in_executor(
                    None,
                    lambda p=prompt, l=f"{lbl}-a{attempt}": _run_inference(p, MAX_NEW_TOKENS_MAP, l)
                )
            except Exception as e:
                logger.error(f"[stream] [{lbl}] inference error: {e}")
                break

            candidate = extract_json(raw)
            if candidate.get("overview") or candidate.get("highlights"):
                parsed = candidate
                break
            else:
                logger.warning(f"[stream] [{lbl}] attempt {attempt}: no usable JSON")

        if not isinstance(parsed, dict) or not (parsed.get("overview") or parsed.get("highlights")):
            parsed = dict(_EMPTY)

        map_results.append(parsed)

        # Collect new highlights that haven't been seen yet
        new_highlights = []
        for h in parsed.get("highlights", []):
            h = h.strip()
            if not h:
                continue
            key = " ".join(h.lower().split())
            if key not in seen_keys:
                seen_keys.add(key)
                all_highlights.append(h)
                new_highlights.append(h)

        yield ("chunk_done", {
            "chunk":                  i + 1,
            "total":                  len(chunks),
            "overview":               parsed.get("overview", ""),
            "new_highlights":         new_highlights,
            "all_highlights_so_far":  list(all_highlights),
        })

        logger.info(
            f"[stream] [{lbl}] done — "
            f"overview={'yes' if parsed.get('overview') else 'empty'} "
            f"new_highlights={len(new_highlights)} total={len(all_highlights)}"
        )

    valid_count = sum(1 for r in map_results if r.get("overview") or r.get("highlights"))
    logger.info(f"[stream] MAP complete — {len(map_results)} total, {valid_count} with content")

    # ── SYNTHESIS: write overview + summary (single LLM call) ─────────────
    synth_overview = ""
    synth_summary  = ""

    if len(map_results) == 1:
        # Single chunk — use map result directly, no synthesis needed
        synth_overview = map_results[0].get("overview", "")
        synth_summary  = map_results[0].get("summary",  "")
    else:
        yield ("synthesis_start", {})
        try:
            synth_prompt = _build_synth_prompt(
                [r for r in map_results if r.get("overview") or r.get("summary")]
            )
            raw_synth = await loop.run_in_executor(
                None,
                lambda: _run_inference(synth_prompt, MAX_NEW_TOKENS_SYNTH, "synthesis")
            )
            parsed_synth   = extract_json(raw_synth)
            synth_overview = parsed_synth.get("overview", "").strip()
            synth_summary  = parsed_synth.get("summary",  "").strip()
            logger.info(
                f"[stream] SYNTHESIS done — "
                f"overview={'yes' if synth_overview else 'empty'} "
                f"summary={'yes' if synth_summary else 'empty'}"
            )
        except Exception as e:
            logger.error(f"[stream] SYNTHESIS failed: {e} — fallback to first chunk")

        # Fallbacks
        if not synth_overview:
            synth_overview = next((r["overview"] for r in map_results if r.get("overview")), "")
        if not synth_summary:
            synth_summary  = next((r["summary"]  for r in map_results if r.get("summary")),  "")

    yield ("synthesis_done", {"overview": synth_overview, "summary": synth_summary})

    logger.info(
        f"[stream] total={time.perf_counter()-t0:.2f}s "
        f"highlights={len(all_highlights)}"
    )
    yield ("done", {})