import asyncio
import os
import time
import logging
import tiktoken
from openai import AsyncOpenAI
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-5-nano")  # ← set in .env, fallback to gpt-4o if missing
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY is not set in environment variables. Please set it in your .env file.")

# GPT-4.1-nano supports 1M token context; large chunks reduce API calls
TOKEN_CHUNK_SIZE    = 50000
TOKEN_CHUNK_OVERLAP = 500

MAP_JSON_RETRY_ATTEMPTS = 2

_client   = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
_encoding = tiktoken.encoding_for_model(MODEL_NAME)   # cl100k_base — same family


# ---------------------------------------------------------------------------
# Token-accurate chunking
# ---------------------------------------------------------------------------
def split_by_tokens(text: str) -> list[str]:
    if not text:
        return []
    t0       = time.perf_counter()
    all_ids  = _encoding.encode(text)
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
        chunks.append(_encoding.decode(chunk_ids))
        start    += TOKEN_CHUNK_SIZE - TOKEN_CHUNK_OVERLAP
    logger.info(f"[split_by_tokens] → {len(chunks)} chunk(s) of ≤{TOKEN_CHUNK_SIZE} tokens")
    return chunks


# ---------------------------------------------------------------------------
# Prompts (OpenAI chat format)
# ---------------------------------------------------------------------------
_MAP_SYSTEM = (
    "You are a document analysis assistant. Analyse the document excerpt provided. "
    "Output ONLY a JSON object with exactly these fields:\n"
    '{"overview":"what this document is — type, subject, and purpose",'
    '"summary":"cover all key points in this excerpt — as long or short as the content requires",'
    '"highlights":["specific fact with number/name/date","fact","fact"]}\n'
    "CRITICAL: overview and summary MUST be plain strings. "
    "highlights MUST be a flat array of strings. "
    "NEVER use nested objects or nested arrays."
)

_MAP_RETRY_SYSTEM = (
    "Output ONLY this JSON object. Nothing else.\n"
    "overview and summary must be plain strings. highlights must be a flat array of strings.\n"
    '{"overview":"...","summary":"...","highlights":["...","...","..."]}'
)

_SYNTH_SYSTEM = (
    "You are given summaries of consecutive sections of a single document. "
    "Write a JSON object with exactly two fields:\n"
    '{"overview":"describe what the entire document is — its type, subject, and main purpose",'
    '"summary":"cover ALL major topics across the entire document. Work through from start to end. '
    "Be specific — include key subjects, people, figures, decisions, and conclusions. "
    'Do NOT repeat topics. Write as much as needed to accurately represent the full document."}\n'
    "CRITICAL: both fields must be plain strings — no nested objects, no arrays. "
    "No prose outside the JSON."
)


def _build_map_messages(text: str, retry: bool = False) -> list[dict]:
    system = _MAP_RETRY_SYSTEM if retry else _MAP_SYSTEM
    return [
        {"role": "system", "content": system},
        {"role": "user",   "content": f"Document:\n---\n{text}\n---"},
    ]


def _build_synth_messages(results: list[dict]) -> list[dict]:
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
    body = "\n\n".join(parts)
    return [
        {"role": "system", "content": _SYNTH_SYSTEM},
        {"role": "user",   "content": f"Section summaries (in document order):\n---\n{body}\n---"},
    ]


# ---------------------------------------------------------------------------
# Highlights merge — pure Python, no LLM, preserves every distinct fact
# ---------------------------------------------------------------------------
def _merge_highlights(results: list[dict]) -> list[str]:
    seen:       set  = set()
    highlights: list = []
    for r in results:
        for h in r.get("highlights", []):
            h = h.strip()
            if not h:
                continue
            key = " ".join(h.lower().split())
            if key not in seen:
                seen.add(key)
                highlights.append(h)
    logger.info(f"[_merge_highlights] {len(highlights)} distinct highlight(s) preserved")
    return highlights


# ---------------------------------------------------------------------------
# Core inference — single OpenAI API call
# ---------------------------------------------------------------------------
async def _run_inference(messages: list[dict], label: str = "") -> str:
    tag = f"[{label}] " if label else ""
    t0  = time.perf_counter()
    try:
        _FIXED_TEMPERATURE_MODELS = {"gpt-5-nano", "gpt-4o-mini", "o1", "o1-mini", "o3-mini"}
        api_kwargs = {
            "model": MODEL_NAME,
            "messages": messages,
            "response_format": {"type": "json_object"},
        }
        if MODEL_NAME not in _FIXED_TEMPERATURE_MODELS:
            api_kwargs["temperature"] = 0.0

        response = await _client.chat.completions.create(**api_kwargs)
        elapsed = time.perf_counter() - t0
        content = response.choices[0].message.content or ""
        usage   = response.usage
        logger.info(
            f"{tag}in={usage.prompt_tokens} out={usage.completion_tokens} "
            f"in {elapsed:.2f}s"
        )
        return content
    except Exception as e:
        logger.exception(f"{tag}OpenAI API call failed: {e}")
        raise


# ---------------------------------------------------------------------------
# generate_analysis — full pipeline, returns complete result
# ---------------------------------------------------------------------------
async def generate_analysis(merged_text: str) -> dict:
    from s_main import extract_json

    _EMPTY = {"overview": "", "summary": "", "highlights": []}
    if not merged_text or not merged_text.strip():
        return dict(_EMPTY)

    t0     = time.perf_counter()
    chunks = split_by_tokens(merged_text)
    logger.info(f"[generate_analysis] {len(chunks)} chunk(s) in {time.perf_counter()-t0:.3f}s")

    t_pipeline  = time.perf_counter()
    map_results = []

    # ── MAP ──────────────────────────────────────────────────────────────
    logger.info(f"[generate_analysis] MAP: {len(chunks)} chunk(s)")
    for i, chunk_text in enumerate(chunks):
        lbl     = f"map {i+1}/{len(chunks)}"
        t_chunk = time.perf_counter()
        parsed  = None

        for attempt in range(1, MAP_JSON_RETRY_ATTEMPTS + 1):
            try:
                messages = _build_map_messages(chunk_text, retry=(attempt > 1))
                if attempt > 1:
                    logger.warning(f"[generate_analysis] [{lbl}] retry {attempt}")
                raw = await _run_inference(messages, f"{lbl}-a{attempt}")
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
                    + ("retrying" if attempt < MAP_JSON_RETRY_ATTEMPTS else "giving up")
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

    # ── SINGLE CHUNK — return directly ───────────────────────────────────
    if len(map_results) == 1:
        logger.info("[generate_analysis] single chunk — returning directly")
        result = map_results[0]

    else:
        # ── HIGHLIGHTS: merge in Python — no LLM, no data loss ────────────
        all_highlights = _merge_highlights(map_results)

        # ── SYNTHESIS: one LLM call for overview + summary only ───────────
        logger.info(f"[generate_analysis] SYNTHESIS: overview+summary from {valid_count} chunk(s)")
        t_synth      = time.perf_counter()
        synth_result = {"overview": "", "summary": ""}
        try:
            synth_messages = _build_synth_messages(
                [r for r in map_results if r.get("overview") or r.get("summary")]
            )
            raw_synth    = await _run_inference(synth_messages, "synthesis")
            parsed_synth = extract_json(raw_synth)

            ov = parsed_synth.get("overview", "")
            sm = parsed_synth.get("summary",  "")
            synth_result["overview"] = (ov if isinstance(ov, str) else str(ov)).strip()
            synth_result["summary"]  = (sm if isinstance(sm, str) else str(sm)).strip()

            logger.info(
                f"[generate_analysis] SYNTHESIS done ({time.perf_counter()-t_synth:.2f}s) "
                f"overview={'yes' if synth_result['overview'] else 'empty'} "
                f"summary={'yes' if synth_result['summary'] else 'empty'}"
            )
        except Exception as e:
            logger.error(f"[generate_analysis] SYNTHESIS failed: {e} — fallback to first chunk")

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
        result = dict(_EMPTY)

    logger.info(
        f"[generate_analysis] total={time.perf_counter()-t_pipeline:.2f}s "
        f"highlights={len(result.get('highlights', []))}"
    )
    return result


# ---------------------------------------------------------------------------
# run_llm — generic reusable runner (used by key-clause extraction & risk detection)
# ---------------------------------------------------------------------------
async def run_llm(
    text: str,
    system_prompt: str,
    max_input_tokens: int = 50000,
) -> str:
    """
    Generic reusable LLM runner.
    Accepts text + system_prompt, returns raw model output string.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": f"Document:\n----------------\n{text}\n----------------"},
    ]
    return await _run_inference(messages, "run_llm")


# ---------------------------------------------------------------------------
# generate_analysis_stream — async generator for SSE streaming endpoint
#
# Yields (event_type, payload) tuples as each pipeline stage completes:
#   ("chunk_start",    {"chunk": N, "total": N})
#   ("chunk_done",     {"chunk": N, "total": N, "overview": str,
#                       "new_highlights": [...], "all_highlights_so_far": [...]})
#   ("synthesis_start", {})
#   ("synthesis_done", {"overview": str, "summary": str})
#   ("done",           {})
# ---------------------------------------------------------------------------
async def generate_analysis_stream(merged_text: str):
    from s_main import extract_json

    _EMPTY = {"overview": "", "summary": "", "highlights": []}
    if not merged_text or not merged_text.strip():
        yield ("done", {})
        return

    t0     = time.perf_counter()
    chunks = split_by_tokens(merged_text)
    logger.info(f"[generate_analysis_stream] {len(chunks)} chunk(s)")

    map_results    = []
    all_highlights: list = []
    seen_keys:      set  = set()

    # ── MAP ──────────────────────────────────────────────────────────────
    for i, chunk_text in enumerate(chunks):
        lbl = f"map {i+1}/{len(chunks)}"
        yield ("chunk_start", {"chunk": i + 1, "total": len(chunks)})

        parsed = None
        for attempt in range(1, MAP_JSON_RETRY_ATTEMPTS + 1):
            try:
                messages = _build_map_messages(chunk_text, retry=(attempt > 1))
                if attempt > 1:
                    logger.warning(f"[stream] [{lbl}] retry {attempt}")
                raw = await _run_inference(messages, f"{lbl}-a{attempt}")
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
            "chunk":                 i + 1,
            "total":                 len(chunks),
            "overview":              parsed.get("overview", ""),
            "new_highlights":        new_highlights,
            "all_highlights_so_far": list(all_highlights),
        })

        logger.info(
            f"[stream] [{lbl}] done — "
            f"overview={'yes' if parsed.get('overview') else 'empty'} "
            f"new_highlights={len(new_highlights)} total={len(all_highlights)}"
        )

    valid_count = sum(1 for r in map_results if r.get("overview") or r.get("highlights"))
    logger.info(f"[stream] MAP complete — {len(map_results)} total, {valid_count} with content")

    # ── SYNTHESIS ────────────────────────────────────────────────────────
    synth_overview = ""
    synth_summary  = ""

    if len(map_results) == 1:
        synth_overview = map_results[0].get("overview", "")
        synth_summary  = map_results[0].get("summary",  "")
    else:
        yield ("synthesis_start", {})
        try:
            synth_messages = _build_synth_messages(
                [r for r in map_results if r.get("overview") or r.get("summary")]
            )
            raw_synth    = await _run_inference(synth_messages, "synthesis")
            parsed_synth = extract_json(raw_synth)
            ov = parsed_synth.get("overview", "")
            sm = parsed_synth.get("summary",  "")
            synth_overview = (ov if isinstance(ov, str) else str(ov)).strip()
            synth_summary  = (sm if isinstance(sm, str) else str(sm)).strip()
            logger.info(
                f"[stream] SYNTHESIS done — "
                f"overview={'yes' if synth_overview else 'empty'} "
                f"summary={'yes' if synth_summary else 'empty'}"
            )
        except Exception as e:
            logger.error(f"[stream] SYNTHESIS failed: {e} — fallback to first chunk")

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
