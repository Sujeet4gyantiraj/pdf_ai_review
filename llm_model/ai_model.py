import asyncio
import os
import time
import logging
import tiktoken
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()


from utils.json_utils import extract_json


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY is not set in environment variables. Please set it in your .env file.")

# Models that only accept default temperature (1) and reject 0.0
_FIXED_TEMPERATURE_MODELS = {"gpt-5-nano", "gpt-4o-mini", "o1", "o1-mini", "o3-mini", "o3"}

# Models that use max_completion_tokens instead of max_tokens
_MAX_COMPLETION_TOKENS_MODELS = {"gpt-4o-mini", "gpt-5-nano", "o1", "o1-mini", "o3-mini", "o3"}

# Token chunking
# 120,000 fits comfortably within gpt-4o / gpt-4o-mini 128k context window
# leaving ~8k tokens headroom for system prompt + output
TOKEN_CHUNK_SIZE    = 120000
TOKEN_CHUNK_OVERLAP = 500

# Max output tokens per API call
# 4096 is enough for a detailed summary + 20 highlights
MAX_OUTPUT_TOKENS = 4096

MAP_JSON_RETRY_ATTEMPTS = 2

# ---------------------------------------------------------------------------
# Module-level semaphore — controls max concurrent OpenAI API calls
# across ALL requests hitting this server simultaneously.
# Reduce to 2 if you hit 429 rate-limit errors.
# ---------------------------------------------------------------------------
_MAP_CONCURRENCY  = 3
_MAP_SEMAPHORE: asyncio.Semaphore | None = None


def _get_semaphore() -> asyncio.Semaphore:
    global _MAP_SEMAPHORE
    if _MAP_SEMAPHORE is None:
        _MAP_SEMAPHORE = asyncio.Semaphore(_MAP_CONCURRENCY)
        logger.info(f"[ai_model] MAP semaphore created (concurrency={_MAP_CONCURRENCY})")
    return _MAP_SEMAPHORE


_client   = AsyncOpenAI(api_key=OPENAI_API_KEY)
_encoding = tiktoken.encoding_for_model("gpt-4o")


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
# Prompts
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
# Highlights merge
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
# Core inference — returns (content, input_tokens, output_tokens)
# ---------------------------------------------------------------------------
async def _run_inference(messages: list[dict], label: str = "") -> tuple[str, int, int]:
    """
    Run one OpenAI API call.
    Returns (response_text, input_tokens, output_tokens).

    Handles model-specific parameter differences:
      - temperature: not supported by o1/o3/gpt-5-nano family
      - max_tokens vs max_completion_tokens: depends on model
    """
    tag = f"[{label}] " if label else ""
    t0  = time.perf_counter()
    try:
        api_kwargs = {
            "model":           MODEL_NAME,
            "messages":        messages,
            "response_format": {"type": "json_object"},
        }

        # temperature: only supported on standard GPT models
        if MODEL_NAME not in _FIXED_TEMPERATURE_MODELS:
            api_kwargs["temperature"] = 0.0

        # max output tokens: newer models use max_completion_tokens
        if MODEL_NAME in _MAX_COMPLETION_TOKENS_MODELS:
            api_kwargs["max_completion_tokens"] = MAX_OUTPUT_TOKENS
        else:
            api_kwargs["max_tokens"] = MAX_OUTPUT_TOKENS

        response      = await _client.chat.completions.create(**api_kwargs)
        elapsed       = time.perf_counter() - t0
        content       = response.choices[0].message.content or ""
        input_tokens  = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        logger.info(
            f"{tag}in={input_tokens} out={output_tokens} "
            f"in {elapsed:.2f}s"
        )
        return content, input_tokens, output_tokens
    except Exception as e:
        logger.exception(f"{tag}OpenAI API call failed: {e}")
        raise


# ---------------------------------------------------------------------------
# _run_map_chunk — one map chunk, returns (parsed_dict, input_tokens, output_tokens)
# ---------------------------------------------------------------------------
async def _run_map_chunk(i: int, total: int, chunk_text: str) -> tuple[dict, int, int]:
    _EMPTY = {"overview": "", "summary": "", "highlights": []}
    sem    = _get_semaphore()
    lbl    = f"map {i+1}/{total}"

    async with sem:
        t_chunk    = time.perf_counter()
        parsed     = None
        in_tokens  = 0
        out_tokens = 0

        for attempt in range(1, MAP_JSON_RETRY_ATTEMPTS + 1):
            try:
                messages = _build_map_messages(chunk_text, retry=(attempt > 1))
                if attempt > 1:
                    logger.warning(f"[generate_analysis] [{lbl}] retry {attempt}")
                raw, i_tok, o_tok = await _run_inference(messages, f"{lbl}-a{attempt}")
                in_tokens  += i_tok
                out_tokens += o_tok
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

        logger.info(
            f"[generate_analysis] [{lbl}] done ({time.perf_counter()-t_chunk:.2f}s) "
            f"overview={'yes' if parsed.get('overview') else 'empty'} "
            f"highlights={len(parsed.get('highlights', []))}"
        )
        return parsed, in_tokens, out_tokens


# ---------------------------------------------------------------------------
# generate_analysis — returns (result_dict, input_tokens, output_tokens)
# ---------------------------------------------------------------------------
async def generate_analysis(merged_text: str) -> tuple[dict, int, int]:
    """
    Full map+synthesise pipeline.
    Returns (result, total_input_tokens, total_output_tokens).
    """
    _EMPTY = {"overview": "", "summary": "", "highlights": []}
    if not merged_text or not merged_text.strip():
        return dict(_EMPTY), 0, 0

    t0     = time.perf_counter()
    chunks = split_by_tokens(merged_text)
    logger.info(f"[generate_analysis] {len(chunks)} chunk(s) in {time.perf_counter()-t0:.3f}s")

    t_pipeline    = time.perf_counter()
    total_in_tok  = 0
    total_out_tok = 0

    # ── MAP ───────────────────────────────────────────────────────────────
    logger.info(
        f"[generate_analysis] MAP: {len(chunks)} chunk(s) "
        f"(parallel, concurrency={_MAP_CONCURRENCY})"
    )
    map_tasks   = [_run_map_chunk(i, len(chunks), ct) for i, ct in enumerate(chunks)]
    raw_results = list(await asyncio.gather(*map_tasks))

    map_results = []
    for parsed, i_tok, o_tok in raw_results:
        map_results.append(parsed)
        total_in_tok  += i_tok
        total_out_tok += o_tok

    valid_count = sum(1 for r in map_results if r.get("overview") or r.get("highlights"))
    logger.info(f"[generate_analysis] MAP complete — {len(map_results)} total, {valid_count} with content")

    # ── SINGLE CHUNK — return directly ───────────────────────────────────
    if len(map_results) == 1:
        logger.info("[generate_analysis] single chunk — returning directly")
        result = map_results[0]

    else:
        # ── HIGHLIGHTS merge ───────────────────────────────────────────────
        all_highlights = _merge_highlights(map_results)

        # ── SYNTHESIS ─────────────────────────────────────────────────────
        logger.info(f"[generate_analysis] SYNTHESIS: overview+summary from {valid_count} chunk(s)")
        t_synth      = time.perf_counter()
        synth_result = {"overview": "", "summary": ""}
        try:
            synth_messages = _build_synth_messages(
                [r for r in map_results if r.get("overview") or r.get("summary")]
            )
            raw_synth, s_in, s_out = await _run_inference(synth_messages, "synthesis")
            total_in_tok  += s_in
            total_out_tok += s_out

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
        f"tokens={total_in_tok}in/{total_out_tok}out "
        f"highlights={len(result.get('highlights', []))}"
    )
    return result, total_in_tok, total_out_tok


# ---------------------------------------------------------------------------
# run_llm — generic reusable runner
# ---------------------------------------------------------------------------
async def run_llm(
    text: str,
    system_prompt: str,
    max_input_tokens: int = 50000,
) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": f"Document:\n----------------\n{text}\n----------------"},
    ]
    content, _, _ = await _run_inference(messages, "run_llm")
    return content


# ---------------------------------------------------------------------------
# _run_inference_stream — streaming OpenAI call, yields token deltas
#
# Yields:  ("delta", str)          — one raw token string at a time
#          ("done",  (int, int))    — (input_tokens, output_tokens) at end
#
# NOTE: response_format="json_object" is incompatible with stream=True on
# some older SDK versions; we omit it here and rely on prompt instructions
# + extract_json() post-processing instead (same strategy as non-stream).
# ---------------------------------------------------------------------------
async def _run_inference_stream(messages: list[dict], label: str = ""):
    tag = f"[{label}] " if label else ""
    t0  = time.perf_counter()

    api_kwargs: dict = {
        "model":    MODEL_NAME,
        "messages": messages,
        "stream":   True,
        # request usage stats in the final chunk (supported since openai>=1.26)
        "stream_options": {"include_usage": True},
    }

    if MODEL_NAME not in _FIXED_TEMPERATURE_MODELS:
        api_kwargs["temperature"] = 0.0

    if MODEL_NAME in _MAX_COMPLETION_TOKENS_MODELS:
        api_kwargs["max_completion_tokens"] = MAX_OUTPUT_TOKENS
    else:
        api_kwargs["max_tokens"] = MAX_OUTPUT_TOKENS

    input_tokens  = 0
    output_tokens = 0

    try:
        async with await _client.chat.completions.create(**api_kwargs) as stream:
            async for chunk in stream:
                # Token deltas arrive in choices[0].delta.content
                if chunk.choices and chunk.choices[0].delta.content:
                    yield ("delta", chunk.choices[0].delta.content)

                # Usage arrives in the final chunk (choices is empty)
                if chunk.usage:
                    input_tokens  = chunk.usage.prompt_tokens
                    output_tokens = chunk.usage.completion_tokens

        logger.info(
            f"{tag}stream done — in={input_tokens} out={output_tokens} "
            f"({time.perf_counter()-t0:.2f}s)"
        )
    except Exception as e:
        logger.exception(f"{tag}streaming OpenAI call failed: {e}")
        raise

    yield ("done", (input_tokens, output_tokens))


# ---------------------------------------------------------------------------
# _run_map_chunk_stream — stream one map chunk, yielding deltas then result
#
# Yields:  ("delta",      str)   — raw token for the client to forward
#          ("chunk_done", dict)  — parsed result after the chunk completes
# ---------------------------------------------------------------------------
async def _run_map_chunk_stream(i: int, total: int, chunk_text: str):
    _EMPTY = {"overview": "", "summary": "", "highlights": []}
    sem    = _get_semaphore()
    lbl    = f"map {i+1}/{total}"

    async with sem:
        t_chunk    = time.perf_counter()
        in_tokens  = 0
        out_tokens = 0
        parsed     = None

        for attempt in range(1, MAP_JSON_RETRY_ATTEMPTS + 1):
            messages   = _build_map_messages(chunk_text, retry=(attempt > 1))
            raw_buffer = ""

            if attempt > 1:
                logger.warning(f"[stream] [{lbl}] retry {attempt}")

            try:
                async for event_type, payload in _run_inference_stream(messages, f"{lbl}-a{attempt}"):
                    if event_type == "delta":
                        raw_buffer += payload
                        yield ("delta", payload)          # ← live tokens to client
                    elif event_type == "done":
                        in_tokens  += payload[0]
                        out_tokens += payload[1]
            except Exception as e:
                logger.error(f"[stream] [{lbl}] inference error: {e}")
                break

            candidate = extract_json(raw_buffer)
            if candidate.get("overview") or candidate.get("highlights"):
                parsed = candidate
                break
            else:
                logger.warning(
                    f"[stream] [{lbl}] attempt {attempt}: no usable JSON — "
                    + ("retrying" if attempt < MAP_JSON_RETRY_ATTEMPTS else "giving up")
                )

        if not isinstance(parsed, dict) or not (parsed.get("overview") or parsed.get("highlights")):
            parsed = dict(_EMPTY)

        logger.info(
            f"[stream] [{lbl}] done ({time.perf_counter()-t_chunk:.2f}s) "
            f"overview={'yes' if parsed.get('overview') else 'empty'} "
            f"highlights={len(parsed.get('highlights', []))}"
        )
        yield ("chunk_done", {**parsed, "_in_tokens": in_tokens, "_out_tokens": out_tokens})


# ---------------------------------------------------------------------------
# generate_analysis_stream — true SSE streaming
#
# Key changes vs previous version:
#   1. Chunks are processed SEQUENTIALLY so the client receives events as
#      each one finishes rather than waiting for all of them.
#   2. _run_map_chunk_stream uses stream=True, so raw token deltas flow to
#      the client in real time during each chunk's inference call.
#   3. The synthesis step also streams tokens live.
# ---------------------------------------------------------------------------
async def generate_analysis_stream(merged_text: str):
    if not merged_text or not merged_text.strip():
        yield ("done", {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
        return

    t0            = time.perf_counter()
    chunks        = split_by_tokens(merged_text)
    total_in_tok  = 0
    total_out_tok = 0
    map_results:  list = []
    all_highlights: list = []
    seen_keys:      set  = set()

    logger.info(f"[generate_analysis_stream] {len(chunks)} chunk(s) — sequential streaming")

    # ── MAP — one chunk at a time so events reach the client immediately ──
    for i, chunk_text in enumerate(chunks):
        yield ("chunk_start", {"chunk": i + 1, "total": len(chunks)})

        parsed = None
        async for event_type, payload in _run_map_chunk_stream(i, len(chunks), chunk_text):

            if event_type == "delta":
                # Forward raw LLM tokens so the client can show live typing
                yield ("token", {"chunk": i + 1, "delta": payload})

            elif event_type == "chunk_done":
                parsed         = payload
                total_in_tok  += payload.pop("_in_tokens",  0)
                total_out_tok += payload.pop("_out_tokens", 0)

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

                map_results.append(parsed)

                yield ("chunk_done", {
                    "chunk":                 i + 1,
                    "total":                 len(chunks),
                    "overview":              parsed.get("overview", ""),
                    "new_highlights":        new_highlights,
                    "all_highlights_so_far": list(all_highlights),
                })
                logger.info(
                    f"[stream] [map {i+1}/{len(chunks)}] yielded — "
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
        yield ("synthesis_done", {"overview": synth_overview, "summary": synth_summary})

    else:
        yield ("synthesis_start", {})
        synth_buffer = ""
        try:
            synth_messages = _build_synth_messages(
                [r for r in map_results if r.get("overview") or r.get("summary")]
            )
            async for event_type, payload in _run_inference_stream(synth_messages, "synthesis"):
                if event_type == "delta":
                    synth_buffer += payload
                    yield ("token", {"chunk": "synthesis", "delta": payload})
                elif event_type == "done":
                    total_in_tok  += payload[0]
                    total_out_tok += payload[1]

            parsed_synth = extract_json(synth_buffer)
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
        f"tokens={total_in_tok}in/{total_out_tok}out "
        f"highlights={len(all_highlights)}"
    )
    yield ("done", {
        "input_tokens":  total_in_tok,
        "output_tokens": total_out_tok,
        "total_tokens":  total_in_tok + total_out_tok,
    })
