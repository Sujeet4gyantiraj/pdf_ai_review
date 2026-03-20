from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.responses import StreamingResponse
import os
import uuid
import json
import re
import time
import asyncio
import logging
import logging.config
from s_padf_utils import load_pdf, get_page_count, all_pages_blank
from s_ai_model import generate_analysis, generate_analysis_stream

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.config.dictConfig({
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        }
    },
    "handlers": {
        "console": {
            "class":     "logging.StreamHandler",
            "formatter": "standard",
            "level":     "INFO",
            "stream":    "ext://sys.stdout",
        },
        "file": {
            "class":     "logging.handlers.RotatingFileHandler",
            "formatter": "standard",
            "level":     "DEBUG",
            "filename":  "app.log",
            "maxBytes":  10 * 1024 * 1024,
            "backupCount": 5,
            "encoding":  "utf-8",
        },
    },
    "root": {"level": "DEBUG", "handlers": ["console", "file"]},
})

logger = logging.getLogger(__name__)

app = FastAPI()

UPLOAD_FOLDER = "temp"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
logger.info(f"Upload folder ready: {os.path.abspath(UPLOAD_FOLDER)}")

# ---------------------------------------------------------------------------
# GPU semaphore — one request on the GPU at a time.
# Others wait on the event loop (zero cost) not in a thread (deadlock risk).
# ---------------------------------------------------------------------------
_gpu_semaphore = asyncio.Semaphore(1)

MAX_PDF_PAGES     = None
_BLANK_PDF_RESULT = {"overview": "", "summary": "", "highlights": []}


# ---------------------------------------------------------------------------
# JSON extraction helpers
# ---------------------------------------------------------------------------

def _fix_json_string(raw: str) -> str:
    raw = raw.replace("```json", "").replace("```", "").strip()
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    raw = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', raw)
    raw = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', raw)
    return raw


def _extract_field_by_regex(text: str, field: str) -> str:
    pattern = rf'"{field}"\s*:\s*"(.*?)"(?=\s*[,}}])'
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else ""


def _extract_highlights_by_regex(text: str) -> list:
    match = re.search(r'"highlights"\s*:\s*\[(.*?)\]', text, re.DOTALL)
    if not match:
        match_open = re.search(r'"highlights"\s*:\s*\[(.*)', text, re.DOTALL)
        if not match_open:
            return []
        inner = match_open.group(1)
    else:
        inner = match.group(1)
    items = re.findall(r'"((?:[^"\\]|\\.)*)"', inner, re.DOTALL)
    return [i.strip() for i in items if i.strip()]


def _flatten_field(value, field_name: str = "") -> str:
    """
    Ensure a text field (overview or summary) is always a plain string.

    Mistral sometimes returns structured objects instead of strings when the
    source document is a form or table. For example:
        "overview": {"type": "Business Proposal", "subject": "...", "purpose": "..."}
    instead of:
        "overview": "Business Proposal — Custom Solutions... Presented by..."

    Strategy:
    - str  → return as-is
    - dict → join all string values in order, separated by ". "
    - list → join all string items in order, separated by ". "
    - anything else → empty string
    """
    if isinstance(value, str):
        return value.strip()

    if isinstance(value, dict):
        parts = []
        for v in value.values():
            if isinstance(v, str) and v.strip():
                parts.append(v.strip())
            elif isinstance(v, dict):
                # One level of nesting — flatten recursively
                parts.append(_flatten_field(v))
            elif isinstance(v, list):
                parts.append(_flatten_field(v))
        result = ". ".join(p for p in parts if p)
        if result:
            logger.debug(f"_flatten_field: flattened dict '{field_name}' → {len(result)} chars")
        return result

    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, str) and item.strip():
                parts.append(item.strip())
            elif isinstance(item, dict):
                parts.append(_flatten_field(item))
        return ". ".join(p for p in parts if p)

    return ""


def _postprocess_highlights(data: dict) -> dict:
    # Flatten overview and summary if they came back as objects/lists
    for field in ("overview", "summary"):
        if field in data and not isinstance(data[field], str):
            original_type = type(data[field]).__name__
            data[field] = _flatten_field(data[field], field)
            logger.warning(
                f"_postprocess_highlights: flattened '{field}' from {original_type} "
                f"to string ({len(data[field])} chars)"
            )

    if "highlights" in data and isinstance(data["highlights"], list):
        cleaned = []
        for item in data["highlights"]:
            # Guard: Mistral sometimes puts dicts instead of strings
            if isinstance(item, dict):
                text = ""
                for key in ("fact", "highlight", "text", "value", "item", "point"):
                    if isinstance(item.get(key), str):
                        text = item[key]
                        break
                if not text:
                    text = next((v for v in item.values() if isinstance(v, str)), "")
                if text:
                    cleaned.append(text.replace('"', '').strip())
                continue
            if not isinstance(item, str):
                continue
            parts = item.split('", "') if '", "' in item else [item]
            for p in parts:
                cleaned.append(p.replace('"', '').strip())
        data["highlights"] = [h for h in cleaned if h]
    return data


def _flatten_field(value) -> str:
    """
    Convert any value to a flat readable string.

    Mistral sometimes returns nested objects instead of strings, e.g.:
      "overview": {"type": "Business Proposal", "subject": "...", "purpose": "..."}
      "summary":  {"aboutUs": {"vision": "...", "mission": "..."}}

    This flattens them into a single readable sentence by joining all
    leaf string values found anywhere in the structure.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        parts = [_flatten_field(v) for v in value]
        return ". ".join(p for p in parts if p)
    if isinstance(value, dict):
        parts = []
        for k, v in value.items():
            flat = _flatten_field(v)
            if flat:
                # Include key as context label: "Type: Business Proposal"
                label = str(k).replace("_", " ").capitalize()
                parts.append(f"{label}: {flat}")
        return ". ".join(parts)
    return str(value).strip()


def _flatten_highlights(value) -> list:
    """
    Convert highlights field to a flat list of strings regardless of
    what structure Mistral returned.

    Handles:
      ["string", ...]                   → as-is
      [{"fact": "..."}, ...]            → extract string values
      {"key": "value", ...}             → convert each key-value to a string
      "string"                          → wrap in list
    """
    if not value:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, dict):
        # e.g. {"personal": "Name, DOB", "address": "Street, City"}
        items = []
        for k, v in value.items():
            flat = _flatten_field(v)
            if flat:
                label = str(k).replace("_", " ").capitalize()
                items.append(f"{label}: {flat}")
        return items
    if isinstance(value, list):
        result = []
        for item in value:
            if isinstance(item, str) and item.strip():
                result.append(item.strip())
            elif isinstance(item, dict):
                flat = _flatten_field(item)
                if flat:
                    result.append(flat)
            elif item is not None:
                s = str(item).strip()
                if s:
                    result.append(s)
        return result
    return []


def _normalize_parsed(data, label: str = "") -> dict:
    """
    Ensure json.loads() output is always a dict with flat string fields.

    Handles:
      - Top-level list instead of object  → merge items
      - Nested objects in overview/summary → flatten to string
      - Nested objects/lists in highlights → flatten to list of strings
    """
    empty = {"overview": "", "summary": "", "highlights": []}

    if isinstance(data, list):
        if not data:
            return empty
        if isinstance(data[0], dict):
            merged: dict = {"overview": "", "summary": "", "highlights": []}
            for item in data:
                if not isinstance(item, dict):
                    continue
                if not merged["overview"] and item.get("overview"):
                    merged["overview"] = item["overview"]
                if not merged["summary"] and item.get("summary"):
                    merged["summary"] = item["summary"]
                merged["highlights"].extend(item.get("highlights") or [])
            logger.warning(f"extract_json {label}: got list of dicts — merged {len(data)} item(s)")
            data = merged
        elif isinstance(data[0], str):
            return {"overview": "", "summary": "", "highlights": [s for s in data if s]}
        else:
            logger.error(f"extract_json {label}: unexpected list type")
            return empty

    if not isinstance(data, dict):
        logger.error(f"extract_json {label}: unexpected type {type(data).__name__}")
        return empty

    # Flatten overview — must be a plain string
    overview = data.get("overview", "")
    if not isinstance(overview, str):
        logger.warning(f"extract_json {label}: overview is {type(overview).__name__} — flattening")
        overview = _flatten_field(overview)

    # Flatten summary — must be a plain string
    summary = data.get("summary", "")
    if not isinstance(summary, str):
        logger.warning(f"extract_json {label}: summary is {type(summary).__name__} — flattening")
        summary = _flatten_field(summary)

    # Flatten highlights — must be a list of strings
    raw_highlights = data.get("highlights", [])
    if not isinstance(raw_highlights, list) or (
        raw_highlights and not isinstance(raw_highlights[0], str)
    ):
        logger.warning(f"extract_json {label}: highlights has non-string items — flattening")
        highlights = _flatten_highlights(raw_highlights)
    else:
        highlights = raw_highlights

    return {
        "overview":   overview.strip(),
        "summary":    summary.strip(),
        "highlights": highlights,
    }


def _recover_truncated_json(text: str) -> dict | None:
    result = {"overview": "", "summary": "", "highlights": []}
    ov_match = re.search(r'"overview"\s*:\s*"((?:[^"\\]|\\.)*)"', text, re.DOTALL)
    if ov_match:
        result["overview"] = ov_match.group(1).strip()
    sm_match = re.search(r'"summary"\s*:\s*"((?:[^"\\]|\\.)*)"', text, re.DOTALL)
    if sm_match:
        result["summary"] = sm_match.group(1).strip()
    result["highlights"] = _extract_highlights_by_regex(text)
    if result["overview"] or result["summary"] or result["highlights"]:
        logger.info(
            f"_recover_truncated_json: recovered "
            f"overview={'yes' if result['overview'] else 'no'} "
            f"summary={'yes' if result['summary'] else 'no'} "
            f"highlights={len(result['highlights'])}"
        )
        return result
    return None


def extract_json(text: str) -> dict:
    """
    Robustly extract JSON from Mistral output.
    Strategy 1 — direct parse
    Strategy 2 — isolate { } block + quote-escape repair
    Strategy 3 — truncation recovery
    Strategy 4 — per-field regex (last resort)
    """
    empty = {"overview": "", "summary": "", "highlights": []}
    if not text or not text.strip():
        return empty

    cleaned = _fix_json_string(text)

    try:
        data = json.loads(cleaned)
        return _postprocess_highlights(_normalize_parsed(data, "strategy-1"))
    except json.JSONDecodeError:
        pass

    brace_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
    if brace_match:
        candidate = brace_match.group()
        try:
            data = json.loads(candidate)
            return _postprocess_highlights(_normalize_parsed(data, "strategy-2"))
        except json.JSONDecodeError as e:
            logger.warning(f"extract_json: strategy 2 failed ({e}), trying repair")
            repaired = re.sub(r'(?<=[^\\])"(?=[^,\]}\n:}{\[])', r'\\"', candidate)
            try:
                data = json.loads(repaired)
                return _postprocess_highlights(_normalize_parsed(data, "strategy-2-repair"))
            except json.JSONDecodeError as e2:
                logger.warning(f"extract_json: strategy 2 repair failed — {e2}")
    else:
        logger.warning("extract_json: no brace block found in output")

    recovered = _recover_truncated_json(cleaned)
    if recovered:
        logger.info("extract_json: strategy 3 (truncation recovery) succeeded")
        return _postprocess_highlights(recovered)

    logger.error("extract_json: all strategies failed — falling back to regex")
    overview   = _extract_field_by_regex(cleaned, "overview")
    summary    = _extract_field_by_regex(cleaned, "summary")
    highlights = _extract_highlights_by_regex(cleaned)
    if overview or summary or highlights:
        return {"overview": overview, "summary": summary, "highlights": highlights}

    logger.error("extract_json: all strategies exhausted — returning empty")
    return empty


# ---------------------------------------------------------------------------
# Response shape enforcement
#
# Guarantees the final response is ALWAYS exactly:
#   {"overview": str, "summary": str, "highlights": [str, str, ...]}
#
# Called as the last step before every return in both endpoints.
# ---------------------------------------------------------------------------

def _to_str(value) -> str:
    """Coerce any value to a plain string."""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        parts = [_to_str(v) for v in value.values()]
        return ". ".join(p for p in parts if p)
    if isinstance(value, list):
        parts = [_to_str(item) for item in value]
        return ". ".join(p for p in parts if p)
    return ""


def _to_str_list(value) -> list:
    """Coerce any value to a list of plain strings."""
    if isinstance(value, list):
        result = []
        for item in value:
            if isinstance(item, str):
                s = item.strip()
                if s:
                    result.append(s)
            elif isinstance(item, dict):
                s = _to_str(item)
                if s:
                    result.append(s)
        return result
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def enforce_response_shape(data: dict, analysis_type: int = 0) -> dict:
    """
    Final enforcement: output is always exactly the standard shape.

    Standard (analysis_type=0):
        {"overview": str, "summary": str, "highlights": [str, ...]}

    Filtered:
        analysis_type=1 → {"overview": str}
        analysis_type=2 → {"summary": str}
        analysis_type=3 → {"highlights": [str, ...]}

    Any field that is not the correct type is coerced before returning.
    Metadata keys (truncated, blank_pdf, pages_analysed, total_pages) are
    passed through unchanged.
    """
    overview   = _to_str(data.get("overview",   ""))
    summary    = _to_str(data.get("summary",    ""))
    highlights = _to_str_list(data.get("highlights", []))

    if not isinstance(data.get("overview"), str):
        logger.warning(
            f"enforce_response_shape: overview was {type(data.get('overview')).__name__} "
            f"— coerced to string ({len(overview)} chars)"
        )
    if not isinstance(data.get("summary"), str):
        logger.warning(
            f"enforce_response_shape: summary was {type(data.get('summary')).__name__} "
            f"— coerced to string ({len(summary)} chars)"
        )

    result: dict = {}
    if analysis_type in (0, 1):
        result["overview"]   = overview
    if analysis_type in (0, 2):
        result["summary"]    = summary
    if analysis_type in (0, 3):
        result["highlights"] = highlights

    # Pass through metadata flags
    for key in ("truncated", "blank_pdf", "pages_analysed", "total_pages"):
        if key in data:
            result[key] = data[key]

    return result


# ---------------------------------------------------------------------------
# Shared pipeline logic (used by both /analyze and /analyze/stream)
# ---------------------------------------------------------------------------

async def _run_pipeline(file: UploadFile, file_path: str, request_id: str):
    """
    Save → page count → extract → blank check → merge.
    Returns (merged_text, total_pages, pages_to_read, was_truncated)
    or raises HTTPException on failure.
    Caller is responsible for deleting file_path in a finally block.
    """
    # Step 1 — save
    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)
    logger.info(f"[{request_id}] saved {len(content):,} bytes")

    # Step 2 — page count
    total_pages   = get_page_count(file_path)
    pages_to_read = total_pages if MAX_PDF_PAGES is None else min(total_pages, MAX_PDF_PAGES)
    was_truncated = MAX_PDF_PAGES is not None and total_pages > MAX_PDF_PAGES
    logger.info(f"[{request_id}] pages={total_pages} analysing={pages_to_read}")

    # Step 3 — extract
    try:
        pages = load_pdf(file_path, max_pages=pages_to_read)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return pages, total_pages, pages_to_read, was_truncated


# ---------------------------------------------------------------------------
# SSE helper
# ---------------------------------------------------------------------------

def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


# ---------------------------------------------------------------------------
# POST /analyze — synchronous response
# ---------------------------------------------------------------------------

@app.post("/analyze")
async def analyze_pdf(
    file: UploadFile = File(...),
    analysis_type: int = Query(0, ge=0, le=3,
        description="0=all, 1=overview, 2=summary, 3=highlights")
):
    """
    Analyse a PDF and return the full result as a single JSON response.
    Use /analyze/stream for real-time streaming progress.
    """
    request_id = str(uuid.uuid4())[:8]
    t_start    = time.perf_counter()

    logger.info(f"[{request_id}] ── NEW REQUEST ──────────────────────────────")
    logger.info(f"[{request_id}] filename='{file.filename}' analysis_type={analysis_type}")

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    file_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}.pdf")
    was_truncated = False
    pages_to_read = 0
    total_pages   = 0

    try:
        pages, total_pages, pages_to_read, was_truncated = await _run_pipeline(
            file, file_path, request_id
        )

        if all_pages_blank(pages):
            logger.warning(f"[{request_id}] blank PDF — skipping inference")
            result = dict(_BLANK_PDF_RESULT)
            result["blank_pdf"] = True
            if was_truncated:
                result.update(truncated=True, pages_analysed=pages_to_read, total_pages=total_pages)
            return enforce_response_shape(result, analysis_type)

        merged_text = "\n\n".join(p.page_content for p in pages)
        logger.info(f"[{request_id}] merged {len(merged_text):,} chars")

        t_wait = time.perf_counter()
        async with _gpu_semaphore:
            wait_time = time.perf_counter() - t_wait
            if wait_time > 0.1:
                logger.info(f"[{request_id}] waited {wait_time:.1f}s for GPU semaphore")
            logger.info(f"[{request_id}] running inference")
            t_infer      = time.perf_counter()
            final_output = await generate_analysis(merged_text)
            logger.info(f"[{request_id}] inference done ({time.perf_counter()-t_infer:.2f}s)")

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"[{request_id}] unhandled error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

    elapsed = time.perf_counter() - t_start
    logger.info(f"[{request_id}] ── COMPLETE — {elapsed:.2f}s ──────")

    if was_truncated:
        final_output.update(truncated=True, pages_analysed=pages_to_read, total_pages=total_pages)

    return enforce_response_shape(final_output, analysis_type)


# ---------------------------------------------------------------------------
# POST /analyze/stream — Server-Sent Events streaming response
#
# Event sequence:
#   status    {"step":"saving"|"extracting"|"inference"|"chunk_done"|"synthesis"|"queued"}
#   overview  {"text":"..."}        — sent after chunk 1, updated after synthesis
#   summary   {"text":"..."}        — sent after synthesis completes
#   highlight {"text":"...","index":N} — one per highlight as each chunk finishes
#   done      {"total_time":N,"total_highlights":N,"pages":N}
#   error     {"message":"..."}
#
# Client: use fetch() POST with ReadableStream — EventSource only supports GET.
# nginx: set proxy_buffering off for this location.
# ---------------------------------------------------------------------------

@app.post("/analyze/stream")
async def analyze_pdf_stream(
    file: UploadFile = File(...),
    analysis_type: int = Query(0, ge=0, le=3,
        description="0=all, 1=overview, 2=summary, 3=highlights")
):
    """
    Analyse a PDF and stream results in real time using Server-Sent Events.
    Highlights are sent one-by-one as each chunk finishes (~8s intervals).
    Overview arrives after chunk 1. Summary arrives after synthesis (~8s after last chunk).
    """
    request_id = str(uuid.uuid4())[:8]
    t_start    = time.perf_counter()

    logger.info(f"[{request_id}] ── NEW STREAM REQUEST ───────────────────────")
    logger.info(f"[{request_id}] filename='{file.filename}' analysis_type={analysis_type}")

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        async def _err():
            yield _sse("error", {"message": "Only PDF files are accepted."})
        return StreamingResponse(_err(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    file_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}.pdf")

    async def _generate():
        was_truncated = False
        pages_to_read = 0
        total_pages   = 0
        final_highlights = []

        try:
            # ── Save + extract ────────────────────────────────────────────
            yield _sse("status", {"step": "saving", "message": "Saving uploaded file..."})
            try:
                pages, total_pages, pages_to_read, was_truncated = await _run_pipeline(
                    file, file_path, request_id
                )
            except HTTPException as e:
                yield _sse("error", {"message": e.detail})
                return

            yield _sse("status", {
                "step":    "extracting",
                "message": f"Extracting text from {pages_to_read} page(s)...",
                "pages":   pages_to_read,
            })

            # ── Blank PDF ─────────────────────────────────────────────────
            if all_pages_blank(pages):
                logger.warning(f"[{request_id}] blank PDF")
                yield _sse("status", {"step": "blank", "message": "PDF has no extractable content."})
                if analysis_type in (0, 1): yield _sse("overview",  {"text": ""})
                if analysis_type in (0, 2): yield _sse("summary",   {"text": ""})
                yield _sse("done", {
                    "total_time": round(time.perf_counter() - t_start, 2),
                    "total_highlights": 0,
                    "pages": total_pages,
                    "blank_pdf": True,
                })
                return

            merged_text = "\n\n".join(p.page_content for p in pages)
            logger.info(f"[{request_id}] merged {len(merged_text):,} chars")

            # ── Wait for GPU ──────────────────────────────────────────────
            t_wait = time.perf_counter()
            async with _gpu_semaphore:
                wait_time = time.perf_counter() - t_wait
                if wait_time > 0.1:
                    logger.info(f"[{request_id}] waited {wait_time:.1f}s for GPU semaphore")
                    yield _sse("status", {
                        "step":    "queued",
                        "message": f"Waiting for GPU ({wait_time:.0f}s queue time)...",
                    })

                # ── Stream from generate_analysis_stream ──────────────────
                overview_sent    = False
                highlight_index  = 0
                final_overview   = ""
                final_summary    = ""

                async for event_type, payload in generate_analysis_stream(merged_text):

                    if event_type == "chunk_start":
                        yield _sse("status", {
                            "step":    "inference",
                            "message": f"Analysing chunk {payload['chunk']} of {payload['total']}...",
                            "chunk":   payload["chunk"],
                            "total":   payload["total"],
                        })

                    elif event_type == "chunk_done":
                        # Send overview from first chunk immediately (enforce string)
                        if not overview_sent and payload.get("overview"):
                            chunk_overview = _to_str(payload["overview"])
                            if analysis_type in (0, 1):
                                yield _sse("overview", {"text": chunk_overview})
                            final_overview = chunk_overview
                            overview_sent  = True

                        # Stream new highlights as they arrive (enforce strings)
                        if analysis_type in (0, 3):
                            for h in payload.get("new_highlights", []):
                                h_str = _to_str(h) if not isinstance(h, str) else h.strip()
                                if h_str:
                                    yield _sse("highlight", {"text": h_str, "index": highlight_index})
                                    highlight_index += 1

                        final_highlights = payload.get("all_highlights_so_far", final_highlights)

                        yield _sse("status", {
                            "step":              "chunk_done",
                            "chunk":             payload["chunk"],
                            "total":             payload["total"],
                            "highlights_so_far": len(final_highlights),
                        })

                    elif event_type == "synthesis_start":
                        yield _sse("status", {
                            "step":    "synthesis",
                            "message": "Writing overview and summary...",
                        })

                    elif event_type == "synthesis_done":
                        # Enforce plain strings before sending to client
                        final_overview = _to_str(payload.get("overview", final_overview))
                        final_summary  = _to_str(payload.get("summary",  ""))

                        if analysis_type in (0, 1):
                            yield _sse("overview", {"text": final_overview})
                        if analysis_type in (0, 2):
                            yield _sse("summary",  {"text": final_summary})

                    elif event_type == "done":
                        pass  # handled below

        except Exception as e:
            logger.exception(f"[{request_id}] stream error: {e}")
            yield _sse("error", {"message": "Internal server error during PDF analysis."})
            return
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)

        elapsed = time.perf_counter() - t_start
        logger.info(f"[{request_id}] ── STREAM COMPLETE — {elapsed:.2f}s ──────")

        done_payload = {
            "total_time":       round(elapsed, 2),
            "total_highlights": len(final_highlights),
            "pages":            total_pages,
        }
        if was_truncated:
            done_payload.update(truncated=True, pages_analysed=pages_to_read, total_pages=total_pages)

        yield _sse("done", done_payload)

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",
            "Connection":        "keep-alive",
        },
    )