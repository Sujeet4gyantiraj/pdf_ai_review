from fastapi import APIRouter, UploadFile, File, Query, HTTPException
from fastapi.responses import StreamingResponse
import os
import io
import uuid
import json
import re
import time
import asyncio
import logging

from s_pdf_utils import load_pdf, get_page_count, all_pages_blank
from s_ai_model import generate_analysis, generate_analysis_stream
from t_key_clause_extraction import classify_document, DOCUMENT_HANDLERS, extract_text_from_upload
from t_risk_detection import analyze_document_risks

logger = logging.getLogger(__name__)

router = APIRouter()

UPLOAD_FOLDER     = "temp"
MAX_PDF_PAGES     = None
_BLANK_PDF_RESULT = {"overview": "", "summary": "", "highlights": []}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


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


def _postprocess_highlights(data: dict) -> dict:
    if "highlights" in data and isinstance(data["highlights"], list):
        cleaned = []
        for item in data["highlights"]:
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
                label = str(k).replace("_", " ").capitalize()
                parts.append(f"{label}: {flat}")
        return ". ".join(parts)
    return str(value).strip()


def _flatten_highlights(value) -> list:
    if not value:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, dict):
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
            return empty

    if not isinstance(data, dict):
        logger.error(f"extract_json {label}: unexpected type {type(data).__name__}")
        return empty

    overview = data.get("overview", "")
    if not isinstance(overview, str):
        logger.warning(f"extract_json {label}: overview is {type(overview).__name__} — flattening")
        overview = _flatten_field(overview)

    summary = data.get("summary", "")
    if not isinstance(summary, str):
        logger.warning(f"extract_json {label}: summary is {type(summary).__name__} — flattening")
        summary = _flatten_field(summary)

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
    Robustly extract JSON from AI output.
    Strategy 1 — direct parse after cleaning
    Strategy 2 — isolate { } block + quote-escape repair
    Strategy 3 — truncation recovery
    Strategy 4 — per-field regex (last resort)
    """
    empty = {"overview": "", "summary": "", "highlights": []}
    if not text or not text.strip():
        logger.warning("extract_json: received empty text")
        return empty

    cleaned = _fix_json_string(text)

    try:
        data = json.loads(cleaned)
        logger.debug("extract_json: strategy 1 succeeded")
        return _postprocess_highlights(_normalize_parsed(data, "strategy-1"))
    except json.JSONDecodeError as e:
        logger.debug(f"extract_json: strategy 1 failed — {e}")

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
# SSE helper
# ---------------------------------------------------------------------------

def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


# ---------------------------------------------------------------------------
# POST /analyze — full JSON response
# ---------------------------------------------------------------------------

@router.post("/analyze")
async def analyze_pdf(
    file: UploadFile = File(...),
    analysis_type: int = Query(0, ge=0, le=3,
        description="0=all, 1=overview, 2=summary, 3=highlights")
):
    """
    Analyse a PDF and return the full result as a single JSON response.
    Use POST /analyze/stream for real-time streaming progress.
    """
    request_id = str(uuid.uuid4())[:8]
    t_start    = time.perf_counter()

    logger.info(f"[{request_id}] ── NEW REQUEST ──────────────────────────────")
    logger.info(f"[{request_id}] filename='{file.filename}' analysis_type={analysis_type}")

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    safe_name     = f"{uuid.uuid4()}.pdf"
    file_path     = os.path.join(UPLOAD_FOLDER, safe_name)
    was_truncated = False
    pages_to_read = 0
    total_pages   = 0

    try:
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        logger.info(f"[{request_id}] Step 1/5 — saved {len(content):,} bytes → '{safe_name}'")

        total_pages   = get_page_count(file_path)
        pages_to_read = total_pages if MAX_PDF_PAGES is None else min(total_pages, MAX_PDF_PAGES)
        was_truncated = MAX_PDF_PAGES is not None and total_pages > MAX_PDF_PAGES
        logger.info(
            f"[{request_id}] Step 2/5 — pages={total_pages} analysing={pages_to_read} "
            f"{'(TRUNCATED)' if was_truncated else '(all pages)'}"
        )

        try:
            t_extract = time.perf_counter()
            pages     = load_pdf(file_path, max_pages=pages_to_read)
            logger.info(f"[{request_id}] Step 3/5 — extracted {len(pages)} page(s) ({time.perf_counter()-t_extract:.2f}s)")
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))

        if all_pages_blank(pages):
            logger.warning(f"[{request_id}] blank PDF — skipping inference")
            result = dict(_BLANK_PDF_RESULT)
            result["blank_pdf"] = True
            if was_truncated:
                result.update(truncated=True, pages_analysed=pages_to_read, total_pages=total_pages)
            if analysis_type == 1: return {"overview": ""}
            if analysis_type == 2: return {"summary": ""}
            if analysis_type == 3: return {"highlights": []}
            return result

        merged_text = "\n\n".join(p.page_content for p in pages)
        logger.info(f"[{request_id}] Step 4/5 — merged {len(merged_text):,} chars")

        logger.info(f"[{request_id}] Step 5/5 — running inference")
        t_infer      = time.perf_counter()
        final_output = await generate_analysis(merged_text)
        logger.info(f"[{request_id}] Step 5/5 — done ({time.perf_counter()-t_infer:.2f}s)")

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"[{request_id}] unhandled error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during PDF analysis.")
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.debug(f"[{request_id}] temp file deleted")

    elapsed = time.perf_counter() - t_start
    logger.info(f"[{request_id}] ── COMPLETE — {elapsed:.2f}s ──────")

    if was_truncated:
        final_output.update(truncated=True, pages_analysed=pages_to_read, total_pages=total_pages)

    if analysis_type == 1: return {"overview":   final_output.get("overview", "")}
    if analysis_type == 2: return {"summary":    final_output.get("summary", "")}
    if analysis_type == 3: return {"highlights": final_output.get("highlights", [])}
    return final_output


# ---------------------------------------------------------------------------
# POST /key-clause-extraction
# ---------------------------------------------------------------------------

@router.post("/key-clause-extraction")
async def key_clause_extraction(file: UploadFile = File(...)):

    # Step 1: text extraction
    text, _, _, request_id, t_start, file_path = await extract_text_from_upload(
        file,
        endpoint="/key-clause-extraction"
    )

    try:
        # Step 2: Classification
        doc_type = await classify_document(text)
        doc_type = doc_type.lower().strip()
        logger.info(f"[{request_id}] Step 3 — classified as: '{doc_type}'")

        # Step 3: Route
        handler = DOCUMENT_HANDLERS.get(doc_type)

        if handler:
            result = await handler(text)
            logger.info(
                f"[{request_id}] ── REQUEST COMPLETE — "
                f"total time: {time.perf_counter() - t_start:.2f}s ──────"
            )
            return result

        # Step 4: Fallback
        logger.warning(f"[{request_id}] No handler found for doc_type='{doc_type}'")
        return {
            "status": "unsupported",
            "document_type": doc_type,
            "message": "Unsupported document type."
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"[{request_id}] Unhandled error during processing: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during key clause extraction.")
    finally:
        # Step 5: Cleanup
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.debug(f"[{request_id}] Temp file deleted: '{file_path}'")


# ---------------------------------------------------------------------------
# POST /detect-risks
# ---------------------------------------------------------------------------

@router.post("/detect-risks")
async def detect_risks(file: UploadFile = File(...)):
    """
    AI Risk Detection Endpoint (Phase 1.2)
    Reuses the extraction logic from Phase 1.1
    """
    text, _, _, request_id, t_start, file_path = await extract_text_from_upload(
        file,
        endpoint="/detect-risks"
    )

    try:
        logger.info(f"[{request_id}] Starting Risk Detection...")
        result = await analyze_document_risks(text)
        return result

    except Exception as e:
        logger.exception(f"[{request_id}] Risk Analysis failed: {e}")
        raise HTTPException(status_code=500, detail="Internal error during risk detection.")
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


# ---------------------------------------------------------------------------
# POST /analyze/stream — Server-Sent Events streaming response
#
# Event sequence:
#   status    {"step":"saving"|"extracting"|"inference"|"chunk_done"|"synthesis"|"queued"}
#   overview  {"text":"..."}        — after chunk 1, updated after synthesis
#   summary   {"text":"..."}        — after synthesis
#   highlight {"text":"...","index":N} — one per highlight as each chunk finishes
#   done      {"total_time":N,"total_highlights":N,"pages":N}
#   error     {"message":"..."}
#
# nginx: add proxy_buffering off to the location block for this route.
# ---------------------------------------------------------------------------

@router.post("/analyze/stream")
async def analyze_pdf_stream(
    file: UploadFile = File(...),
    analysis_type: int = Query(0, ge=0, le=3,
        description="0=all, 1=overview, 2=summary, 3=highlights")
):
    """
    Analyse a PDF and stream results in real time using Server-Sent Events.
    Highlights stream one-by-one as each chunk finishes (~8s intervals).
    Overview arrives after chunk 1. Summary arrives after synthesis.
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

    safe_name = f"{uuid.uuid4()}.pdf"
    file_path = os.path.join(UPLOAD_FOLDER, safe_name)

    async def _generate():
        was_truncated    = False
        pages_to_read    = 0
        total_pages      = 0
        final_highlights = []

        try:
            yield _sse("status", {"step": "saving", "message": "Saving uploaded file..."})
            content = await file.read()
            with open(file_path, "wb") as f:
                f.write(content)
            logger.info(f"[{request_id}] saved {len(content):,} bytes")

            total_pages   = get_page_count(file_path)
            pages_to_read = total_pages if MAX_PDF_PAGES is None else min(total_pages, MAX_PDF_PAGES)
            was_truncated = MAX_PDF_PAGES is not None and total_pages > MAX_PDF_PAGES

            yield _sse("status", {
                "step":    "extracting",
                "message": f"Extracting text from {pages_to_read} page(s)...",
                "pages":   pages_to_read,
            })
            try:
                pages = load_pdf(file_path, max_pages=pages_to_read)
            except ValueError as e:
                yield _sse("error", {"message": str(e)})
                return

            if all_pages_blank(pages):
                logger.warning(f"[{request_id}] blank PDF")
                yield _sse("status", {"step": "blank", "message": "PDF has no extractable content."})
                if analysis_type in (0, 1): yield _sse("overview", {"text": ""})
                if analysis_type in (0, 2): yield _sse("summary",  {"text": ""})
                yield _sse("done", {
                    "total_time": round(time.perf_counter() - t_start, 2),
                    "total_highlights": 0, "pages": total_pages, "blank_pdf": True,
                })
                return

            merged_text = "\n\n".join(p.page_content for p in pages)
            logger.info(f"[{request_id}] merged {len(merged_text):,} chars")

            overview_sent   = False
            highlight_index = 0
            final_overview  = ""
            final_summary   = ""

            async for event_type, payload in generate_analysis_stream(merged_text):

                if event_type == "chunk_start":
                    yield _sse("status", {
                        "step":    "inference",
                        "message": f"Analysing chunk {payload['chunk']} of {payload['total']}...",
                        "chunk":   payload["chunk"],
                        "total":   payload["total"],
                    })

                elif event_type == "chunk_done":
                    if not overview_sent and payload.get("overview"):
                        if analysis_type in (0, 1):
                            yield _sse("overview", {"text": payload["overview"]})
                        final_overview = payload["overview"]
                        overview_sent  = True

                    if analysis_type in (0, 3):
                        for h in payload.get("new_highlights", []):
                            yield _sse("highlight", {"text": h, "index": highlight_index})
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
                        "step": "synthesis", "message": "Writing overview and summary...",
                    })

                elif event_type == "synthesis_done":
                    final_overview = payload.get("overview", final_overview)
                    final_summary  = payload.get("summary", "")
                    if analysis_type in (0, 1):
                        yield _sse("overview", {"text": final_overview})
                    if analysis_type in (0, 2):
                        yield _sse("summary",  {"text": final_summary})

        except Exception as e:
            logger.exception(f"[{request_id}] stream error: {e}")
            yield _sse("error", {"message": "Internal server error during PDF analysis."})
            return
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.debug(f"[{request_id}] temp file deleted")

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


# ---------------------------------------------------------------------------
# POST /convert/pdf-to-docx — PDF to DOCX conversion
# ---------------------------------------------------------------------------

@router.post("/convert/pdf-to-docx")
async def convert_pdf_to_docx(file: UploadFile = File(...)):
    """
    Convert an uploaded PDF to a downloadable DOCX file.

    Handles all PDF types:
      - Native text PDF → PyMuPDF extraction → DOCX
      - Scanned/image PDF → PaddleOCR-VL → DOCX
      - Mixed PDF → best available per page → DOCX

    Returns the .docx file as a binary download.
    No AI inference — fast conversion without GPU queue wait.
    """
    request_id = str(uuid.uuid4())[:8]
    t_start    = time.perf_counter()

    logger.info(f"[{request_id}] ── PDF TO DOCX ───────────────────────────────")
    logger.info(f"[{request_id}] filename='{file.filename}'")

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    safe_name = f"{uuid.uuid4()}.pdf"
    file_path = os.path.join(UPLOAD_FOLDER, safe_name)

    try:
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        logger.info(f"[{request_id}] saved {len(content):,} bytes")

        try:
            pages = load_pdf(file_path)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))

        logger.info(f"[{request_id}] extracted {len(pages)} page(s)")

        def _do_convert():
            from s_pdf_to_docx import _extract_blocks_ocr, _build_docx

            all_blocks = []
            for page in pages:
                blocks = _extract_blocks_ocr(page.page_content)
                all_blocks.append(blocks)

            return _build_docx(all_blocks, file.filename)

        loop = asyncio.get_running_loop()
        docx = await loop.run_in_executor(None, _do_convert)

        buffer = io.BytesIO()
        docx.save(buffer)
        buffer.seek(0)

        output_filename = file.filename.replace(".pdf", ".docx")
        elapsed = time.perf_counter() - t_start
        logger.info(f"[{request_id}] ── DOCX COMPLETE — {elapsed:.2f}s → '{output_filename}'")

        return StreamingResponse(
            buffer,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={
                "Content-Disposition": f'attachment; filename="{output_filename}"',
                "X-Pages":             str(len(pages)),
                "X-Processing-Time":   str(round(elapsed, 2)),
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"[{request_id}] conversion failed: {e}")
        raise HTTPException(status_code=500, detail="PDF to DOCX conversion failed.")
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.debug(f"[{request_id}] temp file deleted")
