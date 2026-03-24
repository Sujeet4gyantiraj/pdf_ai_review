from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.responses import StreamingResponse
import os
import io
import uuid
import json
import re
import time
import hashlib
import asyncio
import logging
import logging.config
from contextlib import asynccontextmanager
from s_pdf_utils import load_pdf, get_page_count, all_pages_blank
from s_ai_model import generate_analysis, generate_analysis_stream
from t_key_clause_extraction import  classify_document, DOCUMENT_HANDLERS, extract_text_from_upload
from t_risk_detection import analyze_document_risks
from redis_cache import init_redis, close_redis, make_cache_key, get_cached, set_cached





@app.post("/analyze")
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

        # ── Cache check ────────────────────────────────────────────────────
        content_hash = hashlib.sha256(content).hexdigest()
        cache_key    = make_cache_key(content_hash, "analyze", analysis_type=analysis_type)
        cached       = await get_cached(cache_key)
        if cached is not None:
            logger.info(f"[{request_id}] Cache HIT — returning cached result")
            os.remove(file_path)
            return cached
        # ──────────────────────────────────────────────────────────────────

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

    if analysis_type == 1:
        response = {"overview": final_output.get("overview", "")}
    elif analysis_type == 2:
        response = {"summary": final_output.get("summary", "")}
    elif analysis_type == 3:
        response = {"highlights": final_output.get("highlights", [])}
    else:
        response = final_output

    await set_cached(cache_key, response)
    return response

@app.post("/key-clause-extraction")
async def key_clause_extraction(file: UploadFile = File(...)):

    # ==============================
    # Step 1: text extraction
    # ==============================

    text, _, _, request_id, t_start, file_path = await extract_text_from_upload(
        file,
        endpoint="/key-clause-extraction"
    )

    # ── Cache check ────────────────────────────────────────────────────────
    raw_bytes    = open(file_path, "rb").read()
    content_hash = hashlib.sha256(raw_bytes).hexdigest()
    cache_key    = make_cache_key(content_hash, "key-clause-extraction")
    cached       = await get_cached(cache_key)
    if cached is not None:
        logger.info(f"[{request_id}] Cache HIT — returning cached result")
        os.remove(file_path)
        return cached
    # ──────────────────────────────────────────────────────────────────────

    try:
        # ==============================
        # Step 2: Classification
        # ==============================

        doc_type = await classify_document(text)
        doc_type = doc_type.lower().strip()
        logger.info(f"[{request_id}] Step 3 — classified as: '{doc_type}'")

        # ==============================
        # Step 3: Route
        # ==============================
        handler = DOCUMENT_HANDLERS.get(doc_type)

        if handler:
            result = await handler(text)
            await set_cached(cache_key, result)
            logger.info(
                f"[{request_id}] ── REQUEST COMPLETE — "
                f"total time: {time.perf_counter() - t_start:.2f}s ──────"
            )
            return result

        # ==============================
        # Step 4: Fallback
        # ==============================
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
        # ==============================
        # Step 5: Cleanup
        # ==============================
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.debug(f"[{request_id}] Temp file deleted: '{file_path}'")

@app.post("/detect-risks")
async def detect_risks(file: UploadFile = File(...)):
    """
    AI Risk Detection Endpoint (Phase 1.2)
    Reuses the extraction logic from Phase 1.1
    """
    from t_key_clause_extraction import extract_text_from_upload

    # Extract text from PDF
    text, _, _, request_id, t_start, file_path = await extract_text_from_upload(
        file,
        endpoint="/detect-risks"
    )

    # ── Cache check ────────────────────────────────────────────────────────
    raw_bytes    = open(file_path, "rb").read()
    content_hash = hashlib.sha256(raw_bytes).hexdigest()
    cache_key    = make_cache_key(content_hash, "detect-risks")
    cached       = await get_cached(cache_key)
    if cached is not None:
        logger.info(f"[{request_id}] Cache HIT — returning cached result")
        os.remove(file_path)
        return cached
    # ──────────────────────────────────────────────────────────────────────

    try:
        logger.info(f"[{request_id}] Starting Risk Detection...")

        result = await analyze_document_risks(text)
        await set_cached(cache_key, result)
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

@app.post("/analyze/stream")
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

            # ── Cache check ────────────────────────────────────────────────
            content_hash = hashlib.sha256(content).hexdigest()
            cache_key    = make_cache_key(content_hash, "analyze_stream", analysis_type=analysis_type)
            cached       = await get_cached(cache_key)
            if cached is not None:
                logger.info(f"[{request_id}] Stream Cache HIT — replaying cached result")
                os.remove(file_path)
                if analysis_type in (0, 1):
                    yield _sse("overview",  {"text": cached.get("overview", "")})
                if analysis_type in (0, 2):
                    yield _sse("summary",   {"text": cached.get("summary", "")})
                if analysis_type in (0, 3):
                    for i, h in enumerate(cached.get("highlights", [])):
                        yield _sse("highlight", {"text": h, "index": i})
                yield _sse("done", {
                    "total_time":       0,
                    "total_highlights": len(cached.get("highlights", [])),
                    "pages":            cached.get("pages", 0),
                    "cache_hit":        True,
                })
                return
            # ──────────────────────────────────────────────────────────────
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

        # ── Store result in cache for future requests ──────────────────────
        await set_cached(cache_key, {
            "overview":   final_overview,
            "summary":    final_summary,
            "highlights": final_highlights,
            "pages":      total_pages,
        })
        # ──────────────────────────────────────────────────────────────────

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
#
# Converts an uploaded PDF to a downloadable DOCX file.
# Uses existing load_pdf() pipeline which handles:
#   - Native text PDFs (PyMuPDF, fast)
#   - Scanned/image PDFs (PaddleOCR-VL, GPU)
#   - Mixed PDFs (both)
#
# Does NOT use GPU inference (no Mistral) — no semaphore needed.
# Runs in thread pool via run_in_executor to avoid blocking event loop.
# ---------------------------------------------------------------------------
 
@app.post("/convert/pdf-to-docx")
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
        # Step 1 — save upload
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        logger.info(f"[{request_id}] saved {len(content):,} bytes")
 
        # Step 2 — extract text (reuses existing pipeline)
        # load_pdf handles native + OCR automatically
        try:
            pages = load_pdf(file_path)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
 
        logger.info(f"[{request_id}] extracted {len(pages)} page(s)")
 
        # Step 3 — convert to DOCX in thread pool
        def _do_convert():
            from s_pdf_to_docx import _extract_blocks_ocr, _build_docx
 
            all_blocks = []
            for page in pages:
                blocks = _extract_blocks_ocr(page.page_content)
                all_blocks.append(blocks)
 
            return _build_docx(all_blocks, file.filename)
 
        loop = asyncio.get_running_loop()
        docx = await loop.run_in_executor(None, _do_convert)
 
        # Step 4 — save to in-memory buffer and return
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
 