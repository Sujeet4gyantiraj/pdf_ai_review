# from fastapi import FastAPI, UploadFile, File, Query, HTTPException
# import os
# import uuid
# import json
# import re
# import time
# import logging
# import logging.config
# from s_padf_utils import load_pdf, get_page_count, all_pages_blank
# from s_ai_model import generate_analysis

# # ---------------------------------------------------------------------------
# # Logging configuration
# # One-time setup: writes INFO+ to console and DEBUG+ to app.log
# # Both handlers use the same structured format so grep works on either.
# # ---------------------------------------------------------------------------
# logging.config.dictConfig({
#     "version": 1,
#     "disable_existing_loggers": False,
#     "formatters": {
#         "standard": {
#             "format": "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
#             "datefmt": "%Y-%m-%d %H:%M:%S",
#         }
#     },
#     "handlers": {
#         "console": {
#             "class":     "logging.StreamHandler",
#             "formatter": "standard",
#             "level":     "INFO",
#             "stream":    "ext://sys.stdout",
#         },
#         "file": {
#             "class":     "logging.handlers.RotatingFileHandler",
#             "formatter": "standard",
#             "level":     "DEBUG",
#             "filename":  "app.log",
#             "maxBytes":  10 * 1024 * 1024,   # 10 MB per file
#             "backupCount": 5,                  # keep 5 rotated files
#             "encoding":  "utf-8",
#         },
#     },
#     "root": {
#         "level":    "DEBUG",
#         "handlers": ["console", "file"],
#     },
# })

# logger = logging.getLogger(__name__)

# app = FastAPI()

# UPLOAD_FOLDER = "temp"
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# logger.info(f"Upload folder ready: {os.path.abspath(UPLOAD_FOLDER)}")


# # ---------------------------------------------------------------------------
# # JSON extraction helpers
# # ---------------------------------------------------------------------------

# def _fix_json_string(raw: str) -> str:
#     """
#     Apply a sequence of targeted repairs to common Mistral JSON output problems,
#     without altering the structure or values of well-formed output.
#     """
#     logger.debug("_fix_json_string: starting string repair")

#     # Remove markdown code fences
#     raw = raw.replace("```json", "").replace("```", "").strip()

#     # Normalise Windows line endings
#     raw = raw.replace("\r\n", "\n").replace("\r", "\n")

#     # Fix illegal (non-JSON) backslash escapes, e.g. \' \, \: \. etc.
#     # Keep valid escapes: \" \\ \/ \b \f \n \r \t \uXXXX
#     raw = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', raw)

#     # Remove ASCII control characters (0x00-0x1F) except \n \r \t
#     raw = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', raw)

#     logger.debug(f"_fix_json_string: repair complete ({len(raw)} chars)")
#     return raw


# def _extract_field_by_regex(text: str, field: str) -> str:
#     """
#     Fallback: extract a single string field value directly with regex
#     when the whole JSON block can't be parsed.
#     """
#     logger.debug(f"_extract_field_by_regex: extracting field '{field}'")
#     pattern = rf'"{field}"\s*:\s*"(.*?)"(?=\s*[,}}])'
#     match = re.search(pattern, text, re.DOTALL)
#     if match:
#         logger.debug(f"_extract_field_by_regex: found '{field}' via regex")
#         return match.group(1).strip()
#     logger.warning(f"_extract_field_by_regex: field '{field}' not found")
#     return ""


# def _extract_highlights_by_regex(text: str) -> list:
#     """
#     Fallback: extract highlights array items directly with regex.
#     """
#     logger.debug("_extract_highlights_by_regex: attempting regex extraction")
#     match = re.search(r'"highlights"\s*:\s*\[(.*?)\]', text, re.DOTALL)
#     if not match:
#         logger.warning("_extract_highlights_by_regex: highlights array not found")
#         return []
#     inner = match.group(1)
#     items = re.findall(r'"(.*?)"(?=\s*[,\]])', inner, re.DOTALL)
#     result = [i.strip() for i in items if i.strip()]
#     logger.debug(f"_extract_highlights_by_regex: found {len(result)} item(s)")
#     return result


# def _postprocess_highlights(data: dict) -> dict:
#     """
#     Post-process highlights list: flatten items the model packed into a single
#     comma-separated string, and strip stray quote characters.
#     """
#     if "highlights" in data and isinstance(data["highlights"], list):
#         cleaned = []
#         for item in data["highlights"]:
#             parts = item.split('", "') if '", "' in item else [item]
#             for p in parts:
#                 cleaned.append(p.replace('"', '').strip())
#         data["highlights"] = [h for h in cleaned if h]
#         logger.debug(f"_postprocess_highlights: {len(data['highlights'])} highlight(s) after cleanup")
#     return data


# def extract_json(text: str) -> dict:
#     """
#     Robustly extract and structure JSON from Mistral output using three
#     fallback strategies so a single malformed character never silently
#     returns an empty result.

#     Strategy 1 — direct parse after cleaning
#     Strategy 2 — isolate outermost { ... } block, then parse; on failure
#                  attempt a conservative in-string quote-escape repair
#     Strategy 3 — per-field regex extraction as last resort
#     """
#     empty = {"overview": "", "summary": "", "highlights": []}

#     if not text or not text.strip():
#         logger.warning("extract_json: received empty text, returning empty result")
#         return empty

#     logger.debug(f"extract_json: input length {len(text)} chars")
#     cleaned = _fix_json_string(text)

#     # --- Strategy 1: parse the whole cleaned string directly ---
#     logger.debug("extract_json: trying strategy 1 — direct json.loads")
#     try:
#         data = json.loads(cleaned)
#         logger.debug("extract_json: strategy 1 succeeded")
#         return _postprocess_highlights(data)
#     except json.JSONDecodeError as e:
#         logger.debug(f"extract_json: strategy 1 failed — {e}")

#     # --- Strategy 2: isolate the outermost { ... } block ---
#     logger.debug("extract_json: trying strategy 2 — isolate brace block")
#     brace_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
#     if brace_match:
#         candidate = brace_match.group()
#         try:
#             data = json.loads(candidate)
#             logger.debug("extract_json: strategy 2 succeeded")
#             return _postprocess_highlights(data)
#         except json.JSONDecodeError as e:
#             logger.warning(f"extract_json: strategy 2 failed ({e}), trying quote-escape repair")

#             repaired = re.sub(
#                 r'(?<=[^\\])"(?=[^,\]}\n:}{\[])',
#                 r'\\"',
#                 candidate
#             )
#             try:
#                 data = json.loads(repaired)
#                 logger.debug("extract_json: strategy 2 repair succeeded")
#                 return _postprocess_highlights(data)
#             except json.JSONDecodeError as e2:
#                 logger.warning(f"extract_json: strategy 2 repair also failed — {e2}")
#     else:
#         logger.warning("extract_json: no brace block found in output")

#     # --- Strategy 3: regex field extraction (last resort) ---
#     logger.error(
#         "extract_json: all JSON parse strategies failed — "
#         "falling back to per-field regex extraction"
#     )
#     overview   = _extract_field_by_regex(cleaned, "overview")
#     summary    = _extract_field_by_regex(cleaned, "summary")
#     highlights = _extract_highlights_by_regex(cleaned)

#     if overview or summary or highlights:
#         logger.info("extract_json: strategy 3 recovered partial data via regex")
#         return {"overview": overview, "summary": summary, "highlights": highlights}

#     logger.error("extract_json: strategy 3 also failed — returning empty result")
#     return empty


# # ---------------------------------------------------------------------------
# # /analyze endpoint
# # ---------------------------------------------------------------------------

# # Set to None to accept PDFs of any size.
# # Set to an integer (e.g. 200) to cap how many pages are processed —
# # pages beyond the cap are skipped but the upload is never rejected.
# MAX_PDF_PAGES = None

# # Empty result returned immediately for fully blank/unreadable PDFs.
# # No model inference is run — avoids wasting GPU time on placeholder text.
# _BLANK_PDF_RESULT = {"overview": "", "summary": "", "highlights": []}


# @app.post("/analyze")
# async def analyze_pdf(
#     file: UploadFile = File(...),
#     analysis_type: int = Query(
#         0,
#         ge=0,
#         le=3,
#         description="0=all, 1=overview, 2=summary, 3=highlights"
#     )
# ):
#     """
#     Analyze uploaded PDF and return:
#     - overview
#     - summary
#     - highlights

#     Large PDFs (> MAX_PDF_PAGES) are accepted but only the first MAX_PDF_PAGES
#     pages are analysed. The response includes a 'truncated' flag and
#     'pages_analysed' / 'total_pages' fields when truncation occurs.

#     Fully blank PDFs (every page is blank or unreadable after extraction)
#     are detected before inference and immediately return empty fields with
#     a 'blank_pdf': true flag — no GPU time is wasted.

#     Uses LangChain RecursiveCharacterTextSplitter for chunking and a
#     direct map-reduce inference pipeline for large-PDF coherence.
#     """
#     request_id = str(uuid.uuid4())[:8]
#     t_start    = time.perf_counter()

#     logger.info(f"[{request_id}] ── NEW REQUEST ──────────────────────────────")
#     logger.info(f"[{request_id}] filename='{file.filename}' analysis_type={analysis_type}")

#     # Validate file type
#     if not file.filename or not file.filename.lower().endswith(".pdf"):
#         logger.warning(f"[{request_id}] Rejected: not a PDF file (filename='{file.filename}')")
#         raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

#     safe_name = f"{uuid.uuid4()}.pdf"
#     file_path = os.path.join(UPLOAD_FOLDER, safe_name)
#     logger.debug(f"[{request_id}] Saving upload to '{file_path}'")

#     # Flag set below; also used in the finally block's truncation metadata
#     was_truncated  = False
#     pages_to_read  = 0
#     total_pages    = 0

#     try:
#         # Step 1 — save upload
#         content = await file.read()
#         with open(file_path, "wb") as f:
#             f.write(content)
#         logger.info(f"[{request_id}] Step 1/5 — saved upload ({len(content):,} bytes → '{safe_name}')")

#         # Step 2 — page count (no rejection — large PDFs are always accepted)
#         total_pages   = get_page_count(file_path)
#         pages_to_read = total_pages if MAX_PDF_PAGES is None else min(total_pages, MAX_PDF_PAGES)
#         was_truncated = MAX_PDF_PAGES is not None and total_pages > MAX_PDF_PAGES

#         logger.info(
#             f"[{request_id}] Step 2/5 — total pages: {total_pages}, "
#             f"will analyse: {pages_to_read} "
#             f"{'(TRUNCATED — large PDF)' if was_truncated else '(all pages)'}"
#         )
#         if was_truncated:
#             logger.warning(
#                 f"[{request_id}] PDF has {total_pages} pages which exceeds "
#                 f"MAX_PDF_PAGES={MAX_PDF_PAGES}. "
#                 f"Only the first {pages_to_read} pages will be analysed. "
#                 "Set MAX_PDF_PAGES = None in main.py to process all pages."
#             )

#         # Step 3 — text extraction (native + OCR fallback per page)
#         logger.info(f"[{request_id}] Step 3/5 — extracting text from {pages_to_read} page(s)")
#         try:
#             t_extract = time.perf_counter()
#             pages     = load_pdf(file_path, max_pages=pages_to_read)
#             logger.info(
#                 f"[{request_id}] Step 3/5 — extraction complete: "
#                 f"{len(pages)} page(s) with text "
#                 f"({time.perf_counter() - t_extract:.2f}s)"
#             )
#         except ValueError as e:
#             logger.error(f"[{request_id}] Step 3/5 — extraction failed: {e}")
#             raise HTTPException(status_code=422, detail=str(e))

#         # ── Blank-PDF short-circuit ───────────────────────────────────────
#         # Checked immediately after extraction, before any merging or inference.
#         # Covers two cases:
#         #   1. Native blank: PDF has pages but every page has zero native text
#         #      AND PaddleOCR-VL also extracted nothing (all blank/placeholder).
#         #   2. Image-only blank: scanned PDF where every page rendered to an
#         #      image but OCR returned no text on any page.
#         # In both cases we skip Steps 4-5 entirely and return empty fields.
#         if all_pages_blank(pages):
#             elapsed = time.perf_counter() - t_start
#             logger.warning(
#                 f"[{request_id}] Blank PDF detected — all {len(pages)} page(s) "
#                 f"are blank or unreadable. Skipping inference. ({elapsed:.2f}s)"
#             )
#             result = dict(_BLANK_PDF_RESULT)
#             result["blank_pdf"] = True
#             if was_truncated:
#                 result["truncated"]      = True
#                 result["pages_analysed"] = pages_to_read
#                 result["total_pages"]    = total_pages
#             if analysis_type == 1:
#                 return {"overview": ""}
#             elif analysis_type == 2:
#                 return {"summary": ""}
#             elif analysis_type == 3:
#                 return {"highlights": []}
#             return result
#         # ─────────────────────────────────────────────────────────────────

#         # Step 4 — merge pages into one text string
#         logger.info(f"[{request_id}] Step 4/5 — merging {len(pages)} page(s) into text")
#         t_merge     = time.perf_counter()
#         merged_text = "\n\n".join(p.page_content for p in pages)
#         logger.info(
#             f"[{request_id}] Step 4/5 — merged: {len(merged_text):,} chars "
#             f"({time.perf_counter() - t_merge:.3f}s)"
#         )

#         # Step 5 — token-accurate map-reduce inference
#         logger.info(f"[{request_id}] Step 5/5 — running map-reduce inference")
#         t_infer      = time.perf_counter()
#         final_output = await generate_analysis(merged_text)
#         logger.info(
#             f"[{request_id}] Step 5/5 — inference complete "
#             f"({time.perf_counter() - t_infer:.2f}s)"
#         )

#     except HTTPException:
#         raise

#     except Exception as e:
#         logger.exception(f"[{request_id}] Unhandled error during processing: {e}")
#         raise HTTPException(status_code=500, detail="Internal server error during PDF analysis.")

#     finally:
#         if os.path.exists(file_path):
#             os.remove(file_path)
#             logger.debug(f"[{request_id}] Temp file deleted: '{file_path}'")

#     elapsed = time.perf_counter() - t_start
#     logger.info(
#         f"[{request_id}] ── REQUEST COMPLETE — total time: {elapsed:.2f}s ──────"
#     )

#     # Attach truncation metadata so the caller knows partial analysis was done
#     if was_truncated:
#         final_output["truncated"]      = True
#         final_output["pages_analysed"] = pages_to_read
#         final_output["total_pages"]    = total_pages
#         logger.info(
#             f"[{request_id}] Response includes truncation metadata "
#             f"(analysed {pages_to_read}/{total_pages} pages)"
#         )

#     if analysis_type == 1:
#         logger.debug(f"[{request_id}] Returning overview only")
#         return {"overview": final_output.get("overview", "")}
#     elif analysis_type == 2:
#         logger.debug(f"[{request_id}] Returning summary only")
#         return {"summary": final_output.get("summary", "")}
#     elif analysis_type == 3:
#         logger.debug(f"[{request_id}] Returning highlights only")
#         return {"highlights": final_output.get("highlights", [])}

#     return final_output


"""
t_main.py
=========
FastAPI application for PDF summarization — supports small, large,
text-based, image-only, scanned, and mixed PDFs.

Architecture
------------
POST /analyze          → streams file to disk, returns job_id (202 Accepted)
GET  /status/{job_id}  → poll progress  {stage, current, total}
GET  /result/{job_id}  → retrieve final JSON when done

Pipeline (3-level hierarchical summarization)
---------------------------------------------
Level 0  extract_text_from_pdf()     parallel batch OCR + native text
Level 1  generate_analysis()         per-chunk AI analysis (GPU-semaphore-gated)
Level 2  synthesize_section()        every SECTION_SIZE chunks → 1 section summary
Level 3  synthesize_final()          all section summaries → 1 final output

Edge cases handled
------------------
A) Image-only PDF, OCR unavailable
   → clear "failed" status with install instructions for PaddleOCR-VL

B) PDF with < MIN_TEXT_CHARS total text after extraction
   → clear "failed" status with exact character count

C) Corrupt / encrypted PDF
   → ValueError from extract_text_from_pdf → "failed" with reason

All fixes from original review retained
----------------------------------------
- Streaming upload  (no full-file RAM read)
- UUID temp filename  (no path traversal)
- clean_text imported from t_pdf_utils  (no local duplicate)
- load_model() called once at startup via lifespan
- asyncio.get_running_loop()  (not deprecated get_event_loop)
- GPU semaphore + retry inside t_ai_model
- Highlight deduplication + cap at MAX_HIGHLIGHTS
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import uuid
from contextlib import asynccontextmanager
from typing import Literal

import fitz
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse

from s_ai_model import (
    generate_analysis,
    load_model,
    synthesize_final,
    synthesize_section,
)
from s_padf_utils import chunk_text, clean_text, extract_text_from_pdf

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
UPLOAD_FOLDER    = "temp"
UPLOAD_CHUNK_MB  = 1                        # streaming read size
MAX_UPLOAD_BYTES = 500 * 1024 * 1024        # 500 MB hard upload limit
SECTION_SIZE     = 20                       # chunks per Level-2 section
MAX_HIGHLIGHTS   = 10                       # cap on final highlight list
MIN_TEXT_CHARS   = 20                       # minimum useful extracted text

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------------------------------------------------------------------
# In-memory job store
# Replace with Redis + a persistent DB for multi-worker deployments.
# ---------------------------------------------------------------------------
JobStatus = Literal[
    "pending", "extracting", "analyzing",
    "synthesizing", "done", "failed",
]

_jobs: dict[str, dict] = {}


def _new_job() -> str:
    jid        = str(uuid.uuid4())
    _jobs[jid] = {
        "status":   "pending",
        "progress": {"stage": "queued", "current": 0, "total": 0},
        "result":   None,
        "error":    None,
    }
    return jid


def _set_progress(
    jid: str,
    stage: str,
    current: int = 0,
    total: int   = 0,
) -> None:
    if jid not in _jobs:
        return
    _jobs[jid]["progress"] = {"stage": stage, "current": current, "total": total}
    _jobs[jid]["status"]   = (
        "extracting"   if stage == "extraction"                          else
        "analyzing"    if stage == "chunk_analysis"                      else
        "synthesizing" if stage in ("section_synthesis", "final_synthesis") else
        _jobs[jid]["status"]
    )


def _fail(jid: str, message: str) -> None:
    _jobs[jid]["status"] = "failed"
    _jobs[jid]["error"]  = message
    logger.error(f"[{jid}] FAILED: {message}")


# ---------------------------------------------------------------------------
# Application lifespan  (model loaded once at startup)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(
    title="PDF Summarizer — Large Document Edition",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# JSON extraction  (3-strategy parser)
# ---------------------------------------------------------------------------

def _fix_json_string(raw: str) -> str:
    """Clean common Mistral output issues before JSON parsing."""
    raw = raw.replace("```json", "").replace("```", "").strip()
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    # Fix illegal backslash escapes (keep valid: \" \\ \/ \b \f \n \r \t \uXXXX)
    raw = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', raw)
    # Strip ASCII control characters except \n \r \t
    raw = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', raw)
    return raw


def _extract_field_by_regex(text: str, field: str) -> str:
    m = re.search(rf'"{field}"\s*:\s*"(.*?)"(?=\s*[,}}])', text, re.DOTALL)
    return m.group(1).strip() if m else ""


def _extract_highlights_by_regex(text: str) -> list[str]:
    m = re.search(r'"highlights"\s*:\s*\[(.*?)\]', text, re.DOTALL)
    if not m:
        return []
    items = re.findall(r'"(.*?)"(?=\s*[,\]])', m.group(1), re.DOTALL)
    return [i.strip() for i in items if i.strip()]


def _postprocess_highlights(data: dict) -> dict:
    """Flatten comma-packed highlight items and strip stray quote characters."""
    if "highlights" in data and isinstance(data["highlights"], list):
        cleaned = []
        for item in data["highlights"]:
            parts = item.split('", "') if '", "' in item else [item]
            for p in parts:
                cleaned.append(p.replace('"', '').strip())
        data["highlights"] = [h for h in cleaned if h]
    return data


def extract_json(text: str) -> dict:
    """
    Parse JSON from raw model output using three fallback strategies:

    Strategy 1 — direct parse after cleaning
    Strategy 2 — isolate outermost { … } block, then parse;
                 on failure attempt a conservative quote-escape repair
    Strategy 3 — per-field regex extraction (last resort)
    """
    empty = {"overview": "", "summary": "", "highlights": []}

    if not text or not text.strip():
        return empty

    cleaned = _fix_json_string(text)

    # Strategy 1
    try:
        return _postprocess_highlights(json.loads(cleaned))
    except json.JSONDecodeError:
        pass

    # Strategy 2
    m = re.search(r'\{.*\}', cleaned, re.DOTALL)
    if m:
        candidate = m.group()
        try:
            return _postprocess_highlights(json.loads(candidate))
        except json.JSONDecodeError as exc:
            logger.warning(f"JSON block parse failed ({exc}); trying repair")
            repaired = re.sub(
                r'(?<=[^\\])"(?=[^,\]}\n:}{\[])', r'\\"', candidate
            )
            try:
                return _postprocess_highlights(json.loads(repaired))
            except json.JSONDecodeError:
                pass

    # Strategy 3
    logger.error("All JSON parse strategies failed — falling back to regex")
    overview   = _extract_field_by_regex(cleaned, "overview")
    summary    = _extract_field_by_regex(cleaned, "summary")
    highlights = _extract_highlights_by_regex(cleaned)
    if overview or summary or highlights:
        return {"overview": overview, "summary": summary, "highlights": highlights}

    return empty


# ---------------------------------------------------------------------------
# Highlight deduplication
# ---------------------------------------------------------------------------

def _dedup_highlights(
    highlights: list[str],
    limit: int = MAX_HIGHLIGHTS,
) -> list[str]:
    """
    Normalise → deduplicate → cap.

    Normalisation: lowercase, strip punctuation, collapse whitespace.
    This catches near-duplicates like "Revenue was $2M" and "Revenue: $2M".
    """
    seen:   set[str]  = set()
    unique: list[str] = []
    for h in highlights:
        key = re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', '', h.lower())).strip()
        if key not in seen:
            seen.add(key)
            unique.append(h)
    return unique[:limit]


# ---------------------------------------------------------------------------
# Hierarchical summarization pipeline
# ---------------------------------------------------------------------------

async def _run_pipeline(
    jid:           str,
    file_path:     str,
    analysis_type: int,
) -> None:
    """
    Full async pipeline — runs as a FastAPI BackgroundTask.

    Stage 0  Text extraction      (parallel page batches, adaptive OCR)
    Stage 1  Chunk analysis       (Level 0 — per chunk, GPU-semaphore-gated)
    Stage 2  Section synthesis    (Level 1 — every SECTION_SIZE chunks)
    Stage 3  Final synthesis      (Level 2 — all section summaries → 1 output)
    """
    try:

        # ── Stage 0: extraction ─────────────────────────────────────────────
        _set_progress(jid, "extraction", 0, 1)
        logger.info(f"[{jid}] Extraction started")

        try:
            text = await extract_text_from_pdf(file_path)
        except RuntimeError as exc:
            # Edge case A: image-only PDF or blank/corrupt PDF
            _fail(jid, str(exc))
            return
        except ValueError as exc:
            # Corrupt / encrypted file
            _fail(jid, f"Could not open PDF: {exc}")
            return

        text = clean_text(text)

        # Edge case B: total extracted text is too short to summarise
        if len(text) < MIN_TEXT_CHARS:
            _fail(
                jid,
                f"Extracted text is too short to summarise "
                f"({len(text)} characters after cleaning). "
                f"The PDF may contain only images, decorative elements, "
                f"or a single very short line of text. "
                f"Minimum required: {MIN_TEXT_CHARS} characters.",
            )
            return

        logger.info(f"[{jid}] Extracted {len(text):,} characters")
        _set_progress(jid, "extraction", 1, 1)

        # ── Stage 1: chunk analysis — Level 0 ───────────────────────────────
        chunks       = chunk_text(text)
        total_chunks = len(chunks)

        # Belt-and-suspenders: chunk_text should never return [] after the
        # MIN_TEXT_CHARS guard above, but handle it cleanly just in case.
        if not chunks:
            _fail(jid, "Text could not be split into processable chunks.")
            return

        logger.info(f"[{jid}] Analysing {total_chunks} chunks")
        _set_progress(jid, "chunk_analysis", 0, total_chunks)

        async def _analyse_chunk(idx: int, chunk: str) -> dict:
            raw    = await generate_analysis(chunk)
            parsed = extract_json(raw)
            _set_progress(jid, "chunk_analysis", idx + 1, total_chunks)
            logger.info(f"[{jid}] Chunk {idx+1}/{total_chunks} done")
            return parsed

        # All chunks run concurrently; GPU semaphore inside t_ai_model
        # limits actual simultaneous model.generate() calls.
        chunk_results: list[dict] = list(
            await asyncio.gather(
                *[_analyse_chunk(i, c) for i, c in enumerate(chunks)]
            )
        )

        # ── Stage 2: section synthesis — Level 1 ────────────────────────────
        sections       = [
            chunk_results[i : i + SECTION_SIZE]
            for i in range(0, total_chunks, SECTION_SIZE)
        ]
        total_sections = len(sections)
        logger.info(f"[{jid}] Synthesizing {total_sections} sections")
        _set_progress(jid, "section_synthesis", 0, total_sections)

        async def _synthesize_sec(idx: int, sec: list[dict]) -> dict:
            label  = f"section {idx+1} of {total_sections}"
            raw    = await synthesize_section(sec, label)
            parsed = extract_json(raw)
            _set_progress(jid, "section_synthesis", idx + 1, total_sections)
            logger.info(f"[{jid}] Section {idx+1}/{total_sections} done")
            return parsed

        section_results: list[dict] = list(
            await asyncio.gather(
                *[_synthesize_sec(i, s) for i, s in enumerate(sections)]
            )
        )

        # ── Stage 3: final synthesis — Level 2 ──────────────────────────────
        _set_progress(jid, "final_synthesis", 0, 1)
        logger.info(f"[{jid}] Final synthesis across {total_sections} sections")

        # Detect page count for the final prompt (best-effort)
        try:
            with fitz.open(file_path) as doc:
                total_pages = len(doc)
        except Exception:
            total_pages = total_chunks   # reasonable fallback

        final_raw    = await synthesize_final(section_results, total_pages)
        final_parsed = extract_json(final_raw)

        # Deduplicate and cap highlights
        final_parsed["highlights"] = _dedup_highlights(
            final_parsed.get("highlights", [])
        )

        # Filter output by analysis_type
        if analysis_type == 1:
            result = {"overview":   final_parsed.get("overview",   "")}
        elif analysis_type == 2:
            result = {"summary":    final_parsed.get("summary",    "")}
        elif analysis_type == 3:
            result = {"highlights": final_parsed.get("highlights", [])}
        else:
            result = final_parsed

        _jobs[jid]["status"] = "done"
        _jobs[jid]["result"] = result
        _set_progress(jid, "final_synthesis", 1, 1)
        logger.info(f"[{jid}] Pipeline complete ✓")

    except Exception as exc:
        logger.exception(f"[{jid}] Unexpected pipeline error")
        _fail(jid, f"Unexpected error: {exc}")

    finally:
        # Always delete the temp file, even on failure
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"[{jid}] Temp file deleted")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/analyze", status_code=202)
async def analyze_pdf(
    background_tasks: BackgroundTasks,
    file:             UploadFile = File(...),
    analysis_type:    int        = Query(
        0, ge=0, le=3,
        description=(
            "What to return: "
            "0=all fields  "
            "1=overview only  "
            "2=summary only  "
            "3=highlights only"
        ),
    ),
):
    """
    Upload a PDF and start async analysis.

    The file is streamed to disk chunk-by-chunk — no full-file RAM spike.
    Returns HTTP 202 with a job_id immediately.

    Poll GET /status/{job_id} to track progress.
    Retrieve the result with GET /result/{job_id}.

    Supports text-based, scanned, image-only, and mixed PDFs up to 500 MB.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    safe_name  = f"{uuid.uuid4()}.pdf"
    file_path  = os.path.join(UPLOAD_FOLDER, safe_name)
    total_read = 0
    chunk_size = UPLOAD_CHUNK_MB * 1024 * 1024

    try:
        with open(file_path, "wb") as out:
            while True:
                data = await file.read(chunk_size)
                if not data:
                    break
                total_read += len(data)
                if total_read > MAX_UPLOAD_BYTES:
                    out.close()
                    os.remove(file_path)
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            f"File exceeds the "
                            f"{MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit."
                        ),
                    )
                out.write(data)
    except HTTPException:
        raise
    except Exception as exc:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Upload failed: {exc}")

    jid = _new_job()
    logger.info(
        f"[{jid}] Accepted {safe_name} "
        f"({total_read / 1024 / 1024:.1f} MB), "
        f"analysis_type={analysis_type}"
    )

    background_tasks.add_task(_run_pipeline, jid, file_path, analysis_type)

    return JSONResponse(
        status_code=202,
        content={
            "job_id":  jid,
            "message": "File accepted. Poll /status/{job_id} for progress.",
        },
    )


@app.get("/status/{job_id}")
async def get_status(job_id: str):
    """
    Poll job progress.

    Response
    --------
    job_id    : str
    status    : pending | extracting | analyzing | synthesizing | done | failed
    progress  : {stage: str, current: int, total: int}
    error     : str  (only present when status == "failed")

    Common error messages
    ----------------------
    Image-only PDF, no OCR:
      "This PDF is entirely image-based … Install PaddleOCR-VL …"
    Too little text:
      "Extracted text is too short to summarise (N characters after cleaning)."
    Corrupt / encrypted:
      "Could not open PDF: …"
    """
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")

    resp: dict = {
        "job_id":   job_id,
        "status":   job["status"],
        "progress": job["progress"],
    }
    if job["status"] == "failed":
        resp["error"] = job["error"]
    return resp


@app.get("/result/{job_id}")
async def get_result(job_id: str):
    """
    Retrieve the final analysis result.

    HTTP 200  — processing done; result JSON returned.
    HTTP 202  — still processing; check /status for progress.
    HTTP 404  — unknown job_id.
    HTTP 500  — job failed; detail contains the human-readable reason.

    Result JSON (analysis_type=0)
    ------------------------------
    {
      "overview":   "1-2 sentence document description",
      "summary":    "5-7 sentence executive summary",
      "highlights": ["fact 1", "fact 2", … up to 10]
    }
    """
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")

    if job["status"] == "failed":
        raise HTTPException(
            status_code=500,
            detail=job.get("error", "Unknown error. Check /status for details."),
        )

    if job["status"] != "done":
        return JSONResponse(
            status_code=202,
            content={
                "job_id":   job_id,
                "status":   job["status"],
                "progress": job["progress"],
                "message":  "Processing in progress. Try again shortly.",
            },
        )

    return job["result"]