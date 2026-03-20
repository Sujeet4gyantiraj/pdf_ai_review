from fastapi import FastAPI, UploadFile, File, Query, HTTPException
import os
import uuid
import json
import re
import time
import asyncio
import logging
import logging.config
from contextlib import asynccontextmanager
from s_padf_utils import load_pdf, get_page_count, all_pages_blank
from s_ai_model import generate_analysis
from t_key_clause_extraction import  classify_document, DOCUMENT_HANDLERS, extract_text_from_upload
from t_risk_detection import analyze_document_risks
from t_ai_model import load_model


# ---------------------------------------------------------------------------
# Logging configuration
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
    "root": {
        "level":    "DEBUG",
        "handlers": ["console", "file"],
    },
})

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the AI model on startup
    logger.info("Starting up and loading AI model...")
    load_model()
    yield
    # No cleanup specified, but this is where it would go
    logger.info("Shutting down.")


app = FastAPI(lifespan=lifespan)

UPLOAD_FOLDER = "temp"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
logger.info(f"Upload folder ready: {os.path.abspath(UPLOAD_FOLDER)}")

# ---------------------------------------------------------------------------
# GPU semaphore — serialises all inference calls to prevent concurrent GPU
# access which causes deadlock under multiple simultaneous requests.
#
# Without this: two requests both call run_in_executor → two threads both
# block on GPU → thread pool exhausted → event loop freezes.
#
# With this: only one request enters generate_analysis() at a time.
# All others await here on the event loop (zero cost, no thread consumed).
# ---------------------------------------------------------------------------
_gpu_semaphore = asyncio.Semaphore(1)


# ---------------------------------------------------------------------------
# JSON extraction helpers
# ---------------------------------------------------------------------------

def _fix_json_string(raw: str) -> str:
    """
    Apply a sequence of targeted repairs to common Mistral JSON output problems,
    without altering the structure or values of well-formed output.
    """
    logger.debug("_fix_json_string: starting string repair")

    # Remove markdown code fences
    raw = raw.replace("```json", "").replace("```", "").strip()

    # Normalise Windows line endings
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")

    # Fix illegal (non-JSON) backslash escapes, e.g. \' \, \: \. etc.
    # Keep valid escapes: \" \\ \/ \b \f \n \r \t \uXXXX
    raw = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', raw)

    # Remove ASCII control characters (0x00-0x1F) except \n \r \t
    raw = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', raw)

    logger.debug(f"_fix_json_string: repair complete ({len(raw)} chars)")
    return raw


def _extract_field_by_regex(text: str, field: str) -> str:
    logger.debug(f"_extract_field_by_regex: extracting field '{field}'")
    pattern = rf'"{field}"\s*:\s*"(.*?)"(?=\s*[,}}])'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        logger.debug(f"_extract_field_by_regex: found '{field}' via regex")
        return match.group(1).strip()
    logger.warning(f"_extract_field_by_regex: field '{field}' not found")
    return ""


def _extract_highlights_by_regex(text: str) -> list:
    logger.debug("_extract_highlights_by_regex: attempting regex extraction")
    match = re.search(r'"highlights"\s*:\s*\[(.*?)\]', text, re.DOTALL)
    if not match:
        # FIX (Issue 2): also try extracting from a truncated/unclosed array —
        # when Mistral hits MAX_NEW_TOKENS the highlights array is cut off mid-way.
        # Find the opening bracket and grab all complete quoted strings before cutoff.
        match_open = re.search(r'"highlights"\s*:\s*\[(.*)', text, re.DOTALL)
        if not match_open:
            logger.warning("_extract_highlights_by_regex: highlights array not found")
            return []
        inner = match_open.group(1)
        logger.debug("_extract_highlights_by_regex: found truncated highlights array")
    else:
        inner = match.group(1)

    items = re.findall(r'"((?:[^"\\]|\\.)*)"', inner, re.DOTALL)
    result = [i.strip() for i in items if i.strip()]
    logger.debug(f"_extract_highlights_by_regex: found {len(result)} item(s)")
    return result


def _postprocess_highlights(data: dict) -> dict:
    if "highlights" in data and isinstance(data["highlights"], list):
        cleaned = []
        for item in data["highlights"]:
            parts = item.split('", "') if '", "' in item else [item]
            for p in parts:
                cleaned.append(p.replace('"', '').strip())
        data["highlights"] = [h for h in cleaned if h]
        logger.debug(f"_postprocess_highlights: {len(data['highlights'])} highlight(s) after cleanup")
    return data


def _normalize_parsed(data, label: str = "") -> dict:
    """
    Ensure json.loads() output is always a dict with the expected keys.
    Handles the case where Mistral returns a JSON array instead of an object.
    """
    empty = {"overview": "", "summary": "", "highlights": []}
    if isinstance(data, dict):
        return data
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
            return merged
        if isinstance(data[0], str):
            return {"overview": "", "summary": "", "highlights": [s for s in data if s]}
    logger.error(f"extract_json {label}: unexpected type {type(data).__name__} — using empty sentinel")
    return empty


def _recover_truncated_json(text: str) -> dict | None:
    """
    FIX (Issue 2): Handle the case where Mistral hits MAX_NEW_TOKENS and the
    JSON output is cut off mid-stream (always at exactly 600 tokens_out).

    The model reliably outputs fields in order: overview → summary → highlights.
    So a truncated response has overview and summary complete, but highlights
    is an unclosed array. We extract what we can from each field individually.
    """
    result = {"overview": "", "summary": "", "highlights": []}

    # Extract overview — usually always complete even in truncated output
    ov_match = re.search(r'"overview"\s*:\s*"((?:[^"\\]|\\.)*)"', text, re.DOTALL)
    if ov_match:
        result["overview"] = ov_match.group(1).strip()

    # Extract summary — usually complete too
    sm_match = re.search(r'"summary"\s*:\s*"((?:[^"\\]|\\.)*)"', text, re.DOTALL)
    if sm_match:
        result["summary"] = sm_match.group(1).strip()

    # Extract whatever highlights were written before truncation
    result["highlights"] = _extract_highlights_by_regex(text)

    if result["overview"] or result["summary"] or result["highlights"]:
        logger.info(
            f"_recover_truncated_json: recovered overview={'yes' if result['overview'] else 'no'} "
            f"summary={'yes' if result['summary'] else 'no'} "
            f"highlights={len(result['highlights'])}"
        )
        return result
    return None


def extract_json(text: str) -> dict:
    """
    Robustly extract and structure JSON from Mistral output using four
    strategies so truncation and malformed output never silently lose data.

    Strategy 1 — direct parse after cleaning
    Strategy 2 — isolate outermost { ... } block, then parse; on failure
                 attempt a conservative in-string quote-escape repair
    Strategy 3 — truncation recovery (for outputs cut at MAX_NEW_TOKENS)
    Strategy 4 — per-field regex extraction as last resort
    """
    empty = {"overview": "", "summary": "", "highlights": []}

    if not text or not text.strip():
        logger.warning("extract_json: received empty text, returning empty result")
        return empty

    logger.debug(f"extract_json: input length {len(text)} chars")
    cleaned = _fix_json_string(text)

    # --- Strategy 1: parse the whole cleaned string directly ---
    logger.debug("extract_json: trying strategy 1 — direct json.loads")
    try:
        data = json.loads(cleaned)
        logger.debug("extract_json: strategy 1 succeeded")
        return _postprocess_highlights(_normalize_parsed(data, "strategy-1"))
    except json.JSONDecodeError as e:
        logger.debug(f"extract_json: strategy 1 failed — {e}")

    # --- Strategy 2: isolate the outermost { ... } block ---
    logger.debug("extract_json: trying strategy 2 — isolate brace block")
    brace_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
    if brace_match:
        candidate = brace_match.group()
        try:
            data = json.loads(candidate)
            logger.debug("extract_json: strategy 2 succeeded")
            return _postprocess_highlights(_normalize_parsed(data, "strategy-2"))
        except json.JSONDecodeError as e:
            logger.warning(f"extract_json: strategy 2 failed ({e}), trying quote-escape repair")

            repaired = re.sub(
                r'(?<=[^\\])"(?=[^,\]}\n:}{\[])',
                r'\\"',
                candidate
            )
            try:
                data = json.loads(repaired)
                logger.debug("extract_json: strategy 2 repair succeeded")
                return _postprocess_highlights(_normalize_parsed(data, "strategy-2-repair"))
            except json.JSONDecodeError as e2:
                logger.warning(f"extract_json: strategy 2 repair also failed — {e2}")
    else:
        logger.warning("extract_json: no brace block found in output")

    # --- Strategy 3: truncation recovery ---
    # Handles the frequent case where 600 tokens_out = output cut mid-JSON.
    # Recovers overview + summary (usually complete) + partial highlights.
    logger.debug("extract_json: trying strategy 3 — truncation recovery")
    recovered = _recover_truncated_json(cleaned)
    if recovered:
        logger.info("extract_json: strategy 3 (truncation recovery) succeeded")
        return _postprocess_highlights(recovered)

    # --- Strategy 4: regex field extraction (last resort) ---
    logger.error(
        "extract_json: all JSON parse strategies failed — "
        "falling back to per-field regex extraction"
    )
    overview   = _extract_field_by_regex(cleaned, "overview")
    summary    = _extract_field_by_regex(cleaned, "summary")
    highlights = _extract_highlights_by_regex(cleaned)

    if overview or summary or highlights:
        logger.info("extract_json: strategy 4 recovered partial data via regex")
        return {"overview": overview, "summary": summary, "highlights": highlights}

    logger.error("extract_json: strategy 4 also failed — returning empty result")
    return empty


# ---------------------------------------------------------------------------
# /analyze endpoint
# ---------------------------------------------------------------------------

MAX_PDF_PAGES = None
_BLANK_PDF_RESULT = {"overview": "", "summary": "", "highlights": []}


@app.post("/analyze")
async def analyze_pdf(
    file: UploadFile = File(...),
    analysis_type: int = Query(
        0,
        ge=0,
        le=3,
        description="0=all, 1=overview, 2=summary, 3=highlights"
    )
):
    """
    Analyze uploaded PDF and return overview, summary, and highlights.

    Concurrent requests are handled safely:
    - Upload, page count, and text extraction run freely in parallel.
    - GPU inference is serialised via asyncio.Semaphore(1) — only one request
      runs inference at a time, others wait on the event loop (not in a thread)
      so the server never deadlocks or freezes under concurrent load.
    """
    request_id = str(uuid.uuid4())[:8]
    t_start    = time.perf_counter()

    logger.info(f"[{request_id}] ── NEW REQUEST ──────────────────────────────")
    logger.info(f"[{request_id}] filename='{file.filename}' analysis_type={analysis_type}")

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        logger.warning(f"[{request_id}] Rejected: not a PDF file (filename='{file.filename}')")
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    safe_name = f"{uuid.uuid4()}.pdf"
    file_path = os.path.join(UPLOAD_FOLDER, safe_name)
    logger.debug(f"[{request_id}] Saving upload to '{file_path}'")

    was_truncated = False
    pages_to_read = 0
    total_pages   = 0

    try:
        # Step 1 — save upload
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        logger.info(f"[{request_id}] Step 1/5 — saved upload ({len(content):,} bytes → '{safe_name}')")

        # Step 2 — page count
        total_pages   = get_page_count(file_path)
        pages_to_read = total_pages if MAX_PDF_PAGES is None else min(total_pages, MAX_PDF_PAGES)
        was_truncated = MAX_PDF_PAGES is not None and total_pages > MAX_PDF_PAGES

        logger.info(
            f"[{request_id}] Step 2/5 — total pages: {total_pages}, "
            f"will analyse: {pages_to_read} "
            f"{'(TRUNCATED — large PDF)' if was_truncated else '(all pages)'}"
        )

        # Step 3 — text extraction (runs freely, no semaphore needed)
        logger.info(f"[{request_id}] Step 3/5 — extracting text from {pages_to_read} page(s)")
        try:
            t_extract = time.perf_counter()
            pages     = load_pdf(file_path, max_pages=pages_to_read)
            logger.info(
                f"[{request_id}] Step 3/5 — extraction complete: "
                f"{len(pages)} page(s) with text "
                f"({time.perf_counter() - t_extract:.2f}s)"
            )
        except ValueError as e:
            logger.error(f"[{request_id}] Step 3/5 — extraction failed: {e}")
            raise HTTPException(status_code=422, detail=str(e))

        # Blank PDF short-circuit — checked before acquiring the semaphore
        # so blank PDFs never consume a GPU slot
        if all_pages_blank(pages):
            elapsed = time.perf_counter() - t_start
            logger.warning(
                f"[{request_id}] Blank PDF detected — all {len(pages)} page(s) "
                f"are blank or unreadable. Skipping inference. ({elapsed:.2f}s)"
            )
            result = dict(_BLANK_PDF_RESULT)
            result["blank_pdf"] = True
            if was_truncated:
                result["truncated"]      = True
                result["pages_analysed"] = pages_to_read
                result["total_pages"]    = total_pages
            if analysis_type == 1:
                return {"overview": ""}
            elif analysis_type == 2:
                return {"summary": ""}
            elif analysis_type == 3:
                return {"highlights": []}
            return result

        # Step 4 — merge pages
        logger.info(f"[{request_id}] Step 4/5 — merging {len(pages)} page(s) into text")
        t_merge     = time.perf_counter()
        merged_text = "\n\n".join(p.page_content for p in pages)
        logger.info(
            f"[{request_id}] Step 4/5 — merged: {len(merged_text):,} chars "
            f"({time.perf_counter() - t_merge:.3f}s)"
        )

        # Step 5 — GPU inference, serialised by semaphore.
        # Other requests wait here on the event loop (non-blocking) until
        # the current inference finishes and releases the semaphore.
        t_wait = time.perf_counter()
        async with _gpu_semaphore:
            wait_time = time.perf_counter() - t_wait
            if wait_time > 0.1:
                logger.info(f"[{request_id}] waited {wait_time:.1f}s for GPU semaphore")

            logger.info(f"[{request_id}] Step 5/5 — running map-reduce inference")
            t_infer      = time.perf_counter()
            final_output = await generate_analysis(merged_text)
            logger.info(
                f"[{request_id}] Step 5/5 — inference complete "
                f"({time.perf_counter() - t_infer:.2f}s)"
            )

    except HTTPException:
        raise

    except Exception as e:
        logger.exception(f"[{request_id}] Unhandled error during processing: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during PDF analysis.")

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.debug(f"[{request_id}] Temp file deleted: '{file_path}'")

    elapsed = time.perf_counter() - t_start
    logger.info(f"[{request_id}] ── REQUEST COMPLETE — total time: {elapsed:.2f}s ──────")

    if was_truncated:
        final_output["truncated"]      = True
        final_output["pages_analysed"] = pages_to_read
        final_output["total_pages"]    = total_pages

    if analysis_type == 1:
        logger.debug(f"[{request_id}] Returning overview only")
        return {"overview": final_output.get("overview", "")}
    elif analysis_type == 2:
        logger.debug(f"[{request_id}] Returning summary only")
        return {"summary": final_output.get("summary", "")}
    elif analysis_type == 3:
        logger.debug(f"[{request_id}] Returning highlights only")
        return {"highlights": final_output.get("highlights", [])}

    return final_output

@app.post("/key-clause-extraction")
async def key_clause_extraction(file: UploadFile = File(...)):

    # ==============================
    # Step 1: text extraction
    # ==============================

    text, _, _, request_id, t_start, file_path = await extract_text_from_upload(
        file,
        endpoint="/key-clause-extraction"
    )

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
    # Reuse the PDF extraction utility you already wrote
    from t_key_clause_extraction import extract_text_from_upload
    
    # Extract text from PDF
    text, _, _, request_id, t_start, file_path = await extract_text_from_upload(
        file, 
        endpoint="/detect-risks"
    )

    try:
        logger.info(f"[{request_id}] Starting Risk Detection...")
        
        # Run the risk analysis
        result = await analyze_document_risks(text)
        
        # Return the final analysis
        return result

    except Exception as e:
        logger.exception(f"[{request_id}] Risk Analysis failed: {e}")
        raise HTTPException(status_code=500, detail="Internal error during risk detection.")

    finally:
        # Cleanup the temp PDF file
        import os
        if os.path.exists(file_path):
            os.remove(file_path)