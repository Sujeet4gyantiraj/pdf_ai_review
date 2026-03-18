from fastapi import FastAPI, UploadFile, File, Query, HTTPException
import os
import uuid
import json
import re
import time
import logging
import logging.config
from s_padf_utils import load_pdf, split_documents, get_page_count
from s_ai_model import generate_analysis

# ---------------------------------------------------------------------------
# Logging configuration
# One-time setup: writes INFO+ to console and DEBUG+ to app.log
# Both handlers use the same structured format so grep works on either.
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
            "maxBytes":  10 * 1024 * 1024,   # 10 MB per file
            "backupCount": 5,                  # keep 5 rotated files
            "encoding":  "utf-8",
        },
    },
    "root": {
        "level":    "DEBUG",
        "handlers": ["console", "file"],
    },
})

logger = logging.getLogger(__name__)

app = FastAPI()

UPLOAD_FOLDER = "temp"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
logger.info(f"Upload folder ready: {os.path.abspath(UPLOAD_FOLDER)}")


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
    """
    Fallback: extract a single string field value directly with regex
    when the whole JSON block can't be parsed.
    """
    logger.debug(f"_extract_field_by_regex: extracting field '{field}'")
    pattern = rf'"{field}"\s*:\s*"(.*?)"(?=\s*[,}}])'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        logger.debug(f"_extract_field_by_regex: found '{field}' via regex")
        return match.group(1).strip()
    logger.warning(f"_extract_field_by_regex: field '{field}' not found")
    return ""


def _extract_highlights_by_regex(text: str) -> list:
    """
    Fallback: extract highlights array items directly with regex.
    """
    logger.debug("_extract_highlights_by_regex: attempting regex extraction")
    match = re.search(r'"highlights"\s*:\s*\[(.*?)\]', text, re.DOTALL)
    if not match:
        logger.warning("_extract_highlights_by_regex: highlights array not found")
        return []
    inner = match.group(1)
    items = re.findall(r'"(.*?)"(?=\s*[,\]])', inner, re.DOTALL)
    result = [i.strip() for i in items if i.strip()]
    logger.debug(f"_extract_highlights_by_regex: found {len(result)} item(s)")
    return result


def _postprocess_highlights(data: dict) -> dict:
    """
    Post-process highlights list: flatten items the model packed into a single
    comma-separated string, and strip stray quote characters.
    """
    if "highlights" in data and isinstance(data["highlights"], list):
        cleaned = []
        for item in data["highlights"]:
            parts = item.split('", "') if '", "' in item else [item]
            for p in parts:
                cleaned.append(p.replace('"', '').strip())
        data["highlights"] = [h for h in cleaned if h]
        logger.debug(f"_postprocess_highlights: {len(data['highlights'])} highlight(s) after cleanup")
    return data


def extract_json(text: str) -> dict:
    """
    Robustly extract and structure JSON from Mistral output using three
    fallback strategies so a single malformed character never silently
    returns an empty result.

    Strategy 1 — direct parse after cleaning
    Strategy 2 — isolate outermost { ... } block, then parse; on failure
                 attempt a conservative in-string quote-escape repair
    Strategy 3 — per-field regex extraction as last resort
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
        return _postprocess_highlights(data)
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
            return _postprocess_highlights(data)
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
                return _postprocess_highlights(data)
            except json.JSONDecodeError as e2:
                logger.warning(f"extract_json: strategy 2 repair also failed — {e2}")
    else:
        logger.warning("extract_json: no brace block found in output")

    # --- Strategy 3: regex field extraction (last resort) ---
    logger.error(
        "extract_json: all JSON parse strategies failed — "
        "falling back to per-field regex extraction"
    )
    overview   = _extract_field_by_regex(cleaned, "overview")
    summary    = _extract_field_by_regex(cleaned, "summary")
    highlights = _extract_highlights_by_regex(cleaned)

    if overview or summary or highlights:
        logger.info("extract_json: strategy 3 recovered partial data via regex")
        return {"overview": overview, "summary": summary, "highlights": highlights}

    logger.error("extract_json: strategy 3 also failed — returning empty result")
    return empty


# ---------------------------------------------------------------------------
# /analyze endpoint
# ---------------------------------------------------------------------------

MAX_PDF_PAGES = 500   # hard page cap — raise if your GPU can handle larger docs


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
    Analyze uploaded PDF and return:
    - overview
    - summary
    - highlights

    Uses LangChain RecursiveCharacterTextSplitter for chunking and a
    direct map-reduce inference pipeline for large-PDF coherence.
    """
    # Unique ID per request — appears in every log line for this request
    # so you can grep request_id to reconstruct the full trace of one upload
    request_id = str(uuid.uuid4())[:8]
    t_start    = time.perf_counter()

    logger.info(f"[{request_id}] ── NEW REQUEST ──────────────────────────────")
    logger.info(f"[{request_id}] filename='{file.filename}' analysis_type={analysis_type}")

    # Validate file type
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        logger.warning(f"[{request_id}] Rejected: not a PDF file (filename='{file.filename}')")
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    # Use a random UUID filename to prevent path traversal attacks
    safe_name = f"{uuid.uuid4()}.pdf"
    file_path = os.path.join(UPLOAD_FOLDER, safe_name)
    logger.debug(f"[{request_id}] Saving upload to '{file_path}'")

    try:
        # Step 1 — save upload
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        logger.info(f"[{request_id}] Step 1/5 — saved upload ({len(content):,} bytes → '{safe_name}')")

        # Step 2 — page count guard
        page_count = get_page_count(file_path)
        logger.info(f"[{request_id}] Step 2/5 — page count: {page_count} (limit={MAX_PDF_PAGES})")
        if page_count > MAX_PDF_PAGES:
            logger.warning(
                f"[{request_id}] Rejected: PDF has {page_count} pages, "
                f"exceeds limit of {MAX_PDF_PAGES}"
            )
            raise HTTPException(
                status_code=413,
                detail=f"PDF has {page_count} pages. Maximum allowed is {MAX_PDF_PAGES}."
            )

        # Step 3 — text extraction (native + OCR fallback per page)
        logger.info(f"[{request_id}] Step 3/5 — extracting text from {page_count} page(s)")
        try:
            t_extract = time.perf_counter()
            pages     = load_pdf(file_path)
            logger.info(
                f"[{request_id}] Step 3/5 — extraction complete: "
                f"{len(pages)} page(s) with text "
                f"({time.perf_counter() - t_extract:.2f}s)"
            )
        except ValueError as e:
            logger.error(f"[{request_id}] Step 3/5 — extraction failed: {e}")
            raise HTTPException(status_code=422, detail=str(e))

        # Step 4 — chunking
        logger.info(f"[{request_id}] Step 4/5 — splitting pages into chunks")
        t_chunk = time.perf_counter()
        chunks  = split_documents(pages)
        logger.info(
            f"[{request_id}] Step 4/5 — chunking complete: "
            f"{len(chunks)} chunk(s) from {len(pages)} page(s) "
            f"({time.perf_counter() - t_chunk:.2f}s)"
        )

        # Step 5 — map-reduce inference
        logger.info(
            f"[{request_id}] Step 5/5 — running map-reduce inference "
            f"({len(chunks)} chunk(s))"
        )
        t_infer      = time.perf_counter()
        final_output = await generate_analysis(chunks)
        logger.info(
            f"[{request_id}] Step 5/5 — inference complete "
            f"({time.perf_counter() - t_infer:.2f}s)"
        )

    except HTTPException:
        # Re-raise FastAPI HTTP errors without wrapping them
        raise

    except Exception as e:
        logger.exception(f"[{request_id}] Unhandled error during processing: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during PDF analysis.")

    finally:
        # Always clean up the uploaded file from disk
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.debug(f"[{request_id}] Temp file deleted: '{file_path}'")

    elapsed = time.perf_counter() - t_start
    logger.info(
        f"[{request_id}] ── REQUEST COMPLETE — total time: {elapsed:.2f}s ──────"
    )

    # Return selected section if requested
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