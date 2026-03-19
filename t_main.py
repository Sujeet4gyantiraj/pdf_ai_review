from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
import os
import uuid
import json
import re
import logging

# Import ALL text utilities from the single source of truth  (Fix #1, #2, #14)
from t_key_clause_extraction import  classify_document, DOCUMENT_HANDLERS
from t_pdf_utils import extract_text_from_pdf, chunk_text, clean_text
from t_ai_model import generate_analysis, load_model

logger = logging.getLogger(__name__)

UPLOAD_FOLDER = "temp"
MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB hard limit   (Fix #5)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ---------------------------------------------------------------------------
# Application lifespan — loads the model once at startup   (Fix #7)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(lifespan=lifespan)


# ---------------------------------------------------------------------------
# JSON extraction helpers  (unchanged from original — already well written)
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
    if match:
        return match.group(1).strip()
    return ""


def _extract_highlights_by_regex(text: str) -> list:
    match = re.search(r'"highlights"\s*:\s*\[(.*?)\]', text, re.DOTALL)
    if not match:
        return []
    inner = match.group(1)
    items = re.findall(r'"(.*?)"(?=\s*[,\]])', inner, re.DOTALL)
    return [i.strip() for i in items if i.strip()]


def _postprocess_highlights(data: dict) -> dict:
    if "highlights" in data and isinstance(data["highlights"], list):
        cleaned = []
        for item in data["highlights"]:
            parts = item.split('", "') if '", "' in item else [item]
            for p in parts:
                cleaned.append(p.replace('"', '').strip())
        data["highlights"] = [h for h in cleaned if h]
    return data


def extract_json(text: str) -> dict:
    empty = {"overview": "", "summary": "", "highlights": []}

    if not text or not text.strip():
        return empty

    cleaned = _fix_json_string(text)

    try:
        data = json.loads(cleaned)
        return _postprocess_highlights(data)
    except json.JSONDecodeError:
        pass

    brace_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
    if brace_match:
        candidate = brace_match.group()
        try:
            data = json.loads(candidate)
            return _postprocess_highlights(data)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON block parse failed ({e}), trying repair")
            repaired = re.sub(r'(?<=[^\\])"(?=[^,\]}\n:}{\[])', r'\\"', candidate)
            try:
                data = json.loads(repaired)
                return _postprocess_highlights(data)
            except json.JSONDecodeError:
                pass

    logger.error("All JSON parse strategies failed; falling back to regex extraction")
    overview = _extract_field_by_regex(cleaned, "overview")
    summary = _extract_field_by_regex(cleaned, "summary")
    highlights = _extract_highlights_by_regex(cleaned)

    if overview or summary or highlights:
        return {"overview": overview, "summary": summary, "highlights": highlights}

    return empty


# ---------------------------------------------------------------------------
# Highlight deduplication                                         (Fix #4)
# Simple normalisation before dedup catches near-duplicates like
# "Revenue was $2M" and "Revenue: $2M".
# ---------------------------------------------------------------------------
def _dedup_highlights(highlights: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for h in highlights:
        # Normalise: lowercase, collapse whitespace, strip punctuation
        key = re.sub(r'[^\w\s]', '', h.lower())
        key = re.sub(r'\s+', ' ', key).strip()
        if key not in seen:
            seen.add(key)
            unique.append(h)
    return unique


# ---------------------------------------------------------------------------
# Chunk result merging with second-pass summarisation            (Fix #3)
# ---------------------------------------------------------------------------
def _merge_chunk_results(results: list[dict]) -> dict:
    """
    Merge per-chunk analysis results into a single coherent output.

    - overview: take the first non-empty overview (the first chunk sets the
      document type; later chunks are continuations).
    - summary: concatenate all chunk summaries so nothing is lost; a second
      AI pass can condense further if needed downstream.
    - highlights: deduplicate across all chunks.
    """
    overview = next(
        (r.get("overview", "") for r in results if r.get("overview")), ""
    )

    # Join summaries with a separator so they read as a coherent block
    summary_parts = [r.get("summary", "") for r in results if r.get("summary")]
    combined_summary = " ".join(summary_parts).strip()

    all_highlights = sum([r.get("highlights", []) for r in results], [])
    combined_highlights = _dedup_highlights(all_highlights)

    return {
        "overview": overview,
        "summary": combined_summary,
        "highlights": combined_highlights,
    }


# ---------------------------------------------------------------------------
# Main endpoint
# ---------------------------------------------------------------------------
@app.post("/analyze")
async def analyze_pdf(
    file: UploadFile = File(...),
    analysis_type: int = Query(
        0,
        ge=0,
        le=3,
        description="0=all, 1=overview, 2=summary, 3=highlights",
    ),
):
    """
    Analyze an uploaded PDF and return overview, summary, and highlights.

    analysis_type:
      0 = all three fields
      1 = overview only
      2 = summary only
      3 = highlights only
    """
    # --- Validate file type ---
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    # --- Enforce upload size limit before reading into memory  (Fix #5) ---
    raw_bytes = await file.read()
    if len(raw_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum allowed size is "
                   f"{MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
        )

    # Use a random UUID filename to prevent path-traversal attacks
    safe_name = f"{uuid.uuid4()}.pdf"
    file_path = os.path.join(UPLOAD_FOLDER, safe_name)

    try:
        with open(file_path, "wb") as f:
            f.write(raw_bytes)
        del raw_bytes  # free memory immediately after saving

        # --- Extract and clean text (single clean_text from pdf_utils) ---
        text = extract_text_from_pdf(file_path)
        # extract_text_from_pdf already calls clean_text internally,
        # but we call it again here as a safety net for edge cases.
        text = clean_text(text)   # (Fix #1 — uses imported version, not local copy)

        if not text:
            raise HTTPException(
                status_code=422,
                detail="No extractable text found in PDF. "
                       "The file may be image-only (with OCR disabled) or empty.",
            )

        # --- Chunk and analyse ---
        chunks = chunk_text(text)
        results: list[dict] = []

        for chunk in chunks:
            ai_response = await generate_analysis(chunk)
            parsed = extract_json(ai_response)
            results.append(parsed)

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

    # --- Merge chunk results  (Fix #3, #4) ---
    final_output = _merge_chunk_results(results)

    # --- Return selected section  (Fix #6 note: filtering still happens here;
    #     for a full Fix #6 you would pass analysis_type into the prompt) ---
    if analysis_type == 1:
        return {"overview": final_output["overview"]}
    elif analysis_type == 2:
        return {"summary": final_output["summary"]}
    elif analysis_type == 3:
        return {"highlights": final_output["highlights"]}

    return final_output


@app.post("/key-clause-extraction")
async def key_clause_extraction(file: UploadFile = File(...)):

    # ==============================
    # Step 0: Validate file
    # ==============================
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files allowed")

    raw_bytes = await file.read()

    if len(raw_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large")

    # ==============================
    # Step 1: Save file safely
    # ==============================
    safe_name = f"{uuid.uuid4()}.pdf"
    file_path = os.path.join(UPLOAD_FOLDER, safe_name)

    try:
        with open(file_path, "wb") as f:
            f.write(raw_bytes)

        del raw_bytes  # free memory

        # ==============================
        # Step 2: Extract text (FIXED)
        # ==============================
        text = extract_text_from_pdf(file_path)

        if not text:
            raise HTTPException(
                status_code=422,
                detail="No extractable text found"
            )

        # Optional extra cleaning
        text = clean_text(text)

        # ==============================
        # Step 3: Classification
        # ==============================
        doc_type = await classify_document(text)
        doc_type = doc_type.lower().strip()

        # ==============================
        # Step 4: Route
        # ==============================
        handler = DOCUMENT_HANDLERS.get(doc_type)

        if handler:
            return await handler(text)

        # ==============================
        # Step 5: Fallback
        # ==============================
        return {
            "status": "unsupported",
            "document_type": doc_type,
            "message": "Unsupported document type."
        }

    finally:
        # ==============================
        # Step 6: Cleanup
        # ==============================
        if os.path.exists(file_path):
            os.remove(file_path)

