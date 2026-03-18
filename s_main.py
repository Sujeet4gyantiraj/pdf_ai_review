from fastapi import FastAPI, UploadFile, File, Query, HTTPException
import os
import uuid
import json
import re
import logging
from s_padf_utils import load_pdf, split_documents, get_page_count
from s_ai_model import generate_analysis

logger = logging.getLogger(__name__)

app = FastAPI()

UPLOAD_FOLDER = "temp"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def _fix_json_string(raw: str) -> str:
    """
    Apply a sequence of targeted repairs to common Mistral JSON output problems,
    without altering the structure or values of well-formed output.
    """
    # Remove markdown code fences
    raw = raw.replace("```json", "").replace("```", "").strip()

    # Normalise Windows line endings
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")

    # Fix illegal (non-JSON) backslash escapes, e.g. \' \, \: \. etc.
    # Keep valid escapes: \" \\ \/ \b \f \n \r \t \uXXXX
    raw = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', raw)

    # Remove ASCII control characters (0x00-0x1F) except \n \r \t which are valid in JSON strings
    raw = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', raw)

    return raw


def _extract_field_by_regex(text: str, field: str) -> str:
    """
    Fallback: extract a single string field value directly with regex
    when the whole JSON block can't be parsed.
    """
    pattern = rf'"{field}"\s*:\s*"(.*?)"(?=\s*[,}}])'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def _extract_highlights_by_regex(text: str) -> list:
    """
    Fallback: extract highlights array items directly with regex.
    """
    match = re.search(r'"highlights"\s*:\s*\[(.*?)\]', text, re.DOTALL)
    if not match:
        return []
    inner = match.group(1)
    items = re.findall(r'"(.*?)"(?=\s*[,\]])', inner, re.DOTALL)
    return [i.strip() for i in items if i.strip()]


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
        return empty

    cleaned = _fix_json_string(text)

    # --- Strategy 1: parse the whole cleaned string directly ---
    try:
        data = json.loads(cleaned)
        return _postprocess_highlights(data)
    except json.JSONDecodeError:
        pass

    # --- Strategy 2: isolate the outermost { ... } block ---
    brace_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
    if brace_match:
        candidate = brace_match.group()
        try:
            data = json.loads(candidate)
            return _postprocess_highlights(data)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON block parse failed ({e}), trying character-level repair")

            # Sub-strategy: attempt to repair unescaped double-quotes inside string values.
            # Replace any " that is NOT preceded by \ and NOT a structural delimiter
            # (i.e. not after : [ , { or before ] } ,) with \".
            # This is intentionally conservative — only fixes the most common model error.
            repaired = re.sub(
                r'(?<=[^\\])"(?=[^,\]}\n:}{\[])',
                r'\\"',
                candidate
            )
            try:
                data = json.loads(repaired)
                return _postprocess_highlights(data)
            except json.JSONDecodeError:
                pass

    # --- Strategy 3: regex field extraction (last resort) ---
    logger.error("Extraction Error: all JSON parse strategies failed, falling back to regex extraction")
    overview = _extract_field_by_regex(cleaned, "overview")
    summary = _extract_field_by_regex(cleaned, "summary")
    highlights = _extract_highlights_by_regex(cleaned)

    if overview or summary or highlights:
        return {"overview": overview, "summary": summary, "highlights": highlights}

    return empty


MAX_PDF_PAGES = 200   # hard page cap — raise if your GPU can handle larger docs

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

    # Validate file type
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    # Use a random UUID filename to prevent path traversal attacks
    safe_name = f"{uuid.uuid4()}.pdf"
    file_path = os.path.join(UPLOAD_FOLDER, safe_name)

    try:
        # Save uploaded file
        with open(file_path, "wb") as f:
            f.write(await file.read())

        # Page count guard — reject oversized PDFs before extraction
        page_count = get_page_count(file_path)
        if page_count > MAX_PDF_PAGES:
            raise HTTPException(
                status_code=413,
                detail=f"PDF has {page_count} pages. Maximum allowed is {MAX_PDF_PAGES}."
            )

        # Load PDF into LangChain Documents (one per page, image pages filtered)
        # Raises ValueError → 422 if no text found or file unreadable
        try:
            pages = load_pdf(file_path)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))

        # Split pages into overlapping chunks with RecursiveCharacterTextSplitter
        chunks = split_documents(pages)

        # Run direct map-reduce inference over all chunks.
        # generate_analysis handles map + reduce internally and returns
        # a fully parsed dict — no extract_json call needed here.
        final_output = await generate_analysis(chunks)

    finally:
        # Always clean up the uploaded file from disk
        if os.path.exists(file_path):
            os.remove(file_path)

    # Return selected section if requested
    if analysis_type == 1:
        return {"overview": final_output.get("overview", "")}
    elif analysis_type == 2:
        return {"summary": final_output.get("summary", "")}
    elif analysis_type == 3:
        return {"highlights": final_output.get("highlights", [])}

    return final_output