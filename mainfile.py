from fastapi import FastAPI, UploadFile, File, Query, HTTPException
import os
import uuid
import json
import re
import logging
import time
from pdf_util import extract_text_from_pdf, chunk_text
from ai_models import generate_analysis

# ----------------- LOGGING CONFIG -----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("main")

# --------------------------------------------------

app = FastAPI()

UPLOAD_FOLDER = "temp"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def clean_text(text: str) -> str:
    return text.replace("\n\n", "\n").strip()


def _fix_json_string(raw: str) -> str:
    raw = raw.replace("```json", "").replace("```", "").strip()
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    raw = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', raw)
    raw = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', raw)
    return raw


def extract_json(text: str) -> dict:
    empty = {"overview": "", "summary": "", "highlights": []}

    if not text:
        logger.warning("Empty AI response received.")
        return empty

    cleaned = _fix_json_string(text)

    try:
        data = json.loads(cleaned)
        return data
    except json.JSONDecodeError:
        logger.warning("Direct JSON parsing failed. Trying fallback extraction.")

    match = re.search(r'\{.*\}', cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError as e:
            logger.error(f"Fallback JSON parsing failed: {e}")

    logger.error("All JSON parsing strategies failed.")
    return empty


@app.post("/analyze")
async def analyze_pdf(
    file: UploadFile = File(...),
    analysis_type: int = Query(0, ge=0, le=3)
):
    start_time = time.time()
    logger.info(f"Received file: {file.filename}")

    if not file.filename.lower().endswith(".pdf"):
        logger.warning("Invalid file type uploaded.")
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    safe_name = f"{uuid.uuid4()}.pdf"
    file_path = os.path.join(UPLOAD_FOLDER, safe_name)

    try:
        # Save file
        with open(file_path, "wb") as f:
            f.write(await file.read())
        logger.info(f"File saved as {safe_name}")

        # Extract text
        logger.info("Starting text extraction...")
        text = extract_text_from_pdf(file_path)
        text = clean_text(text)

        if not text:
            logger.warning("No extractable text found.")
            raise HTTPException(
                status_code=422,
                detail="No extractable text found in PDF."
            )

        logger.info(f"Extracted text length: {len(text)} characters")

        # Chunking
        chunks = chunk_text(text)
        logger.info(f"Document split into {len(chunks)} chunks")

        results = []

        for i, chunk in enumerate(chunks, 1):
            logger.info(f"Processing chunk {i}/{len(chunks)} (size={len(chunk)})")
            ai_response = await generate_analysis(chunk)
            parsed = extract_json(ai_response)
            results.append(parsed)

    except Exception as e:
        logger.exception(f"Processing failed: {e}")
        raise

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info("Temporary file removed.")

    # Merge results (preserve order)
    combined_overview = " ".join(r.get("overview", "") for r in results).strip()
    combined_summary = " ".join(r.get("summary", "") for r in results).strip()

    seen = set()
    combined_highlights = []
    for r in results:
        for h in r.get("highlights", []):
            if h not in seen:
                seen.add(h)
                combined_highlights.append(h)

    total_time = round(time.time() - start_time, 2)
    logger.info(f"Request completed in {total_time} seconds")

    final_output = {
        "overview": combined_overview,
        "summary": combined_summary,
        "highlights": combined_highlights
    }

    if analysis_type == 1:
        return {"overview": final_output["overview"]}
    elif analysis_type == 2:
        return {"summary": final_output["summary"]}
    elif analysis_type == 3:
        return {"highlights": final_output["highlights"]}

    return final_output