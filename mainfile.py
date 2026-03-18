from fastapi import FastAPI, UploadFile, File, Query, HTTPException
import os
import uuid
import json
import re

from pdf_util import extract_text_from_pdf, chunk_text
from ai_models import generate_analysis

import logging
from pdf_util import extract_text_from_pdf, chunk_text
from ai_models import generate_analysis


logger = logging.getLogger(__name__)

app = FastAPI()

UPLOAD_FOLDER = "temp"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def clean_text(text: str) -> str:
    text = text.replace("\n\n", "\n")
    return text.strip()


def _fix_json_string(raw: str) -> str:
    raw = raw.replace("```json", "").replace("```", "").strip()
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    raw = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', raw)
    raw = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', raw)
    return raw


def extract_json(text: str) -> dict:
    empty = {"overview": "", "summary": "", "highlights": []}
    if not text:
        return empty

    cleaned = _fix_json_string(text)

    try:
        data = json.loads(cleaned)
        return data
    except:
        pass

    match = re.search(r'\{.*\}', cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except:
            pass

    return empty


@app.post("/analyze")
async def analyze_pdf(
    file: UploadFile = File(...),
    analysis_type: int = Query(0, ge=0, le=3)
):

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    safe_name = f"{uuid.uuid4()}.pdf"
    file_path = os.path.join(UPLOAD_FOLDER, safe_name)

    try:
        with open(file_path, "wb") as f:
            f.write(await file.read())

        text = extract_text_from_pdf(file_path)
        text = clean_text(text)

        if not text:
            raise HTTPException(
                status_code=422,
                detail="No extractable text found in PDF."
            )

        chunks = chunk_text(text)

        results = []
        for chunk in chunks:
            ai_response = await generate_analysis(chunk)
            parsed = extract_json(ai_response)
            results.append(parsed)

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

    # Merge safely while preserving order
    combined_overview = " ".join(r.get("overview", "") for r in results).strip()
    combined_summary = " ".join(r.get("summary", "") for r in results).strip()

    seen = set()
    combined_highlights = []
    for r in results:
        for h in r.get("highlights", []):
            if h not in seen:
                seen.add(h)
                combined_highlights.append(h)

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
