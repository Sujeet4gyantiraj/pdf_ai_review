from fastapi import FastAPI, UploadFile, File, HTTPException
import os
import uuid

import json
import re

from pdf_util import extract_text_from_pdf, chunk_text
from ai_models import generate_analysis


import logging
from transformers import AutoTokenizer



logger = logging.getLogger(__name__)

# ----------------- LOGGING CONFIG -----------------

from pdf_utils import extract_text_from_pdf, chunk_by_tokens
from ai_model import summarize_chunk, generate_final_summary

# ---------------- LOGGING ----------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ---------------- APP ----------------
app = FastAPI()

UPLOAD_FOLDER = "temp"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")


@app.post("/summarize")
async def summarize_pdf(file: UploadFile = File(...)):

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF allowed")

    file_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_FOLDER, f"{file_id}.pdf")

    try:
        with open(file_path, "wb") as f:
            f.write(await file.read())

        logger.info("Extracting text from PDF...")
        text = extract_text_from_pdf(file_path)

        if not text.strip():
            raise HTTPException(status_code=422, detail="No text extracted")

        logger.info("Chunking text by tokens...")
        chunks = chunk_by_tokens(text, tokenizer)

        logger.info(f"Total chunks created: {len(chunks)}")

        # -------- Level 1 Summaries --------
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            logger.info(f"Summarizing chunk {i+1}/{len(chunks)}")
            summary = summarize_chunk(chunk)
            chunk_summaries.append(summary)

        # -------- Level 2 Final Summary --------
        logger.info("Generating final hierarchical summary...")
        final_summary = generate_final_summary(chunk_summaries)

        return {"result": final_summary}

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

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

            os.remove(file_path)

