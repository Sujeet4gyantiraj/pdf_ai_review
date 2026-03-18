from fastapi import FastAPI, UploadFile, File, HTTPException
import os
import uuid
import logging
from transformers import AutoTokenizer

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