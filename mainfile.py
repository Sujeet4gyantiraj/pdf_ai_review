

#==================this is final code ===========================

from fastapi import FastAPI, UploadFile, File, Query
import os
import json
import re
from pdf_util import extract_text_from_pdf, chunk_text
from ai_models import generate_analysis

app = FastAPI()

UPLOAD_FOLDER = "temp"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def clean_text(text: str) -> str:
    """Clean extracted PDF text"""
    text = text.replace("\n\n", "\n")
    return text.strip()

def extract_json(text: str):
    """Cleanly extract and structure JSON from Mistral output"""
    try:
        # 1. Clean up "Markdown-isms" like ```json ... ```
        text = text.replace("```json", "").replace("```", "").strip()
        
        # 2. Extract content between first { and last }
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            clean_str = match.group()
            # Fix illegal backslashes often found in model outputs
            clean_str = re.sub(r'(?<!\\)\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})', r'\\\\', clean_str)
            data = json.loads(clean_str)

            # 3. Post-Process 'highlights' to remove double-quotes if they exist
            if "highlights" in data and isinstance(data["highlights"], list):
                # Flatten the list and strip inner escaped quotes
                cleaned_highlights = []
                for item in data["highlights"]:
                    # Split if the model put everything in one string separated by commas
                    parts = item.split('", "') if '", "' in item else [item]
                    for p in parts:
                        cleaned_highlights.append(p.replace('"', '').strip())
                data["highlights"] = cleaned_highlights
            
            return data
    except Exception as e:
        print(f"Extraction Error: {e}")

    return {"overview": "", "summary": "", "highlights": []}




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
    """

    # Save uploaded file
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)

    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Extract text from PDF
    text = extract_text_from_pdf(file_path)
    text = clean_text(text)

    # Split into chunks (for large PDFs)
    chunks = chunk_text(text)

    results = []

    for chunk in chunks:
        ai_response = generate_analysis(chunk)
        parsed = extract_json(ai_response)
        results.append(parsed)

    # Merge results from all chunks
    combined_overview = " ".join(
        [r.get("overview", "") for r in results]
    ).strip()

    combined_summary = " ".join(
        [r.get("summary", "") for r in results]
    ).strip()

    combined_highlights = list(set(
        sum([r.get("highlights", []) for r in results], [])
    ))

    final_output = {
        "overview": combined_overview,
        "summary": combined_summary,
        "highlights": combined_highlights
    }

    # Return selected section if requested
    if analysis_type == 1:
        return {"overview": final_output["overview"]}
    elif analysis_type == 2:
        return {"summary": final_output["summary"]}
    elif analysis_type == 3:
        return {"highlights": final_output["highlights"]}

    return final_output

