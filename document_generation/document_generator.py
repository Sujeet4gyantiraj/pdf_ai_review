import json
import os
import uuid
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from fastapi.responses import HTMLResponse, JSONResponse
from s_ai_model import run_llm
from .prompt_templates import prompt_templates, REGENERATE_PROMPT

router = APIRouter()

# --- Simple JSON File Storage Logic ---
DB_FILE = "html_db.json"

def load_db():
    if not os.path.exists(DB_FILE):
        return {}
    with open(DB_FILE, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except:
            return {}

def save_to_db(doc_id, html_content):
    db = load_db()
    db[doc_id] = html_content
    with open(DB_FILE, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=4)

def get_from_db(doc_id):
    db = load_db()
    return db.get(doc_id)

# --- Request Models ---

class DocumentGenerationRequest(BaseModel):
    document_type: str
    user_prompt: str

class DocumentRegenerationRequest(BaseModel):
    document_id: str  # We only need the ID now!
    modification_query: str

# --- Helper to clean AI output ---

def clean_html_output(raw_html: str) -> str:
    cleaned = raw_html.strip()
    if "```" in cleaned:
        cleaned = cleaned.replace("```html", "").replace("```", "").strip()
    
    # Ensure we only extract the HTML block
    start = cleaned.find("<html")
    end = cleaned.rfind("</html>")
    if start != -1 and end != -1:
        cleaned = cleaned[start : end + 7]
    return cleaned

# --- Endpoints ---

@router.post("/generate-html")
async def generate_document_html(request: DocumentGenerationRequest):
    """
    Generates HTML, saves it to html_db.json, and returns the ID + HTML.
    """
    selected_template = prompt_templates.get(request.document_type)
    if not selected_template:
        raise HTTPException(status_code=404, detail="Template not found")

    system_prompt = selected_template.format(user_request=request.user_prompt)

    try:
        # 1. AI Generation
        generated_raw = await run_llm(text=request.user_prompt, system_prompt=system_prompt)
        cleaned_html = clean_html_output(generated_raw)

        # 2. Save to JSON file
        doc_id = str(uuid.uuid4())[:8] # Short unique ID
        save_to_db(doc_id, cleaned_html)

        # 3. Return JSON (so frontend knows the ID)
        return {
            "document_id": doc_id,
            "html": cleaned_html
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@router.post("/regenerate-html")
async def regenerate_document_html(request: DocumentRegenerationRequest):
    """
    Reads existing HTML from JSON file by ID, modifies it, and saves it back.
    """
    # 1. Load existing HTML from the file
    existing_html = get_from_db(request.document_id)
    if not existing_html:
        raise HTTPException(status_code=404, detail="Document ID not found in storage.")

    # 2. Prepare AI instructions
    system_prompt = REGENERATE_PROMPT.format(
        existing_html=existing_html,
        modification_query=request.modification_query
    )

    try:
        # 3. Call AI
        updated_raw = await run_llm(
            text=f"Modify document {request.document_id}: {request.modification_query}",
            system_prompt=system_prompt,
        )
        cleaned_html = clean_html_output(updated_raw)

        # 4. Save the updated version back to the JSON file
        save_to_db(request.document_id, cleaned_html)

        return {
            "document_id": request.document_id,
            "html": cleaned_html
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Regeneration failed: {str(e)}")

@router.get("/get-html/{doc_id}", response_class=HTMLResponse)
async def get_html_by_id(doc_id: str):
    """Utility endpoint to view the saved HTML directly."""
    html = get_from_db(doc_id)
    if not html:
        raise HTTPException(status_code=404, detail="Not found")
    return html