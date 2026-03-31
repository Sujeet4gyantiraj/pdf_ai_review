import json
import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from fastapi.responses import HTMLResponse
from s_ai_model import run_llm
from .prompt_templates import SimulatedPromptTemplate, prompt_templates, REGENERATE_PROMPT 

router = APIRouter()

# --- Local File Storage Helpers ---
DB_FILE = "html_db.json"

def get_storage():
    """Load the JSON storage file."""
    if not os.path.exists(DB_FILE):
        return {}
    try:
        with open(DB_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return {}

def update_storage(doc_id: str, html_content: str):
    """Save or update a document in the JSON storage."""
    db = get_storage()
    db[doc_id] = html_content
    with open(DB_FILE, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=4)

# --- Request Models ---

class DocumentGenerationRequest(BaseModel):
    document_id: str
    document_type: str
    user_prompt: str

class DocumentRegenerationRequest(BaseModel):
    document_id: str  # Use ID to look up the saved HTML
    modification_query: str

# --- Output Cleaning ---

def clean_html_output(raw_html: str) -> str:
    """Removes common LLM artifacts like markdown code blocks."""
    cleaned = raw_html.strip()
    if "```" in cleaned:
        cleaned = cleaned.replace("```html", "").replace("```", "").strip()
    
    # Extract only the content between <html> tags if AI added extra text
    start = cleaned.find("<html")
    end = cleaned.rfind("</html>")
    if start != -1 and end != -1:
        cleaned = cleaned[start : end + 7]
    return cleaned

# --- Endpoints ---

@router.post("/generate-html", response_class=HTMLResponse)
async def generate_document_html(request: DocumentGenerationRequest):
    """
    Generates HTML, saves it to html_db.json using the provided document_id, 
    and returns the raw HTML.
    """
    selected_template = prompt_templates.get(request.document_type)

    if not selected_template:
        raise HTTPException(status_code=404, detail=f"Prompt template not found: {request.document_type}")

    # Format prompt
    if request.document_type == "example_type":
        system_prompt = selected_template.format(
            user_prompt=request.user_prompt,
            document_id=request.document_id
        )
    else:
        system_prompt = selected_template.format(user_request=request.user_prompt)

    try:
        # 1. Generate
        generated_raw = await run_llm(
            text=request.user_prompt,
            system_prompt=system_prompt,
        )
        cleaned_html = clean_html_output(generated_raw)

        # 2. Store in JSON file using provided document_id
        update_storage(request.document_id, cleaned_html)

        # 3. Return Raw HTML (Response structure unchanged)
        return HTMLResponse(content=cleaned_html)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI model generation failed: {str(e)}")


@router.post("/regenerate-html", response_class=HTMLResponse)
async def regenerate_document_html(request: DocumentRegenerationRequest):
    """
    Looks up HTML from storage by document_id, applies modifications, 
    updates storage, and returns modified raw HTML.
    """
    # 1. Retrieve the saved HTML from the local JSON file
    db = get_storage()
    existing_html = db.get(request.document_id)

    if not existing_html:
        raise HTTPException(
            status_code=404, 
            detail=f"No document found with ID '{request.document_id}'. Generate it first."
        )

    # 2. Prepare the modification prompt using the stored HTML
    system_prompt = REGENERATE_PROMPT.format(
        existing_html=existing_html,
        modification_query=request.modification_query
    )

    try:
        # 3. Call AI to apply changes
        updated_raw = await run_llm(
            text=f"Instructions: {request.modification_query}",
            system_prompt=system_prompt,
        )
        cleaned_html = clean_html_output(updated_raw)

        # 4. Update the storage with the new version
        update_storage(request.document_id, cleaned_html)

        # 5. Return Raw HTML (Consistent with generate-html)
        return HTMLResponse(content=cleaned_html)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Regeneration failed: {str(e)}")