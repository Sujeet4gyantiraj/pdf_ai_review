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
    # Temporarily bypass cleaning for debugging
    return raw_html
    
    # cleaned = raw_html.strip()
    # if "```" in cleaned:
    #     cleaned = cleaned.replace("```html", "").replace("```", "").strip()
    
    # # Extract only the content between <html> tags if AI added extra text
    # start = cleaned.find("<html")
    # end = cleaned.rfind("</html>")
    # if start != -1 and end != -1:
    #     cleaned = cleaned[start : end + 7]
    # return cleaned

# --- Endpoints ---

@router.post("/generate-html", response_class=HTMLResponse)
async def generate_document_html(request: DocumentGenerationRequest):
    """
    Generates HTML, saves it to html_db.json using the provided document_id, 
    and returns the raw HTML.
    """
    selected_template = prompt_templates.get(request.document_type)

    if not selected_template:
        # Use the "default" template if the requested document_type is not found
        selected_template = prompt_templates.get("default")
        if not selected_template:
            # This should ideally not happen if "default" is always present
            raise HTTPException(status_code=500, detail="Default prompt template not found.")
        # Optionally, log that a default template is being used
        print(f"Warning: Document type '{request.document_type}' not found. Using default template.")


    # Format prompt
    # The default template only uses 'user_request'
    if request.document_type == "example_type":
        system_prompt = selected_template.format(
            user_prompt=request.user_prompt,
            document_id=request.document_id
        )
    elif request.document_type in ["offer_letter", "invoice", "default"]: # Explicitly include "default" here
        system_prompt = selected_template.format(user_request=request.user_prompt)
    else: # Fallback for any other custom templates that might exist
        system_prompt = selected_template.format(user_request=request.user_prompt)

    try:
        # 1. Generate
        generated_raw = await run_llm(
            text=request.user_prompt,
            system_prompt=system_prompt,
        )
        
        # Temporarily bypass cleaning and error checking to see raw LLM output
        # 2. Store in JSON file using provided document_id (using generated_raw directly)
        update_storage(request.document_id, generated_raw)

        # 3. Return Raw HTML (Response structure unchanged)
        return HTMLResponse(content=generated_raw)

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