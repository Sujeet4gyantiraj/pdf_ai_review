from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from fastapi.responses import HTMLResponse
from s_ai_model import run_llm
from .prompt_templates import SimulatedPromptTemplate, prompt_templates, REGENERATE_PROMPT 

router = APIRouter()

class DocumentGenerationRequest(BaseModel):
    document_id: str
    document_type: str
    user_prompt: str

@router.post("/generate-html", response_class=HTMLResponse)
async def generate_document_html(request: DocumentGenerationRequest):
    """
    Generates HTML for a document based on the provided document type and user prompt using prompt templates.
    """
    selected_template = prompt_templates.get(request.document_type)

    if not selected_template:
        raise HTTPException(status_code=404, detail=f"Prompt template not found for document type: {request.document_type}")

    # Format the system prompt using the selected template and request data
    if request.document_type == "example_type":
        system_prompt = selected_template.format(
            user_prompt=request.user_prompt,
            document_id=request.document_id
        )
    else:
        system_prompt = selected_template.format(user_request=request.user_prompt)

    # Call the AI model
    try:
        generated_html = await run_llm(
            text=request.user_prompt, # The user's actual prompt goes here for the LLM to process
            system_prompt=system_prompt,
        )
        return generated_html
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI model generation failed: {str(e)}")
    


class DocumentRegenerationRequest(BaseModel):
    existing_html: str
    modification_query: str


def clean_html_output(raw_html: str) -> str:
    """Removes common LLM artifacts like markdown code blocks."""
    cleaned = raw_html.replace("```html", "").replace("```", "").strip()
    return cleaned

@router.post("/regenerate-html", response_class=HTMLResponse)
async def regenerate_document_html(request: DocumentRegenerationRequest):
    """
    Updates an existing HTML response based on a modification query.
    """
    if not request.existing_html.strip():
        raise HTTPException(status_code=400, detail="Existing HTML content is empty.")

    # Format the specialized regeneration prompt
    system_prompt = REGENERATE_PROMPT.format(
        existing_html=request.existing_html,
        modification_query=request.modification_query
    )

    try:
        # Call AI with the modification request
        updated_raw = await run_llm(
            text=f"Modify this document as follows: {request.modification_query}",
            system_prompt=system_prompt,
        )

        # Clean backticks and return as HTML
        cleaned_html = clean_html_output(updated_raw)
        return HTMLResponse(content=cleaned_html)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Regeneration failed: {str(e)}")