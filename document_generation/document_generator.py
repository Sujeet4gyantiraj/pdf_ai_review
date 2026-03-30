from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from fastapi.responses import HTMLResponse
from s_ai_model import run_llm
from .prompt_templates import SimulatedPromptTemplate, prompt_templates # Import from new file

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
