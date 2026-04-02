import io
import json
import os
import logging
from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from openai import AsyncOpenAI
from dotenv import load_dotenv
from .prompt_templates import DOCUMENT_GENERATION_PROMPT, REGENERATE_PROMPT

load_dotenv()

logger = logging.getLogger(__name__)
router = APIRouter()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_MODEL      = os.environ.get("MODEL_NAME", "gpt-5-nano")
_API_KEY    = os.environ.get("OPENAI_API_KEY", "")
_CLIENT     = AsyncOpenAI(api_key=_API_KEY)

# HTML documents can be large — use a higher output token limit than the
# default 4096 used by run_llm (which is designed for text analysis).
_HTML_MAX_OUTPUT_TOKENS = 16000

# Models that do not support the temperature parameter
_FIXED_TEMPERATURE_MODELS = {
    "gpt-5-nano", "gpt-4.1-nano", "gpt-4o-mini",
    "o1", "o1-mini", "o3-mini", "o3",
}

# Models that use max_completion_tokens instead of max_tokens
_MAX_COMPLETION_TOKENS_MODELS = {
    "gpt-5-nano", "gpt-4.1-nano", "gpt-4o-mini",
    "o1", "o1-mini", "o3-mini", "o3",
}

# ---------------------------------------------------------------------------
# Storage — absolute path anchored to this file's directory
# ---------------------------------------------------------------------------

_DB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "html_db.json")


def get_storage() -> dict:
    if not os.path.exists(_DB_FILE):
        return {}
    try:
        with open(_DB_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def update_storage(doc_id: str, html_content: str) -> None:
    db = get_storage()
    db[doc_id] = html_content
    with open(_DB_FILE, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=4)


# ---------------------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------------------

class DocumentGenerationRequest(BaseModel):
    document_id: str
    user_prompt: str


class DocumentRegenerationRequest(BaseModel):
    document_id: str
    modification_query: str


class HtmlToPdfRequest(BaseModel):
    document_id: str | None = None   # fetch HTML from html_db.json
    html: str | None = None          # or pass raw HTML directly


# ---------------------------------------------------------------------------
# LLM caller — dedicated for HTML generation with higher output token limit
# ---------------------------------------------------------------------------

async def _call_llm(system_prompt: str, user_message: str) -> str:
    model = os.environ.get("MODEL_NAME", _MODEL)

    kwargs: dict = {
        "model":    model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ],
    }

    if model not in _FIXED_TEMPERATURE_MODELS:
        kwargs["temperature"] = 0.3

    if model in _MAX_COMPLETION_TOKENS_MODELS:
        kwargs["max_completion_tokens"] = _HTML_MAX_OUTPUT_TOKENS
    else:
        kwargs["max_tokens"] = _HTML_MAX_OUTPUT_TOKENS

    try:
        response = await _CLIENT.chat.completions.create(**kwargs)
        content  = response.choices[0].message.content or ""
        logger.info(
            f"[html-gen] in={response.usage.prompt_tokens} "
            f"out={response.usage.completion_tokens} "
            f"model={model}"
        )
        return content
    except Exception as e:
        logger.exception(f"[html-gen] OpenAI call failed: {e}")
        raise


# ---------------------------------------------------------------------------
# Output cleaning
# ---------------------------------------------------------------------------

def _clean_html(raw: str) -> str:
    cleaned = raw.strip()
    if "```" in cleaned:
        cleaned = cleaned.replace("```html", "").replace("```", "").strip()
    start = cleaned.find("<html")
    end   = cleaned.rfind("</html>")
    if start != -1 and end != -1:
        cleaned = cleaned[start : end + 7]
    return cleaned


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/generate-html", response_class=HTMLResponse)
async def generate_document_html(request: DocumentGenerationRequest):
    """
    Generates an HTML document of any type based on user_prompt.
    The LLM infers the document type automatically from the prompt.
    Saves to html_db.json and returns raw HTML.
    """
    system_prompt = DOCUMENT_GENERATION_PROMPT.format(user_request=request.user_prompt)

    try:
        raw_html     = await _call_llm(system_prompt, request.user_prompt)
        cleaned_html = _clean_html(raw_html)

        if not cleaned_html.strip():
            raise HTTPException(
                status_code=500,
                detail=f"AI generated empty HTML. Raw output: {raw_html[:500]}"
            )

        update_storage(request.document_id, cleaned_html)
        return HTMLResponse(content=cleaned_html)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@router.post("/regenerate-html", response_class=HTMLResponse)
async def regenerate_document_html(request: DocumentRegenerationRequest):
    """
    Looks up HTML by document_id, applies user modifications,
    updates storage, and returns the modified HTML.
    """
    db            = get_storage()
    existing_html = db.get(request.document_id)

    if not existing_html:
        raise HTTPException(
            status_code=404,
            detail=f"No document found with ID '{request.document_id}'. Generate it first."
        )

    system_prompt = REGENERATE_PROMPT.format(
        existing_html=existing_html,
        modification_query=request.modification_query
    )

    try:
        raw_html     = await _call_llm(system_prompt, request.modification_query)
        cleaned_html = _clean_html(raw_html)

        update_storage(request.document_id, cleaned_html)
        return HTMLResponse(content=cleaned_html)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Regeneration failed: {str(e)}")


@router.get("/get-html/{document_id}", response_class=HTMLResponse)
async def get_document_html(document_id: str):
    """
    Fetches previously generated HTML from html_db.json by document_id.
    """
    db = get_storage()
    html = db.get(document_id)
    if not html:
        raise HTTPException(
            status_code=404,
            detail=f"No document found with ID '{document_id}'. Generate it first via /generate-html."
        )
    return HTMLResponse(content=html)


@router.post("/html-to-pdf")
async def html_to_pdf(request: HtmlToPdfRequest):
    """
    Converts HTML to a PDF file.
    Provide either:
      - document_id  → fetches HTML from html_db.json
      - html         → uses the raw HTML string directly
    Returns a downloadable PDF.
    """
    try:
        from weasyprint import HTML as WeasyprintHTML
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="weasyprint is not installed. Run: pip install weasyprint"
        )

    if request.document_id:
        db = get_storage()
        html_content = db.get(request.document_id)
        if not html_content:
            raise HTTPException(
                status_code=404,
                detail=f"No document found with ID '{request.document_id}'."
            )
        filename = f"{request.document_id}.pdf"
    elif request.html:
        html_content = request.html
        filename = "document.pdf"
    else:
        raise HTTPException(
            status_code=400,
            detail="Provide either 'document_id' or 'html' in the request body."
        )

    try:
        pdf_bytes = WeasyprintHTML(string=html_content).write_pdf()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF conversion failed: {str(e)}")

    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
