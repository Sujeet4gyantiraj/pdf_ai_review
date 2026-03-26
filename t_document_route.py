"""
t_document_route.py

FastAPI router for document generation.

Endpoints:
  GET  /documents/types              List all supported document types + schemas
  POST /documents/generate           Generate document → JSON + PDF download link
  POST /documents/generate/download  Generate document → direct PDF file download
  POST /documents/extract-fields     Extract fields only (no generation)
"""

import json
import time
import uuid
import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel, Field

from t_document_generator import (
    generate_document,
    resolve_document_type,
    SUPPORTED_DOCUMENT_TYPES,
    _SCHEMAS,
    _extract_fields,
    _get_missing_fields,
    _render_pdf,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["Document Generation"])


# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------

class DocumentGenerateRequest(BaseModel):
    document_type: str = Field(
        ...,
        description="Type of document to generate. Use GET /documents/types for all options.",
        examples=["nda", "job_offer", "employment_contract"],
    )
    user_query: str = Field(
        ...,
        min_length=20,
        description=(
            "Free-text description of the document you need. "
            "Include party names, dates, amounts, jurisdiction, and any special clauses. "
            "The more detail you provide, the more complete the document will be."
        ),
        examples=[
            "NDA between TechCorp Inc (disclosing party) and John Smith (receiving party). "
            "Purpose: sharing our AI product roadmap for partnership evaluation. "
            "Duration: 2 years. Effective date: April 1, 2026. Governed by California law."
        ],
    )


# ---------------------------------------------------------------------------
# SSE helper
# ---------------------------------------------------------------------------

def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


# ---------------------------------------------------------------------------
# GET /documents/types
# ---------------------------------------------------------------------------

@router.get(
    "/types",
    summary="List all supported document types",
)
async def list_document_types() -> dict:
    """
    Returns all 7 supported document types with their required and optional fields.
    Use the slug as the document_type value in generate requests.
    """
    types = []
    for slug, display_name in SUPPORTED_DOCUMENT_TYPES.items():
        schema = _SCHEMAS.get(slug, {"required": [], "optional": []})
        types.append({
            "slug":            slug,
            "display_name":    display_name,
            "required_fields": schema.get("required", []),
            "optional_fields": schema.get("optional", []),
        })
    return {
        "status":         "success",
        "total":          len(types),
        "document_types": types,
    }


# ---------------------------------------------------------------------------
# POST /documents/extract-fields
# ---------------------------------------------------------------------------

@router.post(
    "/extract-fields",
    summary="Extract document fields from description (without generating document)",
)
async def extract_document_fields(request: DocumentGenerateRequest) -> dict:
    """
    Extracts structured fields from user_query WITHOUT generating the document.

    Use this in a multi-step UI:
    1. Call this to preview what fields were detected
    2. Let user review / correct them
    3. Call POST /documents/generate with the refined description
    """
    request_id = str(uuid.uuid4())[:8]

    doc_type = resolve_document_type(request.document_type)
    if not doc_type:
        raise HTTPException(
            status_code=400,
            detail={
                "error":           "Unsupported document type",
                "provided":        request.document_type,
                "supported_types": list(SUPPORTED_DOCUMENT_TYPES.keys()),
            }
        )

    try:
        fields  = await _extract_fields(doc_type, request.user_query)
        missing = _get_missing_fields(doc_type, fields)
    except Exception as e:
        logger.exception(f"[{request_id}] field extraction failed: {e}")
        raise HTTPException(status_code=500, detail="Field extraction failed.")

    return {
        "status":         "missing_fields" if missing else "success",
        "document_type":  doc_type,
        "document_name":  SUPPORTED_DOCUMENT_TYPES[doc_type],
        "fields":         fields,
        "missing_fields": missing,
        "message": (
            "All required fields found. Ready to generate."
            if not missing else
            f"Missing {len(missing)} required field(s): {', '.join(missing)}. "
            "Add these details to your query for a complete document."
        ),
    }


# ---------------------------------------------------------------------------
# POST /documents/generate
# Returns JSON with document text + base64 PDF
# ---------------------------------------------------------------------------

@router.post(
    "/generate",
    summary="Generate a legal document — returns JSON with text and PDF",
)
async def generate_document_endpoint(request: DocumentGenerateRequest) -> dict:
    """
    Generates a complete legal document.

    Returns:
    - `fields` — all extracted structured fields
    - `missing_fields` — required fields not found (document still generated with defaults)
    - `document` — full document as plain text
    - `pdf_base64` — base64-encoded PDF for download
    - `word_count`, `generation_time_s`

    For a direct PDF file download use POST /documents/generate/download instead.
    """
    import base64

    request_id = str(uuid.uuid4())[:8]
    t_start    = time.perf_counter()

    logger.info(f"[{request_id}] ── DOC GENERATE — type='{request.document_type}'")

    doc_type = resolve_document_type(request.document_type)
    if not doc_type:
        raise HTTPException(
            status_code=400,
            detail={
                "error":           "Unsupported document type",
                "provided":        request.document_type,
                "supported_types": list(SUPPORTED_DOCUMENT_TYPES.keys()),
                "supported_names": list(SUPPORTED_DOCUMENT_TYPES.values()),
            }
        )

    try:
        result = await generate_document(doc_type, request.user_query)
    except Exception as e:
        logger.exception(f"[{request_id}] document generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Document generation failed: {str(e)}")

    elapsed = time.perf_counter() - t_start
    logger.info(
        f"[{request_id}] ── COMPLETE — {elapsed:.2f}s "
        f"words={result.get('word_count', 0)} "
        f"missing={result.get('missing_fields', [])}"
    )

    # Convert PDF bytes to base64
    pdf_bytes  = result.pop("pdf_bytes", b"")
    pdf_base64 = base64.b64encode(pdf_bytes).decode("utf-8") if pdf_bytes else ""

    return {
        "status":             result["status"],
        "document_type":      result["document_type"],
        "document_name":      result["document_name"],
        "fields":             result["fields"],
        "missing_fields":     result["missing_fields"],
        "document":           result["document"],
        "pdf_base64":         pdf_base64,
        "word_count":         result["word_count"],
        "input_tokens":       result.get("input_tokens", 0),
        "output_tokens":      result.get("output_tokens", 0),
        "generation_time_s":  round(elapsed, 2),
        "request_id":         request_id,
    }


# ---------------------------------------------------------------------------
# POST /documents/generate/download
# Returns the PDF directly as a file download
# ---------------------------------------------------------------------------

@router.post(
    "/generate/download",
    summary="Generate a legal document — returns PDF file for download",
    response_class=Response,
)
async def generate_document_download(request: DocumentGenerateRequest):
    """
    Generates a complete legal document and returns it as a downloadable PDF file.

    The response is a binary PDF with Content-Disposition: attachment.
    Response headers include:
    - `X-Document-Type` — canonical document type slug
    - `X-Word-Count` — word count of the generated document
    - `X-Missing-Fields` — comma-separated list of missing required fields (if any)
    - `X-Generation-Time` — seconds taken to generate
    - `X-Status` — "success" or "missing_fields"
    """
    request_id = str(uuid.uuid4())[:8]
    t_start    = time.perf_counter()

    logger.info(f"[{request_id}] ── DOC DOWNLOAD — type='{request.document_type}'")

    doc_type = resolve_document_type(request.document_type)
    if not doc_type:
        raise HTTPException(
            status_code=400,
            detail={
                "error":           "Unsupported document type",
                "provided":        request.document_type,
                "supported_types": list(SUPPORTED_DOCUMENT_TYPES.keys()),
            }
        )

    try:
        result = await generate_document(doc_type, request.user_query)
    except Exception as e:
        logger.exception(f"[{request_id}] document generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Document generation failed: {str(e)}")

    elapsed   = time.perf_counter() - t_start
    pdf_bytes = result.get("pdf_bytes", b"")
    doc_name  = result["document_name"].replace(" ", "_")
    missing   = result.get("missing_fields", [])

    logger.info(
        f"[{request_id}] ── DOWNLOAD COMPLETE — {elapsed:.2f}s "
        f"words={result.get('word_count', 0)} "
        f"pdf={len(pdf_bytes):,} bytes"
    )

    filename = f"{doc_type}_{request_id}.pdf"

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition":  f'attachment; filename="{filename}"',
            "X-Document-Type":      doc_type,
            "X-Document-Name":      result["document_name"],
            "X-Word-Count":         str(result.get("word_count", 0)),
            "X-Missing-Fields":     ",".join(missing) if missing else "",
            "X-Generation-Time":    str(round(elapsed, 2)),
            "X-Status":             result["status"],
            "X-Request-Id":         request_id,
        }
    )


# ---------------------------------------------------------------------------
# POST /documents/generate/stream
# SSE: fields first, then document token by token
# ---------------------------------------------------------------------------

@router.post(
    "/generate/stream",
    summary="Generate a legal document with real-time token streaming",
)
async def generate_document_stream(request: DocumentGenerateRequest):
    """
    Streams the document generation in real time via Server-Sent Events.

    Event sequence:
    - `status`  — pipeline stage updates
    - `fields`  — extracted fields (sent before generation starts)
    - `token`   — one raw token delta from the LLM
    - `pdf`     — base64-encoded PDF (sent after all tokens complete)
    - `done`    — final metadata
    - `error`   — if something fails
    """
    import base64
    import os
    from t_document_generator import (
        _extract_fields as ef,
        _get_missing_fields as gmf,
        _DOCUMENT_INSTRUCTIONS,
        _render_pdf,
        _get_client,
        _get_model_kwargs,
    )

    request_id = str(uuid.uuid4())[:8]
    t_start    = time.perf_counter()

    logger.info(f"[{request_id}] ── DOC STREAM — type='{request.document_type}'")

    doc_type = resolve_document_type(request.document_type)
    if not doc_type:
        async def _err():
            yield _sse("error", {
                "message":         "Unsupported document type",
                "provided":        request.document_type,
                "supported_types": list(SUPPORTED_DOCUMENT_TYPES.keys()),
            })
        return StreamingResponse(_err(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    async def _generate():
        try:
            # Step 1: Extract fields
            yield _sse("status", {"step": "extracting_fields",
                                   "message": "Extracting details from your description..."})

            fields  = await ef(doc_type, request.user_query)
            missing = gmf(doc_type, fields)

            yield _sse("fields", {
                "fields":         fields,
                "missing_fields": missing,
                "document_type":  doc_type,
                "document_name":  SUPPORTED_DOCUMENT_TYPES[doc_type],
            })

            # Step 2: Stream document generation
            doc_name     = SUPPORTED_DOCUMENT_TYPES[doc_type]
            instructions = _DOCUMENT_INSTRUCTIONS[doc_type]

            field_lines = "\n".join(
                f"  {k.replace('_', ' ').title()}: {v}"
                for k, v in fields.items()
                if v and v != "Not Specified"
            )

            system = f"""You are a senior legal document drafter with 20+ years of experience.
Generate a complete, professional, legally-sound {doc_name}.

{instructions}

Formatting rules:
- Section headings in ALL CAPS followed by a colon
- Numbered sub-clauses within each section (1.1, 1.2, etc.)
- Formal legal language throughout
- Use actual names, dates, and amounts provided
- If a value is "Not Specified" use [TO BE AGREED] placeholder
- Plain text only — no markdown"""

            user = f"""Generate a complete {doc_name} using these details:

{field_lines}

Additional context:
\"\"\"{request.user_query}\"\"\"

Write the full document now."""

            yield _sse("status", {"step": "generating_document",
                                   "message": f"Drafting {doc_name}..."})

            from s_ai_model import (
                MODEL_NAME,
                _FIXED_TEMPERATURE_MODELS,
                _MAX_COMPLETION_TOKENS_MODELS,
            )
            from openai import AsyncOpenAI

            client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))

            api_kwargs = {
                "model":    MODEL_NAME,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                "stream":         True,
                "stream_options": {"include_usage": True},
            }
            if MODEL_NAME not in _FIXED_TEMPERATURE_MODELS:
                api_kwargs["temperature"] = 0.1
            if MODEL_NAME in _MAX_COMPLETION_TOKENS_MODELS:
                api_kwargs["max_completion_tokens"] = 4096
            else:
                api_kwargs["max_tokens"] = 4096

            full_document = ""
            in_tok        = 0
            out_tok       = 0

            async with await client.chat.completions.create(**api_kwargs) as stream:
                async for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        delta          = chunk.choices[0].delta.content
                        full_document += delta
                        yield _sse("token", {"delta": delta})
                    if chunk.usage:
                        in_tok  = chunk.usage.prompt_tokens
                        out_tok = chunk.usage.completion_tokens

            # Step 3: Render PDF
            yield _sse("status", {"step": "rendering_pdf", "message": "Rendering PDF..."})
            pdf_bytes  = _render_pdf(full_document, doc_name)
            pdf_base64 = base64.b64encode(pdf_bytes).decode("utf-8")

            yield _sse("pdf", {
                "pdf_base64": pdf_base64,
                "size_bytes": len(pdf_bytes),
            })

            elapsed = time.perf_counter() - t_start
            logger.info(
                f"[{request_id}] ── STREAM COMPLETE — {elapsed:.2f}s "
                f"words={len(full_document.split())} "
                f"tokens={in_tok}in/{out_tok}out"
            )

            yield _sse("done", {
                "word_count":        len(full_document.split()),
                "generation_time_s": round(elapsed, 2),
                "input_tokens":      in_tok,
                "output_tokens":     out_tok,
                "total_tokens":      in_tok + out_tok,
                "missing_fields":    missing,
                "request_id":        request_id,
            })

        except Exception as e:
            logger.exception(f"[{request_id}] stream error: {e}")
            yield _sse("error", {"message": f"Document generation failed: {str(e)}"})

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",
            "Connection":        "keep-alive",
        },
    )