"""
t_document_route.py

FastAPI router for document generation.

Endpoints:
  GET  /documents/types                 List all supported document types
  POST /documents/generate              Generate → JSON (text + base64 docx)
  POST /documents/generate/download     Generate → DOCX file download
  POST /documents/extract-fields        Extract fields only (no generation)
"""

import json
import time
import uuid
import base64
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from t_document_generator import (
    generate_document,
    classify_intent,
    resolve_document_type,
    SUPPORTED_DOCUMENT_TYPES,
    _SCHEMAS,
    _extract_fields,
    _get_missing_fields,
    _render_docx,
    _GENERATION_PROMPTS,
    _api_kwargs,
    _get_client,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["Document Generation"])


# ---------------------------------------------------------------------------
# Request model — document_type is optional
# ---------------------------------------------------------------------------

class DocumentGenerateRequest(BaseModel):
    document_type: Optional[str] = Field(
        default=None,
        description=(
            "Type of document to generate. "
            "If omitted, the type is automatically detected from user_query. "
            "Supported: nda, job_offer, freelancer_agreement, service_agreement, "
            "consulting_agreement, lease_agreement, employment_contract"
        ),
        examples=["nda", "job_offer", None],
    )
    user_query: str = Field(
        ...,
        min_length=20,
        description=(
            "Free-text description of the document you need. "
            "Include party names, dates, amounts, jurisdiction, and special clauses. "
            "If document_type is not provided, this text is used to detect the type automatically."
        ),
        examples=[
            "I need an NDA between TechCorp Inc and John Smith, covering our AI roadmap "
            "for 2 years, governed by California law. Effective April 1, 2026.",
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
    summary="List all supported document types with field schemas",
)
async def list_document_types() -> dict:
    """
    Returns all supported document types with their required and optional fields.
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
    summary="Extract document fields without generating the document",
)
async def extract_document_fields(request: DocumentGenerateRequest) -> dict:
    """
    Extracts structured fields from user_query without generating the document.
    If document_type is omitted, it is automatically detected.
    """
    request_id = str(uuid.uuid4())[:8]

    if request.document_type:
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
        detected = False
    else:
        doc_type = await classify_intent(request.user_query)
        detected = True
        if not doc_type:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "Could not determine document type from description.",
                    "hint":  "Please specify document_type explicitly.",
                    "supported_types": list(SUPPORTED_DOCUMENT_TYPES.keys()),
                }
            )

    try:
        fields  = await _extract_fields(doc_type, request.user_query)
        missing = _get_missing_fields(doc_type, fields)
    except Exception as e:
        logger.exception(f"[{request_id}] field extraction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Field extraction failed: {str(e)}")

    return {
        "status":            "missing_fields" if missing else "success",
        "document_type":     doc_type,
        "document_name":     SUPPORTED_DOCUMENT_TYPES[doc_type],
        "type_was_detected": detected,
        "fields":            fields,
        "missing_fields":    missing,
        "message": (
            "All required fields found. Ready to generate."
            if not missing else
            f"Missing {len(missing)} required field(s): {', '.join(missing)}. "
            "Add these details to your query for a complete document."
        ),
    }


# ---------------------------------------------------------------------------
# POST /documents/generate
# Returns JSON with plain text + base64 DOCX
# ---------------------------------------------------------------------------

@router.post(
    "/generate",
    summary="Generate a legal document — returns JSON with text and DOCX",
)
async def generate_document_endpoint(request: DocumentGenerateRequest) -> dict:
    """
    Generates a complete legal document.

    - document_type is optional — auto-detected if omitted
    - Returns document as plain text AND base64-encoded DOCX
    - For direct DOCX download use POST /documents/generate/download
    """
    request_id = str(uuid.uuid4())[:8]
    t_start    = time.perf_counter()

    logger.info(
        f"[{request_id}] ── DOC GENERATE — "
        f"type='{request.document_type or 'auto-detect'}'"
    )

    try:
        result = await generate_document(request.document_type, request.user_query)
    except Exception as e:
        logger.exception(f"[{request_id}] error: {e}")
        raise HTTPException(status_code=500, detail=f"Document generation failed: {str(e)}")

    if result["status"] == "unknown_type":
        raise HTTPException(
            status_code=422,
            detail={
                "error":           result.get("message", "Unknown document type"),
                "supported_types": list(SUPPORTED_DOCUMENT_TYPES.keys()),
            }
        )

    elapsed    = time.perf_counter() - t_start
    docx_bytes = result.pop("docx_bytes", b"")
    docx_b64   = base64.b64encode(docx_bytes).decode("utf-8") if docx_bytes else ""

    logger.info(
        f"[{request_id}] ── COMPLETE — {elapsed:.2f}s "
        f"type={result['document_type']} "
        f"words={result.get('word_count', 0)} "
        f"missing={result.get('missing_fields', [])}"
    )

    return {
        "status":             result["status"],
        "document_type":      result["document_type"],
        "document_name":      result["document_name"],
        "type_was_detected":  request.document_type is None,
        "fields":             result["fields"],
        "missing_fields":     result["missing_fields"],
        "document":           result["document"],
        "docx_base64":        docx_b64,
        "word_count":         result["word_count"],
        "input_tokens":       result.get("input_tokens", 0),
        "output_tokens":      result.get("output_tokens", 0),
        "generation_time_s":  round(elapsed, 2),
        "request_id":         request_id,
    }


# ---------------------------------------------------------------------------
# POST /documents/generate/download
# Returns DOCX file directly
# ---------------------------------------------------------------------------

@router.post(
    "/generate/download",
    summary="Generate a legal document — returns DOCX file for download",
    response_class=Response,
)
async def generate_document_download(request: DocumentGenerateRequest):
    """
    Generates a complete legal document and returns it as a downloadable DOCX file.
    document_type is optional — auto-detected if omitted.

    Response headers include X-Document-Type, X-Word-Count, X-Missing-Fields,
    X-Generation-Time, X-Status, X-Request-Id.
    """
    request_id = str(uuid.uuid4())[:8]
    t_start    = time.perf_counter()

    logger.info(
        f"[{request_id}] ── DOC DOWNLOAD — "
        f"type='{request.document_type or 'auto-detect'}'"
    )

    try:
        result = await generate_document(request.document_type, request.user_query)
    except Exception as e:
        logger.exception(f"[{request_id}] error: {e}")
        raise HTTPException(status_code=500, detail=f"Document generation failed: {str(e)}")

    if result["status"] == "unknown_type":
        raise HTTPException(
            status_code=422,
            detail={
                "error":           result.get("message", "Unknown document type"),
                "supported_types": list(SUPPORTED_DOCUMENT_TYPES.keys()),
            }
        )

    elapsed    = time.perf_counter() - t_start
    docx_bytes = result.get("docx_bytes", b"")
    doc_type   = result["document_type"]
    missing    = result.get("missing_fields", [])
    detected   = request.document_type is None
    filename   = f"{doc_type}_{request_id}.docx"

    logger.info(
        f"[{request_id}] ── DOWNLOAD COMPLETE — {elapsed:.2f}s "
        f"type={doc_type} words={result.get('word_count', 0)} "
        f"docx={len(docx_bytes):,} bytes"
    )

    return Response(
        content=docx_bytes,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "X-Document-Type":     doc_type,
            "X-Document-Name":     result["document_name"],
            "X-Type-Detected":     str(detected).lower(),
            "X-Word-Count":        str(result.get("word_count", 0)),
            "X-Missing-Fields":    ",".join(missing) if missing else "",
            "X-Generation-Time":   str(round(elapsed, 2)),
            "X-Status":            result["status"],
            "X-Request-Id":        request_id,
        }
    )


# ---------------------------------------------------------------------------
# POST /documents/generate/stream
# SSE streaming — fields first, then tokens, then done with DOCX
# ---------------------------------------------------------------------------

@router.post(
    "/generate/stream",
    summary="Generate a legal document with real-time token streaming",
)
async def generate_document_stream(request: DocumentGenerateRequest):
    """
    Streams document generation via Server-Sent Events.
    document_type is optional — auto-detected if omitted.

    Event sequence:
    - status   — pipeline stage updates
    - detected — emitted if type was auto-detected
    - fields   — extracted structured fields
    - token    — one raw text delta from the LLM
    - done     — final metadata including base64 DOCX
    - error    — if something fails
    """
    import os

    request_id = str(uuid.uuid4())[:8]
    t_start    = time.perf_counter()

    logger.info(
        f"[{request_id}] ── DOC STREAM — "
        f"type='{request.document_type or 'auto-detect'}'"
    )

    async def _generate():
        try:
            # Step 0: Resolve or detect type
            detected = False
            if request.document_type:
                doc_type = resolve_document_type(request.document_type)
                if not doc_type:
                    yield _sse("error", {
                        "message":         f"Unsupported document type: '{request.document_type}'",
                        "supported_types": list(SUPPORTED_DOCUMENT_TYPES.keys()),
                    })
                    return
            else:
                yield _sse("status", {
                    "step":    "classifying",
                    "message": "Detecting document type from your description...",
                })
                doc_type = await classify_intent(request.user_query)
                detected = True
                if not doc_type:
                    yield _sse("error", {
                        "message":         "Could not determine document type. Please specify document_type.",
                        "supported_types": list(SUPPORTED_DOCUMENT_TYPES.keys()),
                    })
                    return
                yield _sse("detected", {
                    "document_type": doc_type,
                    "document_name": SUPPORTED_DOCUMENT_TYPES[doc_type],
                    "message":       f"Detected: {SUPPORTED_DOCUMENT_TYPES[doc_type]}",
                })

            doc_name = SUPPORTED_DOCUMENT_TYPES[doc_type]

            # Step 1: Extract fields
            yield _sse("status", {
                "step":    "extracting_fields",
                "message": f"Extracting details for {doc_name}...",
            })
            fields  = await _extract_fields(doc_type, request.user_query)
            missing = _get_missing_fields(doc_type, fields)

            yield _sse("fields", {
                "fields":            fields,
                "missing_fields":    missing,
                "document_type":     doc_type,
                "document_name":     doc_name,
                "type_was_detected": detected,
            })

            # Step 2: Stream document generation
            gen_prompt  = _GENERATION_PROMPTS[doc_type]
            field_lines = "\n".join(
                f"  {k.replace('_', ' ').title()}: {v}"
                for k, v in fields.items()
                if v and v != "Not Specified"
            )

            system = gen_prompt
            user   = f"""Use these details to generate the complete {doc_name}:

{field_lines if field_lines else "(Use standard template values)"}

Additional context:
\"\"\"{request.user_query}\"\"\"

Write the complete {doc_name} now. Start directly with the document title."""

            yield _sse("status", {
                "step":    "generating_document",
                "message": f"Drafting {doc_name}...",
            })

            client = _get_client()
            # CRITICAL: use_json=False for document generation
            kwargs = _api_kwargs(max_tokens=32000, use_json=False)
            kwargs["messages"]       = [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ]
            kwargs["stream"]         = True
            kwargs["stream_options"] = {"include_usage": True}

            full_document = ""
            in_tok        = 0
            out_tok       = 0

            async with await client.chat.completions.create(**kwargs) as stream:
                async for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        delta          = chunk.choices[0].delta.content
                        full_document += delta
                        yield _sse("token", {"delta": delta})
                    if chunk.usage:
                        in_tok  = chunk.usage.prompt_tokens
                        out_tok = chunk.usage.completion_tokens

            # Step 3: Render DOCX
            yield _sse("status", {"step": "rendering_docx", "message": "Rendering DOCX..."})
            docx_bytes = _render_docx(full_document, doc_name)
            docx_b64   = base64.b64encode(docx_bytes).decode("utf-8")

            elapsed = time.perf_counter() - t_start
            logger.info(
                f"[{request_id}] ── STREAM COMPLETE — {elapsed:.2f}s "
                f"type={doc_type} words={len(full_document.split())} "
                f"tokens={in_tok}in/{out_tok}out"
            )

            yield _sse("done", {
                "status":             "missing_fields" if missing else "success",
                "document_type":      doc_type,
                "document_name":      doc_name,
                "type_was_detected":  detected,
                "missing_fields":     missing,
                "word_count":         len(full_document.split()),
                "docx_base64":        docx_b64,
                "docx_size_bytes":    len(docx_bytes),
                "generation_time_s":  round(elapsed, 2),
                "input_tokens":       in_tok,
                "output_tokens":      out_tok,
                "total_tokens":       in_tok + out_tok,
                "request_id":         request_id,
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