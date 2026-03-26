"""
t_document_route.py

FastAPI router for the document generation API.

Endpoints:
  GET  /documents/types              List all supported document types
  POST /documents/generate           Generate a document (JSON response)
  POST /documents/generate/stream    Generate a document (SSE streaming)
"""

import json
import time
import uuid
import logging
import asyncio
import os

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from t_document_generator import (
    generate_document,
    resolve_document_type,
    SUPPORTED_DOCUMENT_TYPES,
    _SCHEMAS,
    _extract_fields,
    _get_missing_fields,
    _generate_document,
)

logger    = logging.getLogger(__name__)
router    = APIRouter(prefix="/documents", tags=["Document Generation"])


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class DocumentGenerateRequest(BaseModel):
    document_type: str = Field(
        ...,
        description="Type of document to generate. "
                    "Use GET /documents/types to see all supported values.",
        examples=["nda", "job_offer", "freelancer_agreement"],
    )
    user_query: str = Field(
        ...,
        min_length=20,
        description="Free-text description of the document you need. "
                    "Include all relevant details: party names, dates, "
                    "amounts, jurisdiction, special clauses, etc.",
        examples=[
            "I need an NDA between my company TechCorp Inc and a contractor "
            "John Smith. The NDA should last 2 years and cover our AI product "
            "roadmap. Governed by California law.",
        ],
    )


class DocumentTypeInfo(BaseModel):
    slug:          str
    display_name:  str
    required_fields: list[str]
    optional_fields: list[str]


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
    summary="List supported document types",
    response_description="All document types with their required and optional fields",
)
async def list_document_types() -> dict:
    """
    Returns all supported document types with their field schemas.
    Use the slug value as document_type in the generate endpoint.
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
        "status":          "success",
        "total":           len(types),
        "document_types":  types,
    }


# ---------------------------------------------------------------------------
# POST /documents/generate — full JSON response
# ---------------------------------------------------------------------------

@router.post(
    "/generate",
    summary="Generate a legal document",
    response_description="Generated document with extracted fields and full text",
)
async def generate_document_endpoint(request: DocumentGenerateRequest) -> dict:
    """
    Generate a complete legal document from a free-text description.

    **How it works:**
    1. Validates and normalises the document_type
    2. Extracts structured fields from your user_query using AI
    3. Generates the full legal document using those fields
    4. Returns both the extracted fields and the complete document

    **Tips for best results:**
    - Include both party names
    - Specify dates, amounts, and jurisdiction
    - Mention any special clauses or requirements
    - The more detail you provide, the better the document

    **Example user_query for NDA:**
    > "NDA between TechCorp Inc (disclosing) and John Smith (receiving).
    >  Purpose: sharing our AI product roadmap for partnership evaluation.
    >  Duration: 2 years. Governed by California law. Must return all
    >  materials within 10 days of termination."
    """
    request_id = str(uuid.uuid4())[:8]
    t_start    = time.perf_counter()

    logger.info(f"[{request_id}] ── NEW DOC GENERATE REQUEST ─────────────────")
    logger.info(f"[{request_id}] document_type='{request.document_type}' query_len={len(request.user_query)}")

    # Resolve document type
    doc_type = resolve_document_type(request.document_type)
    if not doc_type:
        raise HTTPException(
            status_code=400,
            detail={
                "error":             "Unsupported document type",
                "provided":          request.document_type,
                "supported_types":   list(SUPPORTED_DOCUMENT_TYPES.keys()),
                "supported_names":   list(SUPPORTED_DOCUMENT_TYPES.values()),
            }
        )

    try:
        result = await generate_document(doc_type, request.user_query)
    except Exception as e:
        logger.exception(f"[{request_id}] document generation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail="Document generation failed. Please try again."
        )

    elapsed = time.perf_counter() - t_start
    logger.info(
        f"[{request_id}] ── COMPLETE — {elapsed:.2f}s — "
        f"words={result.get('word_count', 0)} "
        f"missing={result.get('missing_fields', [])}"
    )

    result["generation_time_s"] = round(elapsed, 2)
    result["request_id"]        = request_id

    return result


# ---------------------------------------------------------------------------
# POST /documents/generate/stream — SSE streaming response
#
# Event sequence:
#   status       {"step": "extracting_fields"}
#   fields       {"fields": {...}, "missing_fields": [...]}
#   status       {"step": "generating_document"}
#   token        {"delta": "..."} — live token stream from OpenAI
#   done         {"word_count": N, "generation_time_s": N}
#   error        {"message": "..."}
# ---------------------------------------------------------------------------

@router.post(
    "/generate/stream",
    summary="Generate a legal document with real-time streaming",
    response_description="Server-Sent Events stream of the document as it is generated",
)
async def generate_document_stream(request: DocumentGenerateRequest):
    """
    Same as POST /documents/generate but streams the document token-by-token
    as OpenAI generates it — so the client sees output immediately rather than
    waiting for the full document.

    **SSE Events:**
    - `status`  — pipeline stage updates
    - `fields`  — extracted fields (sent before generation starts)
    - `token`   — one raw token delta from the LLM
    - `done`    — final metadata (word count, timing)
    - `error`   — if something goes wrong
    """
    request_id = str(uuid.uuid4())[:8]
    t_start    = time.perf_counter()

    logger.info(f"[{request_id}] ── NEW DOC STREAM REQUEST ────────────────────")
    logger.info(f"[{request_id}] document_type='{request.document_type}' query_len={len(request.user_query)}")

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
            yield _sse("status", {
                "step":    "extracting_fields",
                "message": f"Extracting details from your description...",
            })

            fields  = await _extract_fields(doc_type, request.user_query)
            missing = _get_missing_fields(doc_type, fields)

            yield _sse("fields", {
                "fields":        fields,
                "missing_fields": missing,
                "document_type": doc_type,
                "document_name": SUPPORTED_DOCUMENT_TYPES[doc_type],
            })

            if missing:
                logger.warning(f"[{request_id}] missing fields: {missing}")

            # Step 2: Stream document generation
            yield _sse("status", {
                "step":    "generating_document",
                "message": f"Generating {SUPPORTED_DOCUMENT_TYPES[doc_type]}...",
            })

            # Build the prompts (same as _generate_document but with streaming)
            from s_ai_model import (
                MODEL_NAME,
                _FIXED_TEMPERATURE_MODELS,
                _MAX_COMPLETION_TOKENS_MODELS,
            )
            from t_document_generator import _DOCUMENT_INSTRUCTIONS
            from openai import AsyncOpenAI

            doc_name     = SUPPORTED_DOCUMENT_TYPES[doc_type]
            instructions = _DOCUMENT_INSTRUCTIONS[doc_type]

            field_lines = "\n".join(
                f"  {k.replace('_', ' ').title()}: {v}"
                for k, v in fields.items()
                if v and v != "Not Specified"
            )

            system_prompt = f"""You are a senior legal document drafter with 20+ years of experience.
Generate a complete, professional, legally-sound {doc_name}.

{instructions}

Formatting rules:
- Use clear section headings in ALL CAPS followed by a colon
- Use numbered clauses within each section (1.1, 1.2, etc.)
- Write in formal legal language
- Be specific and unambiguous
- Include all standard legal protections
- Use [SIGNATURE BLOCK] placeholder at the end for signatures
- Do NOT include placeholder brackets like [DATE] — use the actual values provided
- If a value is "Not Specified", use reasonable standard legal defaults
"""

            user_prompt = f"""Generate a complete {doc_name} using these details:

{field_lines}

Additional context from user:
\"\"\"{request.user_query}\"\"\"

Generate the complete document now."""

            client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))

            api_kwargs = {
                "model":    MODEL_NAME,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                "stream":         True,
                "stream_options": {"include_usage": True},
            }

            if MODEL_NAME not in _FIXED_TEMPERATURE_MODELS:
                api_kwargs["temperature"] = 0.2

            if MODEL_NAME in _MAX_COMPLETION_TOKENS_MODELS:
                api_kwargs["max_completion_tokens"] = 4096
            else:
                api_kwargs["max_tokens"] = 4096

            full_document  = ""
            input_tokens   = 0
            output_tokens  = 0

            async with await client.chat.completions.create(**api_kwargs) as stream:
                async for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        delta         = chunk.choices[0].delta.content
                        full_document += delta
                        yield _sse("token", {"delta": delta})

                    if chunk.usage:
                        input_tokens  = chunk.usage.prompt_tokens
                        output_tokens = chunk.usage.completion_tokens

            word_count = len(full_document.split())
            elapsed    = time.perf_counter() - t_start

            logger.info(
                f"[{request_id}] ── STREAM COMPLETE — {elapsed:.2f}s — "
                f"words={word_count} tokens={input_tokens}in/{output_tokens}out"
            )

            yield _sse("done", {
                "word_count":        word_count,
                "generation_time_s": round(elapsed, 2),
                "input_tokens":      input_tokens,
                "output_tokens":     output_tokens,
                "total_tokens":      input_tokens + output_tokens,
                "missing_fields":    missing,
                "request_id":        request_id,
            })

        except Exception as e:
            logger.exception(f"[{request_id}] stream error: {e}")
            yield _sse("error", {"message": "Document generation failed. Please try again."})

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",
            "Connection":        "keep-alive",
        },
    )


# ---------------------------------------------------------------------------
# POST /documents/extract-fields
# Lightweight endpoint — just extracts fields without generating the document.
# Useful for building multi-step UIs where user reviews fields before generation.
# ---------------------------------------------------------------------------

@router.post(
    "/extract-fields",
    summary="Extract document fields from a description (without generating document)",
    response_description="Extracted structured fields and list of any missing required fields",
)
async def extract_document_fields(request: DocumentGenerateRequest) -> dict:
    """
    Extracts and returns the structured fields from the user_query
    **without** generating the full document.

    Use this in a multi-step UI:
    1. Call this endpoint to show the user what was extracted
    2. Let user review / correct the fields
    3. Call POST /documents/generate with the corrected details
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
        "status":        "missing_fields" if missing else "success",
        "document_type": doc_type,
        "document_name": SUPPORTED_DOCUMENT_TYPES[doc_type],
        "fields":        fields,
        "missing_fields": missing,
        "message": (
            f"All required fields found. Ready to generate."
            if not missing else
            f"Missing {len(missing)} required field(s): {', '.join(missing)}. "
            f"Add these details to your query for a complete document."
        ),
    }