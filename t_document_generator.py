"""
t_document_generator.py

Business-level legal document generation engine.

Supported document types:
  - nda                  Non-Disclosure Agreement
  - job_offer            Job Offer Letter
  - freelancer_agreement Freelancer Agreement
  - service_agreement    Service Agreement
  - consulting_agreement Consulting Agreement
  - lease_agreement      Lease Agreement
  - employment_contract  Employment Contract

Flow:
  1. Client sends document_type + user_query
  2. Type is normalised to canonical slug
  3. LLM extracts structured fields from user_query (own JSON extraction — NOT s_json_utils)
  4. LLM generates the full legal document text
  5. Document text is converted to a downloadable PDF
  6. Response returns fields, document text, and PDF bytes
"""

import os
import re
import json
import time
import logging
from io import BytesIO
from typing import Any

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Supported document types
# ---------------------------------------------------------------------------

SUPPORTED_DOCUMENT_TYPES: dict[str, str] = {
    "nda":                   "Non-Disclosure Agreement",
    "job_offer":             "Job Offer Letter",
    "freelancer_agreement":  "Freelancer Agreement",
    "service_agreement":     "Service Agreement",
    "consulting_agreement":  "Consulting Agreement",
    "lease_agreement":       "Lease Agreement",
    "employment_contract":   "Employment Contract",
}

_TYPE_ALIASES: dict[str, str] = {
    "non disclosure agreement":  "nda",
    "non-disclosure agreement":  "nda",
    "non_disclosure_agreement":  "nda",
    "confidentiality agreement": "nda",
    "job offer":                 "job_offer",
    "offer letter":              "job_offer",
    "job offer letter":          "job_offer",
    "freelance agreement":       "freelancer_agreement",
    "freelancer":                "freelancer_agreement",
    "service":                   "service_agreement",
    "consulting":                "consulting_agreement",
    "consultant agreement":      "consulting_agreement",
    "lease":                     "lease_agreement",
    "rental agreement":          "lease_agreement",
    "employment":                "employment_contract",
    "employment agreement":      "employment_contract",
}


def resolve_document_type(raw: str) -> str | None:
    cleaned  = raw.lower().strip().replace("-", "_").replace(" ", "_")
    if cleaned in SUPPORTED_DOCUMENT_TYPES:
        return cleaned
    readable = raw.lower().strip()
    if readable in _TYPE_ALIASES:
        return _TYPE_ALIASES[readable]
    for alias, slug in _TYPE_ALIASES.items():
        if alias in readable or readable in alias:
            return slug
    for slug in SUPPORTED_DOCUMENT_TYPES:
        if slug in cleaned or cleaned in slug:
            return slug
    return None


# ---------------------------------------------------------------------------
# Field schemas
# ---------------------------------------------------------------------------

_SCHEMAS: dict[str, dict[str, list[str]]] = {
    "nda": {
        "required": ["disclosing_party", "receiving_party", "effective_date",
                     "purpose", "duration_years", "governing_law"],
        "optional": ["confidential_info_description", "exclusions",
                     "return_of_materials", "remedies", "signatory_names"],
    },
    "job_offer": {
        "required": ["candidate_name", "company_name", "job_title",
                     "start_date", "salary", "employment_type", "reporting_to"],
        "optional": ["work_location", "probation_period", "benefits",
                     "equity", "signing_bonus", "offer_expiry_date", "hr_contact"],
    },
    "freelancer_agreement": {
        "required": ["client_name", "freelancer_name", "project_description",
                     "start_date", "end_date", "payment_amount",
                     "payment_schedule", "governing_law"],
        "optional": ["deliverables", "revision_rounds", "intellectual_property",
                     "confidentiality", "kill_fee", "late_payment_penalty"],
    },
    "service_agreement": {
        "required": ["service_provider", "client", "services_description",
                     "start_date", "end_date", "fee", "payment_terms", "governing_law"],
        "optional": ["service_levels", "termination_notice",
                     "limitation_of_liability", "indemnification",
                     "insurance_requirements", "dispute_resolution"],
    },
    "consulting_agreement": {
        "required": ["consultant_name", "client_name", "scope_of_work",
                     "start_date", "end_date", "consulting_fee",
                     "payment_terms", "governing_law"],
        "optional": ["expenses_reimbursement", "non_compete_period",
                     "non_solicitation", "intellectual_property",
                     "confidentiality_period", "termination_clause"],
    },
    "lease_agreement": {
        "required": ["landlord_name", "tenant_name", "property_address",
                     "lease_start_date", "lease_end_date", "monthly_rent",
                     "security_deposit", "governing_law"],
        "optional": ["late_fee", "pet_policy", "utilities_included",
                     "maintenance_responsibilities", "subletting_policy",
                     "renewal_terms", "early_termination_fee"],
    },
    "employment_contract": {
        "required": ["employer_name", "employee_name", "job_title",
                     "department", "start_date", "salary",
                     "work_hours", "governing_law"],
        "optional": ["probation_period", "notice_period", "non_compete",
                     "non_solicitation", "benefits", "leave_policy",
                     "termination_clause", "intellectual_property_assignment"],
    },
}


# ---------------------------------------------------------------------------
# Document section instructions
# ---------------------------------------------------------------------------

_DOCUMENT_INSTRUCTIONS: dict[str, str] = {
    "nda": """Generate a complete Non-Disclosure Agreement with these sections:
1. PARTIES
2. DEFINITION OF CONFIDENTIAL INFORMATION
3. OBLIGATIONS OF RECEIVING PARTY
4. EXCLUSIONS FROM CONFIDENTIAL INFORMATION
5. TERM AND TERMINATION
6. RETURN OF MATERIALS
7. REMEDIES
8. GENERAL PROVISIONS
9. SIGNATURES""",

    "job_offer": """Generate a complete Job Offer Letter with these sections:
1. OPENING AND CONGRATULATIONS
2. POSITION DETAILS
3. COMPENSATION AND BENEFITS
4. START DATE AND WORK LOCATION
5. CONDITIONS OF EMPLOYMENT
6. OFFER EXPIRY
7. ACCEPTANCE INSTRUCTIONS
8. CLOSING AND SIGNATURES""",

    "freelancer_agreement": """Generate a complete Freelancer Agreement with these sections:
1. PARTIES
2. SCOPE OF WORK AND DELIVERABLES
3. TIMELINE
4. COMPENSATION AND PAYMENT SCHEDULE
5. INTELLECTUAL PROPERTY OWNERSHIP
6. CONFIDENTIALITY
7. REVISIONS AND CHANGE REQUESTS
8. CANCELLATION AND KILL FEE
9. INDEPENDENT CONTRACTOR STATUS
10. GENERAL PROVISIONS
11. SIGNATURES""",

    "service_agreement": """Generate a complete Service Agreement with these sections:
1. PARTIES
2. SERVICES DESCRIPTION
3. TERM
4. FEES AND PAYMENT TERMS
5. SERVICE LEVELS
6. CONFIDENTIALITY
7. INTELLECTUAL PROPERTY
8. LIMITATION OF LIABILITY
9. INDEMNIFICATION
10. TERMINATION
11. DISPUTE RESOLUTION
12. GENERAL PROVISIONS
13. SIGNATURES""",

    "consulting_agreement": """Generate a complete Consulting Agreement with these sections:
1. PARTIES
2. SCOPE OF CONSULTING SERVICES
3. TERM
4. CONSULTING FEES AND EXPENSES
5. INTELLECTUAL PROPERTY
6. CONFIDENTIALITY
7. NON-COMPETE AND NON-SOLICITATION
8. INDEPENDENT CONTRACTOR STATUS
9. TERMINATION
10. GENERAL PROVISIONS
11. SIGNATURES""",

    "lease_agreement": """Generate a complete Lease Agreement with these sections:
1. PARTIES AND PROPERTY
2. LEASE TERM
3. RENT AND PAYMENT TERMS
4. SECURITY DEPOSIT
5. UTILITIES AND SERVICES
6. MAINTENANCE AND REPAIRS
7. USE OF PROPERTY
8. PETS AND SMOKING POLICY
9. SUBLETTING
10. EARLY TERMINATION
11. RENEWAL TERMS
12. DEFAULT AND REMEDIES
13. GENERAL PROVISIONS
14. SIGNATURES""",

    "employment_contract": """Generate a complete Employment Contract with these sections:
1. PARTIES
2. POSITION AND DUTIES
3. TERM OF EMPLOYMENT
4. COMPENSATION AND BENEFITS
5. WORKING HOURS AND LOCATION
6. PROBATION PERIOD
7. LEAVE ENTITLEMENTS
8. CONFIDENTIALITY AND NON-DISCLOSURE
9. NON-COMPETE AND NON-SOLICITATION
10. INTELLECTUAL PROPERTY ASSIGNMENT
11. TERMINATION AND NOTICE PERIOD
12. GOVERNING LAW AND DISPUTE RESOLUTION
13. GENERAL PROVISIONS
14. SIGNATURES""",
}


# ---------------------------------------------------------------------------
# OpenAI client helper — does NOT use response_format=json_object
# so we can get plain text for the document and parse JSON ourselves
# for fields
# ---------------------------------------------------------------------------

def _get_client() -> AsyncOpenAI:
    return AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))


def _get_model_kwargs(max_tokens: int = 4096) -> dict:
    from s_ai_model import (
        MODEL_NAME,
        _FIXED_TEMPERATURE_MODELS,
        _MAX_COMPLETION_TOKENS_MODELS,
    )
    kwargs: dict = {"model": MODEL_NAME}
    if MODEL_NAME not in _FIXED_TEMPERATURE_MODELS:
        kwargs["temperature"] = 0.1
    if MODEL_NAME in _MAX_COMPLETION_TOKENS_MODELS:
        kwargs["max_completion_tokens"] = max_tokens
    else:
        kwargs["max_tokens"] = max_tokens
    return kwargs


# ---------------------------------------------------------------------------
# Own JSON extractor — does NOT rely on s_json_utils which forces the
# PDF analysis schema {overview, summary, highlights}
# ---------------------------------------------------------------------------

def _extract_fields_from_json(raw: str) -> dict:
    """
    Extract a plain JSON object from LLM output.
    Returns whatever dict the model returned, without enforcing any schema.
    """
    # Strip markdown fences
    text = raw.replace("```json", "").replace("```", "").strip()

    # Direct parse
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    # Find first { ... } block
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass

    logger.warning("[doc_gen] could not parse field JSON — returning empty dict")
    return {}


# ---------------------------------------------------------------------------
# Step 1: Extract structured fields from user_query
# ---------------------------------------------------------------------------

async def _extract_fields(document_type: str, user_query: str) -> dict[str, Any]:
    schema   = _SCHEMAS[document_type]
    doc_name = SUPPORTED_DOCUMENT_TYPES[document_type]
    all_fields    = schema["required"] + schema["optional"]
    required_list = "\n".join(f"- {f}" for f in schema["required"])
    field_list    = "\n".join(f"- {f}" for f in all_fields)

    system = f"""You are a legal assistant that extracts information for a {doc_name}.

Extract ONLY the fields listed below from the user description.

REQUIRED fields:
{required_list}

ALL fields to extract:
{field_list}

Rules:
- Return ONLY a valid JSON object — no explanation, no markdown, no extra text
- Use the exact field names above as JSON keys (snake_case)
- Extract the value from the description if present
- If a field is not mentioned, set its value to "Not Specified"
- Keep values concise and precise
- For dates: "Month DD, YYYY" or "Not Specified"
- For money: include currency symbol (e.g. "$5,000 USD")"""

    user = f"Description:\n\"\"\"{user_query}\"\"\"\n\nExtract all fields as a JSON object."

    client = _get_client()
    kwargs = _get_model_kwargs(max_tokens=1000)
    kwargs["messages"] = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]
    # Use json_object format ONLY for field extraction
    kwargs["response_format"] = {"type": "json_object"}

    response = await client.chat.completions.create(**kwargs)
    raw      = response.choices[0].message.content or "{}"
    logger.info(f"[doc_gen] field extraction tokens={response.usage.prompt_tokens}in/{response.usage.completion_tokens}out")

    fields = _extract_fields_from_json(raw)

    # Ensure all required fields exist
    for field in schema["required"]:
        if field not in fields or not fields[field]:
            fields[field] = "Not Specified"

    return fields


# ---------------------------------------------------------------------------
# Step 2: Validate required fields
# ---------------------------------------------------------------------------

def _get_missing_fields(document_type: str, fields: dict) -> list[str]:
    return [
        f for f in _SCHEMAS[document_type]["required"]
        if not fields.get(f) or fields.get(f) == "Not Specified"
    ]


# ---------------------------------------------------------------------------
# Step 3: Generate full document text
# ---------------------------------------------------------------------------

async def _generate_document_text(
    document_type: str,
    fields: dict[str, Any],
    user_query: str,
) -> tuple[str, int, int]:
    """
    Returns (document_text, input_tokens, output_tokens).
    """
    doc_name     = SUPPORTED_DOCUMENT_TYPES[document_type]
    instructions = _DOCUMENT_INSTRUCTIONS[document_type]

    field_lines = "\n".join(
        f"  {k.replace('_', ' ').title()}: {v}"
        for k, v in fields.items()
        if v and v != "Not Specified"
    )

    system = f"""You are a senior legal document drafter with 20+ years of experience.
Generate a complete, professional, legally-sound {doc_name}.

{instructions}

Formatting rules:
- Section headings in ALL CAPS followed by a colon (e.g. "1. PARTIES:")
- Numbered sub-clauses within each section (1.1, 1.2, etc.)
- Formal legal language throughout
- Be specific — use the actual names, dates, and amounts provided
- If a value is "Not Specified" use a reasonable legal standard/placeholder like [TO BE AGREED]
- End with a SIGNATURES section with lines for each party
- Do NOT wrap output in markdown — plain text only"""

    user = f"""Generate a complete {doc_name} using these details:

{field_lines}

Additional context:
\"\"\"{user_query}\"\"\"

Write the full document now."""

    client = _get_client()
    kwargs = _get_model_kwargs(max_tokens=4096)
    kwargs["messages"] = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]
    # Plain text output — do NOT use json_object format for document generation

    response      = await client.chat.completions.create(**kwargs)
    document_text = response.choices[0].message.content or ""
    in_tok        = response.usage.prompt_tokens
    out_tok       = response.usage.completion_tokens

    logger.info(
        f"[doc_gen] document generated — "
        f"type={document_type} length={len(document_text)} chars "
        f"tokens={in_tok}in/{out_tok}out"
    )
    return document_text, in_tok, out_tok


# ---------------------------------------------------------------------------
# Step 4: Render document text → PDF bytes using reportlab
# ---------------------------------------------------------------------------

def _render_pdf(document_text: str, document_name: str) -> bytes:
    """
    Convert plain-text legal document to a professionally formatted PDF.
    Uses reportlab with a clean business template:
      - Header with document name and date
      - Body with proper fonts, line spacing, section detection
      - Footer with page numbers
    """
    from reportlab.lib.pagesizes import LETTER
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib.colors import HexColor, black, white
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer,
        HRFlowable, Table, TableStyle,
    )
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
    from datetime import datetime

    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=LETTER,
        rightMargin=1.0 * inch,
        leftMargin=1.0 * inch,
        topMargin=1.2 * inch,
        bottomMargin=1.0 * inch,
    )

    # ── Colours ──────────────────────────────────────────────────────────
    DARK_NAVY   = HexColor("#1a2744")
    MID_BLUE    = HexColor("#2c4a8c")
    LIGHT_GREY  = HexColor("#f5f5f5")
    BORDER_GREY = HexColor("#cccccc")
    TEXT_DARK   = HexColor("#1a1a1a")
    TEXT_MUTED  = HexColor("#555555")

    # ── Styles ────────────────────────────────────────────────────────────
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "DocTitle",
        fontName="Helvetica-Bold",
        fontSize=20,
        textColor=white,
        alignment=TA_CENTER,
        spaceAfter=4,
        leading=26,
    )

    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        fontName="Helvetica",
        fontSize=11,
        textColor=HexColor("#ccd6f0"),
        alignment=TA_CENTER,
        spaceAfter=0,
    )

    section_style = ParagraphStyle(
        "SectionHead",
        fontName="Helvetica-Bold",
        fontSize=11,
        textColor=MID_BLUE,
        spaceBefore=18,
        spaceAfter=6,
        leading=15,
        borderPad=0,
    )

    body_style = ParagraphStyle(
        "Body",
        fontName="Helvetica",
        fontSize=10,
        textColor=TEXT_DARK,
        leading=16,
        spaceAfter=4,
        alignment=TA_JUSTIFY,
    )

    clause_style = ParagraphStyle(
        "Clause",
        fontName="Helvetica",
        fontSize=10,
        textColor=TEXT_DARK,
        leading=16,
        spaceAfter=4,
        leftIndent=20,
        alignment=TA_JUSTIFY,
    )

    sig_label_style = ParagraphStyle(
        "SigLabel",
        fontName="Helvetica-Bold",
        fontSize=9,
        textColor=TEXT_MUTED,
        spaceAfter=2,
    )

    sig_value_style = ParagraphStyle(
        "SigValue",
        fontName="Helvetica",
        fontSize=10,
        textColor=TEXT_DARK,
        spaceAfter=16,
    )

    # ── Helper to add page number footer ─────────────────────────────────
    def _on_page(canvas, doc):
        canvas.saveState()
        # Footer line
        canvas.setStrokeColor(BORDER_GREY)
        canvas.setLineWidth(0.5)
        canvas.line(inch, 0.75 * inch, LETTER[0] - inch, 0.75 * inch)
        # Page number
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(TEXT_MUTED)
        canvas.drawCentredString(
            LETTER[0] / 2,
            0.55 * inch,
            f"Page {doc.page}  |  {document_name}  |  Confidential"
        )
        canvas.restoreState()

    # ── Build content ─────────────────────────────────────────────────────
    story = []
    today = datetime.now().strftime("%B %d, %Y")

    # Header banner (dark navy background with title)
    header_data = [[
        Paragraph(document_name.upper(), title_style),
    ]]
    header_table = Table(header_data, colWidths=[6.5 * inch])
    header_table.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, -1), DARK_NAVY),
        ("TOPPADDING",  (0, 0), (-1, -1), 20),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 20),
        ("LEFTPADDING",  (0, 0), (-1, -1), 20),
        ("RIGHTPADDING", (0, 0), (-1, -1), 20),
        ("ROUNDEDCORNERS", [6]),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 6))

    # Date line
    story.append(Paragraph(
        f"<font color='#888888' size='9'>Generated: {today}</font>",
        ParagraphStyle("DateLine", fontName="Helvetica", fontSize=9,
                       textColor=HexColor("#888888"), alignment=TA_LEFT)
    ))
    story.append(Spacer(1, 14))

    # ── Parse and render document text ───────────────────────────────────
    lines = document_text.split("\n")
    in_signature = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            story.append(Spacer(1, 4))
            continue

        # Detect SIGNATURES section
        if re.match(r'^\d+\.\s+SIGNATURE', stripped.upper()):
            in_signature = True

        # Section heading: "1. HEADING:" or "HEADING:" or all-caps short line
        is_section = (
            re.match(r'^\d+\.\s+[A-Z][A-Z\s,/&()-]{3,}[:.]?\s*$', stripped)
            or re.match(r'^[A-Z][A-Z\s,/&()-]{3,}[:.]?\s*$', stripped)
            or (stripped.isupper() and len(stripped) > 4 and len(stripped) < 80)
        )

        if is_section and not in_signature:
            story.append(HRFlowable(
                width="100%", thickness=0.5,
                color=BORDER_GREY, spaceAfter=4, spaceBefore=8
            ))
            story.append(Paragraph(stripped, section_style))
            continue

        # Numbered sub-clause: "1.1 text" or "1.1. text"
        if re.match(r'^\d+\.\d+\.?\s+', stripped) and not in_signature:
            story.append(Paragraph(stripped, clause_style))
            continue

        # Signature block lines
        if in_signature:
            if re.match(r'^(Name|Title|Date|Company|Signature)\s*:', stripped, re.I):
                label, _, rest = stripped.partition(":")
                story.append(Paragraph(label.strip() + ":", sig_label_style))
                val = rest.strip() if rest.strip() else "________________________"
                story.append(Paragraph(val, sig_value_style))
                continue
            if re.match(r'^_{3,}', stripped):
                story.append(Paragraph("________________________", sig_value_style))
                continue

        # Regular paragraph
        story.append(Paragraph(stripped, body_style))

    # ── Build PDF ─────────────────────────────────────────────────────────
    doc.build(story, onFirstPage=_on_page, onLaterPages=_on_page)
    pdf_bytes = buffer.getvalue()
    buffer.close()

    logger.info(f"[doc_gen] PDF rendered — {len(pdf_bytes):,} bytes")
    return pdf_bytes


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def generate_document(
    document_type: str,
    user_query: str,
) -> dict:
    """
    Main entry point. Returns:
      status          : "success" | "missing_fields"
      document_type   : canonical slug
      document_name   : display name
      fields          : extracted structured fields
      missing_fields  : required fields not found in query
      document        : full document text
      pdf_bytes       : rendered PDF as bytes (use separately for download)
      word_count      : word count of document
    """
    doc_name = SUPPORTED_DOCUMENT_TYPES[document_type]
    logger.info(f"[doc_gen] generating '{doc_name}' from query ({len(user_query)} chars)")

    # Step 1: Extract fields
    fields = await _extract_fields(document_type, user_query)
    logger.info(f"[doc_gen] extracted fields: {list(fields.keys())}")

    # Step 2: Check missing
    missing = _get_missing_fields(document_type, fields)
    if missing:
        logger.warning(f"[doc_gen] missing required fields: {missing}")

    # Step 3: Generate document text
    document_text, in_tok, out_tok = await _generate_document_text(
        document_type, fields, user_query
    )

    # Step 4: Render PDF
    pdf_bytes = _render_pdf(document_text, doc_name)

    return {
        "status":        "missing_fields" if missing else "success",
        "document_type": document_type,
        "document_name": doc_name,
        "fields":        fields,
        "missing_fields": missing,
        "document":      document_text,
        "pdf_bytes":     pdf_bytes,
        "word_count":    len(document_text.split()),
        "input_tokens":  in_tok,
        "output_tokens": out_tok,
    }