"""
t_document_generator.py

Business-level legal document generation engine.

Features:
  - Separate extraction + generation prompt per document type
  - document_type is OPTIONAL — auto-detected via intent classification
  - Outputs: plain text + DOCX file bytes
  - gpt-5-nano compatible (no temperature, uses max_completion_tokens)
  - Field extraction uses json_object mode
  - Document generation uses plain text mode (no response_format)

Supported types:
  nda, job_offer, freelancer_agreement, service_agreement,
  consulting_agreement, lease_agreement, employment_contract
"""

import os
import re
import json
import logging
from io import BytesIO
from typing import Any

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model config — read from environment
# ---------------------------------------------------------------------------

_MODEL    = os.environ.get("MODEL_NAME", "gpt-5-nano")
_API_KEY  = os.environ.get("OPENAI_API_KEY", "")

# Models that do NOT support temperature parameter
_NO_TEMP  = {"gpt-5-nano", "gpt-4.1-nano", "gpt-4o-mini", "o1", "o1-mini", "o3-mini", "o3"}

# Models that use max_completion_tokens instead of max_tokens
_MCT      = {"gpt-5-nano", "gpt-4.1-nano", "gpt-4o-mini", "o1", "o1-mini", "o3-mini", "o3"}


def _get_client() -> AsyncOpenAI:
    key = _API_KEY or os.environ.get("OPENAI_API_KEY", "")
    return AsyncOpenAI(api_key=key)


def _api_kwargs(max_tokens: int, use_json: bool = False) -> dict:
    """
    Build OpenAI API kwargs.
    use_json=True  → field extraction   (response_format: json_object)
    use_json=False → document generation (plain text, NO response_format)

    IMPORTANT: setting response_format=json_object on a plain-text generation
    call causes gpt-5-nano to return empty content. Never set it for generation.
    """
    model  = os.environ.get("MODEL_NAME", _MODEL)
    kwargs: dict = {"model": model}

    # Temperature — not supported by gpt-5-nano family
    if model not in _NO_TEMP:
        kwargs["temperature"] = 0.1

    # Token limit parameter name differs by model
    if model in _MCT:
        kwargs["max_completion_tokens"] = max_tokens
    else:
        kwargs["max_tokens"] = max_tokens

    # JSON mode ONLY for field extraction
    if use_json:
        kwargs["response_format"] = {"type": "json_object"}

    return kwargs


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
# Per-document-type EXTRACTION prompts
# ---------------------------------------------------------------------------

_EXTRACTION_PROMPTS: dict[str, str] = {

    "nda": """You are a legal assistant specialising in Non-Disclosure Agreements.
Extract fields from the user description. Return ONLY a valid JSON object.

Required JSON keys:
disclosing_party, receiving_party, effective_date, purpose, duration_years, governing_law

Optional JSON keys:
confidential_info_description, exclusions, return_of_materials, remedies, signatory_names

Rules:
- Use exact key names shown above
- Extract value from description if present
- Set to "Not Specified" if not mentioned
- Return ONLY the JSON object, nothing else""",

    "job_offer": """You are an HR specialist extracting job offer details.
Extract fields from the user description. Return ONLY a valid JSON object.

Required JSON keys:
candidate_name, company_name, job_title, start_date, salary, employment_type, reporting_to

Optional JSON keys:
work_location, probation_period, benefits, equity, signing_bonus, offer_expiry_date, hr_contact

Rules:
- Use exact key names shown above
- Extract value from description if present
- Set to "Not Specified" if not mentioned
- Return ONLY the JSON object, nothing else""",

    "freelancer_agreement": """You are a contracts specialist extracting freelancer agreement details.
Extract fields from the user description. Return ONLY a valid JSON object.

Required JSON keys:
client_name, freelancer_name, project_description, start_date, end_date, payment_amount, payment_schedule, governing_law

Optional JSON keys:
deliverables, revision_rounds, intellectual_property, confidentiality, kill_fee, late_payment_penalty

Rules:
- Use exact key names shown above
- Extract value from description if present
- Set to "Not Specified" if not mentioned
- Return ONLY the JSON object, nothing else""",

    "service_agreement": """You are a contracts specialist extracting service agreement details.
Extract fields from the user description. Return ONLY a valid JSON object.

Required JSON keys:
service_provider, client, services_description, start_date, end_date, fee, payment_terms, governing_law

Optional JSON keys:
service_levels, termination_notice, limitation_of_liability, indemnification, insurance_requirements, dispute_resolution

Rules:
- Use exact key names shown above
- Extract value from description if present
- Set to "Not Specified" if not mentioned
- Return ONLY the JSON object, nothing else""",

    "consulting_agreement": """You are a contracts specialist extracting consulting agreement details.
Extract fields from the user description. Return ONLY a valid JSON object.

Required JSON keys:
consultant_name, client_name, scope_of_work, start_date, end_date, consulting_fee, payment_terms, governing_law

Optional JSON keys:
expenses_reimbursement, non_compete_period, non_solicitation, intellectual_property, confidentiality_period, termination_clause

Rules:
- Use exact key names shown above
- Extract value from description if present
- Set to "Not Specified" if not mentioned
- Return ONLY the JSON object, nothing else""",

    "lease_agreement": """You are a real estate attorney extracting lease agreement details.
Extract fields from the user description. Return ONLY a valid JSON object.

Required JSON keys:
landlord_name, tenant_name, property_address, lease_start_date, lease_end_date, monthly_rent, security_deposit, governing_law

Optional JSON keys:
late_fee, pet_policy, utilities_included, maintenance_responsibilities, subletting_policy, renewal_terms, early_termination_fee

Rules:
- Use exact key names shown above
- Extract value from description if present
- Set to "Not Specified" if not mentioned
- Return ONLY the JSON object, nothing else""",

    "employment_contract": """You are an employment law specialist extracting contract details.
Extract fields from the user description. Return ONLY a valid JSON object.

Required JSON keys:
employer_name, employee_name, job_title, department, start_date, salary, work_hours, governing_law

Optional JSON keys:
probation_period, notice_period, non_compete, non_solicitation, benefits, leave_policy, termination_clause, intellectual_property_assignment

Rules:
- Use exact key names shown above
- Extract value from description if present
- Set to "Not Specified" if not mentioned
- Return ONLY the JSON object, nothing else""",
}


# ---------------------------------------------------------------------------
# Per-document-type GENERATION prompts
# ---------------------------------------------------------------------------

_GENERATION_PROMPTS: dict[str, str] = {

    "nda": """You are a senior attorney specialising in confidentiality law.
Draft a complete, enforceable Non-Disclosure Agreement.

Include these sections:
1. PARTIES
2. RECITALS
3. DEFINITIONS
   3.1 Definition of Confidential Information
   3.2 Exclusions from Confidential Information
4. OBLIGATIONS OF RECEIVING PARTY
   4.1 Non-Disclosure Obligation
   4.2 Standard of Care
   4.3 Permitted Disclosures
5. TERM AND TERMINATION
6. RETURN OR DESTRUCTION OF MATERIALS
7. REMEDIES
   7.1 Injunctive Relief
   7.2 Damages
8. GENERAL PROVISIONS
   8.1 Governing Law
   8.2 Entire Agreement
   8.3 Amendments
   8.4 Severability
9. SIGNATURE BLOCK

Formatting:
- Section headings in ALL CAPS with number (e.g. "1. PARTIES:")
- Sub-clauses numbered 1.1, 1.2 etc.
- Formal legal language
- Plain text only, no markdown
- Use actual values provided, use [TO BE AGREED] for missing values""",

    "job_offer": """You are an experienced HR Director drafting a job offer letter.
Write in a warm, professional tone.

Include these sections:
1. DATE AND ADDRESSEE
2. OPENING — congratulate the candidate
3. POSITION DETAILS
4. COMPENSATION
5. BENEFITS
6. START DATE AND LOCATION
7. EMPLOYMENT CONDITIONS
8. PROBATION PERIOD
9. OFFER EXPIRY
10. NEXT STEPS
11. CLOSING
12. SIGNATURE BLOCK

Formatting:
- Warm professional tone for a letter
- Plain text only, no markdown
- Use actual values provided""",

    "freelancer_agreement": """You are a contracts attorney drafting a Freelancer Agreement.

Include these sections:
1. PARTIES AND RECITALS
2. SCOPE OF WORK
   2.1 Project Description
   2.2 Deliverables
   2.3 Timeline
3. COMPENSATION AND PAYMENT
   3.1 Project Fee
   3.2 Payment Schedule
   3.3 Late Payment
4. REVISIONS AND CHANGE ORDERS
5. INTELLECTUAL PROPERTY
   5.1 Work-for-Hire
   5.2 Assignment of Rights
6. CONFIDENTIALITY
7. INDEPENDENT CONTRACTOR STATUS
8. CANCELLATION AND KILL FEE
9. LIMITATION OF LIABILITY
10. GENERAL PROVISIONS
11. SIGNATURE BLOCK

Formatting:
- Section headings in ALL CAPS with number
- Sub-clauses numbered
- Formal legal language
- Plain text only, no markdown""",

    "service_agreement": """You are a commercial contracts attorney drafting a Service Agreement.

Include these sections:
1. PARTIES
2. SERVICES
   2.1 Scope of Services
   2.2 Service Standards
   2.3 Change in Scope
3. TERM AND RENEWAL
4. FEES AND PAYMENT
   4.1 Service Fees
   4.2 Payment Terms
   4.3 Late Payment Interest
5. INTELLECTUAL PROPERTY
6. CONFIDENTIALITY
7. LIMITATION OF LIABILITY
8. INDEMNIFICATION
9. TERMINATION
   9.1 Termination for Convenience
   9.2 Termination for Cause
10. DISPUTE RESOLUTION
11. GENERAL PROVISIONS
12. SIGNATURE BLOCK

Formatting:
- Section headings in ALL CAPS with number
- Sub-clauses numbered
- Formal legal language
- Plain text only, no markdown""",

    "consulting_agreement": """You are a commercial attorney drafting a Consulting Agreement.

Include these sections:
1. PARTIES
2. SCOPE OF CONSULTING SERVICES
3. TERM
4. COMPENSATION AND EXPENSES
   4.1 Consulting Fees
   4.2 Expense Reimbursement
   4.3 Invoicing and Payment
5. INTELLECTUAL PROPERTY
6. CONFIDENTIALITY
7. NON-COMPETE AND NON-SOLICITATION
8. INDEPENDENT CONTRACTOR
9. LIMITATION OF LIABILITY
10. TERMINATION
11. GENERAL PROVISIONS
12. SIGNATURE BLOCK

Formatting:
- Section headings in ALL CAPS with number
- Sub-clauses numbered
- Formal legal language
- Plain text only, no markdown""",

    "lease_agreement": """You are a real estate attorney drafting a Lease Agreement.

Include these sections:
1. PARTIES AND PREMISES
2. LEASE TERM
3. RENT
   3.1 Monthly Rent
   3.2 Due Date and Payment Method
   3.3 Late Fee
4. SECURITY DEPOSIT
5. UTILITIES AND SERVICES
6. USE OF PREMISES
7. MAINTENANCE AND REPAIRS
   7.1 Landlord Obligations
   7.2 Tenant Obligations
8. ASSIGNMENT AND SUBLETTING
9. PETS
10. DEFAULT AND REMEDIES
11. EARLY TERMINATION
12. RENEWAL
13. GENERAL PROVISIONS
14. SIGNATURE BLOCK

Formatting:
- Section headings in ALL CAPS with number
- Sub-clauses numbered
- Formal legal language
- Plain text only, no markdown""",

    "employment_contract": """You are an employment law attorney drafting an Employment Contract.

Include these sections:
1. PARTIES
2. POSITION AND DUTIES
   2.1 Job Title and Department
   2.2 Duties and Responsibilities
   2.3 Reporting Structure
3. COMMENCEMENT AND TERM
4. COMPENSATION
   4.1 Base Salary
   4.2 Performance Review
5. BENEFITS
   5.1 Health Insurance
   5.2 Annual Leave
   5.3 Sick Leave
6. PROBATIONARY PERIOD
7. WORKING HOURS
8. CONFIDENTIALITY AND NON-DISCLOSURE
9. INTELLECTUAL PROPERTY ASSIGNMENT
10. NON-COMPETE AND NON-SOLICITATION
11. TERMINATION
    11.1 Termination by Employer
    11.2 Resignation
    11.3 Notice Period
12. GOVERNING LAW AND DISPUTE RESOLUTION
13. GENERAL PROVISIONS
14. SIGNATURE BLOCK

Formatting:
- Section headings in ALL CAPS with number
- Sub-clauses numbered
- Formal legal language
- Plain text only, no markdown""",
}


# ---------------------------------------------------------------------------
# Intent classification
# ---------------------------------------------------------------------------

_INTENT_SYSTEM = """You are a legal document classifier.
Read the description and return ONLY the matching slug.

Slugs:
nda, job_offer, freelancer_agreement, service_agreement,
consulting_agreement, lease_agreement, employment_contract

Return ONLY the slug. No explanation. If none match return unknown."""


async def classify_intent(user_query: str) -> str | None:
    client = _get_client()
    kwargs = _api_kwargs(max_tokens=20, use_json=False)
    kwargs["messages"] = [
        {"role": "system", "content": _INTENT_SYSTEM},
        {"role": "user",   "content": f"Classify:\n\"\"\"{user_query[:800]}\"\"\""},
    ]
    try:
        response = await client.chat.completions.create(**kwargs)
        raw      = (response.choices[0].message.content or "").strip().lower()
        raw      = re.sub(r'[^a-z_]', '', raw)
        if raw in SUPPORTED_DOCUMENT_TYPES:
            logger.info(f"[doc_gen] intent classified: '{raw}'")
            return raw
        resolved = resolve_document_type(raw)
        if resolved:
            return resolved
        logger.warning(f"[doc_gen] intent unknown: '{raw}'")
        return None
    except Exception as e:
        logger.error(f"[doc_gen] intent classification failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Own JSON extractor — no dependency on s_json_utils
# ---------------------------------------------------------------------------

def _parse_json(raw: str) -> dict:
    text = raw.replace("```json", "").replace("```", "").strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
    logger.warning("[doc_gen] could not parse JSON")
    return {}


# ---------------------------------------------------------------------------
# Step 1: Extract fields — json_object mode
# ---------------------------------------------------------------------------

async def _extract_fields(document_type: str, user_query: str) -> dict[str, Any]:
    schema = _SCHEMAS[document_type]
    system = _EXTRACTION_PROMPTS[document_type]

    client = _get_client()
    # use_json=True → sets response_format=json_object for reliable JSON output
    kwargs = _api_kwargs(max_tokens=4096, use_json=True)
    kwargs["messages"] = [
        {"role": "system", "content": system},
        {"role": "user",   "content": (
            f"Extract all fields from this description as a JSON object.\n\n"
            f"Description:\n\"\"\"{user_query}\"\"\""
        )},
    ]

    response = await client.chat.completions.create(**kwargs)
    raw      = response.choices[0].message.content or "{}"
    logger.info(
        f"[doc_gen] field extraction "
        f"tokens={response.usage.prompt_tokens}in/{response.usage.completion_tokens}out "
        f"finish={response.choices[0].finish_reason}"
    )
    logger.debug(f"[doc_gen] extraction raw: {raw[:300]}")

    fields = _parse_json(raw)

    # Ensure all required fields exist
    for field in schema["required"]:
        if field not in fields or not fields[field]:
            fields[field] = "Not Specified"

    # Log what was actually extracted
    found = {k: v for k, v in fields.items() if v != "Not Specified"}
    logger.info(f"[doc_gen] extracted {len(found)} non-empty fields: {list(found.keys())}")

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
# Step 3: Generate document text — plain text mode, NO response_format
# ---------------------------------------------------------------------------

async def _generate_document_text(
    document_type: str,
    fields: dict[str, Any],
    user_query: str,
) -> tuple[str, int, int]:
    doc_name   = SUPPORTED_DOCUMENT_TYPES[document_type]
    gen_prompt = _GENERATION_PROMPTS[document_type]

    field_lines = "\n".join(
        f"  {k.replace('_', ' ').title()}: {v}"
        for k, v in fields.items()
        if v and v != "Not Specified"
    )

    system = gen_prompt  # generation prompt already contains all formatting rules

    user = f"""Use these details to generate the complete {doc_name}:

{field_lines if field_lines else "(Use standard template values)"}

Additional context from user:
\"\"\"{user_query}\"\"\"

Write the complete {doc_name} now. Start directly with the document title."""

    client = _get_client()
    # CRITICAL: use_json=False — do NOT set response_format for plain text generation
    # Setting response_format=json_object causes gpt-5-nano to return empty content
    kwargs = _api_kwargs(max_tokens=32000, use_json=False)
    kwargs["messages"] = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]

    response      = await client.chat.completions.create(**kwargs)
    document_text = response.choices[0].message.content or ""
    in_tok        = response.usage.prompt_tokens
    out_tok       = response.usage.completion_tokens
    finish        = response.choices[0].finish_reason

    logger.info(
        f"[doc_gen] document generated — type={document_type} "
        f"length={len(document_text)} chars "
        f"tokens={in_tok}in/{out_tok}out "
        f"finish={finish}"
    )

    if not document_text.strip():
        logger.error(
            f"[doc_gen] empty response from model — "
            f"finish_reason={finish} model={os.environ.get('MODEL_NAME', _MODEL)}"
        )

    return document_text, in_tok, out_tok


# ---------------------------------------------------------------------------
# Step 4: Render DOCX
# ---------------------------------------------------------------------------

def _render_docx(document_text: str, document_name: str) -> bytes:
    from docx import Document as DocxDocument
    from docx.shared import Pt, RGBColor, Inches
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement
    from datetime import datetime

    doc = DocxDocument()

    for section in doc.sections:
        section.top_margin    = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin   = Inches(1.25)
        section.right_margin  = Inches(1.25)

    BLUE  = RGBColor(0x2c, 0x4a, 0x8c)
    DARK  = RGBColor(0x1a, 0x1a, 0x1a)
    MUTED = RGBColor(0x55, 0x55, 0x55)
    WHITE = RGBColor(0xFF, 0xFF, 0xFF)

    def _set_cell_bg(cell, hex_color: str):
        tc   = cell._tc
        tcPr = tc.get_or_add_tcPr()
        shd  = OxmlElement("w:shd")
        shd.set(qn("w:val"),   "clear")
        shd.set(qn("w:color"), "auto")
        shd.set(qn("w:fill"),  hex_color)
        tcPr.append(shd)

    def _remove_table_borders(table):
        tbl   = table._tbl
        tblPr = tbl.find(qn("w:tblPr"))
        if tblPr is None:
            tblPr = OxmlElement("w:tblPr")
            tbl.insert(0, tblPr)
        tblBorders = OxmlElement("w:tblBorders")
        for side in ("top", "left", "bottom", "right", "insideH", "insideV"):
            b = OxmlElement(f"w:{side}")
            b.set(qn("w:val"), "none")
            tblBorders.append(b)
        tblPr.append(tblBorders)

    # Header banner
    table = doc.add_table(rows=1, cols=1)
    table.style = "Table Grid"
    cell  = table.cell(0, 0)
    _set_cell_bg(cell, "1a2744")
    _remove_table_borders(table)
    para = cell.paragraphs[0]
    para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run  = para.add_run(document_name.upper())
    run.bold           = True
    run.font.size      = Pt(20)
    run.font.color.rgb = WHITE

    doc.add_paragraph()

    dp  = doc.add_paragraph()
    dr  = dp.add_run(f"Generated: {datetime.now().strftime('%B %d, %Y')}")
    dr.font.size      = Pt(9)
    dr.font.color.rgb = RGBColor(0x88, 0x88, 0x88)
    dp.paragraph_format.space_after = Pt(12)

    lines        = document_text.split("\n")
    in_signature = False

    for line in lines:
        stripped = line.strip()

        if not stripped:
            sp = doc.add_paragraph()
            sp.paragraph_format.space_before = Pt(0)
            sp.paragraph_format.space_after  = Pt(3)
            continue

        if re.match(r'^\d+\.\s+SIGNATURE', stripped.upper()):
            in_signature = True

        is_section = (
            re.match(r'^\d+\.\s+[A-Z][A-Z\s,/&().-]{3,}[:.]?\s*$', stripped)
            or re.match(r'^[A-Z][A-Z\s,/&().-]{3,}[:.]?\s*$', stripped)
            or (stripped.isupper() and 4 < len(stripped) < 80)
        )

        if is_section and not in_signature:
            p    = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(14)
            p.paragraph_format.space_after  = Pt(4)
            pPr  = p._p.get_or_add_pPr()
            pBdr = OxmlElement("w:pBdr")
            top  = OxmlElement("w:top")
            top.set(qn("w:val"),   "single")
            top.set(qn("w:sz"),    "4")
            top.set(qn("w:space"), "1")
            top.set(qn("w:color"), "CCCCCC")
            pBdr.append(top)
            pPr.append(pBdr)
            run = p.add_run(stripped)
            run.bold           = True
            run.font.size      = Pt(11)
            run.font.color.rgb = BLUE
            continue

        if re.match(r'^\d+\.\d+\.?\s+', stripped) and not in_signature:
            p   = doc.add_paragraph(style="Normal")
            p.paragraph_format.left_indent = Inches(0.3)
            p.paragraph_format.space_after = Pt(4)
            run = p.add_run(stripped)
            run.font.size      = Pt(10)
            run.font.color.rgb = DARK
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            continue

        if in_signature and re.match(r'^(Name|Title|Date|Company|Signature)\s*:', stripped, re.I):
            label, _, rest = stripped.partition(":")
            lp  = doc.add_paragraph()
            lr  = lp.add_run(label.strip() + ":")
            lr.bold            = True
            lr.font.size       = Pt(9)
            lr.font.color.rgb  = MUTED
            lp.paragraph_format.space_after = Pt(2)
            vp  = doc.add_paragraph()
            vr  = vp.add_run(rest.strip() if rest.strip() else "________________________")
            vr.font.size      = Pt(10)
            vr.font.color.rgb = DARK
            vp.paragraph_format.space_after = Pt(12)
            continue

        p   = doc.add_paragraph(style="Normal")
        run = p.add_run(stripped)
        run.font.size      = Pt(10)
        run.font.color.rgb = DARK
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.space_after = Pt(4)

    # Footer
    for section in doc.sections:
        footer      = section.footer
        footer_para = footer.paragraphs[0]
        footer_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = footer_para.add_run(f"{document_name}  |  Confidential  |  Page ")
        run.font.size      = Pt(8)
        run.font.color.rgb = RGBColor(0x88, 0x88, 0x88)
        fldChar1              = OxmlElement("w:fldChar")
        fldChar1.set(qn("w:fldCharType"), "begin")
        instrText             = OxmlElement("w:instrText")
        instrText.text        = "PAGE"
        fldChar2              = OxmlElement("w:fldChar")
        fldChar2.set(qn("w:fldCharType"), "end")
        run2 = footer_para.add_run()
        run2.font.size      = Pt(8)
        run2.font.color.rgb = RGBColor(0x88, 0x88, 0x88)
        run2._r.append(fldChar1)
        run2._r.append(instrText)
        run2._r.append(fldChar2)

    buffer     = BytesIO()
    doc.save(buffer)
    docx_bytes = buffer.getvalue()
    buffer.close()

    logger.info(f"[doc_gen] DOCX rendered — {len(docx_bytes):,} bytes")
    return docx_bytes


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def generate_document(
    document_type: str | None,
    user_query: str,
) -> dict:
    """
    Main entry point.
    document_type is optional — auto-detected from user_query if None.
    """
    total_in_tok  = 0
    total_out_tok = 0

    # Resolve or auto-detect document type
    if document_type:
        doc_type = resolve_document_type(document_type)
        if not doc_type:
            return {
                "status":        "unknown_type",
                "document_type": document_type,
                "document_name": "",
                "fields":        {},
                "missing_fields": [],
                "document":      "",
                "docx_bytes":    b"",
                "word_count":    0,
                "input_tokens":  0,
                "output_tokens": 0,
                "message": (
                    f"'{document_type}' is not supported. "
                    f"Supported: {', '.join(SUPPORTED_DOCUMENT_TYPES.keys())}"
                ),
            }
    else:
        logger.info("[doc_gen] document_type not provided — classifying intent...")
        doc_type = await classify_intent(user_query)
        if not doc_type:
            return {
                "status":        "unknown_type",
                "document_type": None,
                "document_name": "",
                "fields":        {},
                "missing_fields": [],
                "document":      "",
                "docx_bytes":    b"",
                "word_count":    0,
                "input_tokens":  total_in_tok,
                "output_tokens": total_out_tok,
                "message": (
                    "Could not determine document type. "
                    "Please specify document_type. "
                    f"Supported: {', '.join(SUPPORTED_DOCUMENT_TYPES.keys())}"
                ),
            }

    doc_name = SUPPORTED_DOCUMENT_TYPES[doc_type]
    logger.info(f"[doc_gen] generating '{doc_name}' ({len(user_query)} chars)")

    # Extract fields
    fields = await _extract_fields(doc_type, user_query)

    # Check missing
    missing = _get_missing_fields(doc_type, fields)
    if missing:
        logger.warning(f"[doc_gen] missing required fields: {missing}")

    # Generate document text
    document_text, in_tok, out_tok = await _generate_document_text(
        doc_type, fields, user_query
    )
    total_in_tok  += in_tok
    total_out_tok += out_tok

    # Render DOCX
    docx_bytes = _render_docx(document_text, doc_name)

    return {
        "status":         "missing_fields" if missing else "success",
        "document_type":  doc_type,
        "document_name":  doc_name,
        "fields":         fields,
        "missing_fields": missing,
        "document":       document_text,
        "docx_bytes":     docx_bytes,
        "word_count":     len(document_text.split()),
        "input_tokens":   total_in_tok,
        "output_tokens":  total_out_tok,
    }