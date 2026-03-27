"""
t_document_generator.py

Business-level legal document generation engine.

Features:
  - Separate extraction + generation prompt per document type
  - document_type is OPTIONAL — if omitted, intent classification detects it
  - Outputs: plain text + DOCX file bytes
  - No dependency on s_json_utils (uses own JSON parser)

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
# Per-document-type field extraction prompts
# ---------------------------------------------------------------------------

_EXTRACTION_PROMPTS: dict[str, str] = {

    "nda": """You are a legal assistant specialising in Non-Disclosure Agreements.

Extract the following fields from the user description.
Return ONLY a valid JSON object — no explanation, no markdown.

Fields to extract:
- disclosing_party     : name of company/person sharing confidential info
- receiving_party      : name of company/person receiving confidential info
- effective_date       : when the NDA takes effect (e.g. "April 1, 2026")
- purpose              : why confidential information is being shared
- duration_years       : how long the NDA lasts (e.g. "2 years")
- governing_law        : jurisdiction/state/country governing the agreement
- confidential_info_description : what specific info is covered (optional)
- exclusions           : what is NOT considered confidential (optional)
- return_of_materials  : must materials be returned on termination? (optional)
- remedies             : what remedies apply for breach (optional)
- signatory_names      : names of people signing (optional)

Rules:
- Use exact field names as JSON keys
- If a field is not mentioned, set its value to "Not Specified"
- Keep values concise and precise""",

    "job_offer": """You are an HR specialist drafting Job Offer Letters.

Extract the following fields from the user description.
Return ONLY a valid JSON object — no explanation, no markdown.

Fields to extract:
- candidate_name   : full name of the person receiving the offer
- company_name     : name of the company making the offer
- job_title        : the position being offered
- start_date       : proposed start date
- salary           : annual salary or hourly rate with currency
- employment_type  : full-time, part-time, contract, etc.
- reporting_to     : manager/supervisor name or title
- work_location    : office location or remote (optional)
- probation_period : length of probation if any (optional)
- benefits         : health insurance, PTO, retirement, etc. (optional)
- equity           : stock options or equity grant (optional)
- signing_bonus    : one-time signing bonus amount (optional)
- offer_expiry_date: deadline to accept the offer (optional)
- hr_contact       : HR contact name and email (optional)

Rules:
- Use exact field names as JSON keys
- If a field is not mentioned, set its value to "Not Specified"
- Keep values concise and precise""",

    "freelancer_agreement": """You are a contracts specialist drafting Freelancer Agreements.

Extract the following fields from the user description.
Return ONLY a valid JSON object — no explanation, no markdown.

Fields to extract:
- client_name         : name of the company/person hiring the freelancer
- freelancer_name     : name of the freelancer
- project_description : what work is being done
- start_date          : when work begins
- end_date            : when work is due to be completed
- payment_amount      : total payment amount with currency
- payment_schedule    : when and how payment is made (milestone/weekly/on-completion)
- governing_law       : jurisdiction governing the agreement
- deliverables        : specific outputs expected (optional)
- revision_rounds     : how many rounds of revisions are included (optional)
- intellectual_property : who owns the work product (optional)
- confidentiality     : any confidentiality requirements (optional)
- kill_fee            : cancellation fee if project is terminated (optional)
- late_payment_penalty: penalty for late payment (optional)

Rules:
- Use exact field names as JSON keys
- If a field is not mentioned, set its value to "Not Specified"
- Keep values concise and precise""",

    "service_agreement": """You are a contracts specialist drafting Service Agreements.

Extract the following fields from the user description.
Return ONLY a valid JSON object — no explanation, no markdown.

Fields to extract:
- service_provider    : name of company/person providing services
- client              : name of company/person receiving services
- services_description: detailed description of the services
- start_date          : service start date
- end_date            : service end date or duration
- fee                 : total fee or rate with currency
- payment_terms       : payment schedule (Net 30, Net 15, monthly, etc.)
- governing_law       : jurisdiction governing the agreement
- service_levels      : SLA or performance standards (optional)
- termination_notice  : notice period required to terminate (optional)
- limitation_of_liability : liability cap amount (optional)
- indemnification     : indemnification terms (optional)
- insurance_requirements : required insurance coverage (optional)
- dispute_resolution  : how disputes are resolved (optional)

Rules:
- Use exact field names as JSON keys
- If a field is not mentioned, set its value to "Not Specified"
- Keep values concise and precise""",

    "consulting_agreement": """You are a contracts specialist drafting Consulting Agreements.

Extract the following fields from the user description.
Return ONLY a valid JSON object — no explanation, no markdown.

Fields to extract:
- consultant_name     : name of the consultant or consulting firm
- client_name         : name of the client company
- scope_of_work       : description of consulting services
- start_date          : engagement start date
- end_date            : engagement end date
- consulting_fee      : fee amount (hourly/daily/project) with currency
- payment_terms       : how and when payment is made
- governing_law       : jurisdiction governing the agreement
- expenses_reimbursement : expense policy (optional)
- non_compete_period  : non-compete duration after engagement (optional)
- non_solicitation    : non-solicitation of employees clause (optional)
- intellectual_property : ownership of work product (optional)
- confidentiality_period : how long confidentiality lasts (optional)
- termination_clause  : termination conditions and notice period (optional)

Rules:
- Use exact field names as JSON keys
- If a field is not mentioned, set its value to "Not Specified"
- Keep values concise and precise""",

    "lease_agreement": """You are a real estate attorney drafting Lease Agreements.

Extract the following fields from the user description.
Return ONLY a valid JSON object — no explanation, no markdown.

Fields to extract:
- landlord_name          : full name of the landlord
- tenant_name            : full name of the tenant(s)
- property_address       : full address of the leased property
- lease_start_date       : when the lease begins
- lease_end_date         : when the lease ends
- monthly_rent           : monthly rent amount with currency
- security_deposit       : security deposit amount with currency
- governing_law          : state/country law governing the lease
- late_fee               : late payment fee amount (optional)
- pet_policy             : whether pets are allowed and any deposit (optional)
- utilities_included     : which utilities are included in rent (optional)
- maintenance_responsibilities : tenant vs landlord maintenance duties (optional)
- subletting_policy      : whether subletting is allowed (optional)
- renewal_terms          : how the lease can be renewed (optional)
- early_termination_fee  : fee for breaking the lease early (optional)

Rules:
- Use exact field names as JSON keys
- If a field is not mentioned, set its value to "Not Specified"
- Keep values concise and precise""",

    "employment_contract": """You are an employment law specialist drafting Employment Contracts.

Extract the following fields from the user description.
Return ONLY a valid JSON object — no explanation, no markdown.

Fields to extract:
- employer_name        : name of the employing company
- employee_name        : full name of the employee
- job_title            : employee's job title
- department           : department or team
- start_date           : employment start date
- salary               : annual salary with currency
- work_hours           : hours per week or shift schedule
- governing_law        : jurisdiction governing the contract
- probation_period     : probationary period duration (optional)
- notice_period        : notice required to resign or terminate (optional)
- non_compete          : non-compete restrictions (optional)
- non_solicitation     : non-solicitation clause details (optional)
- benefits             : health insurance, PTO, retirement, etc. (optional)
- leave_policy         : annual leave, sick leave, etc. (optional)
- termination_clause   : grounds and process for termination (optional)
- intellectual_property_assignment : IP ownership clause (optional)

Rules:
- Use exact field names as JSON keys
- If a field is not mentioned, set its value to "Not Specified"
- Keep values concise and precise""",
}


# ---------------------------------------------------------------------------
# Per-document-type generation prompts
# ---------------------------------------------------------------------------

_GENERATION_PROMPTS: dict[str, str] = {

    "nda": """You are a senior attorney specialising in confidentiality law.
Draft a complete, enforceable Non-Disclosure Agreement.

Required sections:
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
   8.5 Waiver
9. SIGNATURE BLOCK""",

    "job_offer": """You are an experienced HR Director drafting a professional job offer letter.
Write in a warm yet professional tone — this is a letter, not a legal contract.

Required sections:
1. DATE AND ADDRESSEE
2. OPENING — congratulate the candidate
3. POSITION DETAILS
   - Job title, department, reporting structure
4. COMPENSATION
   - Salary, bonus structure if any
5. BENEFITS
   - Health, retirement, PTO, etc.
6. START DATE AND LOCATION
7. EMPLOYMENT CONDITIONS
   - Contingencies (background check, references, work authorisation)
   - At-will statement if applicable
8. PROBATION PERIOD (if applicable)
9. OFFER EXPIRY
10. NEXT STEPS — how to accept
11. CLOSING — welcoming tone
12. SIGNATURE BLOCK""",

    "freelancer_agreement": """You are a contracts attorney specialising in independent contractor law.
Draft a complete Freelancer/Independent Contractor Agreement.

Required sections:
1. PARTIES AND RECITALS
2. SCOPE OF WORK
   2.1 Project Description
   2.2 Deliverables
   2.3 Timeline and Milestones
3. COMPENSATION AND PAYMENT
   3.1 Project Fee
   3.2 Payment Schedule
   3.3 Late Payment
   3.4 Expenses
4. REVISIONS AND CHANGE ORDERS
5. INTELLECTUAL PROPERTY
   5.1 Work-for-Hire
   5.2 Assignment of Rights
   5.3 Freelancer Portfolio Rights
6. CONFIDENTIALITY
7. INDEPENDENT CONTRACTOR STATUS
   7.1 No Employment Relationship
   7.2 Taxes and Benefits
8. CANCELLATION AND KILL FEE
9. WARRANTIES AND REPRESENTATIONS
10. LIMITATION OF LIABILITY
11. GENERAL PROVISIONS
12. SIGNATURE BLOCK""",

    "service_agreement": """You are a commercial contracts attorney drafting a Service Agreement.

Required sections:
1. PARTIES
2. DEFINITIONS
3. SERVICES
   3.1 Scope of Services
   3.2 Service Standards and SLAs
   3.3 Change in Scope
4. TERM AND RENEWAL
5. FEES AND PAYMENT
   5.1 Service Fees
   5.2 Payment Terms
   5.3 Late Payment Interest
   5.4 Taxes
6. INTELLECTUAL PROPERTY
7. CONFIDENTIALITY
8. REPRESENTATIONS AND WARRANTIES
9. LIMITATION OF LIABILITY
10. INDEMNIFICATION
11. INSURANCE
12. TERMINATION
    12.1 Termination for Convenience
    12.2 Termination for Cause
    12.3 Effect of Termination
13. DISPUTE RESOLUTION
14. GENERAL PROVISIONS
15. SIGNATURE BLOCK""",

    "consulting_agreement": """You are a commercial attorney specialising in professional services contracts.
Draft a complete Consulting Agreement.

Required sections:
1. PARTIES
2. ENGAGEMENT AND SCOPE
   2.1 Scope of Consulting Services
   2.2 Deliverables
   2.3 Consultant's Personnel
3. TERM
4. COMPENSATION AND EXPENSES
   4.1 Consulting Fees
   4.2 Expense Reimbursement
   4.3 Invoicing and Payment
5. INTELLECTUAL PROPERTY
   5.1 Work Product Ownership
   5.2 Background IP
   5.3 Licence Grant
6. CONFIDENTIALITY
7. NON-COMPETE AND NON-SOLICITATION
   7.1 Non-Competition
   7.2 Non-Solicitation of Employees
   7.3 Non-Solicitation of Clients
8. INDEPENDENT CONTRACTOR
9. REPRESENTATIONS AND WARRANTIES
10. LIMITATION OF LIABILITY
11. TERMINATION
12. GENERAL PROVISIONS
13. SIGNATURE BLOCK""",

    "lease_agreement": """You are a real estate attorney drafting a residential/commercial Lease Agreement.

Required sections:
1. PARTIES AND PREMISES
2. LEASE TERM
3. RENT
   3.1 Monthly Rent
   3.2 Due Date and Payment Method
   3.3 Late Fee
   3.4 Returned Payment Fee
4. SECURITY DEPOSIT
   4.1 Amount
   4.2 Use and Return
5. UTILITIES AND SERVICES
6. USE OF PREMISES
7. MAINTENANCE AND REPAIRS
   7.1 Landlord Obligations
   7.2 Tenant Obligations
8. ALTERATIONS AND IMPROVEMENTS
9. ENTRY BY LANDLORD
10. ASSIGNMENT AND SUBLETTING
11. PETS
12. NOISE AND NUISANCE
13. DEFAULT AND REMEDIES
    13.1 Tenant Default
    13.2 Landlord Remedies
14. EARLY TERMINATION
15. RENEWAL
16. MOVE-OUT PROCEDURES
17. GENERAL PROVISIONS
18. SIGNATURE BLOCK""",

    "employment_contract": """You are an employment law attorney drafting an Employment Contract.

Required sections:
1. PARTIES
2. POSITION AND DUTIES
   2.1 Job Title and Department
   2.2 Duties and Responsibilities
   2.3 Reporting Structure
   2.4 Place of Work
3. COMMENCEMENT AND TERM
4. COMPENSATION
   4.1 Base Salary
   4.2 Performance Review
   4.3 Overtime
5. BENEFITS
   5.1 Health and Dental Insurance
   5.2 Retirement Plan
   5.3 Annual Leave
   5.4 Sick Leave
   5.5 Other Benefits
6. PROBATIONARY PERIOD
7. WORKING HOURS
8. CONFIDENTIALITY AND NON-DISCLOSURE
9. INTELLECTUAL PROPERTY ASSIGNMENT
10. NON-COMPETE AND NON-SOLICITATION
    10.1 Non-Competition
    10.2 Non-Solicitation
11. TERMINATION
    11.1 Termination by Employer
    11.2 Resignation by Employee
    11.3 Notice Period
    11.4 Summary Dismissal
12. POST-TERMINATION OBLIGATIONS
13. DATA PROTECTION
14. GOVERNING LAW AND DISPUTE RESOLUTION
15. GENERAL PROVISIONS
16. SIGNATURE BLOCK""",
}


# ---------------------------------------------------------------------------
# Intent classification — detects document type from user_query
# ---------------------------------------------------------------------------

_INTENT_SYSTEM = """You are a legal document classifier.

Read the user's description and classify it into EXACTLY ONE of these document types:

nda                  - Non-Disclosure Agreement, confidentiality agreement
job_offer            - Job offer letter, employment offer
freelancer_agreement - Freelancer contract, independent contractor agreement
service_agreement    - Service agreement, vendor agreement, SLA
consulting_agreement - Consulting agreement, advisory agreement
lease_agreement      - Lease agreement, rental agreement, tenancy agreement
employment_contract  - Employment contract, employment agreement

Rules:
- Return ONLY the slug (e.g. "nda") — no explanation, no extra text
- If multiple types could apply, pick the most specific one
- If none match, return "unknown"

Examples:
"I need an NDA with a contractor" → nda
"Hire John as full-time software engineer" → employment_contract
"Hire John as a freelance designer for a logo project" → freelancer_agreement
"Rent my apartment to a tenant" → lease_agreement
"Engage a consultant for 3 months of strategy work" → consulting_agreement"""


async def classify_intent(user_query: str) -> str | None:
    """
    Classify user_query into a document type slug using the LLM.
    Returns the slug or None if unrecognised.
    """
    client = _get_client()
    kwargs = _get_model_kwargs(max_tokens=20)
    kwargs["messages"] = [
        {"role": "system", "content": _INTENT_SYSTEM},
        {"role": "user",   "content": f"Classify this:\n\"\"\"{user_query[:1000]}\"\"\""},
    ]
    # Plain text — just a single word response
    if "response_format" in kwargs:
        del kwargs["response_format"]

    response = await client.chat.completions.create(**kwargs)
    raw      = (response.choices[0].message.content or "").strip().lower()

    # Clean up in case model adds punctuation
    raw = re.sub(r'[^a-z_]', '', raw)

    if raw in SUPPORTED_DOCUMENT_TYPES:
        logger.info(f"[doc_gen] intent classified as: '{raw}'")
        return raw

    # Try alias lookup
    resolved = resolve_document_type(raw)
    if resolved:
        logger.info(f"[doc_gen] intent resolved via alias: '{raw}' → '{resolved}'")
        return resolved

    logger.warning(f"[doc_gen] intent classification returned unknown: '{raw}'")
    return None


# ---------------------------------------------------------------------------
# OpenAI client helpers
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
# Own JSON extractor — independent of s_json_utils
# ---------------------------------------------------------------------------

def _extract_fields_from_json(raw: str) -> dict:
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
    logger.warning("[doc_gen] could not parse field JSON — returning empty dict")
    return {}


# ---------------------------------------------------------------------------
# Step 1: Extract structured fields using per-type prompt
# ---------------------------------------------------------------------------

async def _extract_fields(document_type: str, user_query: str) -> dict[str, Any]:
    schema         = _SCHEMAS[document_type]
    system_prompt  = _EXTRACTION_PROMPTS[document_type]

    client = _get_client()
    kwargs = _get_model_kwargs(max_tokens=2048)
    kwargs["messages"] = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": f"Description:\n\"\"\"{user_query}\"\"\"\n\nExtract all fields as a JSON object."},
    ]
    kwargs["response_format"] = {"type": "json_object"}

    response = await client.chat.completions.create(**kwargs)
    raw      = response.choices[0].message.content or "{}"
    logger.info(
        f"[doc_gen] field extraction "
        f"tokens={response.usage.prompt_tokens}in/{response.usage.completion_tokens}out"
    )

    fields = _extract_fields_from_json(raw)

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
# Step 3: Generate document text using per-type generation prompt
# ---------------------------------------------------------------------------

async def _generate_document_text(
    document_type: str,
    fields: dict[str, Any],
    user_query: str,
) -> tuple[str, int, int]:
    doc_name      = SUPPORTED_DOCUMENT_TYPES[document_type]
    gen_prompt    = _GENERATION_PROMPTS[document_type]

    field_lines = "\n".join(
        f"  {k.replace('_', ' ').title()}: {v}"
        for k, v in fields.items()
        if v and v != "Not Specified"
    )

    system = f"""{gen_prompt}

Formatting rules:
- Section headings in ALL CAPS followed by a colon (e.g. "1. PARTIES:")
- Numbered sub-clauses (1.1, 1.2, etc.)
- Formal legal language throughout
- Use the actual names, dates, and amounts provided below
- If a value is "Not Specified" use [TO BE AGREED] as placeholder
- Plain text only — no markdown, no asterisks, no bullet symbols
- End with a complete SIGNATURE BLOCK with lines for each party"""

    user = f"""Use these details to generate the complete {doc_name}:

{field_lines}

Additional context from user:
\"\"\"{user_query}\"\"\"

Generate the full document now."""

    client = _get_client()
    kwargs = _get_model_kwargs(max_tokens=4096)
    kwargs["messages"] = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]
    # Plain text — do NOT use json_object format for document generation

    response      = await client.chat.completions.create(**kwargs)
    document_text = response.choices[0].message.content or ""
    in_tok        = response.usage.prompt_tokens
    out_tok       = response.usage.completion_tokens

    logger.info(
        f"[doc_gen] document generated — type={document_type} "
        f"length={len(document_text)} chars "
        f"tokens={in_tok}in/{out_tok}out"
    )
    return document_text, in_tok, out_tok


# ---------------------------------------------------------------------------
# Step 4: Render DOCX
# ---------------------------------------------------------------------------

def _render_docx(document_text: str, document_name: str) -> bytes:
    """
    Convert plain-text legal document to a professionally formatted DOCX.
    Uses python-docx with a clean business template.
    """
    from docx import Document
    from docx.shared import Pt, RGBColor, Inches, Cm
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement
    from datetime import datetime

    doc = Document()

    # ── Page margins ─────────────────────────────────────────────────────
    for section in doc.sections:
        section.top_margin    = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin   = Inches(1.25)
        section.right_margin  = Inches(1.25)

    # ── Colours ──────────────────────────────────────────────────────────
    NAVY   = RGBColor(0x1a, 0x27, 0x44)
    BLUE   = RGBColor(0x2c, 0x4a, 0x8c)
    DARK   = RGBColor(0x1a, 0x1a, 0x1a)
    MUTED  = RGBColor(0x55, 0x55, 0x55)

    def _set_cell_bg(cell, hex_color: str):
        """Set table cell background colour."""
        tc   = cell._tc
        tcPr = tc.get_or_add_tcPr()
        shd  = OxmlElement("w:shd")
        shd.set(qn("w:val"),   "clear")
        shd.set(qn("w:color"), "auto")
        shd.set(qn("w:fill"),  hex_color)
        tcPr.append(shd)

    # ── Header banner ─────────────────────────────────────────────────────
    table = doc.add_table(rows=1, cols=1)
    table.style = "Table Grid"
    cell  = table.cell(0, 0)
    _set_cell_bg(cell, "1a2744")

    para = cell.paragraphs[0]
    para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run  = para.add_run(document_name.upper())
    run.bold      = True
    run.font.size = Pt(20)
    run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

    # Remove table borders
    tbl  = table._tbl
    tblPr = tbl.find(qn("w:tblPr"))
    if tblPr is None:
        tblPr = OxmlElement("w:tblPr")
        tbl.insert(0, tblPr)
    tblBorders = OxmlElement("w:tblBorders")
    for side in ("top", "left", "bottom", "right", "insideH", "insideV"):
        border = OxmlElement(f"w:{side}")
        border.set(qn("w:val"), "none")
        tblBorders.append(border)
    tblPr.append(tblBorders)

    doc.add_paragraph()  # spacer

    # Date line
    date_para = doc.add_paragraph()
    date_run  = date_para.add_run(f"Generated: {datetime.now().strftime('%B %d, %Y')}")
    date_run.font.size  = Pt(9)
    date_run.font.color.rgb = RGBColor(0x88, 0x88, 0x88)
    date_para.paragraph_format.space_after = Pt(12)

    # ── Parse and render document text ───────────────────────────────────
    lines = document_text.split("\n")
    in_signature = False

    for line in lines:
        stripped = line.strip()

        if not stripped:
            # Blank line → small spacer paragraph
            sp = doc.add_paragraph()
            sp.paragraph_format.space_before = Pt(0)
            sp.paragraph_format.space_after  = Pt(3)
            continue

        # Detect SIGNATURES section
        if re.match(r'^\d+\.\s+SIGNATURE', stripped.upper()):
            in_signature = True

        # Section heading detection
        is_section = (
            re.match(r'^\d+\.\s+[A-Z][A-Z\s,/&()-]{3,}[:.]?\s*$', stripped)
            or re.match(r'^[A-Z][A-Z\s,/&()-]{3,}[:.]?\s*$', stripped)
            or (stripped.isupper() and 4 < len(stripped) < 80)
        )

        if is_section and not in_signature:
            # Section heading — styled
            p    = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(14)
            p.paragraph_format.space_after  = Pt(4)
            # Horizontal rule before heading
            pPr   = p._p.get_or_add_pPr()
            pBdr  = OxmlElement("w:pBdr")
            top   = OxmlElement("w:top")
            top.set(qn("w:val"),   "single")
            top.set(qn("w:sz"),    "4")
            top.set(qn("w:space"), "1")
            top.set(qn("w:color"), "CCCCCC")
            pBdr.append(top)
            pPr.append(pBdr)
            run = p.add_run(stripped)
            run.bold      = True
            run.font.size = Pt(11)
            run.font.color.rgb = BLUE
            continue

        # Sub-clause: "1.1 text" or "1.1. text"
        if re.match(r'^\d+\.\d+\.?\s+', stripped) and not in_signature:
            p = doc.add_paragraph(style="Normal")
            p.paragraph_format.left_indent  = Inches(0.3)
            p.paragraph_format.space_after  = Pt(4)
            p.paragraph_format.first_line_indent = Pt(0)
            run = p.add_run(stripped)
            run.font.size = Pt(10)
            run.font.color.rgb = DARK
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            continue

        # Signature block
        if in_signature:
            if re.match(r'^(Name|Title|Date|Company|Signature)\s*:', stripped, re.I):
                label, _, rest = stripped.partition(":")
                lp  = doc.add_paragraph()
                lr  = lp.add_run(label.strip() + ":")
                lr.bold = True
                lr.font.size = Pt(9)
                lr.font.color.rgb = MUTED
                lp.paragraph_format.space_after = Pt(2)

                vp = doc.add_paragraph()
                vr = vp.add_run(rest.strip() if rest.strip() else "________________________")
                vr.font.size = Pt(10)
                vr.font.color.rgb = DARK
                vp.paragraph_format.space_after = Pt(12)
                continue

        # Regular paragraph
        p   = doc.add_paragraph(style="Normal")
        run = p.add_run(stripped)
        run.font.size = Pt(10)
        run.font.color.rgb = DARK
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.space_after = Pt(4)

    # ── Footer with page numbers ──────────────────────────────────────────
    for section in doc.sections:
        footer      = section.footer
        footer_para = footer.paragraphs[0]
        footer_para.alignment = WD_ALIGN_PARAGRAPH.CENTER

        run = footer_para.add_run(f"{document_name}  |  Confidential  |  Page ")
        run.font.size = Pt(8)
        run.font.color.rgb = RGBColor(0x88, 0x88, 0x88)

        # Page number field
        fldChar1 = OxmlElement("w:fldChar")
        fldChar1.set(qn("w:fldCharType"), "begin")
        instrText = OxmlElement("w:instrText")
        instrText.text = "PAGE"
        fldChar2 = OxmlElement("w:fldChar")
        fldChar2.set(qn("w:fldCharType"), "end")

        run2 = footer_para.add_run()
        run2.font.size = Pt(8)
        run2.font.color.rgb = RGBColor(0x88, 0x88, 0x88)
        run2._r.append(fldChar1)
        run2._r.append(instrText)
        run2._r.append(fldChar2)

    # ── Save to bytes ─────────────────────────────────────────────────────
    buffer = BytesIO()
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

    Args:
        document_type : canonical slug OR None (auto-detected from user_query)
        user_query    : free-text description

    Returns dict:
        status          : "success" | "missing_fields" | "unknown_type"
        document_type   : detected/provided slug
        document_name   : display name
        fields          : extracted structured fields
        missing_fields  : required fields not found
        document        : full document as plain text
        docx_bytes      : DOCX file bytes
        word_count      : word count
        input_tokens    : total LLM input tokens used
        output_tokens   : total LLM output tokens used
    """
    total_in_tok  = 0
    total_out_tok = 0

    # ── Step 0: Resolve document type ─────────────────────────────────────
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
                    f"'{document_type}' is not a supported document type. "
                    f"Supported: {', '.join(SUPPORTED_DOCUMENT_TYPES.keys())}"
                ),
            }
    else:
        # Auto-detect from user_query
        logger.info("[doc_gen] document_type not provided — classifying intent...")
        doc_type = await classify_intent(user_query)
        total_in_tok  += 50   # rough estimate for classification call
        total_out_tok += 5

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
                    "Could not determine document type from your description. "
                    "Please specify document_type. "
                    f"Supported: {', '.join(SUPPORTED_DOCUMENT_TYPES.keys())}"
                ),
            }

    doc_name = SUPPORTED_DOCUMENT_TYPES[doc_type]
    logger.info(f"[doc_gen] generating '{doc_name}' from query ({len(user_query)} chars)")

    # ── Step 1: Extract fields ─────────────────────────────────────────────
    fields = await _extract_fields(doc_type, user_query)
    logger.info(f"[doc_gen] extracted {len(fields)} fields")

    # ── Step 2: Check missing ──────────────────────────────────────────────
    missing = _get_missing_fields(doc_type, fields)
    if missing:
        logger.warning(f"[doc_gen] missing required fields: {missing}")

    # ── Step 3: Generate document text ────────────────────────────────────
    document_text, in_tok, out_tok = await _generate_document_text(
        doc_type, fields, user_query
    )
    total_in_tok  += in_tok
    total_out_tok += out_tok

    # ── Step 4: Render DOCX ───────────────────────────────────────────────
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