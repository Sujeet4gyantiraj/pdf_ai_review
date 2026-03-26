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
  1. Client sends document_type + user_query (free-text description of what they need)
  2. Router validates and normalises document_type
  3. Schema for that document type is loaded (required + optional fields)
  4. LLM extracts structured fields from user_query
  5. LLM generates the full legal document from those fields
  6. Response returns both the extracted fields and the full document text
"""

import logging
from typing import Any
from s_ai_model import run_llm
from s_json_utils import extract_json

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Supported document types — canonical slug → display name
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

# Aliases — common alternate names the user might send
_TYPE_ALIASES: dict[str, str] = {
    "non disclosure agreement": "nda",
    "non-disclosure agreement": "nda",
    "non_disclosure_agreement": "nda",
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
    """
    Normalise user-supplied document type string to a canonical slug.
    Returns None if unrecognised.
    """
    cleaned = raw.lower().strip().replace("-", "_").replace(" ", "_")

    # Direct match
    if cleaned in SUPPORTED_DOCUMENT_TYPES:
        return cleaned

    # Alias match (un-slugified)
    readable = raw.lower().strip()
    if readable in _TYPE_ALIASES:
        return _TYPE_ALIASES[readable]

    # Partial match — e.g. user sends "nda agreement"
    for alias, slug in _TYPE_ALIASES.items():
        if alias in readable or readable in alias:
            return slug

    for slug in SUPPORTED_DOCUMENT_TYPES:
        if slug in cleaned or cleaned in slug:
            return slug

    return None


# ---------------------------------------------------------------------------
# Per-document field schemas
# Each schema defines:
#   required  — fields the LLM MUST extract (shown to user if missing)
#   optional  — fields that enrich the document if present
# ---------------------------------------------------------------------------

_SCHEMAS: dict[str, dict[str, list[str]]] = {

    "nda": {
        "required": [
            "disclosing_party",       # company/person sharing info
            "receiving_party",        # company/person receiving info
            "effective_date",
            "purpose",                # why info is being shared
            "duration_years",         # how long NDA lasts
            "governing_law",          # jurisdiction
        ],
        "optional": [
            "confidential_info_description",
            "exclusions",             # what is NOT confidential
            "return_of_materials",    # must materials be returned?
            "remedies",               # injunctive relief etc.
            "signatory_names",
        ],
    },

    "job_offer": {
        "required": [
            "candidate_name",
            "company_name",
            "job_title",
            "start_date",
            "salary",
            "employment_type",        # full-time / part-time
            "reporting_to",
        ],
        "optional": [
            "work_location",
            "probation_period",
            "benefits",               # health, PTO, etc.
            "equity",
            "signing_bonus",
            "offer_expiry_date",
            "hr_contact",
        ],
    },

    "freelancer_agreement": {
        "required": [
            "client_name",
            "freelancer_name",
            "project_description",
            "start_date",
            "end_date",
            "payment_amount",
            "payment_schedule",       # milestone / weekly / on completion
            "governing_law",
        ],
        "optional": [
            "deliverables",
            "revision_rounds",
            "intellectual_property",  # who owns the work
            "confidentiality",
            "kill_fee",               # cancellation fee
            "late_payment_penalty",
        ],
    },

    "service_agreement": {
        "required": [
            "service_provider",
            "client",
            "services_description",
            "start_date",
            "end_date",
            "fee",
            "payment_terms",          # Net 30 / Net 15 etc.
            "governing_law",
        ],
        "optional": [
            "service_levels",         # SLA details
            "termination_notice",
            "limitation_of_liability",
            "indemnification",
            "insurance_requirements",
            "dispute_resolution",
        ],
    },

    "consulting_agreement": {
        "required": [
            "consultant_name",
            "client_name",
            "scope_of_work",
            "start_date",
            "end_date",
            "consulting_fee",
            "payment_terms",
            "governing_law",
        ],
        "optional": [
            "expenses_reimbursement",
            "non_compete_period",
            "non_solicitation",
            "intellectual_property",
            "confidentiality_period",
            "termination_clause",
        ],
    },

    "lease_agreement": {
        "required": [
            "landlord_name",
            "tenant_name",
            "property_address",
            "lease_start_date",
            "lease_end_date",
            "monthly_rent",
            "security_deposit",
            "governing_law",
        ],
        "optional": [
            "late_fee",
            "pet_policy",
            "utilities_included",
            "maintenance_responsibilities",
            "subletting_policy",
            "renewal_terms",
            "early_termination_fee",
        ],
    },

    "employment_contract": {
        "required": [
            "employer_name",
            "employee_name",
            "job_title",
            "department",
            "start_date",
            "salary",
            "work_hours",
            "governing_law",
        ],
        "optional": [
            "probation_period",
            "notice_period",
            "non_compete",
            "non_solicitation",
            "benefits",
            "leave_policy",
            "termination_clause",
            "intellectual_property_assignment",
        ],
    },
}


# ---------------------------------------------------------------------------
# Step 1: Extract structured fields from user_query
# ---------------------------------------------------------------------------

async def _extract_fields(
    document_type: str,
    user_query: str,
) -> dict[str, Any]:
    """
    Use LLM to extract structured fields from the user's free-text query.
    Returns a dict of field_name → value (or "Not Specified").
    """
    schema   = _SCHEMAS[document_type]
    doc_name = SUPPORTED_DOCUMENT_TYPES[document_type]

    all_fields = schema["required"] + schema["optional"]
    field_list = "\n".join(f"- {f}" for f in all_fields)

    required_list = "\n".join(f"- {f}" for f in schema["required"])

    system_prompt = f"""You are a legal document assistant specialising in {doc_name} documents.

Extract structured information from the user's description for a {doc_name}.

REQUIRED fields (must be present):
{required_list}

ALL fields to extract:
{field_list}

Rules:
- Return ONLY a valid JSON object
- Use the exact field names listed above as JSON keys
- If a field is mentioned in the description, extract its value
- If a field is NOT mentioned, set its value to "Not Specified"
- Do NOT add fields not in the list
- Keep values concise and precise
- For dates use format: "Month DD, YYYY" or "Not Specified"
- For monetary values include currency (e.g. "$5,000 USD")
"""

    user_prompt = f"""User description:
\"\"\"{user_query}\"\"\"

Extract all fields for a {doc_name} from the above description.
Return ONLY a JSON object with the field names as keys."""

    messages_text = system_prompt + "\n\n" + user_prompt
    raw = await run_llm(user_query, messages_text)
    logger.info(f"[doc_gen] field extraction raw length: {len(raw)}")

    fields = extract_json(raw)

    # Ensure all required fields exist
    for field in schema["required"]:
        if field not in fields:
            fields[field] = "Not Specified"

    return fields


# ---------------------------------------------------------------------------
# Step 2: Validate required fields — return list of missing ones
# ---------------------------------------------------------------------------

def _get_missing_fields(document_type: str, fields: dict) -> list[str]:
    schema   = _SCHEMAS[document_type]
    missing  = []
    for field in schema["required"]:
        val = fields.get(field, "Not Specified")
        if not val or val == "Not Specified":
            missing.append(field)
    return missing


# ---------------------------------------------------------------------------
# Step 3: Generate the full legal document
# ---------------------------------------------------------------------------

_DOCUMENT_INSTRUCTIONS: dict[str, str] = {

    "nda": """Generate a professional Non-Disclosure Agreement with these sections:
1. Parties
2. Definition of Confidential Information
3. Obligations of Receiving Party
4. Exclusions from Confidential Information
5. Term and Termination
6. Return of Materials
7. Remedies
8. General Provisions (governing law, entire agreement, amendments)
9. Signatures""",

    "job_offer": """Generate a professional Job Offer Letter with these sections:
1. Opening / Congratulations
2. Position Details (title, department, reporting structure)
3. Compensation and Benefits
4. Start Date and Work Location
5. Employment Type and Hours
6. Conditions of Employment (background check, at-will etc.)
7. Offer Expiry
8. Acceptance Instructions
9. Closing and Signatures""",

    "freelancer_agreement": """Generate a professional Freelancer Agreement with these sections:
1. Parties
2. Scope of Work and Deliverables
3. Timeline
4. Compensation and Payment Schedule
5. Intellectual Property Ownership
6. Confidentiality
7. Revisions and Change Requests
8. Cancellation and Kill Fee
9. Independent Contractor Status
10. General Provisions
11. Signatures""",

    "service_agreement": """Generate a professional Service Agreement with these sections:
1. Parties
2. Services Description
3. Term
4. Fees and Payment Terms
5. Service Levels and Performance
6. Confidentiality
7. Intellectual Property
8. Limitation of Liability
9. Indemnification
10. Termination
11. Dispute Resolution
12. General Provisions
13. Signatures""",

    "consulting_agreement": """Generate a professional Consulting Agreement with these sections:
1. Parties
2. Scope of Consulting Services
3. Term
4. Consulting Fees and Expenses
5. Intellectual Property
6. Confidentiality
7. Non-Compete and Non-Solicitation
8. Independent Contractor Status
9. Termination
10. General Provisions
11. Signatures""",

    "lease_agreement": """Generate a professional Lease Agreement with these sections:
1. Parties and Property
2. Lease Term
3. Rent and Payment Terms
4. Security Deposit
5. Utilities and Services
6. Maintenance and Repairs
7. Use of Property
8. Pets and Smoking Policy
9. Subletting
10. Early Termination
11. Renewal Terms
12. Default and Remedies
13. General Provisions
14. Signatures""",

    "employment_contract": """Generate a professional Employment Contract with these sections:
1. Parties
2. Position and Duties
3. Term of Employment
4. Compensation and Benefits
5. Working Hours and Location
6. Probation Period
7. Leave Entitlements
8. Confidentiality and Non-Disclosure
9. Non-Compete and Non-Solicitation
10. Intellectual Property Assignment
11. Termination and Notice Period
12. Governing Law and Dispute Resolution
13. General Provisions
14. Signatures""",
}


async def _generate_document(
    document_type: str,
    fields: dict[str, Any],
    user_query: str,
) -> str:
    """
    Generate the full legal document text using extracted fields.
    Returns the complete document as a formatted string.
    """
    doc_name     = SUPPORTED_DOCUMENT_TYPES[document_type]
    instructions = _DOCUMENT_INSTRUCTIONS[document_type]

    # Format fields for the prompt
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
\"\"\"{user_query}\"\"\"

Generate the complete document now. Use formal legal language throughout."""

    full_prompt = system_prompt + "\n\n" + user_prompt

    # For document generation we don't need JSON — use run_llm directly
    # but bypass the json_object response format by calling the model directly
    from openai import AsyncOpenAI
    import os
    from s_ai_model import (
        MODEL_NAME,
        _FIXED_TEMPERATURE_MODELS,
        _MAX_COMPLETION_TOKENS_MODELS,
    )

    client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))

    api_kwargs = {
        "model":    MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        "max_completion_tokens" if MODEL_NAME in _MAX_COMPLETION_TOKENS_MODELS else "max_tokens": 4096,
    }

    if MODEL_NAME not in _FIXED_TEMPERATURE_MODELS:
        api_kwargs["temperature"] = 0.2   # slightly creative for natural legal language

    response = await client.chat.completions.create(**api_kwargs)
    document_text = response.choices[0].message.content or ""

    logger.info(
        f"[doc_gen] document generated — "
        f"type={document_type} "
        f"length={len(document_text)} chars "
        f"tokens={response.usage.prompt_tokens}in/{response.usage.completion_tokens}out"
    )

    return document_text


# ---------------------------------------------------------------------------
# Public API — main entry point called by the router
# ---------------------------------------------------------------------------

async def generate_document(
    document_type: str,
    user_query: str,
) -> dict:
    """
    Main entry point for document generation.

    Args:
        document_type : canonical slug (e.g. "nda", "job_offer")
        user_query    : free-text description of what the user needs

    Returns dict with:
        status          : "success" | "missing_fields" | "error"
        document_type   : canonical slug
        document_name   : display name
        fields          : extracted structured fields
        missing_fields  : list of required fields not found in query (if any)
        document        : full generated document text
        word_count      : approximate word count of generated document
    """
    doc_name = SUPPORTED_DOCUMENT_TYPES[document_type]
    logger.info(f"[doc_gen] generating '{doc_name}' from query ({len(user_query)} chars)")

    # Step 1: Extract fields
    fields = await _extract_fields(document_type, user_query)
    logger.info(f"[doc_gen] extracted {len(fields)} fields")

    # Step 2: Check for missing required fields
    missing = _get_missing_fields(document_type, fields)
    if missing:
        logger.warning(f"[doc_gen] missing required fields: {missing}")

    # Step 3: Generate document (even with missing fields — use defaults)
    document_text = await _generate_document(document_type, fields, user_query)

    word_count = len(document_text.split())

    return {
        "status":        "missing_fields" if missing else "success",
        "document_type": document_type,
        "document_name": doc_name,
        "fields":        fields,
        "missing_fields": missing,
        "document":      document_text,
        "word_count":    word_count,
    }