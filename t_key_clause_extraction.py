from fastapi import HTTPException, UploadFile
import time
import uuid
import os
import logging

from s_ai_model import run_llm
from s_padf_utils import load_pdf, get_page_count, all_pages_blank
from pdf_ai_review.backupfiles.t_utils import extract_json_from_text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

UPLOAD_FOLDER = "temp"                      # ← was missing, caused NameError
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ==============================
# CLASSIFICATION
# ==============================
async def classify_document(text: str):
    prompt = f"""
Classify the document into EXACTLY ONE of the following categories:

contract
resume
invoice
report
other

Definitions:
- contract → legal agreements (NDA, Service Agreement, Lease, Terms & Conditions)
- resume → CV, job profile, candidate details
- invoice → billing documents, receipts, payment summaries
- report → business reports, analysis documents, research papers
- other → anything else (stories, books, random text)

Rules:
- Return ONLY one word from the list above
- Do NOT add explanation
- Do NOT return multiple categories
- If unsure, return "other"

Examples:
"Non Disclosure Agreement between two parties..." → contract
"John Doe, Software Engineer, Skills: Python..." → resume
"Invoice No: 12345, Total: $500..." → invoice
"Quarterly sales analysis shows growth..." → report
"Once upon a time..." → other

Text:
\"\"\"{text[:1500]}\"\"\"
"""

    result = await run_llm(text[:1500], prompt)

    # Safety cleanup
    result = result.lower().strip()

    valid_labels = {"contract", "resume", "invoice", "report", "other"}

    # Handle messy LLM outputs like "contract\n" or "contract document"
    for label in valid_labels:
        if label in result:
            return label

    return "other"

# ==============================
# CONTRACT HANDLER
# ==============================
async def handle_contract(text: str):
    prompt = f"""
Extract key contract clauses from the given text and return in STRICT JSON format.

Fields:
- Agreement Type (Service Agreement, NDA, Lease, etc.)
- Effective Date
- Contract Term
- Renewal Conditions
- Governing Law
- Payment Terms
- Termination Notice Period

Rules:
- Return ONLY valid JSON (no explanation, no extra text)
- Use EXACT field names as listed above
- Do NOT rename keys
- Do NOT add extra fields
- If any field is missing, return "Not Found"
- Keep values short and precise

Example Output:
{{
  "Agreement Type": "Service Agreement",
  "Effective Date": "Jan 1, 2026",
  "Contract Term": "3 years",
  "Renewal Conditions": "Auto-renew yearly",
  "Governing Law": "Alberta",
  "Payment Terms": "Net 30",
  "Termination Notice Period": "60 days"
}}

Text:
\"\"\"{text}\"\"\"
"""

    result = await run_llm(text, prompt)
    print("Raw LLM Output:", result)  # Debugging: see the unprocessed output
    parsed_data = extract_json_from_text(result)
    return {
        "status": "success",
        "document_type": "contract",
        "data": parsed_data
    }


# ==============================
# RESUME HANDLER
# ==============================
async def handle_resume(text: str):
    prompt = f"""
Extract structured resume information from the given text and return in STRICT JSON format.

Fields:
- Name
- Skills (list)
- Experience (list of roles or summary)
- Education (list)
- Missing Sections (list of missing sections like Skills, Experience, Education, Summary, Projects)

Rules:
- Return ONLY valid JSON (no explanation, no extra text)
- Use EXACT field names as listed above
- Do NOT rename keys
- Do NOT add extra fields
- If a field is not found, return "Not Found"
- Lists should be arrays ([])
- Keep values concise

Example Output:
{{
  "Name": "John Doe",
  "Skills": ["Python", "Machine Learning", "FastAPI"],
  "Experience": ["Software Engineer at ABC Corp (2022-2025)"],
  "Education": ["B.Tech in Computer Science"],
  "Missing Sections": ["Projects", "Summary"]
}}

Text:
\"\"\"{text}\"\"\"
"""

    result = await run_llm(text, prompt)
    print("Raw LLM Output:", result)  # Debugging: see the unprocessed output
    parsed_data = extract_json_from_text(result)
    return {
        "status": "success",
        "document_type": "resume",
        "data": parsed_data
    }


# ==============================
# INVOICE HANDLER
# ==============================
async def handle_invoice(text: str):
    prompt = f"""
Extract structured invoice information from the given text and return in STRICT JSON format.

Fields:
- Invoice Number
- Date
- Vendor Name
- Total Amount
- Tax
- Line Items (list of objects with: Description, Quantity, Unit Price, Amount)

Rules:
- Return ONLY valid JSON (no explanation, no extra text)
- Use EXACT field names as listed above
- Do NOT rename keys
- Do NOT add extra fields
- If any field is missing, return "Not Found"
- Line Items must be an array of objects
- If line items are not clearly available, return []

Example Output:
{{
  "Invoice Number": "INV-1023",
  "Date": "2026-01-15",
  "Vendor Name": "ABC Pvt Ltd",
  "Total Amount": "1500 USD",
  "Tax": "150 USD",
  "Line Items": [
    {{
      "Description": "Software License",
      "Quantity": "1",
      "Unit Price": "1000 USD",
      "Amount": "1000 USD"
    }},
    {{
      "Description": "Support Fee",
      "Quantity": "1",
      "Unit Price": "500 USD",
      "Amount": "500 USD"
    }}
  ]
}}

Text:
\"\"\"{text}\"\"\"
"""

    result = await run_llm(text, prompt)
    print("Raw LLM Output:", result)  # Debugging: see the unprocessed output
    parsed_data = extract_json_from_text(result)
    return {
        "status": "success",
        "document_type": "invoice",
        "data": parsed_data
    }

# ==============================
# ROUTER MAP
# ==============================
DOCUMENT_HANDLERS = {
    "contract": handle_contract,
    "resume": handle_resume,
    "invoice": handle_invoice
}

async def extract_text_from_upload(
    file: UploadFile,
    *,
    endpoint: str = "",
    max_pages: int | None = None,
) -> tuple[str, int, int, str, float, str]:
    logger.info(f"Received file: {file.filename} for endpoint: {endpoint}")
    request_id = str(uuid.uuid4())[:8]
    t_start = time.perf_counter()

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    safe_name = f"{uuid.uuid4()}.pdf"
    file_path = os.path.join(UPLOAD_FOLDER, safe_name)

    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    total_pages = get_page_count(file_path)
    pages_to_read = total_pages if max_pages is None else min(total_pages, max_pages)

    try:
        pages = load_pdf(file_path, max_pages=pages_to_read)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    if all_pages_blank(pages):
        raise HTTPException(status_code=422, detail="No extractable text found in PDF.")

    text = "\n\n".join(p.page_content for p in pages)

    return text, pages_to_read, total_pages, request_id, t_start, file_path