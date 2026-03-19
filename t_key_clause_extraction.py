from t_ai_model import run_llm

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

    return {
        "status": "success",
        "document_type": "contract",
        "data": result
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

    return {
        "status": "success",
        "document_type": "resume",
        "data": result
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

    return {
        "status": "success",
        "document_type": "invoice",
        "data": result
    }

# ==============================
# ROUTER MAP
# ==============================
DOCUMENT_HANDLERS = {
    "contract": handle_contract,
    "resume": handle_resume,
    "invoice": handle_invoice
}

