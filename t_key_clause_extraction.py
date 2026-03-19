from t_ai_model import run_llm

# ==============================
# PARSE RESPONSE
# ==============================
def parse_extraction_response(api_response):
    if not api_response.get("status"):
        return None

    pages = api_response.get("data", [])

    full_text = "\n".join(
        page.get("content", "") for page in pages
    )

    return full_text.strip()

# ==============================
# CLASSIFICATION
# ==============================
async def classify_document(text: str):
    prompt = """
Classify the document into ONE of the following categories:
contract, resume, invoice, report, other

Rules:
- If legal agreement → contract
- If CV/job profile → resume
- If billing/payment doc → invoice
- If business/analysis → report
- If story/book/random → other

Return ONLY one word.

"""

    result = await run_llm(text[:1500], prompt)
    return result.lower().strip()

# ==============================
# CONTRACT HANDLER
# ==============================
async def handle_contract(text: str):
    prompt = """
Extract contract fields in JSON format:

Fields:
- Agreement Type
- Effective Date
- Contract Term
- Renewal Conditions
- Governing Law
- Payment Terms
- Termination Notice Period

Rules:
- Return ONLY valid JSON
- If field not found → "Not Found"
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
    prompt = """
Extract resume details in JSON:

Fields:
- Name
- Skills
- Experience
- Education
- Missing Sections

Rules:
- Return ONLY valid JSON
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
    prompt = """
Extract invoice details in JSON:

Fields:
- Invoice Number
- Date
- Vendor Name
- Total Amount
- Tax
- Line Items

Rules:
- Return ONLY valid JSON
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

