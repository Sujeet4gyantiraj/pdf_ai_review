# u_risk_detection.py
import logging
from llm_model.ai_model import run_llm
from utils.json_utils import extract_json_raw as extract_json_from_text

logger = logging.getLogger(__name__)

async def analyze_document_risks(text: str):
    """
    Scans document for legal/financial risks and returns structured JSON.
    Reuses: s_ai_model.run_llm
    """
    
    prompt = """
    Analyze the provided document text for legal and financial risks. 
    Identify clauses that are unfavorable to the signer (e.g., founders or business owners).

    Specific Risks to detect:
    - Auto-renewal: Clauses that commit the user to another term automatically.
    - Indemnity: Clauses where the user takes on heavy financial liability.
    - Termination Penalties: Excessive costs for exiting the contract.
    - Non-compete: Restrictions on future business or career moves.
    - Missing Liability Caps: No limit on how much the user might owe in damages.
    - Jurisdiction: Legal disputes happening in a far-away or unfavorable location.

    Output ONLY valid JSON in this format:
    {
      "risk_score": 0-100,
      "detected_risks": [
        {
          "risk_name": "Name of category",
          "severity": "High/Medium/Low",
          "summary": "1 sentence describing the clause",
          "impact": "Why this is dangerous",
          "mitigation": "How to negotiate or change this"
        }
      ],
      "overall_assessment": "Executive summary of the document's risk profile."
    }

    Document Text:
    ---
    """ + text[:12000] + "\n---"

    logger.info("[risk_detection] Running AI analysis...")
    
    # Reusing your core LLM runner
    raw_output = await run_llm(text, prompt)
    print("Raw LLM Output:", raw_output)  # Debugging: see the unprocessed output
    # Using the new generic JSON parser
    analysis = extract_json_from_text(raw_output)
    
    return {
        "status": "success",
        "analysis_type": "risk_detection",
        "data": analysis
    }
