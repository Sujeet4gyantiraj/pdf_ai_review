# document_generation/prompt_templates.py

# Simulate LangChain's PromptTemplate for this example
# In a real scenario, you'd install and import from langchain_core.prompts
class SimulatedPromptTemplate:
    def __init__(self, template: str, input_variables: list[str]):
        self.template = template
        self.input_variables = input_variables

    def format(self, **kwargs) -> str:
        formatted_template = self.template
        for var in self.input_variables:
            if var in kwargs:
                # Replace placeholders with actual values, ensuring proper escaping if necessary
                formatted_template = formatted_template.replace(f"{{{var}}}", str(kwargs[var]))
        return formatted_template

# Define prompt templates for different document types
# These could be loaded from external files or a database in a real application
prompt_templates = {
    "offer_letter": SimulatedPromptTemplate(
        template="""You are an AI assistant that generates professional offer letters in HTML format.
Generate a complete, well-structured HTML document for an offer letter based on the following details:

User Request: {user_request}

Ensure the HTML includes `<html>`, `<head>`, `<style>`, `<body>`, `<p>`, `<h1>`, etc., as appropriate for a formal letter.
For key fields like employee name, salary, position, and dates, include the `contenteditable='true'` attribute on the HTML element containing that information.
Do NOT include any markdown, backticks, or extra text outside the HTML structure.""",
        input_variables=["user_request"]
    ),
    "invoice": SimulatedPromptTemplate(
        template="""You are an AI assistant that generates invoices in HTML format.
Generate a complete, well-structured HTML document for an invoice based on the following details:

User Request: {user_request}

Include typical invoice fields like client name, item list, quantities, prices, total amount, and date. For each of these key pieces of information, add the `contenteditable='true'` attribute to their respective HTML elements.
Ensure the HTML is clean and formatted for readability.
Do NOT include any markdown, backticks, or extra text outside the HTML structure.""",
        input_variables=["user_request"]
    ),
    "example_type": SimulatedPromptTemplate(
        template="""You are an AI assistant that generates example HTML documents.
Generate a complete HTML document based on the user's prompt and document ID.

User Prompt: {user_prompt}
Document ID: {document_id}

For any prominent text content that could be edited, add the `contenteditable='true'` attribute to its HTML element (e.g., a paragraph or heading).
Ensure the HTML is well-formed. Do NOT include any markdown or extra text outside the HTML.""",
        input_variables=["user_prompt", "document_id"]
    )
}
