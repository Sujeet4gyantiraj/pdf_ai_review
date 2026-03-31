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

Ensure the HTML includes `<html>`, `<head>`, `<style>`, and `<body>`. Make the entire `<body>` tag (or its direct primary content container, like a main `div`) editable by adding the `contenteditable='true'` attribute. All text content inside the body should then be editable.
Do NOT include any markdown, backticks, or extra text outside the HTML structure.""",
        input_variables=["user_request"]
    ),
    "invoice": SimulatedPromptTemplate(
        template="""You are an AI assistant that generates invoices in HTML format.
Generate a complete, well-structured HTML document for an invoice based on the following details:

User Request: {user_request}

Ensure the HTML includes `<html>`, `<head>`, `<style>`, and `<body>`. Make the entire `<body>` tag (or its direct primary content container, like a main `div`) editable by adding the `contenteditable='true'` attribute. All text content inside the body should then be editable.
Do NOT include any markdown, backticks, or extra text outside the HTML structure.""",
        input_variables=["user_request"]
    ),
    "example_type": SimulatedPromptTemplate(
        template="""You are an AI assistant that generates example HTML documents.
Generate a complete HTML document based on the user's prompt and document ID.

User Prompt: {user_prompt}
Document ID: {document_id}

Ensure the HTML includes `<html>`, `<head>`, `<style>`, and `<body>`. Make the entire `<body>` tag (or its direct primary content container, like a main `div`) editable by adding the `contenteditable='true'` attribute. All text content inside the body should then be editable.
Do NOT include any markdown or extra text outside the HTML.""",
        input_variables=["user_prompt", "document_id"]
    )
}



REGENERATE_PROMPT = SimulatedPromptTemplate(
    template="""You are an expert HTML editor. 
You will be provided with an existing HTML document and a user's request for modification.

GOAL:
Update the existing HTML content based on the user's instructions while maintaining the exact same CSS styles, design, and layout.

STRICT RULES:
1. Return ONLY the complete, updated HTML document starting with <html> and ending with </html>.
2. Ensure the `<body>` (or its primary content container) remains `contenteditable='true'`.
3. Do NOT include markdown backticks (```html), explanations, or notes.
4. Preserve all original <style> blocks and CSS logic.
5. Apply the changes requested by the user accurately.

Existing HTML:
{existing_html}

User Modification Request:
{modification_query}
""",
    input_variables=["existing_html", "modification_query"]
)