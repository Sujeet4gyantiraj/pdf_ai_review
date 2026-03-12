import fitz  # PyMuPDF


def extract_text_from_pdf(file_path: str) -> str:
    """
    Extract ONLY real text from PDF.
    Image-only pages are skipped.
    """
    text_content = []
    with fitz.open(file_path) as doc:
        for page in doc:
            text = page.get_text("text")
            if not text or not text.strip():
                continue
            text_content.append(text.strip())
    return "\n".join(text_content)


def chunk_text(text: str, chunk_size: int = 10000):
    """
    Split large text into smaller chunks
    """
    if not text:
        return []
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
