# import fitz  # PyMuPDF


# def extract_text_from_pdf(file_path: str) -> str:
#     """
#     Extract ONLY real text from PDF.
#     Image-only pages are skipped.
#     """
#     text_content = []
#     with fitz.open(file_path) as doc:
#         for page in doc:
#             text = page.get_text("text")
#             if not text or not text.strip():
#                 continue
#             text_content.append(text.strip())
#     return "\n".join(text_content)


# def chunk_text(text: str, chunk_size: int = 10000):
#     """
#     Split large text into smaller chunks
#     """
#     if not text:
#         return []
#     return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


import re
import fitz  # PyMuPDF
import numpy as np
from paddleocr import PaddleOCR

# Initialize OCR only once (important for performance)
ocr = PaddleOCR(
    use_angle_cls=True,
    lang="en",
    use_gpu=True  # set False if no GPU
)


def clean_text(text: str) -> str:
    """
    Normalize and clean extracted text for better LLM results.
    """
    text = re.sub(r"\r", "\n", text)
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    return text.strip()


def extract_text_from_pdf(file_path: str) -> str:
    """
    Hybrid extraction:
    - Try embedded text first
    - If page has little/no text → use OCR only for that page
    - Works for normal, scanned, and mixed PDFs
    """

    final_text = []

    with fitz.open(file_path) as doc:
        for page_index, page in enumerate(doc):

            # 1️⃣ Try extracting embedded text
            text = page.get_text("text")

            if text and len(text.strip()) > 50:
                final_text.append(text.strip())
                continue

            # 2️⃣ If not enough text → use OCR for this page
            pix = page.get_pixmap(dpi=300)

            # Convert to numpy array (no temp file needed)
            img = np.frombuffer(pix.samples, dtype=np.uint8)
            img = img.reshape(pix.height, pix.width, pix.n)

            result = ocr.ocr(img, cls=True)

            page_text = []
            for line in result:
                for word_info in line:
                    page_text.append(word_info[-1][0])

            final_text.append(" ".join(page_text))

    return clean_text("\n".join(final_text))


def chunk_text(text: str, chunk_size: int = 2500):
    """
    Smart sentence-based chunking.
    Prevents breaking context mid-sentence.
    """

    if not text:
        return []

    sentences = re.split(r'(?<=[.!?]) +', text)

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks
