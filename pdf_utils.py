
# # ==============OCR-VL-1.5-0.9B VERSION (PaddleOCRVL)================

# import re
# import fitz  # PyMuPDF
# import numpy as np
# import paddle # Added for memory management
# from paddleocr import PaddleOCRVL

# # Initialize the 0.9B VLM once
# # "v1.5" is the correct shorthand for PaddleOCR-VL-1.5-0.9B
# ocr_vl = PaddleOCRVL("v1.5")

# # def clean_text(text: str) -> str:
# #     """Normalize and clean extracted text for better LLM results."""
# #     text = re.sub(r"\r", "\n", text)
# #     text = re.sub(r"\n{2,}", "\n", text)
# #     text = re.sub(r"[ \t]+", " ", text)
# #     text = re.sub(r"[^\x00-\x7F]+", " ", text)
# #     return text.strip()

# # def chunk_text(text: str, chunk_size: int = 10000):
# #     """Split large text into smaller chunks."""
# #     if not text:
# #         return []
# #     return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

# def extract_text_from_pdf(file_path: str) -> str:
#     final_text = []

#     with fitz.open(file_path) as doc:
#         for page in doc:
#             # 1. Native Extraction (Fastest)
#             text = page.get_text("text")
#             if len(text.strip()) > 50:
#                 final_text.append(text.strip())
#                 continue

#             # 2. VLM Extraction (For scanned pages)
#             # Dropping DPI to 150 helps prevent the "MemoryError" you faced
#             pix = page.get_pixmap(dpi=150) 
            
#             img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
#             if pix.n == 4: 
#                 img = img[:, :, :3]

#             # Run prediction
#             results = ocr_vl.predict(img)
            
#             # --- FIX: Extract text using the correct attribute ---
#             # Most PaddleX/OCR-VL objects use .res for the final string
#             page_parts = []
#             for res in results:
#                 if hasattr(res, 'res'):
#                     page_parts.append(res.res)
#                 elif isinstance(res, dict) and 'res' in res:
#                     page_parts.append(res['res'])
#                 else:
#                     page_parts.append(str(res))
            
#             final_text.append("\n".join(page_parts))

#             # --- MEMORY CLEANUP ---
#             # Critical for preventing OOM crashes on long PDFs
#             del results
#             if paddle.device.is_compiled_with_cuda():
#                 paddle.device.cuda.empty_cache()

#     return clean_text("\n".join(final_text))




# # updated code



# def extract_text_from_pdf(file_path: str) -> str:
#     """
#     Extract ONLY real text from PDF.
#     Image-only pages are skipped.
#     Raises ValueError if the PDF cannot be opened (e.g. encrypted or corrupt).
#     """
#     try:
#         text_content = []
#         with fitz.open(file_path) as doc:
#             for page in doc:
#                 text = page.get_text("text")
#                 if not text or not text.strip():
#                     continue
#                 text_content.append(text.strip())
#         return "\n".join(text_content)
#     except Exception as e:
#         raise ValueError(f"Could not open PDF: {e}")


def clean_text(text: str) -> str:
    """
    Clean common PDF extraction artifacts:
    - Rejoin hyphenated line breaks
    - Collapse excessive blank lines
    - Collapse repeated spaces
    """
    text = re.sub(r'-\n', '', text)        # rejoin hyphenated words
    text = re.sub(r'\n{3,}', '\n\n', text) # collapse 3+ newlines to double
    text = re.sub(r' {2,}', ' ', text)     # collapse repeated spaces
    return text.strip()


def chunk_text(text: str, max_chars: int = 10000) -> list:
    """
    Split large text into smaller chunks on paragraph boundaries
    to avoid cutting mid-sentence.
    """
    if not text:
        return []

    paragraphs = text.split("\n")
    chunks = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) > max_chars:
            if current.strip():
                chunks.append(current.strip())
            current = para
        else:
            current += "\n" + para

    if current.strip():
        chunks.append(current.strip())

    return chunks

import re
import fitz  # PyMuPDF
import numpy as np
import paddle # Added for memory management
from paddleocr import PaddleOCRVL

# Initialize the 0.9B VLM once
# "v1.5" is the correct shorthand for PaddleOCR-VL-1.5-0.9B
ocr_vl = PaddleOCRVL("v1.5")

def clean_text(text: str) -> str:
    """Normalize and clean extracted text for better LLM results."""
    text = re.sub(r"\r", "\n", text)
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    return text.strip()

def chunk_text(text: str, chunk_size: int = 10000):
    """Split large text into smaller chunks."""
    if not text:
        return []
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def extract_text_from_pdf(file_path: str) -> str:
    final_text = []

    with fitz.open(file_path) as doc:
        for page in doc:
            # 1. Native Extraction (Fastest)
            text = page.get_text("text")
            if len(text.strip()) > 50:
                final_text.append(text.strip())
                continue

            # 2. VLM Extraction (For scanned pages)
            # Dropping DPI to 150 helps prevent the "MemoryError" you faced
            pix = page.get_pixmap(dpi=150) 
            
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
            if pix.n == 4: 
                img = img[:, :, :3]

            # Run prediction
            results = ocr_vl.predict(img)
            
            # --- FIX: Extract text using the correct attribute ---
            # Most PaddleX/OCR-VL objects use .res for the final string
            page_parts = []
            for res in results:
                if hasattr(res, 'res'):
                    page_parts.append(res.res)
                elif isinstance(res, dict) and 'res' in res:
                    page_parts.append(res['res'])
                else:
                    page_parts.append(str(res))
            
            final_text.append("\n".join(page_parts))

            # --- MEMORY CLEANUP ---
            # Critical for preventing OOM crashes on long PDFs
            del results
            if paddle.device.is_compiled_with_cuda():
                paddle.device.cuda.empty_cache()

    return clean_text("\n".join(final_text))

# import re
# import fitz  # PyMuPDF
# import numpy as np
# import paddle
# from paddleocr import PaddleOCRVL


# # ============================
# # Initialize OCR Model (Load Once)
# # ============================

# # "v1.5" = PaddleOCR-VL-1.5-0.9B
# ocr_vl = PaddleOCRVL("v1.5")


# # ============================
# # Clean Extracted Text
# # ============================

# def clean_text(text: str) -> str:
#     """Normalize and clean extracted text for better LLM results."""
#     text = re.sub(r"\r", "\n", text)
#     text = re.sub(r"\n{2,}", "\n", text)
#     text = re.sub(r"[ \t]+", " ", text)
#     text = re.sub(r"[^\x00-\x7F]+", " ", text)
#     return text.strip()


# # ============================
# # Chunk Large Text
# # ============================

# def chunk_text(text: str, chunk_size: int = 10000):
#     """Split large text into smaller chunks."""
#     if not text:
#         return []
#     return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


# # ============================
# # OCR Extraction (Image Page)
# # ============================

# def extract_text_from_image(page):
#     """Run OCR on scanned page."""
    
#     # DPI 180 = balance between clarity & memory
#     pix = page.get_pixmap(dpi=180)

#     img = np.frombuffer(
#         pix.samples,
#         dtype=np.uint8
#     ).reshape(pix.h, pix.w, pix.n)

#     # Remove alpha channel if exists
#     if pix.n == 4:
#         img = img[:, :, :3]

#     # Run OCR
#     results = ocr_vl.predict(img)

#     page_text_parts = []

#     for res in results:
#         # New PaddleOCR-VL output format
#         if isinstance(res, dict) and "text" in res:
#             page_text_parts.append(res["text"])
#         else:
#             page_text_parts.append(str(res))

#     # Memory cleanup (important for long PDFs)
#     del results
#     if paddle.device.is_compiled_with_cuda():
#         paddle.device.cuda.empty_cache()

#     return "\n".join(page_text_parts)


# # ============================
# # Main PDF Extraction Function
# # ============================

# def extract_text_from_pdf(file_path: str) -> str:
#     """
#     Extract text from:
#     - Text-based PDFs
#     - Image/scanned PDFs
#     - Mixed PDFs
#     """

#     final_text = []

#     with fitz.open(file_path) as doc:
#         for page_number, page in enumerate(doc):

#             # 1️⃣ Try native text extraction
#             text = page.get_text("text")

#             # If real readable text exists → use it
#             if text.strip() and any(c.isalpha() for c in text):
#                 final_text.append(text.strip())
#                 continue

#             # 2️⃣ Otherwise → Run OCR
#             print(f"Running OCR on page {page_number + 1}...")
#             ocr_text = extract_text_from_image(page)

#             if ocr_text.strip():
#                 final_text.append(ocr_text)

#     combined_text = "\n".join(final_text)

#     return clean_text(combined_text)


