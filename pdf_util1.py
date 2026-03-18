import re
import fitz  # PyMuPDF
import numpy as np
import paddle
import logging
from paddleocr import PaddleOCRVL

logger = logging.getLogger(__name__)

# ---------------- DEVICE SETUP ----------------
if paddle.is_compiled_with_cuda():
    paddle.set_device("gpu")
    logger.info("PaddleOCR-VL running on GPU")
else:
    paddle.set_device("cpu")
    logger.warning("PaddleOCR-VL running on CPU")

# ---------------- INIT VLM MODEL ----------------
# "v1.5" = PaddleOCR-VL-1.5-0.9B
ocr_vl = PaddleOCRVL("v1.5")


# ---------------- TEXT CLEANING ----------------
def clean_text(text: str) -> str:
    """Clean extracted text for better LLM summarization."""
    text = re.sub(r"\r", "\n", text)
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    return text.strip()


# ---------------- PDF EXTRACTION ----------------
def extract_text_from_pdf(file_path: str) -> str:
    """
    Hybrid extraction:
    1️⃣ Try native extraction
    2️⃣ If page has little/no text → use OCR-VL
    """

    final_text = []

    with fitz.open(file_path) as doc:
        for page_number, page in enumerate(doc):

            logger.info(f"Processing page {page_number}")

            # 1️⃣ Native extraction (fastest)
            text = page.get_text("text")

            if len(text.strip()) > 50:
                final_text.append(text.strip())
                continue

            # 2️⃣ OCR-VL extraction (scanned page)
            logger.info(f"Using OCR-VL on page {page_number}")

            # Lower DPI prevents GPU OOM
            pix = page.get_pixmap(dpi=150)

            img = np.frombuffer(
                pix.samples,
                dtype=np.uint8
            ).reshape(pix.h, pix.w, pix.n)

            # Remove alpha channel
            if pix.n == 4:
                img = img[:, :, :3]

            # Run VLM OCR
            results = ocr_vl.predict(img)

            page_parts = []

            for res in results:
                if hasattr(res, "res"):
                    page_parts.append(res.res)
                elif isinstance(res, dict) and "res" in res:
                    page_parts.append(res["res"])
                else:
                    page_parts.append(str(res))

            page_text = "\n".join(page_parts).strip()

            final_text.append(page_text)

            # -------- MEMORY CLEANUP --------
            del results
            if paddle.is_compiled_with_cuda():
                paddle.device.cuda.empty_cache()

    combined_text = "\n".join(final_text)

    return clean_text(combined_text)




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