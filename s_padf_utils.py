import re
import logging
import numpy as np
import fitz  # PyMuPDF
import paddle

from paddleocr import PaddleOCRVL
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Chunking config — tune to your GPU VRAM:
#   24 GB → CHUNK_SIZE=12000  |  16 GB → 10000  |  8 GB → 6000
# ---------------------------------------------------------------------------
CHUNK_SIZE    = 10000
CHUNK_OVERLAP = 500

# Minimum characters on a page before falling back to OCR.
# Pages with fewer native characters than this are treated as image-only.
NATIVE_TEXT_THRESHOLD = 50

# ---------------------------------------------------------------------------
# PaddleOCR-VL model — loaded once at import time, same as the LLM
# ---------------------------------------------------------------------------
logger.info("Loading PaddleOCR-VL 1.5 ...")
_ocr_vl = PaddleOCRVL("v1.5")
logger.info("PaddleOCR-VL ready.")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ocr_page(page: fitz.Page) -> str:
    """
    Run PaddleOCR-VL on a single fitz page and return the extracted text.
    Renders at 150 DPI to balance quality vs. memory usage.
    Cleans up GPU memory after each page to prevent OOM on long PDFs.
    """
    pix = page.get_pixmap(dpi=150)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)

    # Drop alpha channel if present (RGBA → RGB)
    if pix.n == 4:
        img = img[:, :, :3]

    results = _ocr_vl.predict(img)

    # Extract text — PaddleOCR-VL result objects expose text via .res
    page_parts = []
    for res in results:
        if hasattr(res, "res"):
            page_parts.append(res.res)
        elif isinstance(res, dict) and "res" in res:
            page_parts.append(res["res"])
        else:
            page_parts.append(str(res))

    # GPU memory cleanup — critical for multi-page scanned PDFs
    del results
    if paddle.device.is_compiled_with_cuda():
        paddle.device.cuda.empty_cache()

    return "\n".join(page_parts)


def clean_text(text: str) -> str:
    """
    Clean common PDF/OCR extraction artifacts:
    - Rejoin hyphenated line breaks
    - Collapse excessive blank lines
    - Collapse repeated spaces
    """
    text = re.sub(r"-\n",    "",     text)  # rejoin hyphenated words
    text = re.sub(r"\n{3,}", "\n\n", text)  # collapse 3+ newlines
    text = re.sub(r" {2,}",  " ",    text)  # collapse repeated spaces
    return text.strip()


# ---------------------------------------------------------------------------
# Public API — matches what main.py and ai_model.py expect
# ---------------------------------------------------------------------------

def load_pdf(file_path: str) -> list[Document]:
    """
    Load a PDF and return one LangChain Document per page.

    Extraction strategy per page (in priority order):
      1. Native PyMuPDF text  — fast, zero GPU cost, used when text is present
      2. PaddleOCR-VL 1.5    — used for scanned / image-only pages

    Page metadata (page number, source file path) is attached to every
    Document so it propagates through splitting into every chunk.

    Raises ValueError if the file cannot be opened or yields no text at all.
    """
    try:
        doc = fitz.open(file_path)
    except Exception as e:
        raise ValueError(f"Could not open PDF: {e}")

    pages: list[Document] = []

    with doc:
        for page in doc:
            page_num = page.number + 1  # 1-based for readability in logs

            # --- Strategy 1: native text extraction ---
            native_text = page.get_text("text").strip()

            if len(native_text) >= NATIVE_TEXT_THRESHOLD:
                logger.info(f"Page {page_num}: native extraction "
                            f"({len(native_text)} chars)")
                text = native_text

            else:
                # --- Strategy 2: PaddleOCR-VL for scanned/image pages ---
                logger.info(f"Page {page_num}: native text too short "
                            f"({len(native_text)} chars) — running OCR-VL")
                try:
                    text = _ocr_page(page)
                except Exception as e:
                    logger.warning(f"Page {page_num}: OCR-VL failed ({e}), skipping.")
                    continue

            text = clean_text(text)
            if not text:
                logger.info(f"Page {page_num}: no text after cleaning, skipping.")
                continue

            pages.append(Document(
                page_content=text,
                metadata={
                    "page":   page_num,
                    "source": file_path,
                },
            ))

    if not pages:
        raise ValueError(
            "No extractable text found. "
            "The file may be fully image-based with unrecognisable content, "
            "encrypted, or empty."
        )

    logger.info(f"Loaded {len(pages)} page(s) from {file_path}")
    return pages


def split_documents(pages: list[Document]) -> list[Document]:
    """
    Split LangChain Document pages into smaller overlapping chunks using
    RecursiveCharacterTextSplitter, which respects paragraph / sentence /
    word boundaries in priority order before falling back to characters.

    Page metadata (page number, source path) is preserved on every chunk.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],  # priority order
        length_function=len,
    )

    chunks = splitter.split_documents(pages)
    logger.info(
        f"Split {len(pages)} page(s) into {len(chunks)} chunk(s) "
        f"(chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})"
    )
    return chunks


def get_page_count(file_path: str) -> int:
    """Return the number of pages in a PDF without fully loading it."""
    try:
        with fitz.open(file_path) as doc:
            return len(doc)
    except Exception:
        return 0