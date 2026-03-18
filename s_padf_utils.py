import re
import time
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
logger.info("[pdf_utils] Loading PaddleOCR-VL 1.5 ...")
t0      = time.perf_counter()
_ocr_vl = PaddleOCRVL("v1.5")
logger.info(f"[pdf_utils] PaddleOCR-VL ready ({time.perf_counter() - t0:.2f}s)")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ocr_page(page: fitz.Page) -> str:
    """
    Run PaddleOCR-VL on a single fitz page and return the extracted text.
    Renders at 150 DPI to balance quality vs. memory usage.
    Cleans up GPU memory after each page to prevent OOM on long PDFs.
    """
    page_num = page.number + 1

    logger.debug(f"[pdf_utils] Page {page_num}: rendering pixmap at 150 DPI")
    t_pix = time.perf_counter()
    pix   = page.get_pixmap(dpi=150)
    logger.debug(
        f"[pdf_utils] Page {page_num}: pixmap ready "
        f"{pix.w}x{pix.h}px, {pix.n} channel(s) "
        f"({time.perf_counter() - t_pix:.2f}s)"
    )

    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)

    # Drop alpha channel if present (RGBA → RGB)
    if pix.n == 4:
        logger.debug(f"[pdf_utils] Page {page_num}: dropping alpha channel (RGBA → RGB)")
        img = img[:, :, :3]

    logger.debug(f"[pdf_utils] Page {page_num}: running OCR-VL predict")
    t_ocr   = time.perf_counter()
    results = _ocr_vl.predict(img)
    logger.info(
        f"[pdf_utils] Page {page_num}: OCR-VL predict done "
        f"({time.perf_counter() - t_ocr:.2f}s), "
        f"{len(results)} result block(s)"
    )

    # Extract text — PaddleOCR-VL result objects expose text via .res
    page_parts = []
    for res in results:
        if hasattr(res, "res"):
            page_parts.append(res.res)
        elif isinstance(res, dict) and "res" in res:
            page_parts.append(res["res"])
        else:
            logger.debug(f"[pdf_utils] Page {page_num}: unexpected result type {type(res)}, using str()")
            page_parts.append(str(res))

    extracted_chars = sum(len(p) for p in page_parts)
    logger.debug(
        f"[pdf_utils] Page {page_num}: extracted {extracted_chars} chars "
        f"from {len(page_parts)} block(s)"
    )

    # GPU memory cleanup — critical for multi-page scanned PDFs
    del results
    if paddle.device.is_compiled_with_cuda():
        paddle.device.cuda.empty_cache()
        logger.debug(f"[pdf_utils] Page {page_num}: paddle GPU cache cleared")

    return "\n".join(page_parts)


def clean_text(text: str) -> str:
    """
    Clean common PDF/OCR extraction artifacts:
    - Rejoin hyphenated line breaks
    - Collapse excessive blank lines
    - Collapse repeated spaces
    """
    before = len(text)
    text = re.sub(r"-\n",    "",     text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}",  " ",    text)
    text = text.strip()
    logger.debug(f"[pdf_utils] clean_text: {before} → {len(text)} chars")
    return text


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
    logger.info(f"[pdf_utils] load_pdf: opening '{file_path}'")
    t_load = time.perf_counter()

    try:
        doc = fitz.open(file_path)
    except Exception as e:
        logger.error(f"[pdf_utils] load_pdf: failed to open file — {e}")
        raise ValueError(f"Could not open PDF: {e}")

    total_pages = len(doc)
    logger.info(f"[pdf_utils] load_pdf: {total_pages} page(s) found")

    pages:        list[Document] = []
    native_count: int = 0
    ocr_count:    int = 0
    skip_count:   int = 0

    with doc:
        for page in doc:
            page_num = page.number + 1

            # --- Strategy 1: native text extraction ---
            native_text = page.get_text("text").strip()
            logger.debug(
                f"[pdf_utils] Page {page_num}/{total_pages}: "
                f"native text = {len(native_text)} chars"
            )

            if len(native_text) >= NATIVE_TEXT_THRESHOLD:
                logger.info(
                    f"[pdf_utils] Page {page_num}/{total_pages}: "
                    f"✓ native ({len(native_text)} chars)"
                )
                text = native_text
                native_count += 1

            else:
                # --- Strategy 2: PaddleOCR-VL for scanned/image pages ---
                logger.info(
                    f"[pdf_utils] Page {page_num}/{total_pages}: "
                    f"native text too short ({len(native_text)} chars) "
                    f"— falling back to OCR-VL"
                )
                try:
                    text = _ocr_page(page)
                    ocr_count += 1
                except Exception as e:
                    logger.error(
                        f"[pdf_utils] Page {page_num}/{total_pages}: "
                        f"OCR-VL failed — {e} — skipping page"
                    )
                    skip_count += 1
                    continue

            text = clean_text(text)
            if not text:
                logger.warning(
                    f"[pdf_utils] Page {page_num}/{total_pages}: "
                    "no text after cleaning — skipping page"
                )
                skip_count += 1
                continue

            pages.append(Document(
                page_content=text,
                metadata={
                    "page":   page_num,
                    "source": file_path,
                },
            ))

    elapsed = time.perf_counter() - t_load
    logger.info(
        f"[pdf_utils] load_pdf complete in {elapsed:.2f}s — "
        f"{len(pages)} page(s) loaded "
        f"(native={native_count}, ocr={ocr_count}, skipped={skip_count})"
    )

    if not pages:
        logger.error(
            f"[pdf_utils] load_pdf: no text extracted from any page in '{file_path}'"
        )
        raise ValueError(
            "No extractable text found. "
            "The file may be fully image-based with unrecognisable content, "
            "encrypted, or empty."
        )

    return pages


def split_documents(pages: list[Document]) -> list[Document]:
    """
    Split LangChain Document pages into smaller overlapping chunks using
    RecursiveCharacterTextSplitter, which respects paragraph / sentence /
    word boundaries in priority order before falling back to characters.

    Page metadata (page number, source path) is preserved on every chunk.
    """
    total_chars = sum(len(p.page_content) for p in pages)
    logger.info(
        f"[pdf_utils] split_documents: splitting {len(pages)} page(s), "
        f"{total_chars:,} total chars "
        f"(chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})"
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],  # priority order
        length_function=len,
    )

    t_split = time.perf_counter()
    chunks  = splitter.split_documents(pages)
    elapsed = time.perf_counter() - t_split

    avg_chunk = sum(len(c.page_content) for c in chunks) / len(chunks) if chunks else 0
    logger.info(
        f"[pdf_utils] split_documents: {len(chunks)} chunk(s) produced "
        f"in {elapsed:.3f}s (avg {avg_chunk:.0f} chars/chunk)"
    )

    return chunks


def get_page_count(file_path: str) -> int:
    """Return the number of pages in a PDF without fully loading it."""
    try:
        with fitz.open(file_path) as doc:
            count = len(doc)
            logger.debug(f"[pdf_utils] get_page_count: '{file_path}' has {count} page(s)")
            return count
    except Exception as e:
        logger.error(f"[pdf_utils] get_page_count: failed to open '{file_path}' — {e}")
        return 0