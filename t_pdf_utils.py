import re
import fitz  # PyMuPDF
import numpy as np
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional PaddleOCR-VL import — only loaded if available.
# If not installed the module still imports cleanly; scanned pages will be
# skipped with a warning instead of crashing the whole server.
# ---------------------------------------------------------------------------
try:
    import paddle
    from paddleocr import PaddleOCRVL
    _ocr_vl = PaddleOCRVL("v1.5")
    _PADDLE_AVAILABLE = True
except Exception as e:
    logger.warning(f"PaddleOCR-VL not available, OCR disabled: {e}")
    _ocr_vl = None
    _PADDLE_AVAILABLE = False

# Maximum pages processed per PDF (prevents multi-hour hangs on huge files)
MAX_PAGES = 100


def _page_to_rgb(pix: fitz.Pixmap) -> np.ndarray:
    """
    Convert a PyMuPDF Pixmap to an HxWx3 uint8 RGB numpy array.
    Handles grayscale (n=1), RGB (n=3), and RGBA (n=4) pixmaps.
    """
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)

    if pix.n == 1:
        # Grayscale → RGB  (Fix #16)
        img = np.repeat(img, 3, axis=2)
    elif pix.n == 4:
        # RGBA → RGB
        img = img[:, :, :3]
    # pix.n == 3 is already correct

    return img


def _run_ocr(img: np.ndarray) -> str:
    """
    Run PaddleOCR-VL on a single-page RGB image and return the page text.
    Safely extracts text regardless of whether res.res is a str or dict.  (Fix #15)
    Cleans up GPU memory after each page.
    """
    if _ocr_vl is None:
        return ""

    try:
        results = _ocr_vl.predict(img)
        page_parts = []

        for res in results:
            raw = None

            # PaddleOCR-VL result can be an object with .res or a plain dict
            if hasattr(res, "res"):
                raw = res.res
            elif isinstance(res, dict) and "res" in res:
                raw = res["res"]
            else:
                raw = str(res)

            # .res is usually a dict like {"rec_text": "...", ...}   (Fix #15)
            if isinstance(raw, dict):
                text = raw.get("rec_text") or raw.get("text") or " ".join(
                    str(v) for v in raw.values() if isinstance(v, str)
                )
            else:
                text = str(raw)

            if text and text.strip():
                page_parts.append(text.strip())

        return "\n".join(page_parts)

    except Exception as e:
        logger.error(f"OCR failed on page: {e}")
        return ""

    finally:
        # Always free GPU memory after each page   (Fix #17)
        try:
            del results
        except Exception:
            pass
        if _PADDLE_AVAILABLE:
            try:
                if paddle.device.is_compiled_with_cuda():
                    paddle.device.cuda.empty_cache()
            except Exception:
                pass


def extract_text_from_pdf(file_path: str) -> str:
    """
    Extract text from a PDF file.

    Strategy per page:
      1. Native PyMuPDF text extraction (fastest, works for text-based PDFs).
      2. PaddleOCR-VL (for scanned / image-only pages) if paddle is available.
      3. Skip the page with a warning if neither yields text.

    Raises ValueError if the file cannot be opened (corrupt / encrypted).
    Limits processing to MAX_PAGES pages.   (Fix #18)
    """
    try:
        doc = fitz.open(file_path)
    except Exception as e:
        raise ValueError(f"Could not open PDF: {e}")

    final_text = []

    with doc:
        total_pages = len(doc)
        if total_pages > MAX_PAGES:
            logger.warning(
                f"PDF has {total_pages} pages; processing only the first {MAX_PAGES}."
            )

        for page_num, page in enumerate(doc):
            if page_num >= MAX_PAGES:
                break

            # --- Strategy 1: native text ---
            text = page.get_text("text")
            if text and len(text.strip()) > 50:
                final_text.append(text.strip())
                continue

            # --- Strategy 2: OCR ---
            if not _PADDLE_AVAILABLE:
                logger.warning(
                    f"Page {page_num + 1} has no native text and PaddleOCR is "
                    "unavailable — page skipped."
                )
                continue

            try:
                pix = page.get_pixmap(dpi=150)
                img = _page_to_rgb(pix)
                ocr_text = _run_ocr(img)
                if ocr_text:
                    final_text.append(ocr_text)
                else:
                    logger.warning(f"Page {page_num + 1}: OCR returned no text.")
            except MemoryError:
                logger.error(f"Page {page_num + 1}: MemoryError during OCR, skipping.")
            except Exception as e:
                logger.error(f"Page {page_num + 1}: unexpected OCR error: {e}")

    return clean_text("\n".join(final_text))


def clean_text(text: str) -> str:
    """
    Clean common PDF extraction artefacts:
      - Rejoin hyphenated line breaks
      - Collapse 3+ blank lines to a double newline
      - Collapse repeated spaces
    """
    text = re.sub(r"-\n", "", text)         # rejoin hyphenated words
    text = re.sub(r"\n{3,}", "\n\n", text)  # collapse excessive blank lines
    text = re.sub(r" {2,}", " ", text)      # collapse repeated spaces
    return text.strip()


def chunk_text(text: str, max_chars: int = 10_000) -> list:
    """
    Split large text into chunks on paragraph boundaries so that sentences
    are never cut mid-way.  Returns an empty list for empty input.
    """
    if not text:
        return []

    paragraphs = text.split("\n")
    chunks: list[str] = []
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