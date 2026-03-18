import re
import time
import logging
import numpy as np
import fitz  # PyMuPDF
import paddle
from concurrent.futures import ThreadPoolExecutor, as_completed

from paddleocr import PaddleOCRVL
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Chunking config
# ---------------------------------------------------------------------------
CHUNK_SIZE    = 40000
CHUNK_OVERLAP = 200

# Pre-compiled regex patterns
_RE_HYPHEN    = re.compile(r"-\n")
_RE_MULTILINE = re.compile(r"\n{3,}")
_RE_SPACES    = re.compile(r" {2,}")

# Minimum native chars before falling back to OCR
NATIVE_TEXT_THRESHOLD = 50

# Minimum chars Tesseract must return to be considered successful.
# If Tesseract returns fewer chars than this, PaddleOCR-VL is used instead.
TESSERACT_MIN_CHARS = 30

# Parallel workers for native extraction (Pass 1)
_NATIVE_EXTRACT_WORKERS = 4

# ---------------------------------------------------------------------------
# Tesseract availability check — done once at import time
# pytesseract wraps the system tesseract binary.
# Install: sudo apt install tesseract-ocr && pip install pytesseract pillow
# ---------------------------------------------------------------------------
_TESSERACT_AVAILABLE = False
try:
    import pytesseract
    from PIL import Image as PILImage
    pytesseract.get_tesseract_version()   # raises if binary not found
    _TESSERACT_AVAILABLE = True
    logger.info("[pdf_utils] Tesseract available — will use as fast OCR fallback")
except Exception as e:
    logger.info(
        f"[pdf_utils] Tesseract not available ({e}) — "
        "scanned pages will go directly to PaddleOCR-VL. "
        "Install for faster OCR: sudo apt install tesseract-ocr && pip install pytesseract pillow"
    )

# ---------------------------------------------------------------------------
# PaddleOCR-VL — loaded once at import time
# ---------------------------------------------------------------------------
logger.info("[pdf_utils] Loading PaddleOCR-VL 1.5 ...")
t0      = time.perf_counter()
_ocr_vl = PaddleOCRVL("v1.5")
logger.info(f"[pdf_utils] PaddleOCR-VL ready ({time.perf_counter() - t0:.2f}s)")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Clean PDF/OCR artifacts using pre-compiled patterns."""
    before = len(text)
    text = _RE_HYPHEN.sub("",       text)
    text = _RE_MULTILINE.sub("\n\n", text)
    text = _RE_SPACES.sub(" ",      text)
    text = text.strip()
    logger.debug(f"[pdf_utils] clean_text: {before} → {len(text)} chars")
    return text


def _extract_native(page: fitz.Page) -> str | None:
    """
    Strategy 1 — PyMuPDF native text extraction.
    Fast, zero GPU cost. Returns cleaned text if above threshold, else None.
    Thread-safe — safe to call from ThreadPoolExecutor.
    """
    text = page.get_text("text").strip()
    if len(text) >= NATIVE_TEXT_THRESHOLD:
        return clean_text(text)
    return None


def _page_to_image(page: fitz.Page, dpi: int = 200) -> np.ndarray:
    """
    Render a fitz page to a numpy uint8 RGB array.
    Shared by both Tesseract and PaddleOCR paths.
    DPI=200 gives a good balance of quality vs speed for A4 pages.
    """
    pix = page.get_pixmap(dpi=dpi)
    img = np.ascontiguousarray(
        np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
    )
    if pix.n == 4:
        img = img[:, :, :3]  # drop alpha
    return img


def _tesseract_page(page: fitz.Page) -> str | None:
    """
    Strategy 2 — Tesseract OCR (CPU, fast).

    Why Tesseract first:
      - Speed: ~0.3–0.8s per page on CPU vs ~1–3s on GPU for PaddleOCR-VL
      - Works well on clean scanned documents and mixed text/image pages
      - PaddleOCR-VL is reserved only for pages where Tesseract struggles
        (degraded scans, complex layouts, tables, handwriting)

    Returns cleaned text if above TESSERACT_MIN_CHARS, else None.
    None signals that PaddleOCR-VL should be tried next.
    """
    if not _TESSERACT_AVAILABLE:
        return None

    page_num = page.number + 1
    t_tess   = time.perf_counter()

    try:
        img      = _page_to_image(page, dpi=200)
        pil_img  = PILImage.fromarray(img)
        # oem 3 = best LSTM engine, psm 3 = fully automatic page segmentation
        text     = pytesseract.image_to_string(pil_img, config="--oem 3 --psm 3")
        elapsed  = time.perf_counter() - t_tess
        char_count = len(text.strip())

        if char_count >= TESSERACT_MIN_CHARS:
            logger.info(
                f"[pdf_utils] Page {page_num}: ✓ Tesseract "
                f"({char_count} chars, {elapsed:.2f}s)"
            )
            return clean_text(text)
        else:
            logger.info(
                f"[pdf_utils] Page {page_num}: Tesseract returned only "
                f"{char_count} chars ({elapsed:.2f}s) — escalating to PaddleOCR-VL"
            )
            return None

    except Exception as e:
        logger.warning(
            f"[pdf_utils] Page {page_num}: Tesseract failed ({e}) "
            "— escalating to PaddleOCR-VL"
        )
        return None


def _paddle_page(page: fitz.Page) -> str:
    """
    Strategy 3 — PaddleOCR-VL (GPU, thorough).

    Used only when both native extraction and Tesseract fail or return
    insufficient text. Handles complex layouts, tables, degraded scans,
    mixed-language pages, and handwriting better than Tesseract.

    GPU memory is cleared after each page to prevent OOM on long PDFs.
    """
    page_num = page.number + 1
    t_ocr    = time.perf_counter()

    img     = _page_to_image(page, dpi=150)   # 150 DPI is enough for PaddleOCR-VL
    results = _ocr_vl.predict(img)

    elapsed = time.perf_counter() - t_ocr
    logger.info(
        f"[pdf_utils] Page {page_num}: ✓ PaddleOCR-VL "
        f"({len(results)} block(s), {elapsed:.2f}s)"
    )

    page_parts = []
    for res in results:
        if hasattr(res, "res"):
            page_parts.append(res.res)
        elif isinstance(res, dict) and "res" in res:
            page_parts.append(res["res"])
        else:
            page_parts.append(str(res))

    # Free GPU memory immediately — critical for multi-page scanned PDFs
    del results
    if paddle.device.is_compiled_with_cuda():
        paddle.device.cuda.empty_cache()

    return "\n".join(page_parts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_pdf(file_path: str, max_pages: int | None = None) -> list[Document]:
    """
    Load a PDF and return one LangChain Document per page.

    Three-tier extraction pipeline per page:

      Tier 1 — PyMuPDF native  (parallel, ~0.01s/page, free)
        ↓ fails (< 50 chars)
      Tier 2 — Tesseract OCR   (serial CPU, ~0.5s/page)
        ↓ fails (< 30 chars)
      Tier 3 — PaddleOCR-VL   (serial GPU, ~1-3s/page)

    Most PDFs are a mix of native-text pages and scanned pages.
    Tesseract handles the majority of scanned pages quickly on CPU,
    so PaddleOCR-VL only runs on truly difficult pages (degraded scans,
    complex tables, handwriting). This minimises GPU usage and total time.

    Pass 1 runs native extraction in parallel across all pages.
    Pass 2 runs OCR serially on only the pages that need it, trying
    Tesseract first and PaddleOCR-VL only as a last resort.
    """
    logger.info(
        f"[pdf_utils] load_pdf: '{file_path}'"
        + (f" max_pages={max_pages}" if max_pages else "")
    )
    t_load = time.perf_counter()

    try:
        doc = fitz.open(file_path)
    except Exception as e:
        logger.error(f"[pdf_utils] load_pdf: open failed — {e}")
        raise ValueError(f"Could not open PDF: {e}")

    total_pages      = len(doc)
    pages_to_process = total_pages if max_pages is None else min(total_pages, max_pages)
    logger.info(f"[pdf_utils] {total_pages} page(s) total, processing {pages_to_process}")

    fitz_pages = [doc[i] for i in range(pages_to_process)]

    # ── Pass 1: parallel native extraction ───────────────────────────────
    logger.info(
        f"[pdf_utils] Pass 1 — parallel native extraction "
        f"({_NATIVE_EXTRACT_WORKERS} workers)"
    )
    t_pass1        = time.perf_counter()
    native_results: dict[int, str | None] = {}

    def _native_worker(idx_page):
        idx, page = idx_page
        return idx, _extract_native(page)

    with ThreadPoolExecutor(max_workers=_NATIVE_EXTRACT_WORKERS) as pool:
        futures = {pool.submit(_native_worker, (i, p)): i for i, p in enumerate(fitz_pages)}
        for future in as_completed(futures):
            idx, text = future.result()
            native_results[idx] = text

    native_hit  = sum(1 for v in native_results.values() if v is not None)
    native_miss = pages_to_process - native_hit
    logger.info(
        f"[pdf_utils] Pass 1 done ({time.perf_counter() - t_pass1:.2f}s) — "
        f"native={native_hit}, need_ocr={native_miss}"
    )

    # ── Pass 2: OCR for pages that need it (Tesseract → PaddleOCR-VL) ────
    if native_miss > 0:
        logger.info(
            f"[pdf_utils] Pass 2 — OCR pipeline for {native_miss} page(s) "
            f"(Tesseract{'→PaddleOCR-VL' if _TESSERACT_AVAILABLE else ' unavailable →PaddleOCR-VL'})"
        )

    pages:          list[Document] = []
    tesseract_count: int = 0
    paddle_count:   int = 0
    skip_count:     int = 0

    for idx, fitz_page in enumerate(fitz_pages):
        page_num  = idx + 1
        native_ok = native_results.get(idx)

        if native_ok is not None:
            # Tier 1 succeeded — no OCR needed
            text = native_ok

        else:
            # Tier 2 — try Tesseract first (fast CPU OCR)
            logger.info(
                f"[pdf_utils] Page {page_num}/{pages_to_process}: "
                "native short — trying Tesseract"
            )
            tess_text = _tesseract_page(fitz_page)

            if tess_text is not None:
                # Tesseract succeeded
                text = tess_text
                tesseract_count += 1
            else:
                # Tier 3 — PaddleOCR-VL (GPU, last resort)
                logger.info(
                    f"[pdf_utils] Page {page_num}/{pages_to_process}: "
                    "Tesseract insufficient — using PaddleOCR-VL"
                )
                try:
                    raw  = _paddle_page(fitz_page)
                    text = clean_text(raw)
                    paddle_count += 1
                except Exception as e:
                    logger.error(
                        f"[pdf_utils] Page {page_num}: PaddleOCR-VL failed — {e} — skipping"
                    )
                    skip_count += 1
                    continue

        if not text:
            logger.warning(f"[pdf_utils] Page {page_num}: empty after cleaning — skipping")
            skip_count += 1
            continue

        pages.append(Document(
            page_content=text,
            metadata={"page": page_num, "source": file_path},
        ))

    doc.close()
    elapsed = time.perf_counter() - t_load
    logger.info(
        f"[pdf_utils] load_pdf done in {elapsed:.2f}s — "
        f"{len(pages)} page(s) loaded "
        f"(native={native_hit}, tesseract={tesseract_count}, "
        f"paddle={paddle_count}, skipped={skip_count})"
    )

    if not pages:
        raise ValueError(
            "No extractable text found. "
            "File may be image-only, encrypted, or empty."
        )

    return pages


def merge_pages(pages: list[Document]) -> list[Document]:
    """Concatenate all pages into one Document so the splitter produces full-size chunks."""
    if not pages:
        return pages

    logger.info(f"[pdf_utils] merge_pages: merging {len(pages)} page(s) into single document")
    combined_text = "\n\n".join(p.page_content for p in pages)
    merged = Document(
        page_content=combined_text,
        metadata={
            "page":   f"1-{pages[-1].metadata.get('page', len(pages))}",
            "source": pages[0].metadata.get("source", ""),
        },
    )
    logger.info(
        f"[pdf_utils] merge_pages: {len(combined_text):,} chars "
        f"(~{len(combined_text) // CHUNK_SIZE + 1} chunks)"
    )
    return [merged]


def split_documents(pages: list[Document]) -> list[Document]:
    """Merge all pages then split into CHUNK_SIZE chunks."""
    merged      = merge_pages(pages)
    total_chars = sum(len(p.page_content) for p in merged)
    logger.info(
        f"[pdf_utils] split_documents: {total_chars:,} chars, "
        f"chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}"
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    t_split = time.perf_counter()
    chunks  = splitter.split_documents(merged)
    avg     = sum(len(c.page_content) for c in chunks) / len(chunks) if chunks else 0
    logger.info(
        f"[pdf_utils] split_documents: {len(chunks)} chunk(s) in "
        f"{time.perf_counter() - t_split:.3f}s (avg {avg:.0f} chars/chunk)"
    )
    logger.info(
        f"[pdf_utils] Coverage: ALL {len(pages)} pages analysed "
        f"across {len(chunks)} inference call(s)"
    )
    return chunks


def get_page_count(file_path: str) -> int:
    """Return page count without fully loading the PDF."""
    try:
        with fitz.open(file_path) as doc:
            count = len(doc)
            logger.debug(f"[pdf_utils] get_page_count: {count} page(s)")
            return count
    except Exception as e:
        logger.error(f"[pdf_utils] get_page_count failed: {e}")
        return 0