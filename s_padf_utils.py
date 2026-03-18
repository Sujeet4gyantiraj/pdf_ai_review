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
# Larger chunks = fewer inference calls = faster total time.
# Safe values given ~17 GB VRAM free after PaddleOCR-VL loads:
#   CHUNK_SIZE=12000 → ~4000 tokens per chunk, well inside Mistral's 8192 limit
# ---------------------------------------------------------------------------
CHUNK_SIZE    = 8000   # ~2500 tokens — fits well in Mistral's context with prompt overhead
CHUNK_OVERLAP = 200    # small overlap is enough between merged page chunks

# Pre-compile regex patterns once at import time — not on every clean_text call
_RE_HYPHEN    = re.compile(r"-\n")
_RE_MULTILINE = re.compile(r"\n{3,}")
_RE_SPACES    = re.compile(r" {2,}")

# Native text threshold — pages with fewer chars than this fall back to OCR
NATIVE_TEXT_THRESHOLD = 50

# Max worker threads for parallel native-text extraction.
# OCR pages are still processed serially (GPU serialises them anyway).
_NATIVE_EXTRACT_WORKERS = 4

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
    """
    Clean common PDF/OCR extraction artifacts using pre-compiled patterns.
    Pre-compiling at module level removes per-call regex compilation overhead.
    """
    before = len(text)
    text = _RE_HYPHEN.sub("",    text)
    text = _RE_MULTILINE.sub("\n\n", text)
    text = _RE_SPACES.sub(" ",   text)
    text = text.strip()
    logger.debug(f"[pdf_utils] clean_text: {before} → {len(text)} chars")
    return text


def _extract_native(page: fitz.Page) -> str | None:
    """
    Extract native text from one page.
    Returns the cleaned text if it meets the threshold, else None.
    This function is safe to call from a thread pool.
    """
    text = page.get_text("text").strip()
    if len(text) >= NATIVE_TEXT_THRESHOLD:
        return clean_text(text)
    return None


def _ocr_page(page: fitz.Page) -> str:
    """
    Run PaddleOCR-VL on a single page.
    Speed optimisation: use np.ascontiguousarray to ensure the pixmap buffer
    is C-contiguous before passing to PaddleOCR — avoids an internal copy.
    """
    page_num = page.number + 1

    t_pix = time.perf_counter()
    pix   = page.get_pixmap(dpi=150)
    logger.debug(
        f"[pdf_utils] Page {page_num}: pixmap {pix.w}x{pix.h}px "
        f"({time.perf_counter() - t_pix:.2f}s)"
    )

    # Speed: ascontiguousarray avoids an internal copy inside PaddleOCR
    img = np.ascontiguousarray(
        np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
    )
    if pix.n == 4:
        img = img[:, :, :3]

    t_ocr   = time.perf_counter()
    results = _ocr_vl.predict(img)
    logger.info(
        f"[pdf_utils] Page {page_num}: OCR done "
        f"({time.perf_counter() - t_ocr:.2f}s), {len(results)} block(s)"
    )

    page_parts = []
    for res in results:
        if hasattr(res, "res"):
            page_parts.append(res.res)
        elif isinstance(res, dict) and "res" in res:
            page_parts.append(res["res"])
        else:
            page_parts.append(str(res))

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

    Speed optimisation — two-pass parallel extraction:
      Pass 1 (parallel):  run native PyMuPDF extraction on ALL pages
                          simultaneously using a thread pool. fitz is
                          thread-safe for read operations.
      Pass 2 (serial):    for pages that failed the native threshold,
                          run PaddleOCR-VL serially (GPU serialises anyway).

    This means a 100-page native-text PDF completes extraction in the time
    it takes to extract ~25 pages serially — a ~4x speedup on that step.
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

    # ── Pass 1: parallel native extraction ───────────────────────────────
    # Collect all fitz.Page objects first (must happen inside doc context)
    fitz_pages = [doc[i] for i in range(pages_to_process)]

    native_results: dict[int, str | None] = {}

    def _native_worker(idx_page):
        idx, page = idx_page
        return idx, _extract_native(page)

    logger.info(f"[pdf_utils] Pass 1: parallel native extraction ({_NATIVE_EXTRACT_WORKERS} workers)")
    t_pass1 = time.perf_counter()

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

    # ── Pass 2: serial OCR for pages that need it ─────────────────────────
    pages:     list[Document] = []
    ocr_count: int = 0
    skip_count: int = 0

    for idx, fitz_page in enumerate(fitz_pages):
        page_num  = idx + 1
        native_ok = native_results.get(idx)

        if native_ok is not None:
            text = native_ok
        else:
            logger.info(
                f"[pdf_utils] Page {page_num}/{pages_to_process}: "
                "native text short — running OCR-VL"
            )
            try:
                raw  = _ocr_page(fitz_page)
                text = clean_text(raw)
                ocr_count += 1
            except Exception as e:
                logger.error(f"[pdf_utils] Page {page_num}: OCR-VL failed — {e} — skipping")
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
        f"(native={native_hit}, ocr={ocr_count}, skipped={skip_count})"
    )

    if not pages:
        raise ValueError(
            "No extractable text found. "
            "File may be image-only, encrypted, or empty."
        )

    return pages


def merge_pages(pages: list[Document]) -> list[Document]:
    """
    Concatenate adjacent pages into larger combined Documents before splitting.

    Why this matters:
      Your Finance Bill has 232 pages averaging ~1900 chars each.
      With CHUNK_SIZE=8000 the splitter should merge ~4 pages per chunk.
      But RecursiveCharacterTextSplitter only splits — it never MERGES
      documents that are already smaller than chunk_size.
      So 232 pages → 232 tiny chunks → 232 map calls → hours of inference.

    This function concatenates all page text into a single Document first,
    then split_documents will correctly break it into CHUNK_SIZE pieces,
    each spanning multiple pages.

    Result: 232 pages × ~1900 chars = ~440K chars ÷ 8000 = ~55 chunks.
    With MAX_CHUNKS=10, the first 10 chunks cover ~80K chars = ~40 pages.
    """
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
        f"[pdf_utils] merge_pages: combined {len(combined_text):,} chars "
        f"(will split into ~{len(combined_text) // CHUNK_SIZE + 1} chunks)"
    )
    return [merged]


def split_documents(pages: list[Document]) -> list[Document]:
    """
    Merge all pages into one document then split into CHUNK_SIZE chunks.

    This ensures small pages are combined rather than kept as tiny individual
    chunks, which would waste inference calls on near-empty content.
    """
    # Merge pages first so the splitter produces full-size chunks
    merged = merge_pages(pages)

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

    return chunks


def get_page_count(file_path: str) -> int:
    """Return the number of pages without fully loading the PDF."""
    try:
        with fitz.open(file_path) as doc:
            count = len(doc)
            logger.debug(f"[pdf_utils] get_page_count: {count} page(s) in '{file_path}'")
            return count
    except Exception as e:
        logger.error(f"[pdf_utils] get_page_count failed: {e}")
        return 0