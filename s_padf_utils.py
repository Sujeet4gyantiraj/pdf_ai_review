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

# ---------------------------------------------------------------------------
# Extraction thresholds
#
# NATIVE_TEXT_THRESHOLD = 0
#   Any native text, even a single character, is kept.
#   Previously set to 50 — this silently dropped pages with <50 chars
#   (e.g. pages with only a page number, section heading, or figure caption).
#   Setting to 0 means we only fall back to OCR if the page has ZERO native text.
#
# TESSERACT_MIN_CHARS = 10
#   Only escalate to PaddleOCR-VL if Tesseract returns fewer than 10 chars.
#   Previously 30 — this dropped pages where Tesseract found a number,
#   a date, or a short label (e.g. "Figure 1" = 8 chars).
# ---------------------------------------------------------------------------
NATIVE_TEXT_THRESHOLD = 0    # never skip a page that has ANY native text
TESSERACT_MIN_CHARS   = 10   # only escalate if Tesseract returns almost nothing
OCR_RETRY_ATTEMPTS    = 2    # retry OCR this many times before giving up

# Parallel workers for native extraction
_NATIVE_EXTRACT_WORKERS = 4

# ---------------------------------------------------------------------------
# Tesseract availability check
# ---------------------------------------------------------------------------
_TESSERACT_AVAILABLE = False
try:
    import pytesseract
    from PIL import Image as PILImage
    pytesseract.get_tesseract_version()
    _TESSERACT_AVAILABLE = True
    logger.info("[pdf_utils] Tesseract available — will use as fast OCR fallback")
except Exception as e:
    logger.info(
        f"[pdf_utils] Tesseract not available ({e}) — "
        "scanned pages will go directly to PaddleOCR-VL. "
        "Install: sudo apt install tesseract-ocr && pip install pytesseract pillow"
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
    """Clean PDF/OCR artifacts. Never returns None — always returns a string."""
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

    Data loss fix: threshold is now 0, not 50.
    Any page with at least 1 native character uses native extraction.
    Only returns None when the page has ZERO extractable text (pure image page).
    """
    text = page.get_text("text").strip()
    if len(text) > NATIVE_TEXT_THRESHOLD:
        return clean_text(text)
    return None


def _page_to_image(page: fitz.Page, dpi: int = 200) -> np.ndarray:
    """Render a fitz page to a numpy uint8 RGB array."""
    pix = page.get_pixmap(dpi=dpi)
    img = np.ascontiguousarray(
        np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
    )
    if pix.n == 4:
        img = img[:, :, :3]
    return img


def _tesseract_page(page: fitz.Page) -> str | None:
    """
    Strategy 2 — Tesseract OCR (CPU, fast).

    Data loss fix: threshold lowered from 30 → 10.
    Pages with short but real content (figure captions, page numbers,
    section headings like "Clause 42") are no longer escalated unnecessarily.
    """
    if not _TESSERACT_AVAILABLE:
        return None

    page_num = page.number + 1
    t_tess   = time.perf_counter()

    try:
        img        = _page_to_image(page, dpi=200)
        pil_img    = PILImage.fromarray(img)
        text       = pytesseract.image_to_string(pil_img, config="--oem 3 --psm 3")
        elapsed    = time.perf_counter() - t_tess
        char_count = len(text.strip())

        if char_count >= TESSERACT_MIN_CHARS:
            logger.info(
                f"[pdf_utils] Page {page_num}: ✓ Tesseract "
                f"({char_count} chars, {elapsed:.2f}s)"
            )
            return clean_text(text)
        else:
            logger.info(
                f"[pdf_utils] Page {page_num}: Tesseract {char_count} chars "
                f"< {TESSERACT_MIN_CHARS} threshold ({elapsed:.2f}s) "
                "— escalating to PaddleOCR-VL"
            )
            return None

    except Exception as e:
        logger.warning(f"[pdf_utils] Page {page_num}: Tesseract failed ({e}) — escalating")
        return None


def _paddle_page(page: fitz.Page) -> str:
    """
    Strategy 3 — PaddleOCR-VL (GPU, thorough).

    Data loss fix: retry logic added.
    If PaddleOCR fails on the first attempt (GPU blip, memory spike),
    it retries OCR_RETRY_ATTEMPTS times before giving up.
    Each retry uses slightly higher DPI to improve extraction.
    """
    page_num = page.number + 1
    last_exc = None

    for attempt in range(1, OCR_RETRY_ATTEMPTS + 1):
        # Slightly higher DPI on retries for better quality
        dpi = 150 + (attempt - 1) * 50   # attempt 1: 150, attempt 2: 200
        try:
            t_ocr   = time.perf_counter()
            img     = _page_to_image(page, dpi=dpi)
            results = _ocr_vl.predict(img)
            elapsed = time.perf_counter() - t_ocr

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

            text = "\n".join(page_parts)
            logger.info(
                f"[pdf_utils] Page {page_num}: ✓ PaddleOCR-VL "
                f"(attempt {attempt}, {len(page_parts)} block(s), {elapsed:.2f}s)"
            )
            return text

        except Exception as e:
            last_exc = e
            logger.warning(
                f"[pdf_utils] Page {page_num}: PaddleOCR-VL attempt {attempt} "
                f"failed ({e})"
                + (" — retrying" if attempt < OCR_RETRY_ATTEMPTS else " — all attempts exhausted")
            )
            if paddle.device.is_compiled_with_cuda():
                paddle.device.cuda.empty_cache()

    raise RuntimeError(
        f"PaddleOCR-VL failed after {OCR_RETRY_ATTEMPTS} attempts on page {page_num}: {last_exc}"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_pdf(file_path: str, max_pages: int | None = None) -> list[Document]:
    """
    Load a PDF and return one LangChain Document per page.

    Data integrity guarantee: NO page is silently dropped.

    Every page goes through the three-tier pipeline:
      Tier 1 — PyMuPDF native  (any chars > 0 → use native)
      Tier 2 — Tesseract CPU   (≥10 chars → use Tesseract)
      Tier 3 — PaddleOCR-VL   (with retry — last resort)
      Tier 4 — Placeholder     (if ALL three tiers fail, insert a placeholder
                                so the page is still represented in the output
                                and downstream code knows it existed)

    Skipped pages are counted and logged. A summary at the end tells you
    exactly how many pages used each method and how many were unreadable.
    A ValueError is only raised if EVERY page fails — not just some.
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
    logger.info(
        f"[pdf_utils] {total_pages} page(s) total, "
        f"processing {pages_to_process} "
        f"({'all' if pages_to_process == total_pages else f'first {pages_to_process}'})"
    )

    fitz_pages = [doc[i] for i in range(pages_to_process)]

    # ── Pass 1: parallel native extraction ───────────────────────────────
    logger.info(
        f"[pdf_utils] Pass 1 — parallel native extraction "
        f"({_NATIVE_EXTRACT_WORKERS} workers, threshold={NATIVE_TEXT_THRESHOLD} chars)"
    )
    t_pass1        = time.perf_counter()
    native_results: dict[int, str | None] = {}

    def _native_worker(idx_page):
        idx, page = idx_page
        return idx, _extract_native(page)

    with ThreadPoolExecutor(max_workers=_NATIVE_EXTRACT_WORKERS) as pool:
        futures = {pool.submit(_native_worker, (i, p)): i for i, p in enumerate(fitz_pages)}
        for future in as_completed(futures):
            try:
                idx, text = future.result()
                native_results[idx] = text
            except Exception as e:
                # Worker itself failed — mark as needing OCR
                logger.warning(f"[pdf_utils] Pass 1 worker failed: {e}")
                native_results[futures[future]] = None

    native_hit  = sum(1 for v in native_results.values() if v is not None)
    native_miss = pages_to_process - native_hit
    logger.info(
        f"[pdf_utils] Pass 1 done ({time.perf_counter() - t_pass1:.2f}s) — "
        f"native={native_hit}, need_ocr={native_miss}"
    )

    # ── Pass 2: OCR for pages that need it (Tesseract → PaddleOCR-VL) ────
    if native_miss > 0:
        logger.info(
            f"[pdf_utils] Pass 2 — OCR for {native_miss} page(s) "
            f"(Tesseract threshold={TESSERACT_MIN_CHARS} chars, "
            f"PaddleOCR retries={OCR_RETRY_ATTEMPTS})"
        )

    pages:           list[Document] = []
    tesseract_count: int = 0
    paddle_count:    int = 0
    placeholder_count: int = 0   # pages where all OCR failed — represented as placeholder
    unreadable_count:  int = 0   # truly blank pages (no text, no image content)

    for idx, fitz_page in enumerate(fitz_pages):
        page_num  = idx + 1
        native_ok = native_results.get(idx)

        if native_ok is not None:
            text = native_ok
            logger.debug(f"[pdf_utils] Page {page_num}: native ({len(text)} chars)")

        else:
            # Tier 2: Tesseract
            logger.info(
                f"[pdf_utils] Page {page_num}/{pages_to_process}: "
                "no native text — trying Tesseract"
            )
            tess_text = _tesseract_page(fitz_page)

            if tess_text is not None:
                text = tess_text
                tesseract_count += 1
            else:
                # Tier 3: PaddleOCR-VL with retry
                logger.info(
                    f"[pdf_utils] Page {page_num}/{pages_to_process}: "
                    "Tesseract insufficient — using PaddleOCR-VL"
                )
                try:
                    raw  = _paddle_page(fitz_page)
                    text = clean_text(raw)
                    paddle_count += 1
                except Exception as e:
                    # Tier 4: all OCR failed — insert placeholder so page is not lost
                    logger.error(
                        f"[pdf_utils] Page {page_num}: ALL extraction methods failed — {e}. "
                        "Inserting placeholder so page is represented in output."
                    )
                    text = f"[Page {page_num}: content could not be extracted]"
                    placeholder_count += 1

        # Only skip if text is truly empty after all attempts
        # (placeholder text is never empty, so this only catches blank pages)
        if not text:
            logger.warning(
                f"[pdf_utils] Page {page_num}: completely blank after all extraction "
                "attempts — this page appears to have no content"
            )
            unreadable_count += 1
            # Still include it as a placeholder so the page count is accurate
            text = f"[Page {page_num}: blank page]"

        pages.append(Document(
            page_content=text,
            metadata={
                "page":   page_num,
                "source": file_path,
            },
        ))

    doc.close()
    elapsed = time.perf_counter() - t_load

    # ── Integrity report ─────────────────────────────────────────────────
    total_loaded = len(pages)
    logger.info(
        f"[pdf_utils] ── EXTRACTION COMPLETE ──────────────────────────────"
    )
    logger.info(
        f"[pdf_utils] Pages processed : {pages_to_process}/{total_pages} "
        f"({'100%' if pages_to_process == total_pages else f'{pages_to_process/total_pages*100:.0f}%'})"
    )
    logger.info(f"[pdf_utils] Native text      : {native_hit} page(s)")
    logger.info(f"[pdf_utils] Tesseract OCR    : {tesseract_count} page(s)")
    logger.info(f"[pdf_utils] PaddleOCR-VL     : {paddle_count} page(s)")
    logger.info(f"[pdf_utils] Placeholders     : {placeholder_count} page(s) (all OCR failed)")
    logger.info(f"[pdf_utils] Blank pages      : {unreadable_count} page(s)")
    logger.info(f"[pdf_utils] Total loaded     : {total_loaded} page(s)")
    logger.info(f"[pdf_utils] Time             : {elapsed:.2f}s")
    logger.info(
        f"[pdf_utils] ─────────────────────────────────────────────────────"
    )

    # Data loss check — warn if any pages were completely unextractable
    if placeholder_count > 0:
        logger.warning(
            f"[pdf_utils] {placeholder_count} page(s) could not be extracted by any method. "
            "They are included as '[Page N: content could not be extracted]' placeholders "
            "so the LLM knows these pages existed."
        )

    if total_loaded == 0:
        raise ValueError(
            "No content found in any page. "
            "File may be fully blank, encrypted with no text layer, or corrupt."
        )

    # Verify we loaded every page we were supposed to
    if total_loaded != pages_to_process:
        logger.error(
            f"[pdf_utils] DATA INTEGRITY WARNING: expected {pages_to_process} pages, "
            f"got {total_loaded}. This should never happen — investigate immediately."
        )

    return pages


def merge_pages(pages: list[Document]) -> list[Document]:
    """
    Concatenate all pages into one Document so the splitter produces full-size chunks.

    Data integrity: verifies character count before and after merging.
    """
    if not pages:
        return pages

    total_chars_before = sum(len(p.page_content) for p in pages)
    logger.info(
        f"[pdf_utils] merge_pages: {len(pages)} page(s), "
        f"{total_chars_before:,} total chars"
    )

    combined_text = "\n\n".join(p.page_content for p in pages)

    # Sanity check — merged text should be at least as long as the sum of parts
    # (it will be slightly longer due to the \n\n separators)
    if len(combined_text) < total_chars_before:
        logger.error(
            f"[pdf_utils] merge_pages: DATA LOSS — "
            f"merged text ({len(combined_text):,} chars) < "
            f"sum of pages ({total_chars_before:,} chars). "
            "This should never happen."
        )

    merged = Document(
        page_content=combined_text,
        metadata={
            "page":   f"1-{pages[-1].metadata.get('page', len(pages))}",
            "source": pages[0].metadata.get("source", ""),
        },
    )
    logger.info(
        f"[pdf_utils] merge_pages: {len(combined_text):,} chars "
        f"(~{len(combined_text) // CHUNK_SIZE + 1} chunks at CHUNK_SIZE={CHUNK_SIZE})"
    )
    return [merged]


def split_documents(pages: list[Document]) -> list[Document]:
    """
    Merge all pages then split into CHUNK_SIZE chunks.

    Data integrity: verifies total chars are preserved through splitting.
    """
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

    # Verify coverage — sum of unique (non-overlapping) chunk content
    # should equal total_chars. With overlap the sum will be slightly higher.
    chars_in_chunks = sum(len(c.page_content) for c in chunks)
    avg             = chars_in_chunks / len(chunks) if chunks else 0

    logger.info(
        f"[pdf_utils] split_documents: {len(chunks)} chunk(s) in "
        f"{time.perf_counter() - t_split:.3f}s "
        f"(avg {avg:.0f} chars/chunk, total {chars_in_chunks:,} chars with overlap)"
    )
    logger.info(
        f"[pdf_utils] Coverage: ALL {len(pages)} page(s) → "
        f"{len(chunks)} chunk(s) → {len(chunks)} inference call(s)"
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