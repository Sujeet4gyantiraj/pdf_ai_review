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
CHUNK_SIZE    = 12000  # ~5640 tokens for dense legal text — safe under 7500 token limit
CHUNK_OVERLAP = 200

# Pre-compiled regex patterns
_RE_HYPHEN    = re.compile(r"-\n")
_RE_MULTILINE = re.compile(r"\n{3,}")
_RE_SPACES    = re.compile(r" {2,}")

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
NATIVE_TEXT_THRESHOLD = 0   # any native char -> use native; only 0 falls back to OCR
OCR_RETRY_ATTEMPTS    = 2   # PaddleOCR-VL retries before inserting placeholder

# Parallel workers for native extraction (fitz is thread-safe for reads)
_NATIVE_EXTRACT_WORKERS = 4

# How many pages to sample when detecting PDF type
_PDF_TYPE_SAMPLE_PAGES = 5

# ---------------------------------------------------------------------------
# PaddleOCR-VL — loaded once at import time
# ---------------------------------------------------------------------------
logger.info("[pdf_utils] Loading PaddleOCR-VL 1.5 ...")
t0      = time.perf_counter()
_ocr_vl = PaddleOCRVL("v1.5")
logger.info(f"[pdf_utils] PaddleOCR-VL ready ({time.perf_counter() - t0:.2f}s)")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Clean PDF/OCR artifacts. Never returns None."""
    before = len(text)
    text = _RE_HYPHEN.sub("",        text)
    text = _RE_MULTILINE.sub("\n\n", text)
    text = _RE_SPACES.sub(" ",       text)
    text = text.strip()
    logger.debug(f"[pdf_utils] clean_text: {before} -> {len(text)} chars")
    return text


def _detect_pdf_type(doc: fitz.Document, pages_to_process: int) -> str:
    """
    Sample the first N pages to classify the PDF before extraction begins.

    Returns:
      'native'     - all sampled pages have text  -> text-based PDF
      'image_only' - NO sampled pages have text   -> fully scanned/image PDF
      'mixed'      - some have text, some do not  -> hybrid PDF
    """
    sample       = min(_PDF_TYPE_SAMPLE_PAGES, pages_to_process)
    native_count = 0

    for i in range(sample):
        text = doc[i].get_text("text").strip()
        if len(text) > 0:
            native_count += 1

    if native_count == sample:
        pdf_type = "native"
    elif native_count == 0:
        pdf_type = "image_only"
    else:
        pdf_type = "mixed"

    logger.info(
        f"[pdf_utils] PDF type: sampled {sample} page(s), "
        f"{native_count} had native text -> '{pdf_type}'"
    )
    return pdf_type


def _extract_native(page: fitz.Page) -> str | None:
    """Strategy 1 - PyMuPDF native. Returns text if any chars present, else None."""
    text = page.get_text("text").strip()
    if len(text) > NATIVE_TEXT_THRESHOLD:
        return clean_text(text)
    return None


def _page_to_image(page: fitz.Page, dpi: int = 150) -> np.ndarray:
    """Render a fitz page to a C-contiguous uint8 RGB numpy array."""
    pix = page.get_pixmap(dpi=dpi)
    img = np.ascontiguousarray(
        np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
    )
    if pix.n == 4:
        img = img[:, :, :3]
    return img


# ---------------------------------------------------------------------------
# Blank-PDF detection
# ---------------------------------------------------------------------------

# Matches the placeholder strings inserted by load_pdf for unreadable/blank pages
_PLACEHOLDER_RE = re.compile(
    r"^\[Page \d+: (blank page|content could not be extracted)\]$"
)


def all_pages_blank(pages: list[Document]) -> bool:
    """
    Return True if every page in the list is a blank/placeholder page —
    i.e. no real content was extracted from the PDF at all.

    Blank pages are those whose content either:
      - is empty after stripping, or
      - matches the placeholder patterns inserted by load_pdf:
          "[Page N: blank page]"
          "[Page N: content could not be extracted]"

    Called by s_main.py after load_pdf() to short-circuit inference
    and return empty fields immediately without hitting the model.
    """
    if not pages:
        return True
    for page in pages:
        text = page.page_content.strip()
        if text and not _PLACEHOLDER_RE.match(text):
            return False  # at least one page with real content
    return True


# ---------------------------------------------------------------------------
# OCR result text extractor
# ---------------------------------------------------------------------------

# These keys appear in PaddleOCR-VL's internal result objects.
# Only "rec_text" and "text" carry actual recognised text — all others
# are metadata, numpy arrays, or model settings that must be ignored.
_OCR_TEXT_KEYS = ("rec_text", "text")

# Heuristic: if the extracted string contains these substrings it is almost
# certainly PaddleOCR's internal debug repr rather than document text.
_OCR_JUNK_MARKERS = (
    "numpy.ndarray",
    "layout_det",
    "rec_score",
    "table_res",
    "input_path",
    "model_settings",
    "parsing_res",
    "spotting_res",
    "page_id",
)


def _extract_ocr_text(res) -> str:
    """
    Safely extract the human-readable OCR text from a single PaddleOCR-VL
    result item, ignoring internal metadata/debug fields.

    PaddleOCR-VL result objects can be:
      - A plain string                         → return as-is if non-empty
      - A dict with "rec_text" or "text" key   → return that value
      - An object with .rec_text / .text attr  → return that value
      - An object with .res attr               → recurse into .res
      - A dict  with "res"  key               → recurse into res value
      - Anything else                          → return "" (never call str())

    The critical rule: **never fall back to str(res)**.
    str() on a PaddleOCR result object serialises the entire internal state
    (numpy arrays, model config, layout boxes …) and that junk text gets
    passed to the LLM which then happily "summarises" it.
    """
    if res is None:
        return ""

    # Plain string — sanity-check it isn't a debug repr
    if isinstance(res, str):
        s = res.strip()
        if any(marker in s for marker in _OCR_JUNK_MARKERS):
            logger.debug(f"_extract_ocr_text: discarding junk string ({s[:60]!r}…)")
            return ""
        return s

    # Dict — look for known text keys first, then recurse into "res"
    if isinstance(res, dict):
        for key in _OCR_TEXT_KEYS:
            if key in res:
                val = res[key]
                return val.strip() if isinstance(val, str) else ""
        if "res" in res:
            return _extract_ocr_text(res["res"])
        return ""

    # Object — check .rec_text / .text attributes first
    for key in _OCR_TEXT_KEYS:
        if hasattr(res, key):
            val = getattr(res, key)
            return val.strip() if isinstance(val, str) else ""

    # Object with .res — recurse
    if hasattr(res, "res"):
        return _extract_ocr_text(res.res)

    # Unknown type — log and discard rather than calling str()
    logger.debug(f"_extract_ocr_text: unrecognised result type {type(res).__name__!r} — skipping")
    return ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_pdf(file_path: str, max_pages: int | None = None) -> list[Document]:
    """
    Load a PDF and return one LangChain Document per page.

    Extraction pipeline per page:

      Tier 1 — PyMuPDF native  (parallel, ~0.01s/page, zero GPU cost)
                Any page with native text uses this path.
        |
        v  only if page has ZERO native text
      Tier 2 — PaddleOCR-VL   (serial GPU, ~1-3s/page, with retry)
                Handles scanned pages, images, complex layouts.
        |
        v  only if ALL OCR attempts fail
      Tier 3 — Placeholder     ("[Page N: content could not be extracted]")
                Page is still included so page count stays accurate.

    PDF type detection:
      'native'     -> Pass 1 (parallel native) only
      'image_only' -> Pass 1 skipped, all pages go to PaddleOCR-VL
      'mixed'      -> Pass 1 for text pages, PaddleOCR-VL for image pages

    Data integrity: every page is always returned. No page is silently dropped.
    """
    logger.info(
        f"[pdf_utils] load_pdf: '{file_path}'"
        + (f" max_pages={max_pages}" if max_pages else "")
    )
    t_load = time.perf_counter()

    try:
        doc = fitz.open(file_path)
    except Exception as e:
        logger.error(f"[pdf_utils] load_pdf: open failed -- {e}")
        raise ValueError(f"Could not open PDF: {e}")

    total_pages      = len(doc)
    pages_to_process = total_pages if max_pages is None else min(total_pages, max_pages)
    logger.info(
        f"[pdf_utils] {total_pages} total page(s), "
        f"processing {pages_to_process} "
        f"({'all' if pages_to_process == total_pages else f'first {pages_to_process}'})"
    )

    # Detect PDF type upfront
    pdf_type   = _detect_pdf_type(doc, pages_to_process)
    fitz_pages = [doc[i] for i in range(pages_to_process)]

    # ── Pass 1: parallel native extraction (skipped for image_only) ───────
    native_results: dict[int, str | None] = {}

    if pdf_type == "image_only":
        logger.info(
            "[pdf_utils] Image-only PDF -- skipping native pass, "
            "all pages going to PaddleOCR-VL"
        )
        native_results = {i: None for i in range(pages_to_process)}
    else:
        logger.info(
            f"[pdf_utils] Pass 1 -- parallel native extraction "
            f"({_NATIVE_EXTRACT_WORKERS} workers)"
        )
        t_pass1 = time.perf_counter()

        def _native_worker(idx_page):
            idx, page = idx_page
            return idx, _extract_native(page)

        with ThreadPoolExecutor(max_workers=_NATIVE_EXTRACT_WORKERS) as pool:
            futures = {
                pool.submit(_native_worker, (i, p)): i
                for i, p in enumerate(fitz_pages)
            }
            for future in as_completed(futures):
                try:
                    idx, text = future.result()
                    native_results[idx] = text
                except Exception as e:
                    logger.warning(f"[pdf_utils] Pass 1 worker failed: {e}")
                    native_results[futures[future]] = None

        native_hit = sum(1 for v in native_results.values() if v is not None)
        logger.info(
            f"[pdf_utils] Pass 1 done ({time.perf_counter() - t_pass1:.2f}s) -- "
            f"native={native_hit}, need_ocr={pages_to_process - native_hit}"
        )

    native_hit    = sum(1 for v in native_results.values() if v is not None)
    paddle_needed = [i for i, v in native_results.items() if v is None]

    if paddle_needed:
        logger.info(
            f"[pdf_utils] Pass 2 -- PaddleOCR-VL (GPU) for {len(paddle_needed)} page(s)"
        )

    # ── Pass 2: PaddleOCR-VL for all image pages ──────────────────────────
    pre_rendered: dict[int, np.ndarray] = {}

    if paddle_needed:
        logger.info(
            f"[pdf_utils] Pre-rendering {len(paddle_needed)} page image(s) "
            f"in parallel ({_NATIVE_EXTRACT_WORKERS} CPU workers) ..."
        )
        t_render = time.perf_counter()

        def _render_worker(idx: int) -> tuple[int, np.ndarray]:
            return idx, _page_to_image(fitz_pages[idx], dpi=150)

        with ThreadPoolExecutor(max_workers=_NATIVE_EXTRACT_WORKERS) as pool:
            futures = {pool.submit(_render_worker, i): i for i in paddle_needed}
            for future in as_completed(futures):
                try:
                    idx, img = future.result()
                    pre_rendered[idx] = img
                except Exception as e:
                    logger.warning(f"[pdf_utils] Pre-render failed for page {futures[future]+1}: {e}")

        logger.info(
            f"[pdf_utils] Pre-render done ({time.perf_counter()-t_render:.2f}s) — "
            f"{len(pre_rendered)}/{len(paddle_needed)} images ready"
        )

    paddle_results: dict[int, str] = {}

    for idx in paddle_needed:
        fitz_page = fitz_pages[idx]
        page_num  = idx + 1
        pre_img   = pre_rendered.get(idx)
        last_exc  = None

        for attempt in range(1, OCR_RETRY_ATTEMPTS + 1):
            if attempt == 1 and pre_img is not None:
                img = pre_img
                dpi = 150
            else:
                dpi = 150 + (attempt - 1) * 50
                img = _page_to_image(fitz_page, dpi=dpi)

            try:
                t_ocr   = time.perf_counter()
                results = _ocr_vl.predict(img)
                elapsed = time.perf_counter() - t_ocr

                page_parts = []
                for res in results:
                    extracted = _extract_ocr_text(res)
                    if extracted:
                        page_parts.append(extracted)

                del results
                if paddle.device.is_compiled_with_cuda():
                    paddle.device.cuda.empty_cache()

                text = clean_text("\n".join(page_parts))

                # If OCR returned nothing meaningful, treat as blank page rather
                # than storing an empty string that could confuse downstream logic.
                if not text:
                    logger.info(
                        f"[pdf_utils] Page {page_num}: PaddleOCR-VL returned no text "
                        f"(attempt {attempt}, dpi={dpi}, {elapsed:.2f}s) — blank page"
                    )
                    paddle_results[idx] = f"[Page {page_num}: blank page]"
                else:
                    logger.info(
                        f"[pdf_utils] Page {page_num}: PaddleOCR-VL OK "
                        f"(attempt {attempt}, dpi={dpi}, {len(page_parts)} block(s), {elapsed:.2f}s)"
                    )
                    paddle_results[idx] = text
                break

            except Exception as e:
                last_exc = e
                logger.warning(
                    f"[pdf_utils] Page {page_num}: PaddleOCR-VL attempt {attempt} failed ({e})"
                    + (" -- retrying" if attempt < OCR_RETRY_ATTEMPTS else " -- exhausted")
                )
                if paddle.device.is_compiled_with_cuda():
                    paddle.device.cuda.empty_cache()
        else:
            logger.error(f"[pdf_utils] Page {page_num}: all OCR attempts failed -- placeholder")
            paddle_results[idx] = f"[Page {page_num}: content could not be extracted]"

    # ── Assemble results in page order ────────────────────────────────────
    pages:             list[Document] = []
    paddle_count:      int = 0
    placeholder_count: int = 0
    blank_count:       int = 0

    for idx, fitz_page in enumerate(fitz_pages):
        page_num = idx + 1

        if native_results.get(idx) is not None:
            text = native_results[idx]
            logger.debug(f"[pdf_utils] Page {page_num}: native ({len(text)} chars)")

        elif idx in paddle_needed:
            text = paddle_results[idx]
            if text.startswith("[Page "):
                placeholder_count += 1
            else:
                paddle_count += 1

        else:
            logger.error(f"[pdf_utils] Page {page_num}: no result -- placeholder")
            text = f"[Page {page_num}: content could not be extracted]"
            placeholder_count += 1

        if not text:
            logger.warning(f"[pdf_utils] Page {page_num}: blank after extraction")
            text = f"[Page {page_num}: blank page]"
            blank_count += 1

        pages.append(Document(
            page_content=text,
            metadata={"page": page_num, "source": file_path},
        ))

    doc.close()
    elapsed      = time.perf_counter() - t_load
    total_loaded = len(pages)

    # ── Integrity report ──────────────────────────────────────────────────
    logger.info("[pdf_utils] -- EXTRACTION COMPLETE --------------------------")
    logger.info(f"[pdf_utils] PDF type         : {pdf_type}")
    logger.info(f"[pdf_utils] Pages processed  : {pages_to_process}/{total_pages}")
    logger.info(f"[pdf_utils] Native text       : {native_hit} page(s)")
    logger.info(f"[pdf_utils] PaddleOCR-VL      : {paddle_count} page(s)")
    logger.info(f"[pdf_utils] Placeholders      : {placeholder_count} page(s)")
    logger.info(f"[pdf_utils] Blank pages        : {blank_count} page(s)")
    logger.info(f"[pdf_utils] Total loaded      : {total_loaded} page(s)")
    logger.info(f"[pdf_utils] Time              : {elapsed:.2f}s")
    logger.info("[pdf_utils] ------------------------------------------------------")

    if placeholder_count > 0:
        logger.warning(
            f"[pdf_utils] {placeholder_count} page(s) could not be extracted -- "
            "included as placeholders"
        )

    if total_loaded != pages_to_process:
        logger.error(
            f"[pdf_utils] DATA INTEGRITY: expected {pages_to_process}, got {total_loaded}"
        )

    if total_loaded == 0:
        raise ValueError(
            "No content extracted from any page. "
            "File may be blank, encrypted, or corrupt."
        )

    return pages


def merge_pages(pages: list[Document]) -> list[Document]:
    """Concatenate all pages into one Document so the splitter produces full-size chunks."""
    if not pages:
        return pages

    total_chars_before = sum(len(p.page_content) for p in pages)
    logger.info(
        f"[pdf_utils] merge_pages: {len(pages)} page(s), "
        f"{total_chars_before:,} total chars"
    )

    combined_text = "\n\n".join(p.page_content for p in pages)

    if len(combined_text) < total_chars_before:
        logger.error(
            f"[pdf_utils] merge_pages: DATA LOSS -- "
            f"merged ({len(combined_text):,}) < sum of pages ({total_chars_before:,})"
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
        f"[pdf_utils] Coverage: ALL {len(pages)} page(s) -> "
        f"{len(chunks)} inference call(s)"
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