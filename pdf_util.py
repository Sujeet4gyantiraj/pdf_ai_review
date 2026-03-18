import re
import fitz
import numpy as np
import paddle
import logging
from paddleocr import PaddleOCRVL

logger = logging.getLogger("pdf_util")

paddle.set_device("cpu")
logger.info("OCR forced to CPU")

ocr_vl = PaddleOCRVL("v1.5")
logger.info("OCR model loaded")


def clean_text(text: str) -> str:
    text = re.sub(r'-\n', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()


def extract_text_from_pdf(file_path: str) -> str:
    logger.info(f"Opening PDF: {file_path}")
    final_text = []

    with fitz.open(file_path) as doc:
        logger.info(f"PDF has {len(doc)} pages")

        for page in doc:
            logger.info(f"Processing page {page.number}")

            text = page.get_text("text")

            print(f"Extracting page {page.number} with native method...")

            if len(text.strip()) > 50:
                logger.info(f"Page {page.number} extracted using native method")
                final_text.append(text.strip())
                continue

            logger.info(f"Page {page.number} using OCR")

            pix = page.get_pixmap(dpi=150)

            img = np.frombuffer(
                pix.samples,
                dtype=np.uint8
            ).reshape(pix.h, pix.w, pix.n)

            if pix.n == 4:
                img = img[:, :, :3]

            results = ocr_vl.predict(img)

            page_parts = []
            for res in results:
                if hasattr(res, 'res'):
                    page_parts.append(res.res)
                elif isinstance(res, dict) and 'res' in res:
                    page_parts.append(res['res'])
                else:
                    page_parts.append(str(res))

            final_text.append("\n".join(page_parts))

            del results

    logger.info("PDF extraction completed.")
    return clean_text("\n".join(final_text))


def chunk_text(text: str, max_chars: int = 15000) -> list:
    logger.info("Starting text chunking")
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

    logger.info(f"Created {len(chunks)} chunks")
    return chunks

