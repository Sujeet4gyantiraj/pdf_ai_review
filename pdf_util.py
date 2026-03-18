import re
import fitz
import numpy as np
import paddle
from paddleocr import PaddleOCRVL

# Force OCR to CPU (VERY IMPORTANT)
paddle.set_device("cpu")

ocr_vl = PaddleOCRVL("v1.5")


def clean_text(text: str) -> str:
    text = re.sub(r'-\n', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()


def extract_text_from_pdf(file_path: str) -> str:
    final_text = []

    with fitz.open(file_path) as doc:
        for page in doc:

            text = page.get_text("text")

            if len(text.strip()) > 50:
                final_text.append(text.strip())
                continue

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

    return clean_text("\n".join(final_text))


def chunk_text(text: str, max_chars: int = 15000) -> list:

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