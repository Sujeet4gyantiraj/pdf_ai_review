import fitz
import logging
import numpy as np
import paddle
from paddleocr import PaddleOCR

logger = logging.getLogger(__name__)

# -------- PaddleOCR GPU --------
ocr = PaddleOCR(
    use_angle_cls=True,
    use_gpu=True,
    show_log=False
)

BATCH_SIZE = 4  # Adjust based on GPU VRAM


def extract_text_from_pdf(file_path):

    full_text = ""

    doc = fitz.open(file_path)

    # -------- Native Extraction --------
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text()

        if len(text.strip()) > 50:
            full_text += text + "\n"

    if len(full_text.strip()) > 200:
        logger.info("Native extraction sufficient.")
        doc.close()
        return full_text

    logger.warning("Switching to PaddleOCR batch mode...")

    # -------- Batch OCR --------
    images_batch = []
    full_text = ""

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap(dpi=200)

        img = np.frombuffer(pix.samples, dtype=np.uint8)
        img = img.reshape(pix.height, pix.width, pix.n)

        if pix.n == 4:
            img = img[:, :, :3]

        images_batch.append(img)

        if len(images_batch) == BATCH_SIZE:
            full_text += run_ocr_batch(images_batch)
            images_batch.clear()

    if images_batch:
        full_text += run_ocr_batch(images_batch)

    doc.close()
    return full_text.strip()


def run_ocr_batch(images):

    text_output = ""

    results = ocr.ocr(images, cls=True)

    for page_result in results:
        if not page_result:
            continue


            print(f"Extracting page {page.number} with native method...")

            if len(text.strip()) > 50:
                logger.info(f"Page {page.number} extracted using native method")
                final_text.append(text.strip())
                continue

        for line in page_result:
            text_output += line[1][0] + " "


        text_output += "\n"

    if paddle.device.is_compiled_with_cuda():
        paddle.device.cuda.empty_cache()

    return text_output


# -------- Token-based chunking --------
def chunk_by_tokens(text, tokenizer, max_tokens=1400):

    tokens = tokenizer.encode(text)
    chunks = []

    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i + max_tokens]
        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text)


    if current.strip():
        chunks.append(current.strip())


    return chunks

    logger.info(f"Created {len(chunks)} chunks")
    return chunks


    return chunks

