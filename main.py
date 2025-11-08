"""
Step-by-step PDF → Image → OCR pipeline (Thai-focused).

Run with: `python main.py`
"""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import List

import fitz  # PyMuPDF
import numpy as np
from PIL import Image

# --- Optional: EasyOCR for recognition (installed via `pip install easyocr`) ---
try:
    from easyocr import Reader

    EASY_OCR_AVAILABLE = True
except ImportError:
    EASY_OCR_AVAILABLE = False


# --------------------------------------------------------------------------- #
# 1) CONFIGURATION
# --------------------------------------------------------------------------- #
PDF_PATH = Path("./data/บัญชีทรัพย์สินและหนี้สิน.pdf")
OUTPUT_DIR = Path("./output")
OUTPUT_IMAGE_DIR = OUTPUT_DIR / "images"
OUTPUT_TEXT_DIR = OUTPUT_DIR / "text"
OCR_OUTPUT_PATH = OUTPUT_TEXT_DIR / "ocr_output.csv"
ZOOM_X = 3.0  # 1.0 = 72 DPI; 3.0 ≈ 216 DPI; adjust to ~4.17 for ~300 DPI if needed
ZOOM_Y = 3.0


# --------------------------------------------------------------------------- #
# 2) PDF → PIL IMAGES
# --------------------------------------------------------------------------- #
def convert_pdf_to_images(pdf_path: Path) -> List[Image.Image]:
    """Render each PDF page into a high-resolution PIL image."""
    print(f"📄  Opening PDF: {pdf_path}")
    doc = fitz.open(pdf_path)
    images: List[Image.Image] = []

    try:
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            print(f"🔄  Rendering page {page_num + 1}/{len(doc)} ...")

            # Configure zoom (controls DPI scaling)
            matrix = fitz.Matrix(ZOOM_X, ZOOM_Y)

            # Render vector page into pixel buffer
            pix = page.get_pixmap(matrix=matrix, alpha=False)

            # (Optional) keep PIL copies in memory for OCR post-processing
            img_data = pix.tobytes("ppm")
            image = Image.open(io.BytesIO(img_data)).convert("RGB")
            images.append(image)

            # Save PNG snapshot for inspection
            image_path = OUTPUT_IMAGE_DIR / f"output_page_{page_num}.png"
            image.save(image_path)
    finally:
        doc.close()

    print(f"✅  Rendered {len(images)} pages to {OUTPUT_DIR.resolve()}")
    return images


# --------------------------------------------------------------------------- #
# 3) OCR (EASYOCR)
# --------------------------------------------------------------------------- #
def run_easyocr(images: List[Image.Image]) -> None:
    """Recognise Thai/English text on each page using EasyOCR and export CSV."""
    if not EASY_OCR_AVAILABLE:
        print("⚠️  EasyOCR not installed. Skip OCR step or `pip install easyocr`.")
        return

    print("👁️  Initialising EasyOCR (langs=th+en, cpu mode)...")
    reader = Reader(["th", "en"], gpu=False)
    with OCR_OUTPUT_PATH.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["page", "text"])
        writer.writeheader()

        for page_num, image in enumerate(images, start=1):
            print(f"🔍  OCR page {page_num} ...")
            # EasyOCR expects a NumPy array (RGB)
            np_image = np.array(image)

            # detail=0 returns just strings; join into multi-line block
            page_text_lines = reader.readtext(np_image, detail=0, paragraph=True)
            page_text = "\n".join(page_text_lines)

            writer.writerow({"page": page_num, "text": page_text})
            csvfile.flush()

            text_output_path = OUTPUT_TEXT_DIR / f"ocr_page_{page_num}.txt"
            text_output_path.write_text(page_text, encoding="utf-8")
            print(f"💾  Saved text for page {page_num} to {text_output_path.resolve()}")

    print(f"✅  OCR results saved to {OCR_OUTPUT_PATH.resolve()}")


# --------------------------------------------------------------------------- #
# 4) MAIN WORKFLOW
# --------------------------------------------------------------------------- #
def main() -> None:
    """Orchestrate the pipeline."""
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_TEXT_DIR.mkdir(parents=True, exist_ok=True)
    print("📂  Output directories ready.")

    # Step 1: Convert PDF to images
    images = convert_pdf_to_images(PDF_PATH)

    # Step 2: Run OCR (optional, requires EasyOCR)
    run_easyocr(images)

    print("🎉  Pipeline complete. Review PNGs and CSV for accuracy.")


if __name__ == "__main__":
    main()
