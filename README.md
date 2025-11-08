# OCR Guide (Python & Thai Focus)

## Hackathon Context: ข้อมูลงาน "Hackathon ข้อมูลบัญชีทรัพย์สินฯ"
- **ชื่องาน**: Hackathon ข้อมูลบัญชีทรัพย์สินฯ (15 พ.ย. – 6 ธ.ค. 2568; นำเสนอรอบสุดท้าย 13 ธ.ค. 2568)
- **เป้าหมาย**: พัฒนานวัตกรรมที่แปลงข้อมูลบัญชีทรัพย์สินจากเอกสาร PDF/ดิจิทัล ให้เป็นข้อมูลมาตรฐานพร้อมใช้งานด้วยความแม่นยำสูง
- **ชุดข้อมูล**: ไฟล์ PDF (ทั้ง text-based และสแกนภาพ) + ไฟล์ดิจิทัลที่เกี่ยวข้อง
- **รูปแบบการแข่งขัน**: เดี่ยวหรือทีม (ควรมีทักษะผสม เช่น OCR, Data Engineering, Visualization, Domain Knowledge ด้านบัญชี/กฎหมาย)
- **ข้อจำกัด**: เครื่องมือ/เทคโนโลยีเปิดกว้าง แต่ต้องอธิบายวิธีใช้และฐานความรู้ที่เกี่ยวข้อง (Python/R, OCR, PDF processing)
- **สิทธิ์ผู้ผ่านคัดเลือก**: 60 ทีมแรกในวันเปิดงาน (15 พ.ย. 68) จะได้สิทธิ์ใช้เครื่องมือระดับมืออาชีพฟรีตลอดการแข่งขัน

### แนวทางเตรียมตัวก่อน Hackathon
- **เตรียมสภาพแวดล้อม**: ตั้งค่าสภาพแวดล้อม Python/R พร้อมไลบรารี OCR/PDF (ดูหัวข้อ Environment Setup)
- **ฝึกเวิร์กโฟลว์**: ทดลอง pipeline แปลง PDF → Image → OCR → CSV/DB พร้อมการตรวจสอบคุณภาพข้อมูล
- **ไอเดียโครงการ**:
  - แปลง PDF โครงสร้างซับซ้อน เป็น CSV/ฐานข้อมูลอัตโนมัติ
  - ระบบตรวจความผิดปกติในบัญชีทรัพย์สิน (anomaly detection)
  - Dashboard สรุปสินทรัพย์/หนี้สินพร้อมตัวชี้วัดสำคัญ
- **บริหารเวลา 3 สัปดาห์**:
  - สัปดาห์ที่ 1: เข้าใจโจทย์, สำรวจข้อมูล, ตั้ง baseline OCR
  - สัปดาห์ที่ 2: พัฒนา preprocessing + โครงสร้างข้อมูล + UI/รายงาน
  - สัปดาห์ที่ 3: ปรับปรุง accuracy, ทำ automation, ซ้อมนำเสนอ
- **เตรียมเอกสารนำเสนอ**: สลาย pipeline เป็น flowchart, เน้นคุณภาพข้อมูลและการตรวจสอบผลลัพธ์, ระบุผลกระทบ/ประโยชน์เชิงนโยบาย

## 1. Overview
- **Goal**: Build an end-to-end OCR workflow in Python for Thai and multilingual documents (images & PDFs).
- **Use cases**: Scanned paperwork, receipts, invoices, forms, printed books, street signs, mixed-language documents.
- **Pipeline summary**: Data acquisition → preprocessing → OCR inference → post-processing → export (text/CSV) → optional fine-tuning or LLM correction.

## 2. Environment Setup
- **Python ≥3.10**; create an isolated env with `python -m venv .venv && source .venv/bin/activate`.
- **Core tooling**:
  - `pip install opencv-python pillow numpy pandas matplotlib`
  - `pip install pytesseract` (requires native Tesseract install)
  - `pip install easyocr` (built-in Thai models)
  - `pip install paddleocr` (strong multilingual support, incl. Thai)
  - `pip install pdf2image pikepdf` for PDFs → images.
- **Native dependencies**:
  - Tesseract OCR (Homebrew: `brew install tesseract tesseract-lang`).
  - Poppler (`brew install poppler`) for `pdf2image`.
  - CUDA/cuDNN if you plan to run GPU-accelerated models (PaddleOCR, EasyOCR, TrOCR, etc.).

## 3. Data Acquisition & Storage
- **Sources**: Scanner output (TIFF/PDF), smartphone photos, datasets (ICDAR, ThaiOCR, LST20 OCR extensions).
- **File organization**: `raw/`, `processed/`, `ocr_output/`, `models/`.
- **Metadata tracking**: store filename, capture device, resolution, language mix, preprocessing steps applied.
- **Versioning**: use DVC or Git LFS if datasets are large.

## 4. Preprocessing Steps
- **Conversion**: PDF → image using `pdf2image.convert_from_path`.
- **Resolution**: aim 300 DPI for PDFs; upscale low-resolution images via `cv2.dnn_superres`.
- **Noise reduction**: bilateral filter, median blur, non-local means (`cv2.fastNlMeansDenoisingColored`).
- **Binarization**: Otsu threshold, adaptive threshold, Sauvola (from `skimage`).
- **Deskewing**: Hough transform or OpenCV `cv2.getRotationMatrix2D`.
- **Perspective correction**: four-point transform with `cv2.findContours` + `cv2.warpPerspective`.
- **Morphological ops**: dilate/erode for character isolation.
- **Segmentation**:
  - Line/word segmentation via `cv2.connectedComponents`.
  - Layout analysis with `layoutparser`, `detectron2`, or `PaddleOCR` detection module.
- **Language-specific cleanup**: remove guide boxes, dotted lines; handle Thai tone marks by ensuring minimal erosion.
- **Augmentation** (for training/fine-tuning): rotation, blur, Gaussian noise, background textures.

## 5. OCR Engines & Libraries
- **Tesseract OCR** (`pytesseract`):
  - Install Thai language data (`tesseract-ocr-tha`).
  - Custom configs: `--psm` (page segmentation), `--oem` (engine mode), `-l tha+eng`.
  - Post-processing with `PyThaiNLP` for spell-checking or tokenization.
- **EasyOCR**:
  - Ready-to-use Thai model; simple API.
  - Works well for mixed Thai/Latin; GPU support.
- **PaddleOCR**:
  - Detection + recognition pipeline; strong multilingual accuracy.
  - Supports training/fine-tuning; CLI & Python API.
- **Google Cloud Vision / Document AI**:
  - High accuracy for forms; async batch processing; supports Thai.
  - Consider cost & data privacy.
- **AWS Textract**:
  - Form/table extraction; Thai support improving (check latest docs).
- **Microsoft Azure Computer Vision / Form Recognizer**:
  - Good for structured documents; confirm Thai support status.
- **Hugging Face Transformers**:
  - `microsoft/trocr-base-thai` (if available) or multilingual TrOCR; requires GPU.
  - Combine with layout models (`LayoutLM`, `Donut`) for document understanding.
- **Other frameworks**:
  - `kraken` (custom OCR training).
  - `Calamari OCR` (OCR with CTC + LSTM).
  - `lama-cleaner`, `imgaug`, `albumentations` for preprocessing/augmentation combos.

## 6. Inference Workflow (Python)
- **Basic example** (Tesseract):

```python
from pdf2image import convert_from_path
import pytesseract, pandas as pd

images = convert_from_path("input.pdf", dpi=300)
results = []
for page_idx, img in enumerate(images):
    text = pytesseract.image_to_string(img, lang="tha+eng", config="--psm 6")
    results.append({"page": page_idx + 1, "text": text})

df = pd.DataFrame(results)
df.to_csv("ocr_output/output.csv", index=False)
```

- **Detection + recognition** (PaddleOCR) for bounding boxes and text:

```python
from paddleocr import PaddleOCR
ocr = PaddleOCR(lang="en", use_angle_cls=True, det=True, rec=True, show_log=False)
result = ocr.ocr("invoice.jpg", cls=True)
```

- **Post-processing**:
  - Tokenize and normalize Thai text using `pythainlp`.
  - Spell correction: `pythainlp.correct`, custom dictionaries, LLM-based correction.
  - Regex templates for structured extraction (dates, amounts, IDs).

## 7. Exporting Results
- **Plain text**: save per-page `.txt` or a single concatenated file.
- **CSV/TSV**:
  - Use pandas DataFrame with columns like `page`, `bbox`, `confidence`, `text`.
  - Flatten nested results from detection models before saving.
- **JSON**:
  - Preserve layout (e.g., bounding boxes, block hierarchy).
  - Useful for downstream search/indexing.
- **Databases**: insert into SQLite/Postgres; enable full-text search via `fts5` or `pg_trgm`.

## 8. Thai Language Considerations
- **Fonts & handwriting**: evaluate dataset coverage; consider Thai handwriting datasets.
- **Spacing & segmentation**: Thai lacks spaces; use `pythainlp.word_tokenize`.
- **Tone marks & diacritics**: ensure preprocessing preserves accents; prefer grayscale + adaptive thresholding.
- **Custom dictionaries**: domain-specific lexicons improve correction accuracy.
- **Benchmark datasets**: Thai OCR dataset from NECTEC, Thai Street View Text (if accessible).

## 9. Fine-tuning & Model Customization
- **Tesseract**:
  - Train with `tesstrain` or Tesseract 4+ LSTM training pipeline.
  - Requires prepared `.box` files and ground truth.
- **PaddleOCR**:
  - Supports detection/recognition training with YAML configs.
  - Utilize synthetic data generation (e.g., `SynthText`, `TextRecognitionDataGenerator`).
- **EasyOCR**:
  - Custom training via repo scripts; adjust backbone/language packs.
- **Transformers (TrOCR, Donut)**:
  - Fine-tune with labeled datasets using Hugging Face Trainer.
  - Use mixed precision training for speed (`fp16`).
- **LLM Post-correction**:
  - Prompt LLM (GPT-4o, Claude, Llama 3.1) to clean OCR text given domain-specific guidelines.
  - Fine-tune instruction LLMs with pairs of OCR-noisy → clean text.
- **Frameworks**:
  - `LangChain`, `LlamaIndex` for hybrid OCR + LLM pipelines.
  - `SentencePiece` / `huggingface/tokenizers` for custom Thai tokenizers.

## 10. Evaluation & QA
- **Metrics**: Character Error Rate (CER), Word Error Rate (WER), BLEU for post-corrected text.
- **Benchmarking**: hold-out validation set; evaluate across varying lighting/fonts.
- **Human-in-the-loop**: build simple review UI with `streamlit` or `gradio`.
- **Logging**: capture confidence scores, bounding boxes, errors for analytics.

## 11. Automation & Deployment
- **Batch pipelines**: orchestrate with `Airflow`, `Prefect`, or `Dagster`.
- **Serverless**: deploy inference endpoints via AWS Lambda, Google Cloud Functions with lightweight models.
- **Containerization**: Docker image with preinstalled dependencies for consistent deployment.
- **Monitoring**: track throughput, latency, error rates; reprocess low-confidence pages automatically.

## 12. Research & Further Reading
- **Papers**:
  - PaddleOCR: `PP-OCRv3/PP-OCRv4`.
  - Google DocAI & TrOCR publications.
  - Thai OCR corpora from NECTEC, LST20.
- **Repos**:
  - `https://github.com/PaddlePaddle/PaddleOCR`
  - `https://github.com/JaidedAI/EasyOCR`
  - `https://github.com/tesseract-ocr/tesseract`
  - `https://github.com/microsoft/unilm` (TrOCR, LayoutLM, Donut).
- **Datasets**:
  - `https://www.aiforthai.in.th` (Thai language resources).
  - `https://github.com/vistec-AIT/` for Thai NLP datasets.

## 13. Next Steps for the Team
- Stand up a prototype with EasyOCR or PaddleOCR to validate accuracy on internal docs.
- Collect failed cases; annotate for fine-tuning.
- Evaluate LLM-based post-correction for high-precision use cases.
- Document pipeline scripts/tests; integrate into CI for regression checks.