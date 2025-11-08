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

## Quickstart Walkthrough (Step-by-step)
- **1. เตรียมสภาพแวดล้อม**  
  - ติดตั้ง Python ≥3.10 แล้วรัน `python -m venv .venv`, `.\.venv\Scripts\activate` (Windows)  
  - `pip install -r requirements.txt` (จะดึง PyMuPDF, Pillow, numpy, EasyOCR)
- **2. เตรียมข้อมูล**  
  - ใส่ไฟล์ PDF เป้าหมายไว้ใน `data/` (ตัวอย่าง: `บัญชีทรัพย์สินและหนี้สิน.pdf`)
- **3. แปลง PDF → ภาพ**  
  - รัน `python main.py`  
  - สคริปต์จะเรนเดอร์ทุกหน้าเป็น `.png` ที่ `output/images/output_page_*.png`
- **4. รัน OCR ภาษาไทย**  
  - ขั้นตอนในสคริปต์จะใช้ EasyOCR (ถ้าติดตั้งไว้) แปลงภาพเป็นข้อความ  
  - ผลลัพธ์บันทึกเป็นไฟล์ `output/text/ocr_output.csv` (คอลัมน์ `page`, `text`) พร้อมไฟล์แยกต่อหน้า `output/text/ocr_page_*.txt`
- **5. ตรวจสอบคุณภาพ**  
  - เปิดไฟล์ภาพและ CSV ตรวจตัวอย่างข้อความ  
  - ปรับค่า `ZOOM_X` / `ZOOM_Y` หรือ preprocessing เพิ่มเติมในสคริปต์ตามต้องการ
- **6. ขยายต่อยอด**  
  - หากต้องการ layout/table extraction → ใช้ PaddleOCR หรือ Document AI  
  - เพิ่ม post-processing ภาษาไทยด้วย `pythainlp` สำหรับตัดคำ/สะกด  
  - รวมผลในฐานข้อมูล หรือสร้าง Dashboard สรุปข้อมูล

## Installation & Project Setup
- **Clone repository**
  ```powershell
  git clone https://github.com/<your-org>/ocr-guide.git
  cd ocr-guide
  ```
- **Create virtual environment (แนะนำ)**
  ```powershell
  python -m venv .venv
  .\.venv\Scripts\activate
  ```
  > macOS/Linux: `python3 -m venv .venv && source .venv/bin/activate`
- **Install dependencies**
  ```powershell
  pip install --upgrade pip
  pip install -r requirements.txt
  ```
  - `requirements.txt` รวม EasyOCR (ต้องโหลด weight ครั้งแรก), PyMuPDF, Pillow, numpy และ Torch (CPU build)
  - หากต้องการลดขนาด สามารถลบแพ็กเกจที่ไม่ใช้แล้วรัน `pip install` เฉพาะส่วนที่ต้องการ
- **Optional GPU (ถ้ามี CUDA)**: ติดตั้ง torch/torchvision ที่ตรงกับเวอร์ชัน CUDA ของคุณจาก [PyTorch.org](https://pytorch.org/get-started/locally/)
- **ตั้งค่าเอกสารตัวอย่าง**
  - วางไฟล์ PDF ใน `data/` แล้วแก้ `PDF_PATH` ใน `main.py` หากต้องการใช้ชื่ออื่น
  - รันเทสด้วย `python main.py` เพื่อยืนยันว่า folder `output/images` และ `output/text` ถูกสร้างและมีผลลัพธ์

## 1.1 Code-first Overview (`main.py`)
- **Entry point**: `python main.py` จะเรียก `main()` ซึ่งเตรียมโฟลเดอร์ `output/`, `output/images/`, `output/text/`
- **Rendering**: `convert_pdf_to_images()` ใช้ PyMuPDF (`fitz`) แปลงทุกหน้าของ `PDF_PATH` เป็นภาพความละเอียดสูง แล้วเซฟเป็น `output/images/output_page_{n}.png`
- **OCR Streaming**: `run_easyocr()` สร้างอินสแตนซ์ EasyOCR Thai+English แล้ววนทีละหน้า  
  - แปลง `PIL.Image` → NumPy array → ส่งเข้า `Reader.readtext()`  
  - บันทึกผลทีละบรรทัดลง `output/text/ocr_output.csv` พร้อม `flush()` เพื่อให้ดูผลได้ทันที  
  - สร้างไฟล์ข้อความรายหน้า `output/text/ocr_page_{n}.txt`
- **Configuration knobs**:  
  - `PDF_PATH`: ที่อยู่ไฟล์ PDF  
  - `ZOOM_X`, `ZOOM_Y`: ตัวคูณ DPI (3.0 ≈ 216 DPI, 4.17 ≈ 300 DPI)  
  - `EASY_OCR_AVAILABLE`: ตรวจจับอัตโนมัติ (ติดตั้ง easyocr ก่อนรันหรือจะสั่ง `pip install easyocr`)  
  - ปรับค่าเหล่านี้โดยตรงใน `main.py` เพื่อให้ตรงกับเอกสารของคุณ
- **Dependencies ใน `requirements.txt`**: ควรติดตั้งตามไฟล์ (PyMuPDF, Pillow, numpy, EasyOCR ฯลฯ) เพื่อให้ฟังก์ชันใน `main.py` ใช้งานได้ครบ

### เส้นทางพัฒนาถัดไป & แนวทางเพิ่มคุณภาพ
- **Preprocessing เฉพาะเอกสารไทย**: ลองใช้ OpenCV เพื่อลด noise, ปรับ contrast, deskew ก่อน OCR เพื่อเพิ่มความแม่นยำของตัวอักษรที่มีวรรณยุกต์
- **Model Ensemble**: เรียกใช้ EasyOCR พร้อม PaddleOCR หรือ Tesseract แล้วรวมผล (เลือกข้อความที่มี confidence สูงสุด)
- **Post-correction ภาษาไทย**: ใช้ `pythainlp` ทำการตัดคำและตรวจสะกด หรือใช้ LLM ช่วยปรับแก้ข้อความตาม domain
- **ระบบ Feedback Loop**: สร้างสคริปต์เปรียบเทียบผล OCR กับคำตอบที่ถูกต้อง (ground truth) เพื่อ track CER/WER และระบุหน้าที่ควรปรับปรุง
- **รองรับเอกสารจำนวนมาก**: แตก pipeline เป็น job ต่อหน้า (multiprocessing) และเพิ่ม logging/monitoring เพื่อตรวจผลการรัน

### แนวคิดสำหรับการ Extract ตารางจาก PDF/ภาพ
- **PaddleOCR Layout/Table Modules**: ใช้ `PaddleDetection` สำหรับหาเส้นตาราง + `PaddleOCR` สำหรับอ่าน cell แล้ว map กลับเป็นโครงสร้าง grid
- **Table Transformer (Microsoft)**: โมเดล `microsoft/table-transformer` บน Hugging Face สามารถ detect โครงสร้างตารางเป็น HTML หรือ Markdown
- **LLM-based Parsing**: ส่ง bounding boxes + ข้อความ (จาก OCR) เข้า LLM เช่น GPT-4o พร้อม prompt ให้วิเคราะห์และสร้าง CSV/JSON ตาราง
- **OpenCV Table Detection**: ใช้การ threshold + `cv2.findContours` เพื่อหาคอลัมน์/แถว แล้วตัดภาพย่อยก่อนส่งเข้า OCR (ช่วยลดการอ่านผิดตำแหน่ง)
- **Document AI Services**: ถ้าไม่ติดข้อจำกัดด้านข้อมูล ลอง Google Document AI, AWS Textract, Azure Form Recognizer ซึ่งรองรับ table extraction พร้อม confidence score และโครงสร้าง cell
- **Post-processing Rules**: หลังดึงข้อมูลมาแล้ว ใช้กฎ domain-specific (regex สำหรับตัวเลข, วันที่, คอลัมน์ที่ต้องมีหน่วย) เพื่อตรวจและจัดรูปแบบตารางก่อนบันทึก

## 2. Environment Setup
- **Python ≥3.10**; create an isolated env with `python -m venv .venv && source .venv/bin/activate`.
- **Core tooling**:
  - `pip install opencv-python pillow numpy pandas matplotlib`
  - `pip install easyocr` (built-in Thai models)
  - `pip install paddleocr` (strong multilingual support, incl. Thai)
- **Native dependencies**:
  - CUDA/cuDNN if you plan to run GPU-accelerated models (PaddleOCR, EasyOCR, TrOCR, etc.).

## 3. Data Acquisition & Storage
- **Sources**: Scanner output (TIFF/PDF), smartphone photos, datasets (ICDAR, ThaiOCR, LST20 OCR extensions).
- **File organization**: `raw/`, `processed/`, `ocr_output/`, `models/`.
- **Metadata tracking**: store filename, capture device, resolution, language mix, preprocessing steps applied.
- **Versioning**: use DVC or Git LFS if datasets are large.

## 4. Preprocessing Steps
- **Conversion**: PDF → image using PyMuPDF (`fitz`) matrix zoom (ดูตัวอย่างใน `main.py`).
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
- **PyMuPDF + EasyOCR pipeline** (อิงโค้ด `main.py` บรรทัด 14-15 เป็นต้นไป):

```python
import io
import numpy as np
import fitz  # PyMuPDF
from PIL import Image
from easyocr import Reader

reader = Reader(["th", "en"], gpu=False)
doc = fitz.open("data/บัญชีทรัพย์สินและหนี้สิน.pdf")

for page_idx in range(len(doc)):
    page = doc.load_page(page_idx)
    matrix = fitz.Matrix(3.0, 3.0)  # ปรับ DPI ได้ด้วย ZOOM_X/ZOOM_Y
    pix = page.get_pixmap(matrix=matrix, alpha=False)
    image = Image.open(io.BytesIO(pix.tobytes("ppm"))).convert("RGB")
    text_lines = reader.readtext(np.array(image), detail=0, paragraph=True)
    print(f"Page {page_idx + 1}:", "\n".join(text_lines))

doc.close()
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
- เสริม preprocessing pipeline (deskew, adaptive thresholding) และวัดผล CER/WER เทียบ baseline ปัจจุบัน
- เปรียบเทียบผล EasyOCR กับ PaddleOCR และพิจารณาใช้ร่วมกันเพื่อเพิ่มความแม่นยำ
- ตั้ง workflow สำหรับ extract ตาราง: ทดลอง PaddleOCR table, Table Transformer, หรือ OpenCV-based table detection แล้วบันทึกเป็น CSV
- ออกแบบ post-correction (PyThaiNLP หรือ LLM) และระบบ feedback loop สำหรับปรับปรุงข้อความ OCR อย่างต่อเนื่อง
- บันทึกสคริปต์/การตั้งค่าปัจจุบัน พร้อมทดสอบอัตโนมัติสำหรับหน้าที่ปรับแล้ว เพื่อป้องกัน regression ในอนาคต