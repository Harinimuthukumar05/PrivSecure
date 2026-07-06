# =============================================================================
# INDIAN ID CARD OCR EXTRACTION SYSTEM — PRODUCTION UPGRADE
# =============================================================================
# Supports: Aadhaar, PAN, Driving License, Voter ID, College ID, Ration Card
#
# REQUIRED PIP INSTALLS (run these in your terminal first):
# ---------------------------------------------------------
# pip install paddlepaddle==2.6.2
# pip install paddleocr==2.7.3
# pip install groq
# pip install python-dotenv
# pip install pymupdf
# pip install pdf2image
# pip install pillow
# pip install opencv-python
# pip install numpy==1.26.4
# pip install pytesseract
# pip install verhoeff
#
# TESSERACT INSTALL:
# ------------------
# Windows: https://github.com/UB-Mannheim/tesseract/wiki
#   After install, set the path below:
#     pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# Linux:   sudo apt install tesseract-ocr
# Mac:     brew install tesseract
#
# NOTE for Windows users:
#   - pdf2image requires poppler. Download from:
#     https://github.com/oschwartz10612/poppler-windows/releases
#     Extract and add the `bin` folder to your Windows PATH.
#   - PaddleOCR may need Visual C++ Redistributable installed.
#
# VS CODE SETUP:
# --------------
# 1. Open VS Code in your project folder
# 2. Create a Python virtual environment:
#       python -m venv venv
#       venv\Scripts\activate        (Windows)
#       source venv/bin/activate     (Mac/Linux)
# 3. Install all packages listed above inside the venv
# 4. Select the venv as your Python interpreter in VS Code
#    (Ctrl+Shift+P → "Python: Select Interpreter" → choose venv)
#
# .ENV FILE SETUP:
# ----------------
# Create a file named `.env` in the same folder as this script:
#
#   GROQ_API_KEY=your_groq_api_key_here
#
# Get your free Groq API key at: https://console.groq.com
#
# HOW TO RUN:
# -----------
#   python indian_id_ocr.py
#   python indian_id_ocr.py path/to/id_card.jpg
#
# FOLDER STRUCTURE:
# -----------------
#   project/
#   ├── indian_id_ocr.py     ← this script
#   ├── .env                 ← your API key
#   ├── venv/                ← virtual environment
#   └── id_images/           ← place your ID card images here
#
# PRODUCTION PIPELINE:
# --------------------
#   Input Image/PDF
#       ↓
#   Multiple Preprocessing Variants
#       ↓
#   PaddleOCR (multiple passes)
#       ↓
#   OCR Deduplication
#       ↓
#   OCR Sorting
#       ↓
#   Text Cleaning
#       ↓
#   ID Type Detection
#       ↓
#   ID-Type-Aware Regex Extraction
#       ↓
#   PAN Correction Logic (PAN only)
#       ↓
#   Tesseract Fallback (PAN only)
#       ↓
#   LLM Structured Extraction (type-aware prompt)
#       ↓
#   JSON Validation
#       ↓
#   Field Validation
#       ↓
#   Final Field Filtering (allowed fields only)
#       ↓
#   Final Structured Output
#
# =============================================================================

import os
import re
import json
import logging
import tempfile
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageFilter, ImageEnhance
from dotenv import find_dotenv, load_dotenv

# Load environment variables from .env file
load_dotenv()
print("ENV FILE:", find_dotenv())
print("LOADED KEY:", os.getenv("GROQ_API_KEY"))

# =============================================================================
# LOGGING SETUP
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("ocr_extraction.log", encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)


# =============================================================================
# LAZY IMPORTS
# =============================================================================

try:
    from paddleocr import PaddleOCR
except ImportError:
    logger.error("PaddleOCR not installed. Run: pip install paddleocr==2.7.3")
    raise

try:
    from groq import Groq
except ImportError:
    logger.error("Groq not installed. Run: pip install groq")
    raise

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
    # Uncomment and set path if on Windows:
    pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"
except ImportError:
    TESSERACT_AVAILABLE = False
    logger.warning("pytesseract not installed. Tesseract PAN fallback disabled.")

try:
    from verhoeff import validateVerhoeff
    VERHOEFF_AVAILABLE = True
except ImportError:
    VERHOEFF_AVAILABLE = False
    logger.warning(
        "verhoeff package not installed. "
        "Aadhaar checksum validation disabled. "
        "Run: pip install verhoeff"
    )

try:
    import fitz  # PyMuPDF    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    logger.warning("PyMuPDF not available. Falling back to pdf2image.")
    try:
        from pdf2image import convert_from_path
    except ImportError:
        logger.error("Neither PyMuPDF nor pdf2image installed.")
        raise


# =============================================================================
# CONFIGURATION
# =============================================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    raise EnvironmentError(
        "GROQ_API_KEY not found. Create a .env file with:\n  GROQ_API_KEY=your_key_here"
    )

# OCR confidence threshold — lowered for PAN cards with tiny text
OCR_CONFIDENCE_THRESHOLD = 0.25

# Image upscale factor before OCR (2x recommended; 3x for very small IDs)
UPSCALE_FACTOR = 2

# LLM model to use
GROQ_MODEL = "llama-3.3-70b-versatile"


# =============================================================================
# REGEX PATTERNS
# =============================================================================

# Aadhaar: 12 digits, optionally grouped as XXXX XXXX XXXX
AADHAAR_RE = re.compile(r"\b\d{4}[ -]?\d{4}[ -]?\d{4}\b")

# VID (Virtual ID): 16 digits, optionally grouped as XXXX XXXX XXXX XXXX
# Used to detect and reject 12-digit substrings that are part of a VID
VID_RE = re.compile(r"\b\d{4}[ -]?\d{4}[ -]?\d{4}[ -]?\d{4}\b")

# PAN: 5 letters + 4 digits + 1 letter (standard format AAAAA9999A)
PAN_RE = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")

# PAN loose: any 10-character alphanumeric block (catches OCR-mangled PANs)
PAN_LOOSE_RE = re.compile(
    r"\b[A-Z0-9]{10}\b",
    re.IGNORECASE
)

# Driving License: State code (2 letters) + year (2 digits) + serial (4-13 digits)
DL_RE = re.compile(r"\b[A-Z]{2}[ -]?\d{2}[ -]?\d{4,13}\b", re.IGNORECASE)

# Voter ID: 3 letters + 7 digits
VOTER_RE = re.compile(r"\b[A-Z]{3}[0-9]{7}\b", re.IGNORECASE)

# Date of Birth: DD/MM/YYYY or DD-MM-YYYY
DOB_RE = re.compile(r"\b\d{2}[/\-\.]\d{2}[/\-\.]\d{4}\b")

# Indian phone: starts with 6-9, total 10 digits
PHONE_RE = re.compile(r"\b[6-9]\d{9}\b")


# =============================================================================
# FIELDS PER ID TYPE
# NOTE: "email" removed from all types — Indian IDs do not reliably have emails
#       and OCR frequently mistakes website/domain text for email addresses.
# =============================================================================

ID_FIELDS = {
    "Aadhaar": [
        "name", "date_of_birth", "gender",
        "address", "aadhaar_number", "phone_number"
    ],
    "PAN": [
        "name", "father_name", "date_of_birth", "pan_number"
    ],
    "Driving License": [
        "name", "father_name", "date_of_birth", "gender",
        "license_number", "address", "issue_date", "expiry_date", "vehicle_class"
    ],
    "Voter ID": [
        "name", "father_name", "gender",
        "date_of_birth", "voter_id_number", "address", "part_number"
    ],
    "College ID": [
        "name", "college_name", "roll_number",
        "department", "course", "year", "validity"
    ],
    "Ration Card": [
        "head_of_family", "ration_card_number",
        "address", "card_type", "members_count"
    ],
    "Other": [
        "name", "date_of_birth", "address", "id_number"
    ]
}

# ---- Metadata fields always preserved in final output ----
# NOTE: Underscore-prefixed fields excluded per data redaction requirements
METADATA_FIELDS = {
    "id_type", "source_file"
}

# =============================================================================
# STRONG KEYWORD GUARDS — prevent misclassification of number sequences
# =============================================================================

# Aadhaar may only be extracted when at least ONE of these is present in text
AADHAAR_STRONG_KEYWORDS = {
    "UIDAI", "AADHAAR", "AADHAR", "UNIQUE IDENTIFICATION",
    "GOVERNMENT OF INDIA", "ENROLMENT NO"
}

# Driving License may only be extracted when at least ONE of these is present
DL_STRONG_KEYWORDS = {
    "DRIVING LICENCE", "DRIVING LICENSE", "DL NO",
    "MOTOR VEHICLES ACT", "TRANSPORT DEPARTMENT", "COV"
}


# =============================================================================
# GLOBAL INITIALIZER — called once, cached
# =============================================================================

_ocr_engine = None
_groq_client = None


def get_ocr_engine() -> "PaddleOCR":
    """Initialize PaddleOCR once and reuse across calls."""
    global _ocr_engine
    if _ocr_engine is None:
        logger.info("Initializing PaddleOCR engine...")
        _ocr_engine = PaddleOCR(
            use_angle_cls=True,
            lang="en",
            det_db_box_thresh=0.3,
            det_db_thresh=0.3,
            cls_thresh=0.85,
            rec_batch_num=6,
            use_gpu=False,
            show_log=False,
        )
        logger.info("PaddleOCR ready.")
    return _ocr_engine


def get_groq_client() -> "Groq":
    """Initialize Groq client once and reuse."""
    global _groq_client
    if _groq_client is None:
        logger.info("Initializing Groq client...")
        _groq_client = Groq(api_key=GROQ_API_KEY)
        logger.info("Groq client ready.")
    return _groq_client


# =============================================================================
# IMAGE PREPROCESSING — STANDARD PIPELINE
# =============================================================================

def preprocess_image(image_input) -> np.ndarray:
    """
    Preprocess an image for optimal OCR results on Indian ID cards.

    Strategy:
    - Upscale 2x with INTER_CUBIC (preserves thin characters)
    - Convert to grayscale
    - Apply light Gaussian blur (reduce noise without destroying edges)
    - Apply CLAHE for contrast enhancement (helps faded/low-contrast IDs)
    - Apply optional sharpening (helps blurry scans)
    - Avoid aggressive thresholding (damages PAN card thin text)
    """
    if isinstance(image_input, (str, Path)):
        img = cv2.imread(str(image_input))
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {image_input}")
    elif isinstance(image_input, np.ndarray):
        img = image_input.copy()
    else:
        raise TypeError(f"Unsupported image type: {type(image_input)}")

    logger.info(f"Original image size: {img.shape[1]}x{img.shape[0]}")

    h, w = img.shape[:2]
    img = cv2.resize(
        img,
        (w * UPSCALE_FACTOR, h * UPSCALE_FACTOR),
        interpolation=cv2.INTER_CUBIC
    )
    logger.info(f"Upscaled to: {img.shape[1]}x{img.shape[0]}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    blurred = cv2.GaussianBlur(gray, (0, 0), 3)
    gray = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)

    result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return result


# =============================================================================
# MULTI PREPROCESSING VARIANTS
# =============================================================================

def generate_image_variants(img: np.ndarray) -> list[np.ndarray]:
    """
    Generate multiple preprocessing variants for OCR retry passes.

    Different ID types respond better to different preprocessing:
    - PAN cards      → strong sharpening
    - Aadhaar cards  → adaptive threshold
    - Driving License → CLAHE stronger contrast

    Running all variants and deduplicating results catches text that
    a single pipeline would miss.
    """
    variants = []

    # Variant 1: Standard preprocess (baseline)
    variants.append(preprocess_image(img))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Variant 2: Strong sharpening — good for PAN cards
    sharp = cv2.GaussianBlur(gray, (0, 0), 3)
    sharp = cv2.addWeighted(gray, 1.8, sharp, -0.8, 0)
    variants.append(cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR))

    # Variant 3: Stronger CLAHE contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cimg = clahe.apply(gray)
    variants.append(cv2.cvtColor(cimg, cv2.COLOR_GRAY2BGR))

    # Variant 4: Adaptive threshold — good for Aadhaar cards
    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )
    variants.append(cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR))

    return variants


# =============================================================================
# OCR DEDUPLICATION
# =============================================================================

def deduplicate_ocr_lines(lines: list[dict]) -> list[dict]:
    """
    Remove duplicate OCR lines that arise from running multiple variant passes.

    Without this step:
    - The LLM prompt becomes noisy with repeated lines
    - Extraction quality drops due to redundant context
    - Token usage increases unnecessarily
    """
    seen = set()
    unique = []

    for line in lines:
        text = line["text"].strip()
        # Normalize: collapse all whitespace and uppercase for dedup comparison.
        # This catches OCR variants like "8896 6991 3234" vs "889669913234"
        # and prevents fragmented-number merges from slipping through.
        normalized = re.sub(r"\s+", "", text.upper())
        if normalized not in seen:
            seen.add(normalized)
            unique.append(line)  # preserve original text in output

    logger.info(f"Deduplication: {len(lines)} → {len(unique)} lines")
    return unique


# =============================================================================
# OCR RUNNER
# =============================================================================

def run_ocr(image_array: np.ndarray) -> list[dict]:
    """
    Run PaddleOCR on a preprocessed image.

    Returns:
        List of dicts: [{"text": str, "confidence": float, "bbox": list}]
    """
    engine = get_ocr_engine()
    ocr_result = engine.ocr(image_array)

    extracted = []
    if not ocr_result:
        return extracted

    for block in ocr_result:
        if not block:
            continue
        for line in block:
            bbox = line[0]
            text = line[1][0]
            conf = line[1][1]

            logger.debug(f"OCR line: '{text}' (conf={conf:.2f})")

            if conf >= OCR_CONFIDENCE_THRESHOLD:
                extracted.append({
                    "text": text,
                    "confidence": conf,
                    "bbox": bbox
                })
            else:
                logger.debug(f"  → Discarded (conf={conf:.2f} < {OCR_CONFIDENCE_THRESHOLD})")

    return extracted


def ocr_lines_to_text(ocr_lines: list[dict]) -> str:
    """
    Merge OCR line dicts into a single text string.

    Sort by (row bucket, x position) to preserve row structure better than
    a pure Y-sort, which mis-orders text on the same horizontal line when
    bounding boxes have slight Y jitter.
    """
    if not ocr_lines:
        return ""

    sorted_lines = sorted(
        ocr_lines,
        key=lambda x: (
            x["bbox"][0][1],   # top-left Y — primary row order
            x["bbox"][0][0]    # top-left X — left-to-right within row
        )
    )
    return "\n".join(line["text"] for line in sorted_lines)


# =============================================================================
# TESSERACT FALLBACK (PAN ONLY)
# =============================================================================

def tesseract_pan_ocr(img: np.ndarray) -> str:
    """
    Tesseract fallback OCR, used ONLY when PaddleOCR fails to detect a PAN
    number on a PAN card.

    Tesseract with --psm 6 (assume uniform block of text) often handles
    small, densely-packed PAN text better than PaddleOCR in some cases.

    Returns:
        Raw OCR text string, or empty string if Tesseract is unavailable.
    """
    if not TESSERACT_AVAILABLE:
        return ""

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray, config="--psm 6")
    return text


# =============================================================================
# PDF HANDLING
# =============================================================================

def pdf_to_images(pdf_path: str) -> list[np.ndarray]:
    """Convert all PDF pages to numpy image arrays."""
    images = []

    if PYMUPDF_AVAILABLE:
        logger.info("Converting PDF using PyMuPDF...")
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            mat = fitz.Matrix(200 / 72, 200 / 72)
            pix = page.get_pixmap(matrix=mat)
            img_data = np.frombuffer(pix.samples, dtype=np.uint8)
            img = img_data.reshape(pix.height, pix.width, pix.n)
            if pix.n == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            elif pix.n == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            images.append(img)
            logger.info(f"  Page {page_num + 1} converted.")
        doc.close()
    else:
        logger.info("Converting PDF using pdf2image...")
        pil_pages = convert_from_path(pdf_path, dpi=200)
        for pil_img in pil_pages:
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            images.append(img)

    logger.info(f"PDF converted: {len(images)} page(s).")
    return images


# =============================================================================
# OCR NOISE CLEANING
# =============================================================================

# Minimum alphanumeric characters a line must contain to be kept
MIN_ALNUM_CHARS = 2

# Lines that are ONLY symbols / punctuation — pure noise
_SYMBOL_ONLY_RE = re.compile(r"^[^A-Za-z0-9]+$")

# Common OCR garbage sequences: repeated pipes, dots, underscores
_OCR_GARBAGE_RE = re.compile(r"[|\\/*~^`]{2,}|\.{4,}|_{3,}")

# Repeated consecutive identical words: "NAME NAME NAME" → "NAME"
_REPEATED_WORD_RE = re.compile(r"\b(\w+)(?:\s+\1){2,}\b", re.IGNORECASE)

# Unicode box-drawing / private-use areas that OCR engines emit as garbage
_UNICODE_NOISE_RE = re.compile(r"[\u2500-\u257F\uE000-\uF8FF\uFFFD]+")


def clean_ocr_lines(ocr_lines: list[dict]) -> list[dict]:
    """
    Filter and normalise individual OCR line dicts BEFORE joining to text.

    Removes:
    - Symbol-only lines (no letters or digits).
    - Lines with fewer than MIN_ALNUM_CHARS alphanumeric characters.
    - Lines containing obvious OCR garbage sequences.
    - Unicode box-drawing / private-use characters.

    Normalises:
    - Strips surrounding whitespace.
    - Collapses internal whitespace runs to a single space.
    - Removes repeated consecutive words.
    """
    cleaned: list[dict] = []
    rejected_count = 0

    for line in ocr_lines:
        raw = line["text"]
        text = raw.strip()

        if not text:
            rejected_count += 1
            continue

        text = _UNICODE_NOISE_RE.sub(" ", text)
        text = re.sub(r"[ \t]+", " ", text).strip()

        if _SYMBOL_ONLY_RE.match(text):
            logger.debug(f"[clean_ocr] Rejected (symbol-only): '{raw}'")
            rejected_count += 1
            continue

        alnum_count = sum(1 for c in text if c.isalnum())
        if alnum_count < MIN_ALNUM_CHARS:
            logger.debug(f"[clean_ocr] Rejected (alnum={alnum_count}): '{raw}'")
            rejected_count += 1
            continue

        if _OCR_GARBAGE_RE.search(text):
            logger.debug(f"[clean_ocr] Rejected (garbage pattern): '{raw}'")
            rejected_count += 1
            continue

        text = _REPEATED_WORD_RE.sub(r"\1", text)

        cleaned.append({**line, "text": text})

    logger.info(
        f"[clean_ocr] Line filter: {len(ocr_lines)} -> {len(cleaned)} "
        f"({rejected_count} rejected)"
    )
    return cleaned


# =============================================================================
# TEXT CLEANING
# =============================================================================

def clean_text(text: str) -> str:
    """
    Post-join text normalisation applied after ocr_lines_to_text().

    Removes control characters, unicode noise, collapses whitespace,
    drops empty lines, and limits consecutive blank lines to 2.
    """
    text = re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", text)
    text = _UNICODE_NOISE_RE.sub(" ", text)

    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]

    result_lines: list[str] = []
    blank_run = 0
    for ln in lines:
        if ln == "":
            blank_run += 1
            if blank_run <= 2:
                result_lines.append(ln)
        else:
            blank_run = 0
            result_lines.append(ln)

    return "\n".join(result_lines).strip()


# =============================================================================
# PAN CORRECTION LOGIC
# =============================================================================

OCR_CHAR_MAP_ALPHA = str.maketrans({
    "0": "O",
    "1": "I",
    "8": "B",
    "5": "S",
})

OCR_CHAR_MAP_DIGIT = str.maketrans({
    "O": "0",
    "o": "0",
    "I": "1",
    "i": "1",
    "l": "1",
    "S": "5",
    "s": "5",
    "B": "8",
    "b": "8",
})


def correct_pan_candidates(raw_text: str) -> str | None:
    """
    Find and auto-correct PAN number candidates from OCR text.

    PAN format: AAAAA9999A
    - Characters 0-4: uppercase letters
    - Characters 5-8: digits
    - Character  9:   uppercase letter
    """
    exact = PAN_RE.search(raw_text.upper())
    if exact:
        logger.info(f"PAN found directly: {exact.group()}")
        return exact.group().upper()

    candidates = PAN_LOOSE_RE.findall(raw_text.upper())
    logger.info(f"PAN loose candidates: {candidates}")

    for candidate in candidates:
        candidate = candidate.upper()
        corrected = list(candidate)

        for i in range(5):
            corrected[i] = candidate[i].translate(OCR_CHAR_MAP_ALPHA)
        for i in range(5, 9):
            corrected[i] = candidate[i].translate(OCR_CHAR_MAP_DIGIT)
        corrected[9] = candidate[9].translate(OCR_CHAR_MAP_ALPHA)

        corrected_str = "".join(corrected).upper()

        if PAN_RE.match(corrected_str):
            logger.info(f"PAN corrected: {candidate} → {corrected_str}")
            return corrected_str

    logger.warning("No valid PAN candidate found after correction.")
    return None


# =============================================================================
# ID TYPE DETECTION — WEIGHTED SCORING
# =============================================================================

ID_KEYWORD_WEIGHTS: dict[str, list[tuple[str, int, bool]]] = {
    "Aadhaar": [
        ("UIDAI",                        12, True),
        ("AADHAAR",                      12, True),
        ("AADHAR",                       10, True),
        ("UNIQUE IDENTIFICATION",         8, False),
        ("ENROLMENT NO",                  5, False),
        ("GOVERNMENT OF INDIA",           3, False),
    ],
    "PAN": [
        ("INCOME TAX DEPARTMENT",        15, True),
        ("PERMANENT ACCOUNT NUMBER",     15, True),
        ("INCOME TAX",                    8, False),
        ("PAN",                           5, False),
        ("GOVT. OF INDIA",                3, False),
    ],
    "Driving License": [
        ("DRIVING LICENCE",              15, True),
        ("DRIVING LICENSE",              15, True),
        ("DL NO",                        10, True),
        ("MOTOR VEHICLES ACT",            8, False),
        ("TRANSPORT DEPARTMENT",          5, False),
        ("COV",                           3, False),
    ],
    "Voter ID": [
        ("ELECTION COMMISSION OF INDIA", 15, True),
        ("ELECTION COMMISSION",          12, True),
        ("ELECTORS PHOTO",               10, True),
        ("VOTER",                         8, False),
        ("EPIC",                          5, False),
        ("BHARAT NIRVACHAN",              8, False),
    ],
    "College ID": [
        ("COLLEGE",                       8, False),
        ("UNIVERSITY",                    8, False),
        ("STUDENT",                       5, False),
        ("INSTITUTE",                     5, False),
        ("DEPARTMENT",                    3, False),
        ("ROLL NO",                       5, False),
        ("REG NO",                        5, False),
        ("CAMPUS",                        4, False),
        ("ENROLLMENT NO",                 5, False),
    ],
    "Ration Card": [
        ("RATION CARD",                  15, True),
        ("NFSA",                          8, True),
        ("FOOD AND CIVIL SUPPLIES",        8, False),
        ("BPL",                           5, False),
        ("APL",                           5, False),
        ("RATION",                        6, False),
    ],
}

ID_MIN_SCORE_THRESHOLD = 5
ID_AMBIGUITY_GAP = 3

COLLEGE_OVERRIDE_KEYWORDS = {
    "COLLEGE", "UNIVERSITY", "STUDENT", "INSTITUTE",
    "DEPARTMENT", "CAMPUS", "ENROLLMENT",
}


def detect_id_type(text: str) -> str:
    """
    Weighted-scoring ID type detection with ambiguity handling.

    - Accumulates a numeric score per ID type (no first-match-wins).
    - Aadhaar is NOT triggered by a 12-digit regex alone; it requires at
      least one strong keyword (UIDAI / AADHAAR / AADHAR).
    - College ID keywords explicitly suppress a false Aadhaar score.
    - If the gap between the top-2 scores < ID_AMBIGUITY_GAP → "Other".
    """
    upper_text = text.upper()
    scores: dict[str, int] = {k: 0 for k in ID_KEYWORD_WEIGHTS}

    for id_type, keyword_list in ID_KEYWORD_WEIGHTS.items():
        for keyword, weight, exclusive in keyword_list:
            if keyword in upper_text:
                scores[id_type] += weight
                logger.debug(
                    f"[ID-detect] '{keyword}' matched -> +{weight} for {id_type}"
                )
                if exclusive and scores[id_type] < ID_MIN_SCORE_THRESHOLD:
                    scores[id_type] = ID_MIN_SCORE_THRESHOLD

    logger.info(f"[ID-detect] Raw keyword scores: {scores}")

    # Conservative regex boosts (strengthen, never decide alone)
    if AADHAAR_RE.search(upper_text):
        scores["Aadhaar"] += 2
    if PAN_RE.search(upper_text):
        scores["PAN"] += 3
    if DL_RE.search(upper_text):
        scores["Driving License"] += 2
    if VOTER_RE.search(upper_text):
        scores["Voter ID"] += 2

    logger.info(f"[ID-detect] Scores after regex boost: {scores}")

    # Aadhaar guard: require at least one strong keyword
    aadhaar_strong = any(
        kw in upper_text
        for kw in ("UIDAI", "AADHAAR", "AADHAR", "UNIQUE IDENTIFICATION")
    )
    if not aadhaar_strong and scores["Aadhaar"] > 0:
        logger.info(
            "[ID-detect] Aadhaar score zeroed — no strong keyword found "
            "(12-digit number alone is not sufficient)"
        )
        scores["Aadhaar"] = 0

    # College override: suppress false Aadhaar when college keywords present
    words_in_text = set(upper_text.split())
    college_hits = COLLEGE_OVERRIDE_KEYWORDS & words_in_text
    for phrase in ("ROLL NO", "REG NO", "ENROLLMENT NO"):
        if phrase in upper_text:
            college_hits.add(phrase)
    if college_hits and scores["College ID"] > 0 and scores["Aadhaar"] > 0:
        logger.info(
            f"[ID-detect] College ID override suppressing Aadhaar "
            f"(matched: {college_hits})"
        )
        scores["Aadhaar"] = 0

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_type,    top_score    = ranked[0]
    second_type, second_score = ranked[1] if len(ranked) > 1 else ("Other", 0)

    logger.info(
        f"[ID-detect] Top: {top_type}={top_score}, "
        f"Second: {second_type}={second_score}"
    )

    if top_score < ID_MIN_SCORE_THRESHOLD:
        logger.info(
            f"[ID-detect] Top score {top_score} below threshold "
            f"{ID_MIN_SCORE_THRESHOLD} -> 'Other'"
        )
        return "Other"

    if (top_score - second_score) < ID_AMBIGUITY_GAP:
        logger.info(
            f"[ID-detect] Score gap {top_score - second_score} < "
            f"{ID_AMBIGUITY_GAP} -> 'Other' (ambiguous)"
        )
        return "Other"

    logger.info(f"[ID-detect] Final classification: {top_type}")
    return top_type


# =============================================================================
# CONTEXTUAL DATE EXTRACTION
# =============================================================================

DOB_CONTEXT_WINDOW = 80

DOB_CONTEXT_KEYWORDS = [
    "DOB", "DATE OF BIRTH", "D.O.B", "BIRTH DATE",
    "BIRTH", "YOB", "YEAR OF BIRTH",
    "\u091c\u0928\u094d\u092e",
    "\u091c\u0928\u094d\u092e \u0924\u093f\u0925\u093f",
]

ISSUE_DATE_KEYWORDS = [
    "ISSUE DATE", "DATE OF ISSUE", "ISSUED ON", "ISSUED DATE", "ISSUE"
]

EXPIRY_DATE_KEYWORDS = [
    "EXPIRY DATE", "DATE OF EXPIRY", "VALID TILL", "VALID UPTO",
    "EXPIRY", "EXPIRE", "VALIDITY UPTO"
]

VALIDITY_KEYWORDS = [
    "VALIDITY", "VALID FROM", "VALID TILL", "VALID UPTO", "VALID"
]

YEAR_ONLY_RE = re.compile(r"\b(19[4-9]\d|20[0-2]\d)\b")


def _find_date_near_keyword(text: str, keywords: list[str]) -> str | None:
    """
    Search for a date pattern within DOB_CONTEXT_WINDOW characters after
    any occurrence of the given keywords in text.

    Returns the first date found, or None.
    """
    upper_text = text.upper()

    for keyword in keywords:
        kw_upper = keyword.upper()
        idx = upper_text.find(kw_upper)
        while idx != -1:
            start = max(0, idx - 10)
            end   = min(len(text), idx + len(keyword) + DOB_CONTEXT_WINDOW)
            window = text[start:end]

            date_match = DOB_RE.search(window)
            if date_match:
                return date_match.group()

            idx = upper_text.find(kw_upper, idx + 1)

    return None


def extract_dob_contextual(text: str) -> str | None:
    """
    Extract Date of Birth ONLY when it appears near a DOB-related keyword.

    Prevents random dates (issue dates, expiry dates) from being labelled
    as DOB.
    """
    dob = _find_date_near_keyword(text, DOB_CONTEXT_KEYWORDS)
    if dob:
        logger.info(f"[dob] Found contextual DOB: {dob}")
        return dob

    # Fallback: year-only DOB near YOB / YEAR OF BIRTH keywords
    upper_text = text.upper()
    for keyword in ("YEAR OF BIRTH", "YOB", "\u091c\u0928\u094d\u092e"):
        kw_upper = keyword.upper()
        idx = upper_text.find(kw_upper)
        if idx != -1:
            window = text[idx: idx + len(keyword) + 30]
            year_match = YEAR_ONLY_RE.search(window)
            if year_match:
                logger.info(f"[dob] Year-only DOB '{year_match.group()}' near '{keyword}'")
                return year_match.group()

    logger.debug("[dob] No contextual DOB found")
    return None


def extract_issue_date_contextual(text: str) -> str | None:
    """Extract issue date only when near an issue-date keyword."""
    result = _find_date_near_keyword(text, ISSUE_DATE_KEYWORDS)
    if result:
        logger.info(f"[date] Found issue_date: {result}")
    return result


def extract_expiry_date_contextual(text: str) -> str | None:
    """Extract expiry date only when near an expiry keyword."""
    result = _find_date_near_keyword(text, EXPIRY_DATE_KEYWORDS)
    if result:
        logger.info(f"[date] Found expiry_date: {result}")
    return result


def extract_validity_contextual(text: str) -> str | None:
    """Extract validity date only when near a validity keyword."""
    result = _find_date_near_keyword(text, VALIDITY_KEYWORDS)
    if result:
        logger.info(f"[date] Found validity: {result}")
    return result


# =============================================================================
# CONTEXTUAL AADHAAR EXTRACTION — VID-SAFE
# =============================================================================

# Keywords that indicate a line is about the Aadhaar number
_AADHAAR_PREFER_KEYWORDS = {"AADHAAR", "AADHAR", "UIDAI", "UNIQUE IDENTIFICATION"}

# Keywords that indicate a line should be skipped (VID line)
_AADHAAR_SKIP_KEYWORDS   = {"VID"}


def extract_aadhaar_contextual(text: str) -> str | None:
    """
    Extract Aadhaar number from OCR text while safely rejecting VID numbers.

    Algorithm:
      1. Collect all 16-digit VID numbers in the full text (stripped of
         spaces/hyphens) to build a rejection set.
      2. Walk each line of the text:
         - Skip any line containing "VID".
         - For every 12-digit candidate on that line:
             a. Strip spaces/hyphens → 12 raw digits.
             b. Reject if those 12 digits appear as a contiguous substring
                inside any detected VID number.
             c. Optionally prefer candidates on lines that contain Aadhaar
                keywords (AADHAAR / AADHAR / UIDAI / UNIQUE IDENTIFICATION).
      3. Return the first accepted 12-digit candidate, preferring keyword
         lines. Return None if nothing valid is found.

    Args:
        text: Cleaned OCR text (multi-line).

    Returns:
        12-digit Aadhaar string (digits only, no spaces/hyphens), or None.
    """
    # Step 1: collect all VID digit-strings present in the full text
    vid_digit_strings: set[str] = set()
    for vid_match in VID_RE.finditer(text):
        vid_digits = re.sub(r"[ -]", "", vid_match.group())
        vid_digit_strings.add(vid_digits)
        logger.debug(f"[aadhaar] VID detected and blacklisted: {vid_digits}")

    preferred: list[str] = []   # candidates on Aadhaar-keyword lines
    fallback:  list[str] = []   # candidates on all other lines

    for line in text.splitlines():
        line_upper = line.upper()

        # Skip lines that are explicitly about the VID
        if any(kw in line_upper for kw in _AADHAAR_SKIP_KEYWORDS):
            logger.debug(f"[aadhaar] Skipping VID line: '{line.strip()}'")
            continue

        is_keyword_line = any(kw in line_upper for kw in _AADHAAR_PREFER_KEYWORDS)

        for match in AADHAAR_RE.finditer(line):
            candidate_digits = re.sub(r"[ -]", "", match.group())

            # Reject if this 12-digit string is a substring of any VID
            if any(candidate_digits in vid for vid in vid_digit_strings):
                logger.debug(
                    f"[aadhaar] Rejected '{candidate_digits}' "
                    f"— substring of a VID number"
                )
                continue

            logger.debug(
                f"[aadhaar] Candidate '{candidate_digits}' "
                f"(keyword_line={is_keyword_line})"
            )
            if is_keyword_line:
                preferred.append(candidate_digits)
            else:
                fallback.append(candidate_digits)

    for candidate in preferred + fallback:
        logger.info(f"[aadhaar] Accepted Aadhaar: {candidate}")
        return candidate

    logger.debug("[aadhaar] No valid Aadhaar found after VID filtering")
    return None


# =============================================================================
# KEYWORD-GUARDED PHONE EXTRACTION
# =============================================================================

# Phone numbers are only extracted when at least one of these keywords
# appears within PHONE_CONTEXT_WINDOW characters of the digit sequence.
PHONE_CONTEXT_KEYWORDS = {"MOBILE", "PHONE", "MOB", "CONTACT", "PH", "TEL"}
PHONE_CONTEXT_WINDOW   = 60   # characters to scan before the phone number


def extract_phone_contextual(text: str, exclude_digits: str = "") -> str | None:
    """
    Extract an Indian mobile number ONLY when it appears near a phone keyword.

    Prevents address digits, enrollment numbers, and other 10-digit sequences
    from being labelled as phone numbers.

    Args:
        text:           Cleaned OCR text.
        exclude_digits: Digit string (e.g. Aadhaar number) to exclude from
                        matching — avoids re-labelling the ID number.

    Returns:
        10-digit phone string, or None.
    """
    upper_text = text.upper()

    for match in PHONE_RE.finditer(text):
        num = match.group()

        # Skip if this number is a substring of the excluded digits
        if exclude_digits and num in exclude_digits:
            logger.debug(f"[phone] Skipping '{num}' — overlaps excluded digits")
            continue

        # Check for a phone keyword within PHONE_CONTEXT_WINDOW chars before
        start = max(0, match.start() - PHONE_CONTEXT_WINDOW)
        context_before = upper_text[start: match.start()]

        if any(kw in context_before for kw in PHONE_CONTEXT_KEYWORDS):
            logger.info(f"[phone] Found contextual phone: {num}")
            return num

        logger.debug(
            f"[phone] Skipping '{num}' — no phone keyword in context window"
        )

    return None


# =============================================================================
# ID-TYPE-AWARE REGEX EXTRACTION
# =============================================================================

def _has_strong_keyword(text_upper: str, keywords: set[str]) -> bool:
    """Return True if any keyword from the set is present in text_upper."""
    return any(kw in text_upper for kw in keywords)


# =============================================================================
# AADHAAR NAME AND FATHER NAME CONTEXTUAL EXTRACTION
# =============================================================================

# Relation prefixes found on Aadhaar cards directly above/after the holder name
_RELATION_KEYWORDS = ["S/O", "D/O", "W/O", "C/O"]

# Header labels that should never be treated as the cardholder name
_AADHAAR_HEADER_REJECT = {"GOVERNMENT", "INDIA", "UIDAI", "AADHAAR", "AADHAR"}


def extract_aadhaar_name(text: str) -> str | None:
    """
    Extract the Aadhaar cardholder name by locating the relation line
    (S/O, D/O, W/O, C/O) and returning the line immediately above it.

    Algorithm:
      - Walk each line.
      - When a relation keyword is found, the previous non-empty line is
        the candidate name.
      - Reject the candidate if it:
          * is too short (≤ 2 chars)
          * itself contains a relation keyword (avoid picking up "S/O ...")
          * contains any digit (names never have digits)
          * is a known header label (GOVERNMENT OF INDIA / UIDAI / etc.)
      - Return the first accepted candidate, or None.

    Args:
        text: Cleaned OCR text (multi-line).

    Returns:
        Cardholder name string, or None.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    for i, line in enumerate(lines):
        upper = line.upper()

        if any(rel in upper for rel in _RELATION_KEYWORDS):
            if i > 0:
                candidate = lines[i - 1].strip()
                candidate_upper = candidate.upper()

                if (
                    len(candidate) > 2
                    and not any(rel in candidate_upper for rel in _RELATION_KEYWORDS)
                    and not re.search(r"\d", candidate)
                    and not any(hdr in candidate_upper for hdr in _AADHAAR_HEADER_REJECT)
                ):
                    logger.info(f"[aadhaar_name] Extracted name: '{candidate}'")
                    return candidate

    logger.debug("[aadhaar_name] No cardholder name found via relation-line method")
    return None


# Regex patterns for each relation prefix — capture the name that follows
_RELATION_PATTERNS = [
    re.compile(r"S/O\s*(.*)", re.IGNORECASE),
    re.compile(r"D/O\s*(.*)", re.IGNORECASE),
    re.compile(r"W/O\s*(.*)", re.IGNORECASE),
    re.compile(r"C/O\s*(.*)", re.IGNORECASE),
]


def extract_father_name(text: str) -> str | None:
    """
    Extract the father/husband/guardian name from an Aadhaar card by
    matching the text that follows a relation keyword (S/O, D/O, W/O, C/O).

    Args:
        text: Cleaned OCR text (multi-line).

    Returns:
        Father/guardian name string, or None.
    """
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        for pattern in _RELATION_PATTERNS:
            match = pattern.search(line)
            if match:
                father = match.group(1).strip()
                # Reject if too short or contains digits (OCR noise)
                if len(father) > 2 and not re.search(r"\d", father):
                    logger.info(f"[father_name] Extracted: '{father}'")
                    return father

    logger.debug("[father_name] No father/guardian name found")
    return None


def extract_regex_data(text: str, id_type: str, preprocessed_image: np.ndarray = None) -> dict:
    """
    Extract fields from OCR text using regex patterns.

    STRICT ID-TYPE-AWARE: only fields relevant to the detected id_type are
    extracted. This prevents phone numbers becoming license numbers, random
    12-digit sequences becoming Aadhaar numbers, etc.

    - Email is NEVER extracted (removed from all ID types).
    - possible_dates is NEVER returned; dates are only extracted contextually
      when they clearly belong to a named date field.
    - Aadhaar number is only extracted for Aadhaar cards or when strong
      Aadhaar keywords exist.
    - DL number is only extracted for Driving License cards or when strong
      DL keywords exist.

    Args:
        text:              Cleaned OCR text
        id_type:           Detected ID type
        preprocessed_image: Optional image for Tesseract PAN fallback

    Returns:
        Dict of extracted fields relevant to the id_type only.
    """
    result = {}
    upper_text = text.upper()

    # ------------------------------------------------------------------
    # AADHAAR
    # ------------------------------------------------------------------
    if id_type == "Aadhaar":
        aadhaar = extract_aadhaar_contextual(text)
        if aadhaar:
            result["aadhaar_number"] = aadhaar
            logger.info(f"Aadhaar found: {result['aadhaar_number']}")

        name = extract_aadhaar_name(text)
        if name:
            result["name"] = name

        father_name = extract_father_name(text)
        if father_name:
            result["father_name"] = father_name

        dob = extract_dob_contextual(text)
        if dob:
            result["date_of_birth"] = dob

        phone = extract_phone_contextual(
            text,
            exclude_digits=result.get("aadhaar_number", "")
        )
        if phone:
            result["phone_number"] = phone

    # ------------------------------------------------------------------
    # PAN
    # ------------------------------------------------------------------
    elif id_type == "PAN":
        corrected_pan = correct_pan_candidates(upper_text)

        # Tesseract fallback when PaddleOCR misses the PAN
        if not corrected_pan and preprocessed_image is not None:
            logger.info("Trying Tesseract PAN fallback...")
            try:
                tesseract_text = tesseract_pan_ocr(preprocessed_image)
                corrected_pan = correct_pan_candidates(tesseract_text)
            except Exception as e:
                logger.warning(f"Tesseract fallback failed: {e}")

        if corrected_pan:
            result["pan_number"] = corrected_pan

        dob = extract_dob_contextual(text)
        if dob:
            result["date_of_birth"] = dob

    # ------------------------------------------------------------------
    # DRIVING LICENSE
    # ------------------------------------------------------------------
    elif id_type == "Driving License":
        dl_match = DL_RE.search(upper_text)
        if dl_match:
            dl_raw = dl_match.group().replace(" ", "").replace("-", "")
            result["license_number"] = dl_raw
            logger.info(f"DL found: {dl_raw}")

        dob = extract_dob_contextual(text)
        if dob:
            result["date_of_birth"] = dob

        issue = extract_issue_date_contextual(text)
        if issue:
            result["issue_date"] = issue

        expiry = extract_expiry_date_contextual(text)
        if expiry:
            result["expiry_date"] = expiry

    # ------------------------------------------------------------------
    # VOTER ID
    # ------------------------------------------------------------------
    elif id_type == "Voter ID":
        voter_match = VOTER_RE.search(upper_text)
        if voter_match:
            result["voter_id_number"] = voter_match.group().upper()
            logger.info(f"Voter ID found: {result['voter_id_number']}")

        dob = extract_dob_contextual(text)
        if dob:
            result["date_of_birth"] = dob

    # ------------------------------------------------------------------
    # COLLEGE ID
    # College IDs do NOT have Aadhaar, PAN, DL, phone, or voter numbers.
    # Only name, college_name, roll_number, department, course, year,
    # and validity are extracted (all by the LLM from text context).
    # Regex only handles validity date if present.
    # ------------------------------------------------------------------
    elif id_type == "College ID":
        validity = extract_validity_contextual(text)
        if validity:
            result["validity"] = validity

    # ------------------------------------------------------------------
    # RATION CARD
    # ------------------------------------------------------------------
    elif id_type == "Ration Card":
        # Ration card numbers are highly variable; leave to LLM.
        # No phone, Aadhaar, PAN, or DL extraction here.
        pass

    # ------------------------------------------------------------------
    # OTHER — minimal extraction only
    # Apply Aadhaar / DL extraction ONLY when strong keywords confirm type
    # ------------------------------------------------------------------
    else:
        # Aadhaar: only if strong Aadhaar keywords present
        if _has_strong_keyword(upper_text, AADHAAR_STRONG_KEYWORDS):
            aadhaar = extract_aadhaar_contextual(text)
            if aadhaar:
                result["aadhaar_number"] = aadhaar
                logger.info(f"[Other] Aadhaar found: {result['aadhaar_number']}")

        # DL: only if strong DL keywords present
        if _has_strong_keyword(upper_text, DL_STRONG_KEYWORDS):
            dl_match = DL_RE.search(upper_text)
            if dl_match:
                dl_raw = dl_match.group().replace(" ", "").replace("-", "")
                result["license_number"] = dl_raw
                logger.info(f"[Other] DL found: {dl_raw}")

        dob = extract_dob_contextual(text)
        if dob:
            result["date_of_birth"] = dob

    logger.info(f"[regex] Extracted fields for {id_type}: {list(result.keys())}")
    return result


# =============================================================================
# FIELD VALIDATION
# =============================================================================

DL_VALIDATE_RE   = re.compile(r"^[A-Z]{2}\d{2}\d{4,13}$", re.IGNORECASE)
VOTER_VALIDATE_RE = re.compile(r"^[A-Z]{3}[0-9]{7}$", re.IGNORECASE)


def _mask_for_log(value: str, keep_last: int = 4) -> str:
    """Return a masked version of a sensitive value safe for log output."""
    if not value or len(value) <= keep_last:
        return "****"
    return "*" * (len(value) - keep_last) + value[-keep_last:]


def validate_extracted_fields(data: dict) -> dict:
    """
    Validate and sanitise every extracted field.

    Invalid fields are set to None (never silently kept).
    A '_validated' boolean is added:
    - True  → all present ID-number fields passed validation.
    - False → at least one ID-number field failed.
    """
    all_valid = True

    # ---- Aadhaar ----
    if "aadhaar_number" in data:
        raw     = str(data["aadhaar_number"])
        aadhaar = re.sub(r"\D", "", raw)
        if len(aadhaar) == 12:
            # Verhoeff checksum validation — reject if checksum fails
            if VERHOEFF_AVAILABLE:
                if validateVerhoeff(aadhaar):
                    data["aadhaar_number"] = aadhaar
                    logger.info(
                        f"[validate] Aadhaar checksum OK: "
                        f"{_mask_for_log(aadhaar)}"
                    )
                else:
                    logger.warning(
                        f"[validate] Aadhaar failed Verhoeff checksum "
                        f"(masked='{_mask_for_log(aadhaar)}') -> nulled"
                    )
                    data["aadhaar_number"] = None
                    all_valid = False
            else:
                # Verhoeff not available — accept on length alone
                data["aadhaar_number"] = aadhaar
        else:
            logger.warning(
                f"[validate] Aadhaar invalid "
                f"(len={len(aadhaar)}, masked='{_mask_for_log(aadhaar)}') -> nulled"
            )
            data["aadhaar_number"] = None
            all_valid = False

    # ---- PAN ----
    if "pan_number" in data:
        pan = str(data["pan_number"]).strip().upper()
        if PAN_RE.fullmatch(pan):
            data["pan_number"] = pan
        else:
            logger.warning(
                f"[validate] PAN invalid "
                f"(masked='{_mask_for_log(pan, 3)}') -> nulled"
            )
            data["pan_number"] = None
            all_valid = False

    # ---- Phone ----
    if "phone_number" in data:
        phone = re.sub(r"\D", "", str(data["phone_number"]))
        if len(phone) == 10 and phone[0] in "6789":
            data["phone_number"] = phone
        else:
            logger.warning(
                f"[validate] Phone invalid (len={len(phone)}) -> nulled"
            )
            data["phone_number"] = None
            all_valid = False

    # ---- Driving License ----
    if "license_number" in data:
        dl = re.sub(r"[\s\-]", "", str(data["license_number"])).upper()
        if DL_VALIDATE_RE.fullmatch(dl):
            data["license_number"] = dl
        else:
            logger.warning(
                f"[validate] DL invalid "
                f"(masked='{_mask_for_log(dl)}') -> nulled"
            )
            data["license_number"] = None
            all_valid = False

    # ---- Voter ID ----
    if "voter_id_number" in data:
        vid = str(data["voter_id_number"]).strip().upper()
        if VOTER_VALIDATE_RE.fullmatch(vid):
            data["voter_id_number"] = vid
        else:
            logger.warning(
                f"[validate] Voter ID invalid "
                f"(masked='{_mask_for_log(vid)}') -> nulled"
            )
            data["voter_id_number"] = None
            all_valid = False

    # Remove None values to keep output clean
    data = {k: v for k, v in data.items() if v is not None}

    # NOTE: _validated field excluded per data redaction requirements
    # logger.info(f"[validate] _validated={all_valid}")
    return data


# =============================================================================
# FINAL FIELD FILTERING — enforce allowed fields per ID type
# =============================================================================

def filter_fields_by_id_type(data: dict, id_type: str) -> dict:
    """
    Remove every field from `data` that is not in the allowed list for
    the detected id_type, preserving all metadata fields.

    This is the last guardrail before the result is returned:
    - Ensures no Aadhaar number leaks into a College ID result.
    - Ensures no DL number leaks into a PAN result.
    - Ensures no email, possible_dates, or other noise fields survive.

    Allowed fields = ID_FIELDS[id_type] ∪ METADATA_FIELDS
    """
    allowed = set(ID_FIELDS.get(id_type, ID_FIELDS["Other"])) | METADATA_FIELDS
    filtered = {k: v for k, v in data.items() if k in allowed}

    removed = set(data.keys()) - set(filtered.keys())
    if removed:
        logger.info(
            f"[field-filter] Removed fields not allowed for {id_type}: "
            f"{sorted(removed)}"
        )

    return filtered


# =============================================================================
# PROMPT BUILDER
# =============================================================================

def build_prompt(cleaned_ocr: str, id_type: str, regex_data: dict) -> str:
    """
    Build an anti-hallucination, type-aware, field-specific LLM extraction prompt.

    Key behaviours:
    - Regex-confirmed values are injected as CONFIRMED DATA.
    - Strict rules prohibit guessing, inferring, or hallucinating any value.
    - 'Omit-if-unclear' rule prevents partial/uncertain values being returned.
    - JSON-only output enforced at both system and user level.
    - PAN-specific correction instructions (PAN type only).
    - Hindi/regional label hints for Aadhaar/Voter ID.
    - Explicitly instructs LLM NOT to extract fields outside the allowed set.
    - No email field in any prompt (removed globally).
    - No possible_dates in any prompt.
    """
    fields = ID_FIELDS.get(id_type, ID_FIELDS["Other"])

    # Inject regex-confirmed data as ground truth
    confirmed_lines: list[str] = []
    for key, value in regex_data.items():
        if value:
            confirmed_lines.append(f'  "{key}": "{value}"')
    confirmed_block = (
        "ALREADY CONFIRMED by regex — do NOT change these values:\n"
        + "{\n" + ",\n".join(confirmed_lines) + "\n}"
        if confirmed_lines
        else "No fields confirmed by regex yet."
    )

    # PAN-specific rules
    pan_block = ""
    if id_type == "PAN":
        pan_block = """
PAN CARD RULES (MANDATORY):
- PAN format: exactly 5 uppercase letters + 4 digits + 1 uppercase letter.
  Example: ABCDE1234F
- Common OCR confusions: O<->0, I<->1, S<->5, B<->8 — correct them.
- 'father_name' is labelled "Father's Name" on the card.
- If you cannot read the PAN clearly, OMIT the field entirely.
"""

    # Aadhaar-specific name extraction rules
    aadhaar_name_block = ""
    if id_type == "Aadhaar":
        aadhaar_name_block = """
AADHAAR NAME EXTRACTION RULES (MANDATORY):
- The cardholder name appears on the line IMMEDIATELY ABOVE S/O, D/O, W/O, or C/O.
- NEVER use S/O, D/O, W/O, or C/O lines as the 'name' field.
- Text after S/O / D/O / W/O / C/O is the father's / husband's / guardian's name
  — place it in 'father_name', NOT in 'name'.
- Ignore these labels entirely when extracting 'name':
    Government of India, UIDAI, Address, DOB, Date of Birth, Male, Female,
    MALE, FEMALE, any numeric sequence, any line that is all-uppercase short label.
- If you cannot confidently identify the cardholder name, OMIT 'name'.
"""

    fields_ref = ", ".join(f'"{f}"' for f in fields)

    # Build a human-readable "DO NOT EXTRACT" warning for LLM
    all_possible_fields = {
        "aadhaar_number", "pan_number", "license_number", "voter_id_number",
        "phone_number", "email", "roll_number", "college_name", "department",
        "course", "year", "validity", "ration_card_number", "vehicle_class",
        "issue_date", "expiry_date", "part_number", "members_count",
        "head_of_family", "card_type", "id_number"
    }
    disallowed_fields = all_possible_fields - set(fields)
    disallowed_ref = ", ".join(f'"{f}"' for f in sorted(disallowed_fields))

    prompt = f"""You are a strict OCR data extraction engine for Indian government ID cards.
The document type is: {id_type}

{'=' * 60}
ABSOLUTE RULES — VIOLATION IS NOT PERMITTED
{'=' * 60}
1. Return ONLY a valid JSON object. No markdown. No explanation. No prose.
2. Extract ONLY text that is CLEARLY AND FULLY VISIBLE in the OCR output.
3. Do NOT guess, infer, or hallucinate any field value.
4. If a field is missing, unclear, partially visible, or ambiguous -> OMIT IT.
5. Do NOT include null, "null", "N/A", "unknown", or empty strings.
6. Do NOT copy confirmed-data values into wrong field names.
7. 'name' must be the cardholder — ignore logos, city names, and labels.
8. 'father_name' must come ONLY from lines labelled S/O, D/O, W/O, Father's Name.
9. 'address' must be the full postal address, not a city name alone.
10. Do NOT fabricate an ID number if it is not clearly readable.
11. Extract ONLY fields relevant to {id_type}. Do NOT infer or extract
    unrelated ID numbers. Ignore random numeric sequences unless they
    clearly and unambiguously belong to the detected document type.
12. NEVER extract: email addresses, website URLs, or domain names.
13. NEVER return "possible_dates" or any list of date candidates.
14. The following fields are NOT allowed for {id_type} and must NEVER appear
    in your output: {disallowed_ref}
{'=' * 60}

EXPECTED FIELDS for {id_type} (include only fields you can extract):
{fields_ref}
{pan_block}{aadhaar_name_block}
REGIONAL LABEL HINTS (Aadhaar / Voter ID):
- \u091c\u0928\u094d\u092e \u0924\u093f\u0925\u093f = Date of Birth
- \u092a\u093f\u0924\u093e = Father | \u092a\u0924\u093e = Address
- \u0932\u093f\u0902\u0917 = Gender | \u092a\u0941\u0930\u0941\u0937 = Male | \u092e\u0939\u093f\u0932\u093e = Female

{confirmed_block}

{'=' * 60}
OCR TEXT (extract ONLY from this source)
{'=' * 60}
{cleaned_ocr}
{'=' * 60}

Respond with a single JSON object containing only the allowed fields you are confident about.
"""
    return prompt


# =============================================================================
# SAFE JSON PARSER
# =============================================================================

def safe_json_load(raw_output: str) -> dict:
    """Robustly parse JSON from LLM output."""
    if not raw_output:
        return {}

    try:
        return json.loads(raw_output)
    except json.JSONDecodeError:
        pass

    cleaned = raw_output.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"```$", "", cleaned, flags=re.MULTILINE)
    cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    cleaned = re.sub(r",\s*}", "}", cleaned)
    cleaned = re.sub(r",\s*]", "]", cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    logger.error("Could not parse JSON from LLM output.")
    return {}


# =============================================================================
# LLM EXTRACTION
# =============================================================================

def call_groq_llm(prompt: str) -> str:
    """Call Groq LLM with the extraction prompt."""
    client = get_groq_client()

    logger.info(f"Calling Groq LLM ({GROQ_MODEL})...")
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict OCR JSON extraction engine for Indian ID cards. "
                    "Return ONLY valid JSON. Never hallucinate. Never add explanations. "
                    "Never include email addresses, website domains, or fields not "
                    "relevant to the detected ID type."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0,
        max_tokens=1024,
    )

    output = response.choices[0].message.content
    logger.info("LLM response received.")
    return output


# =============================================================================
# CONFIDENCE SCORING
# =============================================================================

def score_extraction(
    extracted: dict,
    id_type: str,
    ocr_lines: list[dict] | None = None
) -> dict:
    """
    Compute extraction confidence combining field coverage and OCR quality.

    Two components:
    1. Field coverage score (0.0–1.0): fraction of expected fields filled.
    2. OCR mean confidence (0.0–1.0): average PaddleOCR line confidence.

    Final score = 0.7 × field_coverage + 0.3 × ocr_confidence
    """
    fields = ID_FIELDS.get(id_type, ID_FIELDS["Other"])

    filled = sum(
        1 for f in fields
        if extracted.get(f) not in (None, "", "null")
        and not str(f).startswith("_")
    )
    total = len(fields)
    field_score = round(filled / total, 4) if total > 0 else 0.0

    ocr_conf = None
    if ocr_lines:
        confs = [ln["confidence"] for ln in ocr_lines if "confidence" in ln]
        if confs:
            ocr_conf = round(sum(confs) / len(confs), 4)

    if ocr_conf is not None:
        combined = round(0.7 * field_score + 0.3 * ocr_conf, 4)
    else:
        combined = field_score

    # NOTE: Underscore-prefixed metadata fields excluded per data redaction requirements
    # These fields are NO LONGER added to extracted data

    logger.info(
        f"[score] field_coverage={field_score} ({filled}/{total}), "
        f"ocr_confidence={ocr_conf}, combined={combined}"
    )
    return extracted


# =============================================================================
# REDACTION
# =============================================================================

def _partial_mask(value: str, mask_char: str = "X", keep_last: int = 4) -> str:
    """Mask all but the last `keep_last` characters."""
    digits_only = re.sub(r"\D", "", value)
    if len(digits_only) >= keep_last:
        masked = mask_char * (len(digits_only) - keep_last) + digits_only[-keep_last:]
        if len(digits_only) == 12:
            masked = f"{mask_char*4} {mask_char*4} {digits_only[-4:]}"
        return masked
    if len(value) > keep_last:
        return mask_char * (len(value) - keep_last) + value[-keep_last:]
    return mask_char * len(value)


_REDACT_FIELDS: dict[str, int] = {
    "aadhaar_number":  4,
    "pan_number":      3,
    "phone_number":    4,
    "license_number":  4,
    "voter_id_number": 4,
}


def redact_sensitive_fields(data: dict, mode: str = "partial") -> dict:
    """
    Return a copy of `data` with sensitive fields masked.

    Only validated (non-None) fields are redacted.
    The original `data` dict is NOT mutated.

    Args:
        data:  Extracted data dict (post-validation).
        mode:  'partial' -> mask all but last N chars (default).
               'full'    -> replace entire value with 'REDACTED'.
    """
    redacted = dict(data)

    for field, keep_last in _REDACT_FIELDS.items():
        if field not in redacted or redacted[field] is None:
            continue

        original = str(redacted[field])

        if mode == "full" or keep_last == 0:
            redacted[field] = "REDACTED"
            logger.debug(f"[redact] {field} fully redacted")
        else:
            redacted[field] = _partial_mask(original, keep_last=keep_last)
            logger.debug(f"[redact] {field} partially masked")

    return redacted


# =============================================================================
# MAIN EXTRACTION PIPELINE
# =============================================================================

def extract_id_data(file_path: str) -> dict:
    """
    Full end-to-end production pipeline for Indian ID card OCR extraction.

    Steps:
     1.  Load image or PDF
     2.  Generate multiple preprocessing variants
     3.  Run PaddleOCR on each variant
     4.  Deduplicate OCR lines
     5.  Filter OCR noise lines
     6.  Sort and join to text
     7.  Clean OCR text
     8.  Detect ID type
     9.  ID-type-aware regex extraction (no cross-type field bleed)
    10.  Build type-aware LLM prompt
    11.  Call Groq LLM
    12.  Parse and merge results (regex overrides LLM for ID numbers)
    13.  Add metadata
    14.  Validate extracted fields
    15.  Score extraction quality
    16.  FINAL FIELD FILTER: remove all fields not in allowed list for id_type
    17.  Redact for safe logging

    Args:
        file_path: Path to image (.jpg, .png, etc.) or PDF (.pdf)

    Returns:
        Dict with extracted fields (allowed for id_type only) + metadata
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    logger.info(f"{'='*60}")
    logger.info(f"Processing: {file_path.name}")
    logger.info(f"{'='*60}")

    all_ocr_lines = []
    last_preprocessed = None

    # ------------------------------------------------------------------
    # STEP 1–3: Load → multi-variant preprocessing → multi-pass OCR
    # ------------------------------------------------------------------
    if file_path.suffix.lower() == ".pdf":
        logger.info("PDF detected — converting pages to images...")
        page_images = pdf_to_images(str(file_path))

        for page_num, raw_img in enumerate(page_images):
            logger.info(f"Processing PDF page {page_num + 1}...")

            variants = generate_image_variants(raw_img)
            last_preprocessed = variants[0]

            for idx, variant in enumerate(variants):
                logger.info(f"Running OCR variant {idx + 1}/{len(variants)}...")
                ocr_lines = run_ocr(variant)
                all_ocr_lines.extend(ocr_lines)
    else:
        logger.info("Image file detected.")
        raw_img = cv2.imread(str(file_path))
        if raw_img is None:
            raise FileNotFoundError(f"Cannot read image: {file_path}")

        variants = generate_image_variants(raw_img)
        last_preprocessed = variants[0]

        for idx, variant in enumerate(variants):
            logger.info(f"Running OCR variant {idx + 1}/{len(variants)}...")
            ocr_lines = run_ocr(variant)
            all_ocr_lines.extend(ocr_lines)

    logger.info(f"Total OCR lines before dedup: {len(all_ocr_lines)}")

    # ------------------------------------------------------------------
    # STEP 4: Deduplicate OCR lines
    # ------------------------------------------------------------------
    all_ocr_lines = deduplicate_ocr_lines(all_ocr_lines)

    # ------------------------------------------------------------------
    # STEP 5: Filter OCR noise lines
    # ------------------------------------------------------------------
    all_ocr_lines = clean_ocr_lines(all_ocr_lines)

    # ------------------------------------------------------------------
    # STEP 6: Sort and convert to text
    # ------------------------------------------------------------------
    raw_ocr_text = ocr_lines_to_text(all_ocr_lines)
    logger.debug(f"\n--- RAW OCR TEXT ---\n{raw_ocr_text}\n---")

    # ------------------------------------------------------------------
    # STEP 7: Clean OCR text
    # ------------------------------------------------------------------
    cleaned_ocr = clean_text(raw_ocr_text)
    logger.debug(f"\n--- CLEANED OCR ---\n{cleaned_ocr}\n---")

    # ------------------------------------------------------------------
    # STEP 8: Detect ID type
    # ------------------------------------------------------------------
    id_type = detect_id_type(cleaned_ocr)
    logger.info(f"ID Type: {id_type}")

    # ------------------------------------------------------------------
    # STEP 9: ID-type-aware regex extraction
    # ------------------------------------------------------------------
    regex_data = extract_regex_data(cleaned_ocr, id_type, last_preprocessed)
    logger.info(f"Regex extracted: {list(regex_data.keys())}")

    # ------------------------------------------------------------------
    # STEP 10: Build type-aware LLM prompt
    # ------------------------------------------------------------------
    prompt = build_prompt(cleaned_ocr, id_type, regex_data)

    # ------------------------------------------------------------------
    # STEP 11: Call Groq LLM
    # ------------------------------------------------------------------
    try:
        raw_llm_output = call_groq_llm(prompt)
        logger.debug("[llm] Response received (content not logged for privacy)")
    except Exception as e:
        logger.error(f"[llm] LLM call failed: {e}")
        raw_llm_output = "{}"

    # ------------------------------------------------------------------
    # STEP 12: Parse LLM output and merge with regex findings
    # Regex overrides LLM for ID numbers (regex is more reliable).
    # For Aadhaar, regex name and father_name also take priority over LLM.
    # ------------------------------------------------------------------
    llm_data = safe_json_load(raw_llm_output)

    # Strip any email / possible_dates that LLM may have hallucinated
    llm_data.pop("email", None)
    llm_data.pop("possible_dates", None)

    # Fields where regex is always authoritative (never let LLM overwrite)
    _REGEX_PRIORITY_FIELDS = {
        "aadhaar_number", "pan_number", "license_number", "voter_id_number",
    }
    # For Aadhaar, contextual name extraction is more reliable than LLM inference
    if id_type == "Aadhaar":
        _REGEX_PRIORITY_FIELDS |= {"name", "father_name"}

    for key, value in regex_data.items():
        if value:
            if key in _REGEX_PRIORITY_FIELDS or key not in llm_data:
                llm_data[key] = value

    # ------------------------------------------------------------------
    # STEP 13: Metadata
    # ------------------------------------------------------------------
    llm_data["id_type"]     = id_type
    llm_data["source_file"] = file_path.name

    # ------------------------------------------------------------------
    # STEP 14: Validate extracted fields
    # ------------------------------------------------------------------
    llm_data = validate_extracted_fields(llm_data)

    # ------------------------------------------------------------------
    # STEP 15: Score extraction quality
    # ------------------------------------------------------------------
    scored = score_extraction(llm_data, id_type, ocr_lines=all_ocr_lines)

    # ------------------------------------------------------------------
    # STEP 16: FINAL FIELD FILTER — remove all disallowed fields
    # This is the last guardrail; nothing outside the allowed set for
    # this id_type can survive into the returned result.
    # ------------------------------------------------------------------
    final_result = filter_fields_by_id_type(scored, id_type)

    # ------------------------------------------------------------------
    # STEP 17: Redact for safe logging
    # ------------------------------------------------------------------
    redacted_for_log = redact_sensitive_fields(final_result, mode="partial")
    logger.info(
        f"[pipeline] Final result (redacted for log): "
        f"{json.dumps(redacted_for_log, indent=2, ensure_ascii=False)}"
    )

    # Return UNREDACTED data — redact at the API/UI boundary as needed
    return final_result


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        # Default test file — change this to your ID image path
        file_path = "id_images/pan_card.jpg"

    try:
        result = extract_id_data(file_path)

        print("\n" + "=" * 60)
        print("FINAL EXTRACTION RESULT")
        print("=" * 60)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        print("=" * 60)

    except FileNotFoundError as e:
        logger.error(str(e))
        print(f"\nError: {e}")
        print("Please provide a valid path to an ID card image or PDF.")
        sys.exit(1)

    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        print(f"\nUnexpected error: {e}")
        sys.exit(1)