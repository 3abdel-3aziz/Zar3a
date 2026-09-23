from pathlib import Path

# Base Data Directory for raw inputs (PDFs, raw documents, etc.)
DATA_DIR: Path = Path("Data")

# Unified output directory for all processed JSON files (OCR, Web Scraping, Crawl4AI, etc.)
OUTPUT_JSON_DIR: Path = Path("Data/processed_json")
PROCESSED_JSON_DIR: Path = OUTPUT_JSON_DIR

# Directory for raw Arabic PDF files downloaded via multi-page PDF scraper
ARABIC_DATA_DIR: Path = DATA_DIR / "Arabic_Data"

# Ensure default directories exist upon config load
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_JSON_DIR.mkdir(parents=True, exist_ok=True)
ARABIC_DATA_DIR.mkdir(parents=True, exist_ok=True)

# ── Pipeline Source URLs ──────────────────────────────────────────────────────

# WikiSource URL for direct LLM text extraction (Egyptian Unified Building Law 119/2008)
DIRECT_TEXT_URL: str = (
    "https://ar.wikisource.org/wiki/"
    "\u0642\u0627\u0646\u0648\u0646_\u0627\u0644\u0628\u0646\u0627\u0621"
    "_\u0627\u0644\u0645\u0648\u062d\u062f_119_\u0644\u0633\u0646\u0629"
    "_2008_(\u0645\u0635\u0631)"
)

# Manshurat taxonomy listing URL for multi-page PDF scraping
PAGINATION_PDF_URL: str = (
    "https://manshurat.org/taxonomy/term/56"
    "?sort=search_api_aggregation_1%20DESC"
)