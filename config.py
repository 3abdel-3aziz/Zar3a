from pathlib import Path

# Base Data Directory for raw inputs (PDFs, raw documents, etc.)
DATA_DIR: Path = Path("Data")

# Unified output directory for all processed JSON files (OCR, Web Scraping, Crawl4AI, etc.)
OUTPUT_JSON_DIR: Path = Path("Data/processed_json")
PROCESSED_JSON_DIR: Path = OUTPUT_JSON_DIR

# Ensure default directories exist upon config load
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_JSON_DIR.mkdir(parents=True, exist_ok=True)