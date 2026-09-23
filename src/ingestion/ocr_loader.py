import os 
import json 
from pathlib import Path 
from mistralai.client import Mistral 

import logging 
# pyrefly: ignore [missing-import]
from config import OUTPUT_JSON_DIR


logging.basicConfig(
    level=logging.INFO,  
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


class MistralPDFProcessor:
    """
    A processor class responsible for handling PDF OCR operations using the Mistral OCR API,
    with built-in local caching to optimize costs and speed up development.
    """
    
    def __init__(self, api_key: str | None = None, output_dir: str | Path | None = None):        
        """
        Initializes the Mistral client and ensures the output directory for JSON cache exists.
        """
        
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
        if not self.api_key:
            logger.error("MISTRAL_API_KEY is missing from environment variables.")
            raise ValueError("MISTRAL_API_KEY is missing. Please set it in your environment variables.")

        self.client = Mistral(api_key=self.api_key)    
        self.output_dir = Path(output_dir) if output_dir is not None else OUTPUT_JSON_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"MistralPDFProcessor initialized with cache directory: {self.output_dir}")

    def is_ocr_cached(self, pdf_path: str | Path) -> bool :       
        """
        Checks if the OCR JSON result for a given PDF already exists in the local cache.
        """

        json_path = self._get_expected_json_path(pdf_path)
        cached = json_path.exists()
        return cached 

    def process_pdf(self, pdf_path: str | Path) -> dict:
        """
        Processes a single PDF file: checks cache first, calls Mistral OCR API if missing, 
        saves the response to cache, and returns the result.
        """     

        pdf_path = Path(pdf_path)

    
        if self.is_ocr_cached(pdf_path) :
            logger.info(f"Loading OCR result from cache for: {pdf_path.name}")
            return self._load_from_cache(pdf_path)

        logger.info(f"Calling Mistral OCR API for file: {pdf_path.name}")
        ocr_response = self._call_mistral_api(pdf_path) 

        self._save_to_cache(pdf_path, ocr_response)
        logger.info(f"Successfully processed and cached OCR for: {pdf_path.name}")

        return ocr_response 

    def _call_mistral_api(self, pdf_path: Path) -> dict:
        """
        Handles the actual communication with the Mistral OCR API.
        """     

        try:
            response = self.client.ocr.process(
                model="mistral-ocr-latest",
                document={
                    "type": "document_url",
                    "document_url": f"file://{pdf_path.absolute()}"
                }
            )
            return response.model_dump()
        except Exception as e:
            logger.error(f"Failed to process OCR via Mistral API for {pdf_path.name}: {e}")
            raise e  

    def _get_expected_json_path(self, pdf_path: Path) -> Path:
        """
        Determines the expected local JSON file path for a given PDF.
        """
        pdf_path = Path(pdf_path)
        return self.output_dir / f"{pdf_path.stem}.json"

    def _save_to_cache(self, pdf_path: Path, data: dict)->  None:
        """
        Saves the OCR dictionary response to a local JSON file.
        """       
        json_path = self._get_expected_json_path(pdf_path)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4) 
        logger.info(f"OCR result saved to cache: {json_path}")

    def _load_from_cache(self, pdf_path: Path) -> dict:
        """
        Loads the cached OCR data from a local JSON file.
        """
        json_path = self._get_expected_json_path(pdf_path)
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.debug(f"Loaded JSON cache from: {json_path}")
        return data    

