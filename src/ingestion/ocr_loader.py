import os
import json
import time
import base64
from pathlib import Path
from typing import List
from dotenv import load_dotenv
from mistralai.client import Mistral

import logging
# pyrefly: ignore [missing-import]
from config import OUTPUT_JSON_DIR

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


class MistralPDFProcessor:
    """
    A processor class responsible for handling PDF OCR operations using the Mistral OCR API,
    with built-in local caching, API key rotation across multiple keys, rate-limit delays,
    and exponential backoff / safety retries.
    """

    def __init__(
        self,
        api_key: str | List[str] | None = None,
        output_dir: str | Path | None = None,
        request_delay: float = 2.0,
        max_retries: int = 5,
        initial_backoff: float = 5.0,
        backoff_factor: float = 2.0,
        safety_delay: float = 30.0,
        inter_file_delay: float = 8.0,
    ):
        """
        Initializes the Mistral client pool and ensures the output directory for JSON cache exists.

        Args:
            api_key: Single API key string, comma-separated string of keys, or list of key strings.
                Defaults to MISTRAL_API_KEY or MISTRAL_API_KEYS environment variables.
            output_dir: Path where processed JSON cache files are saved.
            request_delay: Delay in seconds after each successful API request to respect rate limits.
            max_retries: Maximum retry cycles per key when encountering HTTP 429 rate limits.
            initial_backoff: Initial sleep duration in seconds for single-key backoff retries.
            backoff_factor: Multiplier applied to backoff duration after each single-key failed retry.
            safety_delay: Sleep duration in seconds when all keys in the pool fail consecutively.
            inter_file_delay: Sleep duration in seconds after processing or failing a PDF file before next.
        """

        raw_keys = api_key or os.getenv("MISTRAL_API_KEY") or os.getenv("MISTRAL_API_KEYS")
        if isinstance(raw_keys, list):
            self.api_keys: List[str] = [
                k.strip() for k in raw_keys if k and isinstance(k, str) and k.strip()
            ]
        elif isinstance(raw_keys, str):
            self.api_keys = [k.strip() for k in raw_keys.split(",") if k and k.strip()]
        else:
            self.api_keys = []

        if not self.api_keys:
            logger.error("MISTRAL_API_KEY is missing from environment variables.")
            raise ValueError("MISTRAL_API_KEY is missing. Please set it in your environment variables.")

        self.current_key_index: int = 0
        self.output_dir = Path(output_dir) if output_dir is not None else OUTPUT_JSON_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.request_delay = request_delay
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff
        self.backoff_factor = backoff_factor
        self.safety_delay = safety_delay
        self.inter_file_delay = inter_file_delay

        self.client = self._create_client(self.api_keys[self.current_key_index])

        logger.info(
            f"MistralPDFProcessor initialized with {len(self.api_keys)} API key(s). "
            f"Active key: {self._mask_key(self.api_keys[self.current_key_index])}. "
            f"Inter-file delay: {self.inter_file_delay}s. "
            f"Cache directory: {self.output_dir}"
        )

    def _create_client(self, api_key: str) -> Mistral:
        """
        Creates and returns a Mistral API client instance for the given API key.
        """
        return Mistral(api_key=api_key)

    @staticmethod
    def _mask_key(key: str) -> str:
        """
        Safely masks an API key showing only the last 4 characters.
        """
        if len(key) <= 4:
            return "****"
        return f"...{key[-4:]}"

    def _rotate_key(self) -> str:
        """
        Rotates to the next API key in the pool and updates the Mistral client instance.
        """
        if len(self.api_keys) <= 1:
            return self._mask_key(self.api_keys[self.current_key_index])

        prev_index = self.current_key_index
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)

        old_masked = self._mask_key(self.api_keys[prev_index])
        new_masked = self._mask_key(self.api_keys[self.current_key_index])

        logger.info(
            f"Rotating Mistral API key from {old_masked} (index {prev_index}) "
            f"to {new_masked} (index {self.current_key_index})."
        )
        self.client = self._create_client(self.api_keys[self.current_key_index])
        return new_masked

    def is_ocr_cached(self, pdf_path: str | Path) -> bool:
        """
        Checks if the OCR JSON result for a given PDF already exists in the local cache.
        """

        json_path = self._get_expected_json_path(pdf_path)
        cached = json_path.exists()
        return cached

    def process_pdf(self, pdf_path: str | Path) -> dict:
        """
        Processes a single PDF file: checks cache first, calls Mistral OCR API if missing,
        saves response to cache, and enforces inter-file safety delay before completing.
        """

        pdf_path = Path(pdf_path)

        if self.is_ocr_cached(pdf_path):
            logger.info(f"Loading OCR result from cache for: {pdf_path.name}")
            return self._load_from_cache(pdf_path)

        logger.info(f"Calling Mistral OCR API for file: {pdf_path.name}")
        try:
            ocr_response = self._call_mistral_api(pdf_path)
            self._save_to_cache(pdf_path, ocr_response)
            logger.info(f"Successfully processed and cached OCR for: {pdf_path.name}")
            return ocr_response
        finally:
            if self.inter_file_delay > 0:
                logger.info(
                    f"Inter-file safety delay: sleeping for {self.inter_file_delay:.1f}s "
                    f"before next PDF operation..."
                )
                time.sleep(self.inter_file_delay)

    def _call_mistral_api(self, pdf_path: Path) -> dict:
        """
        Handles communication with the Mistral OCR API by reading the local PDF file in
        binary mode and encoding it into a Base64 Data URI string. Implements API key
        rotation upon HTTP 429 rate limit / quota errors, request delays, and safety backoffs.
        """
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()

        base64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")
        data_url = f"data:application/pdf;base64,{base64_pdf}"

        num_keys = len(self.api_keys)
        total_attempts = self.max_retries * num_keys
        consecutive_failures = 0
        backoff = self.initial_backoff

        for attempt in range(1, total_attempts + 1):
            try:
                response = self.client.ocr.process(
                    model="mistral-ocr-latest",
                    document={
                        "type": "document_url",
                        "document_url": data_url
                    }
                )

                if self.request_delay > 0:
                    time.sleep(self.request_delay)

                return response.model_dump()

            except Exception as e:
                err_str = str(e).lower()
                is_rate_limit = (
                    "429" in err_str
                    or "rate limit" in err_str
                    or "rate_limited" in err_str
                    or "quota" in err_str
                )

                if is_rate_limit:
                    if attempt < total_attempts:
                        consecutive_failures += 1

                        if num_keys > 1:
                            self._rotate_key()

                            if consecutive_failures % num_keys == 0:
                                logger.info(
                                    f"All {num_keys} API keys in pool hit rate limits for {pdf_path.name}. "
                                    f"Sleeping for safety delay of {self.safety_delay:.1f}s..."
                                )
                                time.sleep(self.safety_delay)
                        else:
                            logger.info(
                                f"Rate limit hit (429) for {pdf_path.name}. "
                                f"Retrying in {backoff:.1f}s (Attempt {attempt}/{total_attempts})..."
                            )
                            time.sleep(backoff)
                            backoff *= self.backoff_factor
                    else:
                        logger.error(
                            f"All {num_keys} API key(s) failed for {pdf_path.name} after {total_attempts} total attempt(s) "
                            f"due to persistent HTTP 429 Rate Limit / Quota error. Full Error Details: {e}"
                        )
                        raise e
                else:
                    logger.error(f"Failed to process OCR via Mistral API for {pdf_path.name}: {e}")
                    raise e

        raise RuntimeError(
            f"Failed to process OCR via Mistral API for {pdf_path.name} after {total_attempts} attempts."
        )

    def _get_expected_json_path(self, pdf_path: Path) -> Path:
        """
        Determines the expected local JSON file path for a given PDF.
        """
        pdf_path = Path(pdf_path)
        return self.output_dir / f"{pdf_path.stem}.json"

    def _save_to_cache(self, pdf_path: Path, data: dict) -> None:
        """
        Saves the OCR dictionary response to a local JSON file.
        """
        json_path = self._get_expected_json_path(pdf_path)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        logger.info(f"OCR result saved to cache: {json_path}")

    def _load_from_cache(self, pdf_path: Path) -> dict:
        """
        Loads the cached OCR data from a local JSON file.
        """
        json_path = self._get_expected_json_path(pdf_path)
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"Loaded JSON cache from: {json_path}")
        return data    

