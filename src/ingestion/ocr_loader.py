# ── Critical: disable PaddlePaddle OneDNN & PIR before ANY import ─────────────
# These flags MUST be set before `paddle`, `paddleocr`, or `paddlex` are imported.
# The PaddlePaddle C++ runtime reads them at first load; setting them later has
# no effect.
#
#   FLAGS_use_mkldnn=0   – disables Intel OneDNN (MKL-DNN) execution provider.
#   FLAGS_enable_pir_api=0 – disables the PIR (Program Intermediate Representation)
#                            optimizer that causes:
#                            "ConvertPirAttribute2RuntimeAttribute not support
#                             [pir::ArrayAttribute<pir::DoubleAttribute>]"
import os
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"
# ──────────────────────────────────────────────────────────────────────────────

"""
src/ingestion/ocr_loader.py

Robust, Resilient, and Hybrid OCR / PDF Ingestion Engine for Zar3a.
Includes:
  1. LocalPDFTextExtractor  - Inspects digital PDFs & extracts text locally via pypdf (0 API tokens).
  2. PaddleOCRExtractor     - Local offline OCR engine using PaddleOCR (lang='ar') & PyMuPDF (default).
  3. APIKeyPoolManager      - Manages key rotation, pool health, and key cooldowns for API access.
  4. MistralOCRClient       - Executes OCR calls via Mistral API with exponential backoff + jitter + Retry-After.
  5. MistralPDFProcessor     - Facade coordinator defaulting to 'paddle' OCR with fallback support for 'mistral'.
"""


import json
import time
import base64
import random
import re
import tempfile
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv

try:
    import pypdf
except ImportError:
    pypdf = None

try:
    import pymupdf
except ImportError:
    pymupdf = None

try:
    from paddleocr import PaddleOCR
except ImportError:
    PaddleOCR = None

try:
    from mistralai.client import Mistral
except ImportError:
    Mistral = None

# pyrefly: ignore [missing-import]
from config import OUTPUT_JSON_DIR

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("Zar3a.OCRLoader")


# ── 1. Local Digital PDF Text Extractor (pypdf) ──────────────────────────────

class LocalPDFTextExtractor:
    """
    Inspects PDF files locally to determine if they contain digital/selectable text.
    Extracts text page-by-page if digital text count exceeds threshold, saving API tokens.
    """

    def __init__(self, min_char_threshold: int = 100):
        self.min_char_threshold = min_char_threshold

    def is_pypdf_available(self) -> bool:
        return pypdf is not None

    def extract_text(self, pdf_path: Path) -> Optional[Dict[str, Any]]:
        """
        Attempts to extract digital text locally from the PDF file.
        Returns a dictionary payload matching Zar3a OCR schema if digital text is found,
        or None if the PDF is scanned / empty of selectable text.
        """
        if not self.is_pypdf_available():
            logger.debug("pypdf is not available. Skipping local PDF text pre-check.")
            return None

        try:
            reader = pypdf.PdfReader(str(pdf_path))
            if not reader.pages:
                return None

            extracted_pages: List[Dict[str, Any]] = []
            total_chars = 0

            for idx, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                cleaned_text = text.strip()
                total_chars += len(cleaned_text)
                extracted_pages.append({
                    "index": idx,
                    "markdown": cleaned_text,
                })

            if total_chars >= self.min_char_threshold:
                logger.info(
                    "LocalPDFTextExtractor: Extracted %d characters across %d page(s) from '%s' locally.",
                    total_chars,
                    len(extracted_pages),
                    pdf_path.name,
                )
                return {
                    "pages": extracted_pages,
                    "source": "local_digital_pdf",
                    "total_chars": total_chars,
                }
            else:
                logger.debug(
                    "LocalPDFTextExtractor: PDF '%s' has only %d chars (< threshold %d). Requires OCR.",
                    pdf_path.name,
                    total_chars,
                    self.min_char_threshold,
                )
                return None

        except Exception as exc:
            logger.warning("LocalPDFTextExtractor failed for '%s': %s. Falling back to OCR.", pdf_path.name, exc)
            return None


# ── 2. Local PaddleOCR Offline Extractor ────────────────────────────────────

class PaddleOCRExtractor:
    """
    Local offline OCR engine using PaddleOCR with PyMuPDF for rendering PDF pages to images.
    Supports Arabic ('ar') language recognition offline with automatic cleanup of temp files.
    """

    def __init__(self, lang: str = "ar", use_angle_cls: bool = True):
        self.lang = lang
        # use_angle_cls is the correct constructor param in paddleocr 2.x.
        self.use_angle_cls = use_angle_cls

        if PaddleOCR is None:
            raise RuntimeError("paddleocr package is not installed. Install via `uv add paddleocr`.")

        logger.info("Initializing PaddleOCR instance (lang='%s')...", self.lang)
        # Singleton: initialized once in __init__ and reused across all process_pdf() calls.
        # paddleocr 2.x API: use_angle_cls= (use_textline_orientation= only exists in 3.x/paddlex).
        self._ocr = PaddleOCR(use_angle_cls=self.use_angle_cls, lang=self.lang)


    def extract_text(self, pdf_path: Path) -> Dict[str, Any]:
        """
        Renders PDF pages to images using PyMuPDF, runs PaddleOCR on each page,
        formats detected text lines into markdown strings, cleans temporary files,
        and returns the structured OCR payload.
        """
        pdf_path = Path(pdf_path)
        if not pymupdf:
            raise RuntimeError("pymupdf is not installed. Install via `uv add pymupdf`.")

        ocr_engine = self._ocr
        doc = pymupdf.open(str(pdf_path))
        extracted_pages: List[Dict[str, Any]] = []

        logger.info("PaddleOCR processing file: %s (%d page(s))", pdf_path.name, len(doc))

        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_dir_path = Path(tmp_dir)
                for page_idx, page in enumerate(doc):
                    pix = page.get_pixmap(dpi=200)
                    img_path = tmp_dir_path / f"page_{page_idx}.png"
                    pix.save(str(img_path))

                    # Execute PaddleOCR
                    # paddleocr 2.x: cls= kwarg is accepted and passes the angle
                    # classification flag through to the underlying model.
                    ocr_results = ocr_engine.ocr(str(img_path), cls=self.use_angle_cls)
                    page_text_lines: List[str] = []

                    if ocr_results and ocr_results[0]:
                        for line in ocr_results[0]:
                            if line and len(line) >= 2 and line[1]:
                                text_str = line[1][0]
                                if text_str and text_str.strip():
                                    page_text_lines.append(text_str.strip())

                    page_text = "\n".join(page_text_lines)
                    extracted_pages.append({
                        "index": page_idx,
                        "markdown": page_text,
                    })

            logger.info("PaddleOCR completed successfully for: %s", pdf_path.name)
            return {
                "pages": extracted_pages,
                "source": "paddle_ocr_local",
            }
        finally:
            doc.close()


# ── 3. API Key Pool & Cooldown Manager ────────────────────────────────────────

class APIKeyPoolManager:
    """
    Manages API key health, rotation state, and individual key cooldown timers.
    Defaults to 65.0s minimum cooldown to align with per-minute rate limit windows.
    """

    def __init__(self, api_keys: List[str], default_cooldown_seconds: float = 65.0):
        self.api_keys = [k.strip() for k in api_keys if k and isinstance(k, str) and k.strip()]
        if not self.api_keys:
            raise ValueError("APIKeyPoolManager initialized with no valid API keys.")

        self.default_cooldown_seconds = default_cooldown_seconds
        self.current_index = 0
        # Tracks timestamp (time.time()) when key will exit cooldown
        self.cooldowns: Dict[int, float] = {i: 0.0 for i in range(len(self.api_keys))}

    def get_num_keys(self) -> int:
        return len(self.api_keys)

    def mask_key(self, index: int) -> str:
        key = self.api_keys[index]
        return f"...{key[-4:]}" if len(key) > 4 else "****"

    def is_key_available(self, index: int) -> bool:
        return time.time() >= self.cooldowns.get(index, 0.0)

    def get_available_key_index(self) -> Optional[int]:
        """
        Returns an index of an API key that is not currently rate-limited,
        starting search from current_index.
        """
        now = time.time()
        num_keys = len(self.api_keys)

        for offset in range(num_keys):
            idx = (self.current_index + offset) % num_keys
            if now >= self.cooldowns[idx]:
                self.current_index = idx
                return idx
        return None

    def mark_key_rate_limited(self, index: int, cooldown_seconds: Optional[float] = None) -> float:
        """
        Marks an API key as rate-limited for a given cooldown duration (minimum 65s).
        """
        requested_cooldown = cooldown_seconds or self.default_cooldown_seconds
        effective_cooldown = max(requested_cooldown, self.default_cooldown_seconds)
        unlock_time = time.time() + effective_cooldown
        self.cooldowns[index] = unlock_time
        logger.warning(
            "API Key %s (index %d) rate-limited. Placed in cooldown for %.1fs (unlocks in +%.1fs).",
            self.mask_key(index),
            index,
            effective_cooldown,
            effective_cooldown,
        )
        return effective_cooldown

    def is_pool_exhausted(self) -> bool:
        return self.get_available_key_index() is None

    def get_pool_cooldown_remaining(self) -> float:
        """
        Calculates remaining sleep time in seconds until the earliest key exits cooldown.
        """
        now = time.time()
        remaining_times = [max(0.0, unlock - now) for unlock in self.cooldowns.values()]
        return min(remaining_times) if remaining_times else 0.0

    def create_client(self, index: int) -> Mistral:
        if Mistral is None:
            raise RuntimeError("mistralai package is not installed.")
        return Mistral(api_key=self.api_keys[index])


# ── 4. Mistral OCR Client with Exponential Backoff & Jitter ─────────────────

class MistralOCRClient:
    """
    Executes Mistral OCR API calls with exponential backoff (min 60s), randomized jitter,
    strict Retry-After header parsing, and key pool coordination.
    """

    def __init__(
        self,
        pool_manager: APIKeyPoolManager,
        request_delay: float = 2.0,
        max_retries_per_key: int = 3,
        initial_backoff: float = 60.0,
        backoff_factor: float = 2.0,
        max_backoff: float = 180.0,
        jitter_max: float = 3.0,
    ):
        self.pool = pool_manager
        self.request_delay = request_delay
        self.max_retries_per_key = max_retries_per_key
        self.initial_backoff = initial_backoff
        self.backoff_factor = backoff_factor
        self.max_backoff = max_backoff
        self.jitter_max = jitter_max

    def parse_retry_after(self, exc: Exception) -> Optional[float]:
        """
        Parses Retry-After header from exception object or regex match from error string.
        """
        if hasattr(exc, "response") and exc.response is not None:
            headers = getattr(exc.response, "headers", {})
            if headers:
                retry_val = headers.get("retry-after") or headers.get("Retry-After")
                if retry_val:
                    try:
                        return float(retry_val)
                    except ValueError:
                        pass

        err_msg = str(exc)
        match = re.search(r"retry[-_]after[:=\s]+(\d+)", err_msg, re.IGNORECASE)
        if match:
            return float(match.group(1))

        match_secs = re.search(r"try again in (\d+)\s*s", err_msg, re.IGNORECASE)
        if match_secs:
            return float(match_secs.group(1))

        match_sec_full = re.search(r"wait\s+(\d+)\s*sec", err_msg, re.IGNORECASE)
        if match_sec_full:
            return float(match_sec_full.group(1))

        return None

    def calculate_backoff(self, attempt: int) -> float:
        exponential = self.initial_backoff * (self.backoff_factor ** (attempt - 1))
        jitter = random.uniform(0.0, self.jitter_max)
        return min(exponential + jitter, self.max_backoff)

    def execute_ocr(self, pdf_path: Path) -> Dict[str, Any]:
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()

        base64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")
        data_url = f"data:application/pdf;base64,{base64_pdf}"

        num_keys = self.pool.get_num_keys()
        total_attempts = self.max_retries_per_key * num_keys

        for attempt in range(1, total_attempts + 1):
            key_index = self.pool.get_available_key_index()

            if key_index is None:
                cooldown_needed = max(self.pool.get_pool_cooldown_remaining(), 65.0)
                logger.warning(
                    "All %d API key(s) in pool are rate-limited. Circuit Breaker active: sleeping for %.1fs...",
                    num_keys,
                    cooldown_needed,
                )
                time.sleep(cooldown_needed)
                key_index = self.pool.get_available_key_index() or 0

            client = self.pool.create_client(key_index)
            masked_key = self.pool.mask_key(key_index)

            try:
                logger.debug(
                    "Attempt %d/%d: Calling Mistral OCR for '%s' using key %s...",
                    attempt,
                    total_attempts,
                    pdf_path.name,
                    masked_key,
                )
                response = client.ocr.process(
                    model="mistral-ocr-latest",
                    document={
                        "type": "document_url",
                        "document_url": data_url,
                    },
                )

                if self.request_delay > 0:
                    time.sleep(self.request_delay)

                return response.model_dump()

            except Exception as exc:
                err_str = str(exc)
                err_lower = err_str.lower()
                is_rate_limit = any(k in err_lower for k in ["429", "rate limit", "rate_limited", "quota"])

                if is_rate_limit:
                    retry_after = self.parse_retry_after(exc)
                    cooldown = max(retry_after, 65.0) if retry_after is not None else 65.0
                    self.pool.mark_key_rate_limited(key_index, cooldown)

                    if attempt < total_attempts:
                        backoff_sleep = self.calculate_backoff(attempt)
                        effective_sleep = max(retry_after, backoff_sleep) if retry_after is not None else backoff_sleep
                        logger.info(
                            "Rate limit (429) hit on key %s. Retrying in %.1fs (Attempt %d/%d)...",
                            masked_key,
                            effective_sleep,
                            attempt,
                            total_attempts,
                        )
                        time.sleep(effective_sleep)
                    else:
                        logger.error("All %d retries exhausted for '%s'.", total_attempts, pdf_path.name)
                        raise exc
                else:
                    logger.error("Non-rate-limit error during OCR for '%s': %s", pdf_path.name, exc)
                    raise exc

        raise RuntimeError(f"OCR failed for '{pdf_path.name}' after {total_attempts} attempts.")


# ── 5. MistralPDFProcessor Facade Coordinator ─────────────────────────────────

class MistralPDFProcessor:
    """
    Facade Coordinator for PDF processing:
      1. Local JSON Cache check
      2. Local Digital PDF text extraction (0 API tokens via pypdf)
      3. Multi-engine OCR execution:
         - 'paddle' (default offline local OCR via PaddleOCR)
         - 'mistral' (API-based OCR via Mistral AI)
    """

    def __init__(
        self,
        api_key: str | List[str] | None = None,
        output_dir: str | Path | None = None,
        ocr_engine: Optional[str] = None,
        request_delay: float = 2.0,
        max_retries: int = 3,
        initial_backoff: float = 60.0,
        backoff_factor: float = 2.0,
        safety_delay: float = 65.0,
        inter_file_delay: float = 1.0,
        min_char_threshold: int = 100,
        paddle_lang: str = "ar",
    ):
        # Resolve OCR Engine precedence: constructor param -> env var OCR_ENGINE -> 'paddle'
        env_engine = os.getenv("OCR_ENGINE")
        self.ocr_engine = (ocr_engine or env_engine or "paddle").lower()

        self.output_dir = Path(output_dir) if output_dir is not None else OUTPUT_JSON_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.inter_file_delay = inter_file_delay

        # Sub-component initializations
        self.local_extractor = LocalPDFTextExtractor(min_char_threshold=min_char_threshold)

        if self.ocr_engine == "paddle":
            self.paddle_extractor = PaddleOCRExtractor(lang=paddle_lang)
            self.pool_manager = None
            self.ocr_client = None
        elif self.ocr_engine == "mistral":
            self.paddle_extractor = None
            raw_keys = api_key or os.getenv("MISTRAL_API_KEY") or os.getenv("MISTRAL_API_KEYS")
            if isinstance(raw_keys, list):
                keys_list = [k.strip() for k in raw_keys if k and isinstance(k, str) and k.strip()]
            elif isinstance(raw_keys, str):
                keys_list = [k.strip() for k in raw_keys.split(",") if k and k.strip()]
            else:
                keys_list = []

            if not keys_list:
                logger.error("MISTRAL_API_KEY is missing from environment variables.")
                raise ValueError("MISTRAL_API_KEY is missing. Please set it in your environment variables.")

            self.pool_manager = APIKeyPoolManager(keys_list, default_cooldown_seconds=safety_delay)
            self.ocr_client = MistralOCRClient(
                pool_manager=self.pool_manager,
                request_delay=request_delay,
                max_retries_per_key=max_retries,
                initial_backoff=initial_backoff,
                backoff_factor=backoff_factor,
            )
        else:
            raise ValueError(f"Invalid ocr_engine '{self.ocr_engine}'. Allowed choices: 'paddle', 'mistral'.")

        # Statistics tracking
        self.stats = {
            "cache_hits": 0,
            "local_text_hits": 0,
            "paddle_ocr_hits": 0,
            "ocr_api_hits": 0,
            "failures": 0,
        }

        logger.info(
            "MistralPDFProcessor initialized (ocr_engine='%s'). Inter-file delay: %.1fs. Cache dir: %s",
            self.ocr_engine,
            self.inter_file_delay,
            self.output_dir,
        )

    def _get_expected_json_path(self, pdf_path: Path) -> Path:
        pdf_path = Path(pdf_path)
        return self.output_dir / f"{pdf_path.stem}.json"

    def is_ocr_cached(self, pdf_path: str | Path) -> bool:
        json_path = self._get_expected_json_path(Path(pdf_path))
        return json_path.exists()

    def _load_from_cache(self, pdf_path: Path) -> dict:
        json_path = self._get_expected_json_path(pdf_path)
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info("Loaded JSON cache from: %s", json_path)
        return data

    def _save_to_cache(self, pdf_path: Path, data: dict) -> None:
        json_path = self._get_expected_json_path(pdf_path)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        logger.info("OCR result saved to cache: %s", json_path)

    def process_pdf(self, pdf_path: str | Path) -> dict:
        """
        Processes a single PDF file:
          1. Checks local JSON cache first.
          2. Performs local digital PDF pre-check (0 API tokens / 0 OCR runtime if text is available).
          3. Executes selected OCR engine ('paddle' default, or 'mistral' API).
          4. Enforces inter-file throttling in finally block.
        """
        pdf_path = Path(pdf_path)

        try:
            # 1. Local JSON Cache Check
            if self.is_ocr_cached(pdf_path):
                self.stats["cache_hits"] += 1
                logger.info("Cache hit for file: %s", pdf_path.name)
                return self._load_from_cache(pdf_path)

            # 2. Local Digital Text Pre-Check
            local_result = self.local_extractor.extract_text(pdf_path)
            if local_result is not None:
                self.stats["local_text_hits"] += 1
                self._save_to_cache(pdf_path, local_result)
                logger.info("Successfully extracted text locally (0 API/OCR calls used) for: %s", pdf_path.name)
                return local_result

            # 3. OCR Engine Processing
            if self.ocr_engine == "paddle":
                logger.info("Running PaddleOCR engine for file: %s", pdf_path.name)
                ocr_response = self.paddle_extractor.extract_text(pdf_path)
                self._save_to_cache(pdf_path, ocr_response)
                self.stats["paddle_ocr_hits"] += 1
                logger.info("Successfully processed and cached PaddleOCR for: %s", pdf_path.name)
                return ocr_response

            elif self.ocr_engine == "mistral":
                logger.info("Calling Mistral OCR API for file: %s", pdf_path.name)
                ocr_response = self.ocr_client.execute_ocr(pdf_path)
                self._save_to_cache(pdf_path, ocr_response)
                self.stats["ocr_api_hits"] += 1
                logger.info("Successfully processed and cached Mistral OCR API for: %s", pdf_path.name)
                return ocr_response

            else:
                raise ValueError(f"Unsupported OCR engine: {self.ocr_engine}")

        except Exception as exc:
            self.stats["failures"] += 1
            logger.error("Failed to process PDF '%s': %s", pdf_path.name, exc)
            raise exc

        finally:
            if self.inter_file_delay > 0:
                time.sleep(self.inter_file_delay)

    def get_processing_stats(self) -> Dict[str, int]:
        return dict(self.stats)
