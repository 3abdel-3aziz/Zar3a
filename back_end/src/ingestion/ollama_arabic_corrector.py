"""
ollama_arabic_corrector.py
─────────────────────────────────────────────────────────────────────────────
Ollama-based LLM Corrector for Arabic OCR Text & Legal Documents.

Corrects:
- Fragmented / disjointed words from OCR column wrapping
- Misplaced numbers and dates (e.g., '1994نة لسوعلى' -> 'وعلى ... لسنة 1994')
- Reversed or scattered Arabic tokens
- Incomplete phrases caused by newspaper/gazette multi-column layout

Works with local models via Ollama:
- Prefers 'qwen2.5:3b' or 'qwen2.5:7b' (known for superior Arabic grammar & legal terminology).
- Caches corrections to disk (.cache/ollama_arabic_corrections.json) for instant reuse.
- Has graceful fallback to deterministic normalization on timeout/failure.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

from .arabic_text_normalizer import normalize_arabic_text

logger = logging.getLogger("Zar3a.OllamaArabicCorrector")

# Default Ollama endpoint
OLLAMA_API_BASE = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")

# Preferred models in priority order
PREFERRED_MODELS = [
    "qwen2.5:3b",
    "qwen2.5:7b",
    "qwen2.5:1.5b",
    "phi3:latest",
]

_CACHE_DIR = Path(".cache")
_CACHE_FILE = _CACHE_DIR / "ollama_arabic_corrections.json"

# Pre-compiled Unicode-safe corruption detection regexes
# NOTE: \b is intentionally avoided with Arabic chars — causes catastrophic backtracking
_RE_BROKEN_NUMBERS = re.compile(r"\d+\u0646\u0629\b|\b\u0644\u0633\d+")  # '1994نة' or 'لس1912'
_RE_SCATTERED_CHARS = re.compile(
    r"[\u0600-\u06FF]\s+[\u0600-\u06FF]\s+[\u0600-\u06FF]\s+[\u0600-\u06FF]"
)
_RE_GAZETTE_HEADER = re.compile(
    r"(?<![\u0600-\u06FF])\u0628\s+\u0642\u0631\u0627\u0631\s+\u0631\u0626\u064a\u0633"
    r"|\u0643\u064a\u0644\s+\u0623\u0646\s+\u062a\u0634"
    r"|\u0623\s+\u0628\u0645\u0648\u062c\s+\u062f"
)


class OllamaArabicCorrector:
    """
    Local LLM Arabic text corrector using Ollama.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_base: str = OLLAMA_API_BASE,
        timeout_seconds: int = 40,
        enable_cache: bool = True,
    ):
        self.api_base = api_base.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.enable_cache = enable_cache
        self.cache: Dict[str, str] = {}
        self._load_cache()

        # Determine model
        if model_name:
            self.model_name = model_name
        else:
            self.model_name = self._autodetect_best_model()

        logger.info("OllamaArabicCorrector initialized with model='%s'", self.model_name)

    def _load_cache(self) -> None:
        """Loads cached corrections from disk."""
        if not self.enable_cache:
            return
        try:
            if _CACHE_FILE.exists():
                with open(_CACHE_FILE, "r", encoding="utf-8") as f:
                    self.cache = json.load(f)
                logger.info("Loaded %d cached Arabic corrections from %s", len(self.cache), _CACHE_FILE)
        except Exception as exc:
            logger.warning("Could not load Ollama correction cache: %s", exc)
            self.cache = {}

    def _save_cache(self) -> None:
        """Flushes cached corrections to disk."""
        if not self.enable_cache:
            return
        try:
            _CACHE_DIR.mkdir(parents=True, exist_ok=True)
            with open(_CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            logger.warning("Could not save Ollama correction cache: %s", exc)

    def _autodetect_best_model(self) -> str:
        """Queries Ollama for installed models and picks the best one available."""
        try:
            req = urllib.request.Request(f"{self.api_base}/api/tags")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                installed = [m.get("name", "") for m in data.get("models", [])]

            logger.info("Installed Ollama models: %s", installed)
            for preferred in PREFERRED_MODELS:
                if preferred in installed:
                    logger.info("Selected Ollama model '%s' for Arabic correction", preferred)
                    return preferred

            if installed:
                return installed[0]
        except Exception as exc:
            logger.warning("Ollama connection check failed (%s). Defaulting to 'qwen2.5:7b'", exc)

        return "qwen2.5:7b"

    def needs_llm_correction(self, text: str) -> bool:
        """
        Heuristic to detect if a text contains severe OCR corruption patterns
        (e.g., scrambled numbers, reversed fragments, or disorganized gazette column breaks)
        that require generative LLM reconstruction.

        Uses pre-compiled Unicode-safe regexes — no \\b with Arabic chars to prevent
        catastrophic backtracking.
        """
        if not text or len(text.strip()) < 15:
            return False

        stripped = text.strip()

        # 1. Broken numbers/years like '1994نة' or 'لس1912'
        if _RE_BROKEN_NUMBERS.search(stripped):
            return True

        # 2. Scattered single-char Arabic fragments (e.g. 'س و ج ر')
        if _RE_SCATTERED_CHARS.search(stripped):
            return True

        # 3. Disorganized gazette headers
        if _RE_GAZETTE_HEADER.search(stripped):
            return True

        # 4. Inverted words or lines — use the fast pre-compiled version
        from .arabic_text_normalizer import is_reversed_arabic_line
        lines = stripped.splitlines()
        for line in lines:
            # Only check lines that are long enough to matter
            if len(line.strip()) >= 15 and is_reversed_arabic_line(line):
                return True

        return False

    def correct_arabic_passage(self, text: str) -> str:
        """
        Passes a text passage through Ollama for grammar, word-order, and OCR correction.
        """
        if not text or not text.strip():
            return text

        # Step 1: Algorithmic cleaning first
        cleaned = normalize_arabic_text(text, fix_reversal=True)

        # If text doesn't show signs of corruption, return clean version directly
        if not self.needs_llm_corruption_check(cleaned):
            return cleaned

        # Step 2: Check cache
        cache_key = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()
        if self.enable_cache and cache_key in self.cache:
            return self.cache[cache_key]

        system_prompt = (
            "أنت خبير ومدقق لغوي وقانوني للنصوص والوثائق والقرارات الرسمية المصرية باللغة العربية.\n"
            "النص المدخل مستخرج عبر الماسح الضوئي (OCR) وبه تشوهات في ترتيب الكلمات بسبب أعمدة الصحف، "
            "أو أرقام وتواريخ مبعثرة، أو كلمات مجزأة.\n"
            "المطلوب:\n"
            "1. أعد صياغة وترتيب النص ليكون لغة عربية فصيحة وسليمة ومترابطة تماماً.\n"
            "2. حافظ بدقة تامة على كافة أرقام المواد، أرقام القوانين، السنوات، والتواريخ والأسماء دون أي حذف.\n"
            "3. أخرج النص المصحح فقط بدون أي مقدمات أو شروحات أو عبارات مثل 'إليك النص'."
        )

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": cleaned}
            ],
            "stream": False,
            "options": {
                "temperature": 0.0,
                "num_predict": max(len(cleaned.split()) * 2, 256),
            }
        }

        try:
            req = urllib.request.Request(
                f"{self.api_base}/api/chat",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                corrected = data.get("message", {}).get("content", "").strip()

            if corrected and len(corrected) > 10:
                if self.enable_cache:
                    self.cache[cache_key] = corrected
                    self._save_cache()
                return corrected
            return cleaned

        except Exception as exc:
            logger.warning("Ollama LLM correction failed on snippet (%s); using normalized text fallback.", exc)
            return cleaned

    def needs_llm_corruption_check(self, text: str) -> bool:
        """Alias for needs_llm_correction."""
        return self.needs_llm_correction(text)

    def process_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Corrects a list of chunk dictionaries in-place or returns updated copies.
        """
        logger.info("Running Arabic text verification/correction across %d chunks...", len(chunks))
        corrected_count = 0

        for chunk in chunks:
            raw_content = chunk.get("content") or chunk.get("chunk_text") or ""
            if self.needs_llm_correction(raw_content):
                corrected = self.correct_arabic_passage(raw_content)
                if corrected != raw_content:
                    chunk["content"] = corrected
                    if "chunk_text" in chunk:
                        chunk["chunk_text"] = corrected
                    corrected_count += 1

        logger.info("Ollama Arabic correction finished: %d/%d chunks refined.", corrected_count, len(chunks))
        return chunks
