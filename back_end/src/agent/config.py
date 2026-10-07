"""
src/agent/config.py

Centralized configuration for the Zar3a Multi-Agent LangGraph Workflow.
Configured for local, lightweight execution using quantized Qwen models via Ollama.
"""

import logging
import os
import time
import urllib.request
from typing import Any, Optional, Tuple
from langchain_openai import ChatOpenAI

logger = logging.getLogger("Zar3a.AgentConfig")

# ---------------------------------------------------------------------------
# Language & Encoding Defaults
# ---------------------------------------------------------------------------
DEFAULT_LANGUAGE: str = "ar"
TEXT_ENCODING: str = "utf-8"

# ---------------------------------------------------------------------------
# Local LLM Model Configuration (Quantized Qwen 2.5 via Ollama or OpenAI)
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
DEFAULT_OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")

AGENT_LLM_MODEL: str = os.getenv("AGENT_LLM_MODEL", DEFAULT_OLLAMA_MODEL)

ROUTER_LLM_MODEL: str = AGENT_LLM_MODEL
KNOWLEDGE_LLM_MODEL: str = AGENT_LLM_MODEL
CLIMATE_LLM_MODEL: str = AGENT_LLM_MODEL
SYNTHESIZER_LLM_MODEL: str = AGENT_LLM_MODEL

# ---------------------------------------------------------------------------
# LLM Temperature Settings
# ---------------------------------------------------------------------------
ROUTER_TEMPERATURE: float = 0.0
KNOWLEDGE_TEMPERATURE: float = 0.0
CLIMATE_TEMPERATURE: float = 0.0
SYNTHESIZER_TEMPERATURE: float = 0.1

# TTL-based Ollama availability cache: (is_available, last_check_timestamp)
_ollama_available_cache: Tuple[Optional[bool], float] = (None, 0.0)
_OLLAMA_CACHE_TTL_SECONDS: float = 60.0  # Re-check Ollama every 60 seconds


def check_ollama_available(base_url: str = OLLAMA_BASE_URL) -> bool:
    """Checks if the local Ollama server is responsive (TTL-cached, re-checks every 60s)."""
    global _ollama_available_cache
    cached_result, last_check = _ollama_available_cache
    if cached_result is not None and (time.monotonic() - last_check) < _OLLAMA_CACHE_TTL_SECONDS:
        return cached_result
    try:
        req = urllib.request.Request(f"{base_url}/api/tags", headers={"User-Agent": "Zar3a"})
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            result = (resp.status == 200)
            _ollama_available_cache = (result, time.monotonic())
            return result
    except Exception:
        _ollama_available_cache = (False, time.monotonic())
        return False


def build_chat_llm(
    model: Optional[str] = None,
    temperature: float = 0.0,
    api_key: Optional[str] = None,
    timeout: float = 60.0,
) -> Optional[Any]:
    """
    Unified LLM factory supporting OpenAI, local Ollama (Qwen2.5), and graceful degradation.

    Priority:
      1. Real OPENAI_API_KEY -> OpenAI ChatOpenAI.
      2. Local Ollama -> ChatOpenAI client via Ollama OpenAI compatibility (/v1).
      3. Returns None if neither is reachable.
    """
    openai_key = (api_key or os.getenv("OPENAI_API_KEY", "")).strip()

    # 1. Real OpenAI key configured
    if openai_key and not openai_key.startswith("ollama"):
        chosen_model = model or "gpt-4o-mini"
        if chosen_model.startswith("ollama/"):
            chosen_model = "gpt-4o-mini"
        logger.info("Instantiating OpenAI LLM with model '%s'", chosen_model)
        return ChatOpenAI(model=chosen_model, temperature=temperature, api_key=openai_key, timeout=timeout)

    # 2. Local Ollama instance
    if check_ollama_available(OLLAMA_BASE_URL):
        target_model = model or DEFAULT_OLLAMA_MODEL
        if target_model.startswith("ollama/"):
            target_model = target_model[len("ollama/"):]
        if target_model in ("gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"):
            target_model = DEFAULT_OLLAMA_MODEL
        logger.info("Instantiating Ollama LLM at %s/v1 with model '%s'", OLLAMA_BASE_URL, target_model)
        return ChatOpenAI(
            base_url=f"{OLLAMA_BASE_URL}/v1",
            api_key="ollama",
            model=target_model,
            temperature=temperature,
            timeout=timeout,
        )

    logger.warning("No LLM service available (neither OPENAI_API_KEY nor Ollama at %s)", OLLAMA_BASE_URL)
    return None

