"""
src/agent/config.py

Centralized configuration for the Zar3a Multi-Agent LangGraph Workflow.
Configured for local, lightweight execution using quantized Qwen models via Ollama.
"""

# ---------------------------------------------------------------------------
# Language & Encoding Defaults
# ---------------------------------------------------------------------------
DEFAULT_LANGUAGE: str = "ar"
TEXT_ENCODING: str = "utf-8"

# ---------------------------------------------------------------------------
# Local LLM Model Configuration (Quantized Qwen 2.5 via Ollama)
# ---------------------------------------------------------------------------
AGENT_LLM_MODEL: str = "ollama/qwen2.5:7b"

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
