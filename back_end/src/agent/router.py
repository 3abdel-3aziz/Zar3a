"""
src/agent/router.py

Supervisor Router Node for the Zar3a LangGraph multi-agent workflow.

This node acts as the supervisor router for the "Zar3a" platform. It analyzes
the user's latest query and conversation context from `AgentState` and decides
which specialized pipeline(s) to trigger.

Routing destinations:
  - "knowledge": Urban forestry laws, government regulations, or ecological
    text retrieval from the Qdrant RAG vector store.
  - "climate_recommendation": Microclimate temperature impacts, urban heat
    island (UHI) mitigation, and specific tree recommendation constraints.
  - Both routes: Compound queries that involve both regulations and tree/climate
    recommendations, or ambiguous queries (default fallback).
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Literal, Optional, Sequence, cast

from src.agent.config import (
    DEFAULT_LANGUAGE,
    ROUTER_LLM_MODEL,
    ROUTER_TEMPERATURE,
)

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.agent.state import AgentState

# ---------------------------------------------------------------------------
# Logger Configuration (Strict INFO level, no DEBUG)
# ---------------------------------------------------------------------------
logger = logging.getLogger("Zar3a.RouterNode")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Valid Route Literals and Fallback Constants
# ---------------------------------------------------------------------------
RouteDestination = Literal["knowledge", "climate_recommendation"]

VALID_ROUTES: frozenset[str] = frozenset({"knowledge", "climate_recommendation"})
FALLBACK_ROUTES: List[str] = ["knowledge", "climate_recommendation"]


# ---------------------------------------------------------------------------
# Structured Output Schema
# ---------------------------------------------------------------------------

class RouterOutput(BaseModel):
    """
    Structured output schema for the supervisor router LLM.

    Restricts the LLM routing decision to a list containing only the allowed
    route destination literals: "knowledge" and/or "climate_recommendation".
    """

    routes: List[RouteDestination] = Field(
        ...,
        description=(
            "List of specialized pipeline routes to activate. "
            "Must contain one or both of: 'knowledge', 'climate_recommendation'. "
            "Select 'knowledge' for urban forestry laws, regulations, and ecological text. "
            "Select 'climate_recommendation' for microclimate temperature impacts and tree recommendations. "
            "Select BOTH for compound queries or if there is any ambiguity."
        ),
    )

    model_config = {"frozen": True}


# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are the intelligent supervisor router for the "Zar3a" (زرعة) urban resilience \
and tree recommendation platform in Egypt.

Your task is to analyze the user's latest query and conversation context, then decide \
which specialized pipeline(s) must be triggered to fulfill their request.

## Available Routes:

1. "knowledge":
   - Urban forestry laws, ministerial decrees, and Egyptian environmental legislation \
(e.g., Environment Law No. 4/1994, municipal setback decrees, tree preservation acts).
   - Official government standards, permits, utility clearance guidelines, and public space policies.
   - Botanical taxonomy, ecological characteristics, and documentation retrieved from knowledge bases.

2. "climate_recommendation":
   - Microclimate temperature impacts, urban heat island (UHI) mitigation, and shade calculations.
   - Specific tree recommendations based on site constraints (e.g., street width, soil type, \
salinity, water scarcity/drought tolerance, root depth).
   - Selection and ranking of tree species suited to Egyptian governorates and urban zones.

3. BOTH ["knowledge", "climate_recommendation"]:
   - Compound queries that touch BOTH regulations/laws AND tree/climate recommendations \
(e.g., "What trees can I legally plant on a narrow sidewalk in New Cairo that reduce heat?").
   - Ambiguous, broad, or unclear queries where the intent spans multiple domains.
   - **MANDATORY**: If there is ANY ambiguity, ALWAYS select BOTH routes.

## Output Requirement:
You MUST respond strictly using the provided structured schema containing the `routes` list. \
Do not output any additional commentary or free-form text.
"""


# ---------------------------------------------------------------------------
# LLM Chain Factory & Management
# ---------------------------------------------------------------------------

def _build_router_llm(
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Any:
    """
    Constructs the routing LLM configured with structured output.

    Uses a fast, lightweight model (default: gpt-4o-mini) with temperature=0
    to ensure deterministic, low-latency classification.

    Args:
        model: Optional model name override (defaults to env OPENAI_ROUTER_MODEL or 'gpt-4o-mini').
        api_key: Optional API key override (defaults to env OPENAI_API_KEY).

    Returns:
        A LangChain Runnable configured with RouterOutput structured output.
    """
    selected_model = model or ROUTER_LLM_MODEL
    openai_key = api_key or os.getenv("OPENAI_API_KEY")

    kwargs: Dict[str, Any] = {
        "model": selected_model,
        "temperature": ROUTER_TEMPERATURE,
    }
    if openai_key:
        kwargs["api_key"] = openai_key

    llm = ChatOpenAI(**kwargs)
    return llm.with_structured_output(RouterOutput, strict=True)


# Module-level singleton chain
_router_chain: Optional[Any] = None


def get_router_chain() -> Any:
    """Returns the cached router chain singleton, creating it lazily if needed."""
    global _router_chain
    if _router_chain is None:
        logger.info("Initializing supervisor router LLM chain (model=%s, temperature=%.1f).", ROUTER_LLM_MODEL, ROUTER_TEMPERATURE)
        _router_chain = _build_router_llm()
    return _router_chain


def set_router_chain(chain: Optional[Any]) -> None:
    """Sets or resets the cached router chain (useful for testing and dependency injection)."""
    global _router_chain
    _router_chain = chain


# ---------------------------------------------------------------------------
# Keyword-Based Fallback Router (no LLM required)
# ---------------------------------------------------------------------------

# Legal, regulatory, policy, council, and climate change knowledge keywords (Arabic + English)
_KNOWLEDGE_KEYWORDS_AR = {
    # Legal & Regulatory
    "قانون", "تشريع", "لائحة", "مرسوم", "وزار", "بيئ", "حظر", "ترخيص",
    "اشتراط", "بلدي", "نظام", "قرار", "وثيق", "مستند", "حكوم", "عقوب",
    "مخالف", "كود", "مواصف", "معيار", "معايير", "147", "4 لسنة", "مادة",
    "موارد مائية", "مائي", "ري وصرف",
    # Climate Strategy, Councils & Governance
    "مجلس", "استراتيج", "أهداف", "هدف", "خطة", "خطط", "سياس", "وطني",
    "قومي", "رؤية", "تغير المناخ", "المناخية", "تغيرات", "احتباس",
    "انبعاث", "انبعاثات", "تكيف", "تخفيف", "كربون", "بصمة", "تنمية مستدامة",
    # Treaties & Conventions
    "اتفاق", "بروتوكول", "معاهدة", "كيوتو", "باريس", "مؤتمر", "cop",
    "التزام", "التزامات", "تقرير", "بيان",
}
_KNOWLEDGE_KEYWORDS_EN = {
    "law", "regulat", "decree", "permit", "legal", "act", "ordinance",
    "ministry", "municipal", "bylaw", "ban", "prohibited", "guideline",
    "requirement", "compliance", "environmental", "legislation", "code",
    "standard", "penalty", "violation", "article", "decree no",
    "strategy", "council", "objective", "target", "goal", "plan", "policy",
    "national", "vision", "climate change", "emission", "mitigation",
    "adaptation", "sustainable", "governance", "treaty", "protocol",
    "kyoto", "paris", "unfccc", "cop", "convention", "report",
}

# Urban forestry & specific tree recommendation keywords (Arabic + English)
_TREE_REC_KEYWORDS_AR = {
    "شجر", "أشجار", "نبات", "شتل", "شتلات", "تشجير", "تخضير", "غرس",
    "أنواع الأشجار", "أصناف الأشجار", "ظل", "تبريد", "جزيرة حرارية", "سدر", "نخل", "أكاسيا",
    "فيكس", "نيم", "بوانسيانا", "كاسيا", "يوكالبتوس", "كونوكاربس", "كافور",
    "توت", "جهنمية", "ملوحة التربة", "تحمل الجفاف", "شارع ضيق", "تربة زراعية", "شبكة ري",
    "ري بالتنقيط", "حديقة", "رصيف", "رشح", "ترشيح",
}
_TREE_REC_KEYWORDS_EN = {
    "tree", "plant", "species", "shade", "cool", "heat island", "urban heat",
    "greenery", "garden", "narrow street", "sidewalk", "canopy",
    "leaf", "leaves", "trunk", "root", "roots", "water need", "drought",
    "salinity", "sidr", "neem", "acacia", "ficus", "poinciana",
    "recommend", "planting",
}

# Explicit tree recommendation intent keywords (Arabic + English)
_TREE_INTENT_AR = {"شجر", "أشجار", "رشح", "ترشيح", "غرس", "شتل", "أصناف", "أنواع"}
_TREE_INTENT_EN = {"tree", "trees", "recommend", "species", "planting"}


def _contains_keyword(text: str, keywords: set[str]) -> bool:
    """Checks whether text contains any keyword, using token and prefix boundaries for Arabic/English."""
    q = text.lower()
    tokens = set(re.findall(r"[\w]+", q))
    for kw in keywords:
        if " " in kw:
            if kw in q:
                return True
        else:
            if kw in tokens:
                return True
            for t in tokens:
                if t.startswith(("ال", "وال", "بال", "كال", "فال", "لل")):
                    stripped = re.sub(r"^(ال|وال|بال|كال|فال|لل)", "", t)
                    if stripped == kw or (len(kw) >= 4 and stripped.startswith(kw)):
                        return True
                elif len(kw) >= 4 and t.startswith(kw):
                    return True
    return False


def _keyword_route(query: str) -> List[str]:
    """
    Classifies a query into routing destinations using comprehensive keyword matching.

    Returns:
        ["knowledge"] for legal, regulatory, council, strategy, and environmental policy queries.
        ["climate_recommendation"] for specific tree recommendation and planting queries.
        ["knowledge", "climate_recommendation"] for compound inquiries or general topics.
    """
    hits_knowledge = _contains_keyword(query, _KNOWLEDGE_KEYWORDS_AR) or \
                     _contains_keyword(query, _KNOWLEDGE_KEYWORDS_EN)
    hits_tree_rec = _contains_keyword(query, _TREE_REC_KEYWORDS_AR) or \
                    _contains_keyword(query, _TREE_REC_KEYWORDS_EN)
    has_tree_intent = _contains_keyword(query, _TREE_INTENT_AR) or \
                      _contains_keyword(query, _TREE_INTENT_EN)

    # Pure legal / regulatory / council / strategy queries without explicit tree recommendation intent
    if hits_knowledge and not has_tree_intent:
        return ["knowledge"]

    # Compound query: asks about tree species AND legal/regulatory/policy constraints
    if hits_knowledge and hits_tree_rec and has_tree_intent:
        return list(FALLBACK_ROUTES)

    # Specific tree recommendation / species / planting query
    if hits_tree_rec and has_tree_intent:
        return ["climate_recommendation"]

    # Default to knowledge retrieval for general or unclassified questions
    return ["knowledge"]


# ---------------------------------------------------------------------------
# Supervisor Router Node Function
# ---------------------------------------------------------------------------

def route_query(
    state: AgentState,
    llm_chain: Optional[Any] = None,
) -> Dict[str, List[str]]:
    """
    Supervisor router node for the Zar3a LangGraph multi-agent workflow.

    Analyzes the user's latest query and conversation history from `AgentState`,
    invokes the fast structured LLM, and returns a state update dictionary
    updating the `routes` key.

    Routing rules applied:
      - 'knowledge': Regulatory, legal, ecological literature.
      - 'climate_recommendation': Microclimate temperature delta, tree constraints.
      - Both: Compound queries or any ambiguity (safe default).

    Args:
        state: The current `AgentState` containing conversation `messages`.
        llm_chain: Optional preconfigured chain for testing or custom override.

    Returns:
        Dict[str, List[str]]: State dictionary update, e.g. `{"routes": ["knowledge"]}`.
    """
    messages_input: Sequence[Any] = state.get("messages", [])

    # Handle empty message history
    if not messages_input:
        logger.warning("Router received an empty messages history. Defaulting to fallback routes: %s", FALLBACK_ROUTES)
        return {"routes": list(FALLBACK_ROUTES)}

    # Convert/normalize message list to LangChain BaseMessage objects
    normalized_messages: List[BaseMessage] = []
    latest_query_text = ""

    for msg in messages_input:
        if isinstance(msg, BaseMessage):
            normalized_messages.append(msg)
            if isinstance(msg, HumanMessage) or not latest_query_text:
                latest_query_text = str(msg.content)
        elif isinstance(msg, dict):
            role = msg.get("role", "user")
            content = str(msg.get("content", ""))
            if role in ("user", "human"):
                normalized_messages.append(HumanMessage(content=content))
                latest_query_text = content
            elif role in ("assistant", "ai"):
                normalized_messages.append(AIMessage(content=content))
            elif role == "system":
                normalized_messages.append(SystemMessage(content=content))
            else:
                normalized_messages.append(HumanMessage(content=content))
                latest_query_text = content
        elif isinstance(msg, str):
            normalized_messages.append(HumanMessage(content=msg))
            latest_query_text = msg

    # Fallback if no content could be extracted
    latest_query_text = latest_query_text.strip()
    if not latest_query_text and normalized_messages:
        latest_query_text = str(normalized_messages[-1].content).strip()

    if not latest_query_text:
        logger.warning("Router received an empty query string. Defaulting to fallback routes: %s", FALLBACK_ROUTES)
        return {"routes": list(FALLBACK_ROUTES)}

    logger.info("Router evaluating query (%d chars): '%s...'", len(latest_query_text), latest_query_text[:100])

    # --- Fast path: use keyword-based routing when no external LLM is available ---
    # Avoids blocking on an LLM API call when OPENAI_API_KEY is not set
    # and no explicit chain override has been provided.
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    if llm_chain is None and not openai_key:
        keyword_routes = _keyword_route(latest_query_text)
        logger.info("Supervisor router (keyword mode) decision: %s", keyword_routes)
        return {"routes": keyword_routes}

    # Assemble prompt messages: System prompt + conversation history
    prompt_messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        *normalized_messages,
    ]

    try:
        chain = llm_chain if llm_chain is not None else get_router_chain()
        result: RouterOutput = chain.invoke(prompt_messages)

        # Validate and deduplicate extracted routes against allowed literals
        raw_routes = getattr(result, "routes", [])
        chosen_routes: List[str] = list(
            dict.fromkeys(r for r in raw_routes if r in VALID_ROUTES)
        )

        if not chosen_routes:
            logger.warning(
                "Router LLM produced no valid routes (raw output: %s). Using keyword fallback.",
                raw_routes,
            )
            chosen_routes = _keyword_route(latest_query_text)

        logger.info("Supervisor router (LLM mode) decision: %s", chosen_routes)
        return {"routes": chosen_routes}

    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Supervisor router LLM invocation failed: %s. Falling back to keyword routing.",
            exc,
        )
        keyword_routes = _keyword_route(latest_query_text)
        logger.info("Supervisor router (keyword fallback) decision: %s", keyword_routes)
        return {"routes": keyword_routes}
