"""
src/agent/nodes/synthesizer_node.py

Synthesizer Agent Node for the Zar3a LangGraph multi-agent workflow.

This node is the culmination step in the graph before returning the final response
to the user. It aggregates and harmonizes outputs from:
  1. The Knowledge RAG Agent (`rag_context` with legal/environmental citations).
  2. The Climate & Recommendation Agent (`candidate_species` and `climate_metrics`).

Using a fast, lightweight LLM (gpt-4o-mini, temperature=0), it synthesizes these disparate
streams into a single, cohesive, beautifully structured response in Arabic (preserving UTF-8),
with legal grounding and quantitative recommendation metrics presented side by side.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.agent.config import (
    DEFAULT_LANGUAGE,
    SYNTHESIZER_LLM_MODEL,
    SYNTHESIZER_TEMPERATURE,
    TEXT_ENCODING,
)
from src.agent.state import AgentState

# ---------------------------------------------------------------------------
# Logger Configuration (Strict INFO level, no DEBUG)
# ---------------------------------------------------------------------------
logger = logging.getLogger("Zar3a.SynthesizerNode")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# System Prompt for Master Synthesis
# ---------------------------------------------------------------------------

_SYNTHESIZER_SYSTEM_PROMPT = """\
أنت المحلل المرجعي الأول لمنصة "زرعة" (Zar3a) لتعزيز المرونة البيئية والتشجير الحضري في مصر.

مهمتك توحيد مخرجات مستشاري المنصة (مستشار القوانين واللوائح البيئية، ومستشار المناخ وترشيح الأشجار) \
في تقرير نهائي شامل، فائق التنسيق ومكتوب باللغة العربية الفصحى.

## ضوابط ومحددات حاسمة:
1. **اللغة والترميز**:
   - يجب أن تكون جميع الردود باللغة العربية الفصحى الواضحة والمهنية مع ترميز UTF-8 سليم تماماً.
   - استخدم الإنجليزية فقط إذا كان استفسار المستخدم بالكامل بالإنجليزية أو طلب ذلك صراحةً.
   - اجعل أسماء الأشجار ثنائية دائماً: الاسم العربي والاسم العلمي/الإنجليزي (مثال: كاسيا نودوزا Cassia nodosa).

2. **هيكل التقرير المنظم (Markdown)**:
   - 🌿 **مقدمة توجيهية**: إجابة مباشرة وواضحة عن استفسار المستخدم وهدف التخضير.
   - 📜 **الضوابط والاشتراطات القانونية**: تلخيص القوانين والقرارات الوزارية وقواعد التباعد مع الاستشهاد الصريح بصيغة [Document X].
   - 🌡️ **الأثر المناخي وتخفيف الجزر الحرارية**: عرض الأرقام بدقة (مقدار خفض الحرارة بالدرجات المئوية °C، وتحسن مؤشر الغطاء النباتي NDVI).
   - 🌳 **الأشجار المرشحة الملائمة**: قائمة أو جدول بالأشجار مع الفئة، استهلاك المياه، ودرجة التبريد.
   - 💡 **إرشادات الزراعة والتطبيق العملي**: نصائح عن الأرصفة والمرافق التحتية والري بالتنقيط.

3. **الصرامة والأمانة العلمية**:
   - لا تخترع قوانين أو تخالف البيانات المرفقة من نماذج المناخ وقاعدة البيانات.
"""


# ---------------------------------------------------------------------------
# Deterministic Fallback Synthesis
# ---------------------------------------------------------------------------

def _generate_fallback_synthesis(
    query: str,
    rag_context: Optional[str],
    climate_metrics: Optional[Dict[str, Any]],
    candidate_species: Optional[List[Dict[str, Any]]],
    language: str = "ar",
) -> str:
    """
    Constructs a clean, structured Markdown response in Arabic or English
    when LLM synthesis is unavailable or encounters an error.
    """
    is_english = language == "en"

    if is_english:
        sections: List[str] = [
            "## 🌿 Urban Resilience & Green Recommendation Report — Zar3a",
            f"**Inquiry:** {query}",
        ]

        # 1. Legal / RAG Section
        if rag_context and "No relevant" not in rag_context:
            sections.append(
                "### 📜 Legal & Regulatory Guidelines\n"
                f"{rag_context}"
            )

        # 2. Climate Metrics Section
        if climate_metrics:
            temp_delta = climate_metrics.get("predicted_temp_delta_c", -1.8)
            ndvi_gain = climate_metrics.get("ndvi_improvement", 0.12)
            sections.append(
                "### 🌡️ Predicted Microclimate Impact\n"
                f"- **Local Temperature Reduction:** {temp_delta:.1f}°C\n"
                f"- **NDVI Vegetation Improvement:** +{ndvi_gain:.2f}"
            )

        # 3. Recommended Species
        if candidate_species:
            tree_lines = ["### 🌳 Recommended Tree Species"]
            for idx, tree in enumerate(candidate_species, start=1):
                name_en = tree.get("name_en", "Tree")
                name_ar = tree.get("name_ar", "")
                cat = tree.get("category", "General")
                water = tree.get("water_requirement", "Medium")
                cooling = tree.get("cooling_effect_score", 5)
                score = tree.get("final_score", 0.0)
                tree_lines.append(
                    f"{idx}. **{name_en}** ({name_ar})\n"
                    f"   - **Category:** {cat} | **Water Need:** {water} | **Cooling Score:** {cooling}/10\n"
                    f"   - **Suitability Score:** {score:.2f}"
                )
            sections.append("\n".join(tree_lines))

        sections.append(
            "### 💡 Practical Advice\n"
            "- Ensure proper setback distances from underground utility lines and sidewalk curbs.\n"
            "- Adopt drip irrigation to maximize water conservation in arid zones."
        )
        return "\n\n".join(sections)

    # Arabic Fallback (Default)
    sections = [
        "## 🌿 تقرير التوصيات الحضرية والبيئية — منصة زرعة",
        f"**الاستفسار:** {query}",
    ]

    # 1. Legal / RAG Section
    if rag_context and "لم يتم العثور" not in rag_context and "No relevant" not in rag_context:
        sections.append(
            "### 📜 الضوابط والاشتراطات القانونية والبيئية\n"
            f"{rag_context}"
        )

    # 2. Climate Metrics Section
    if climate_metrics:
        temp_delta = climate_metrics.get("predicted_temp_delta_c", -1.8)
        ndvi_gain = climate_metrics.get("ndvi_improvement", 0.12)
        sections.append(
            "### 🌡️ الأثر المناخي المتوقع وتخفيض الحرارة\n"
            f"- **التخفيض المتوقع في درجة الحرارة المحلية:** {abs(temp_delta):.1f} درجة مئوية\n"
            f"- **التحسن في مؤشر الغطاء النباتي (NDVI):** +{ndvi_gain:.2f}"
        )

    # 3. Recommended Species
    if candidate_species:
        tree_lines = ["### 🌳 الأشجار المرشحة الملائمة للبيئة الحضرية"]
        for idx, tree in enumerate(candidate_species, start=1):
            name_ar = tree.get("name_ar", "شجرة")
            name_en = tree.get("name_en", "")
            cat = tree.get("category", "عام")
            water = tree.get("water_requirement", "متوسط")
            cooling = tree.get("cooling_effect_score", 5)
            score = tree.get("final_score", 0.0)
            tree_lines.append(
                f"{idx}. **{name_ar}** ({name_en})\n"
                f"   - **التصنيف:** {cat} | **احتياج المياه:** {water} | **كفاءة التبريد:** {cooling}/10\n"
                f"   - **درجة التوافق الإجمالية:** {score:.2f}"
            )
        sections.append("\n".join(tree_lines))

    # 4. Practical Planting Advice
    sections.append(
        "### 💡 إرشادات الزراعة والتطبيق العملي\n"
        "- **مسافات الأمان:** مراعاة مسافة لا تقل عن 1.5 إلى 2 متر من الأرصفة والمرافق التحتية لضمان سلامة البنية التحتية.\n"
        "- **ترشيد المياه:** استخدام أنظمة الري بالتنقيط في الساعات الأولى من الصباح لتقليل الفاقد بالتبخر.\n"
        "- **الصيانة:** التقليم الدوري للحفاظ على زوايا الرؤية المرورية والإضاءة العامة."
    )

    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# LLM Factory
# ---------------------------------------------------------------------------

_synthesizer_llm_instance: Optional[Any] = None


def _build_synthesizer_llm(model: Optional[str] = None, api_key: Optional[str] = None) -> Any:
    """Builds the fast ChatOpenAI client for master synthesis using centralized config."""
    selected_model = model or SYNTHESIZER_LLM_MODEL
    openai_key = api_key or os.getenv("OPENAI_API_KEY")

    kwargs: Dict[str, Any] = {
        "model": selected_model,
        "temperature": SYNTHESIZER_TEMPERATURE,
    }
    if openai_key:
        kwargs["api_key"] = openai_key

    return ChatOpenAI(**kwargs)


def get_synthesizer_llm() -> Any:
    """Returns cached synthesizer LLM singleton."""
    global _synthesizer_llm_instance
    if _synthesizer_llm_instance is None:
        try:
            _synthesizer_llm_instance = _build_synthesizer_llm()
        except Exception as exc:  # noqa: BLE001
            logger.info("Synthesizer LLM deferred setup: %s", exc)
            return None
    return _synthesizer_llm_instance


def set_synthesizer_llm(llm: Optional[Any]) -> None:
    """Sets or resets the synthesizer LLM (useful for testing)."""
    global _synthesizer_llm_instance
    _synthesizer_llm_instance = llm


# ---------------------------------------------------------------------------
# Synthesizer Node Implementation
# ---------------------------------------------------------------------------

def synthesizer_node(
    state: AgentState,
    llm: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Synthesizer Node for the Zar3a LangGraph workflow.

    Aggregates `rag_context` (legal citations) and `candidate_species` /
    `climate_metrics` (ML recommendations), synthesizes a master response in Arabic
    (maintaining UTF-8), and updates both `state["final_response"]` and `state["messages"]`.

    Args:
        state: Shared `AgentState` containing conversation messages and specialist outputs.
        llm: Optional ChatOpenAI runnable for synthesis.

    Returns:
        Dict[str, Any]: Partial state update:
            {
                "final_response": synthesized_text,
                "messages": [AIMessage(content=synthesized_text)],
            }
    """
    messages_input: Sequence[Any] = state.get("messages", [])
    rag_context: Optional[str] = state.get("rag_context")
    climate_metrics: Optional[Dict[str, Any]] = state.get("climate_metrics")
    candidate_species: Optional[List[Dict[str, Any]]] = state.get("candidate_species")
    language: str = state.get("language", DEFAULT_LANGUAGE)

    # 1. Extract query text from latest user message
    query_text = ""
    for msg in reversed(messages_input):
        if isinstance(msg, HumanMessage):
            query_text = str(msg.content).strip()
            break
        elif isinstance(msg, dict) and msg.get("role") in ("user", "human"):
            query_text = str(msg.get("content", "")).strip()
            break

    if not query_text and messages_input:
        last_item = messages_input[-1]
        if isinstance(last_item, BaseMessage):
            query_text = str(last_item.content).strip()
        elif isinstance(last_item, dict):
            query_text = str(last_item.get("content", "")).strip()
        elif isinstance(last_item, str):
            query_text = last_item.strip()

    logger.info(
        "SynthesizerNode merging outputs (query: '%s...', rag_context: %s, candidates: %d)",
        query_text[:60],
        bool(rag_context),
        len(candidate_species) if candidate_species else 0,
    )

    # 2. Invoke LLM for master synthesis
    active_llm = llm if llm is not None else get_synthesizer_llm()
    synthesized_content = ""

    if active_llm is not None:
        try:
            # Construct structured synthesis prompt
            context_payload = {
                "user_query": query_text,
                "retrieved_legal_context": rag_context or "No legal context requested or retrieved.",
                "climate_metrics": climate_metrics or {},
                "tree_recommendations": candidate_species or [],
                "target_language": language,
            }

            user_prompt = (
                f"Please synthesize the following urban greening and legal data into a unified, "
                f"authoritative, and beautifully formatted response:\n\n"
                f"{json.dumps(context_payload, indent=2, ensure_ascii=False)}"
            )

            prompt_messages = [
                SystemMessage(content=_SYNTHESIZER_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ]

            response = active_llm.invoke(prompt_messages)
            synthesized_content = response.content if hasattr(response, "content") else str(response)
            logger.info("Successfully generated master synthesis via LLM (%d chars).", len(synthesized_content))
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM synthesis failed (%s). Using fallback synthesis.", exc)
            synthesized_content = _generate_fallback_synthesis(
                query=query_text,
                rag_context=rag_context,
                climate_metrics=climate_metrics,
                candidate_species=candidate_species,
                language=language,
            )
    else:
        synthesized_content = _generate_fallback_synthesis(
            query=query_text,
            rag_context=rag_context,
            climate_metrics=climate_metrics,
            candidate_species=candidate_species,
            language=language,
        )

    # 3. Return state update with both final_response and messages
    return {
        "final_response": synthesized_content,
        "messages": [AIMessage(content=synthesized_content)],
    }
