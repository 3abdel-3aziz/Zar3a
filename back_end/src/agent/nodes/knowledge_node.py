"""
src/agent/nodes/knowledge_node.py

Knowledge / RAG Agent Node for the Zar3a LangGraph multi-agent workflow.

This node handles user queries related to urban forestry laws, government regulations,
municipal planting guidelines, and environmental/ecological documentation. It retrieves
grounding document chunks from the Qdrant vector database via the hybrid RAGRetriever
and synthesizes a well-cited, grounded AI response.

Workflow within this node:
  1. Extract and normalize the user's latest query from `state["messages"]`.
  2. Query the hybrid RAGRetriever (dense E5 + BM25 RRF fusion) for relevant chunks.
  3. Format the retrieved chunks into a clean, markdown-structured context string
     and update `state["rag_context"]`.
  4. Synthesize an authoritative answer citing sources using a fast LLM (gpt-4o-mini).
  5. Append the resulting AIMessage to `state["messages"]`.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.agent.config import (
    DEFAULT_LANGUAGE,
    KNOWLEDGE_LLM_MODEL,
    KNOWLEDGE_TEMPERATURE,
)
from src.agent.state import AgentState
from src.rag_database.retriever import RAGRetriever

# ---------------------------------------------------------------------------
# Logger Configuration (Strict INFO level, no DEBUG)
# ---------------------------------------------------------------------------
logger = logging.getLogger("Zar3a.KnowledgeNode")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Default number of chunks to retrieve for knowledge grounding
DEFAULT_TOP_K: int = 5

# ---------------------------------------------------------------------------
# System Prompt for Knowledge Synthesis
# ---------------------------------------------------------------------------
_KNOWLEDGE_SYSTEM_PROMPT = """\
أنت متخصص المعرفة والتشريعات البيئية في منصة "زرعة" (Zar3a) للبنية الحضرية والتخضير في مصر.

مهمتك تقديم إجابات دقيقة وموثقة ومحترفة على الأسئلة المتعلقة بـ:
- التشريعات البيئية المصرية وقوانين الغابات والمراسيم الوزارية (مثال: قانون البيئة رقم 4/1994).
- إرشادات التباعد البلدية، ضوابط مرافق الطرق العامة، وتشريعات تشجير الأرصفة.
- التصنيف النباتي والخصائص البيئية ومعايير التخضير الحضري.

ضوابط الإجابة:
1. **اللغة العربية أولاً**: يجب أن تكون الإجابة باللغة العربية الفصحى الاحترافية في جميع الأحوال. استخدم الإنجليزية فقط إذا طلب المستخدم صراحةً.
2. **التقيد بالمصادر**: ابني إجابتك على "السياق المسترجع" بالكامل. لا تخترع أرقاماً للقوانين أو تفاصيل غير واردة في السياق.
3. **استشهاد صريح**: استخدم الصيغة [Document X: اسم الملف] لكل وثيقة محتج بها.
4. إذا كان السياق غير كافٍ للإجابة بشكل كامل، أوضح ما هو متاح وما هو مفقود بصراحة.
"""


# ---------------------------------------------------------------------------
# Context Formatting Utility
# ---------------------------------------------------------------------------

def format_rag_context(chunks: List[Dict[str, Any]]) -> str:
    """
    Formats a list of retrieved chunk dictionaries into a clean, markdown-formatted
    context string suitable for LLM prompt injection and state persistence.

    Args:
        chunks: List of chunk dictionaries returned by RAGRetriever.

    Returns:
        A structured string containing formatted document headers, metadata, and contents.
    """
    if not chunks:
        return "No relevant legal or environmental documents found in the knowledge base."

    formatted_entries: List[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        content = str(chunk.get("content", "")).strip()
        metadata = chunk.get("metadata", {}) or {}
        file_name = metadata.get("file_name") or f"Doc #{metadata.get('doc_id', idx)}"
        chunk_idx = metadata.get("chunk_index")
        score = float(chunk.get("score", 0.0))

        chunk_label = f" (Chunk {chunk_idx})" if chunk_idx is not None else ""
        header = f"[Document {idx}: {file_name}{chunk_label} | Score: {score:.4f}]"
        formatted_entries.append(f"{header}\n{content}")

    return "\n\n---\n\n".join(formatted_entries)


# ---------------------------------------------------------------------------
# Fallback Synthesis Generator
# ---------------------------------------------------------------------------

def _generate_fallback_synthesis(
    query: str,
    chunks: List[Dict[str, Any]],
    language: str = "ar",
) -> str:
    """
    Generates a structured citation summary when LLM generation is unavailable or fails.

    Args:
        query: The user query string.
        chunks: Retrieved candidate chunks.
        language: Interface language ('ar' | 'en').

    Returns:
        A synthesized fallback response with citations.
    """
    if not chunks:
        if language == "en":
            return (
                "No relevant legal or ecological documentation was found in the "
                "knowledge base matching your inquiry. Please consult local municipal "
                "authorities or refine your search terms."
            )
        return (
            "لم يتم العثور على وثائق قانونية أو بيئية مطابقة لاستفسارك في قاعدة المعرفة. "
            "يرجى مراجعة اللوائح الرسمية للحي أو إعادة صياغة الاستفسار بدقة أكبر."
        )

    is_english = language == "en"
    citations: List[str] = []

    for idx, chunk in enumerate(chunks, start=1):
        meta = chunk.get("metadata", {}) or {}
        file_name = meta.get("file_name") or f"Doc #{idx}"
        chunk_idx = meta.get("chunk_index")
        label = f"{file_name}" + (f" (Chunk {chunk_idx})" if chunk_idx is not None else "")
        snippet = str(chunk.get("content", "")).strip()
        if len(snippet) > 280:
            snippet = snippet[:280] + "..."
        citations.append(f"• **[Document {idx}: {label}]**\n  {snippet}")

    citations_block = "\n\n".join(citations)

    if is_english:
        return (
            f"Based on the official documents retrieved from the Zar3a knowledge base, "
            f"here are the relevant legal and environmental excerpts:\n\n{citations_block}"
        )
    return (
        f"بناءً على الوثائق والمراجع الرسمية المسترجعة من قاعدة معرفة 'زرعة'، "
        f"إليك أهم البنود والمقتطفات المتعلقة باستفسارك:\n\n{citations_block}"
    )


# ---------------------------------------------------------------------------
# Dependency Factories (Retriever & LLM)
# ---------------------------------------------------------------------------

_retriever_instance: Optional[RAGRetriever] = None
_knowledge_llm_instance: Optional[Any] = None


def get_retriever() -> RAGRetriever:
    """Returns the cached RAGRetriever singleton, instantiating it lazily if needed."""
    global _retriever_instance
    if _retriever_instance is None:
        logger.info("Initializing RAGRetriever for knowledge_node...")
        _retriever_instance = RAGRetriever()
    return _retriever_instance


def set_retriever(retriever: Optional[RAGRetriever]) -> None:
    """Overrides or resets the cached retriever (useful for unit testing)."""
    global _retriever_instance
    _retriever_instance = retriever


def _build_knowledge_llm(model: Optional[str] = None, api_key: Optional[str] = None) -> Any:
    """Constructs the LLM client for knowledge response synthesis using config settings."""
    selected_model = model or KNOWLEDGE_LLM_MODEL
    openai_key = api_key or os.getenv("OPENAI_API_KEY")

    kwargs: Dict[str, Any] = {
        "model": selected_model,
        "temperature": KNOWLEDGE_TEMPERATURE,
    }
    if openai_key:
        kwargs["api_key"] = openai_key

    return ChatOpenAI(**kwargs)


def get_knowledge_llm() -> Any:
    """Returns the cached LLM singleton, instantiating it lazily if needed."""
    global _knowledge_llm_instance
    if _knowledge_llm_instance is None:
        logger.info("Initializing knowledge synthesis LLM (model=%s, temperature=%.1f).", KNOWLEDGE_LLM_MODEL, KNOWLEDGE_TEMPERATURE)
        _knowledge_llm_instance = _build_knowledge_llm()
    return _knowledge_llm_instance


def set_knowledge_llm(llm: Optional[Any]) -> None:
    """Overrides or resets the cached LLM (useful for unit testing)."""
    global _knowledge_llm_instance
    _knowledge_llm_instance = llm


# ---------------------------------------------------------------------------
# Knowledge Node Implementation
# ---------------------------------------------------------------------------

def knowledge_node(
    state: AgentState,
    retriever: Optional[RAGRetriever] = None,
    llm: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Knowledge / RAG Agent Node for the Zar3a LangGraph workflow.

    Extracts the latest user message from `state["messages"]`, queries the
    Qdrant hybrid retriever for grounding documents, updates `state["rag_context"]`,
    and appends a synthesized `AIMessage` with source citations to `state["messages"]`.

    Args:
        state: The shared `AgentState` dictionary.
        retriever: Optional custom or mocked retriever instance.
        llm: Optional custom or mocked LLM instance.

    Returns:
        Dict[str, Any]: Partial state update with `rag_context` and `messages`.
    """
    messages_input: Sequence[Any] = state.get("messages", [])
    language: str = state.get("language", DEFAULT_LANGUAGE)

    # 1. Extract latest user query text
    query_text = ""
    for msg in reversed(messages_input):
        if isinstance(msg, HumanMessage):
            query_text = str(msg.content).strip()
            break
        elif isinstance(msg, dict) and msg.get("role") in ("user", "human"):
            query_text = str(msg.get("content", "")).strip()
            break

    # If no explicit HumanMessage found, take the last message content
    if not query_text and messages_input:
        last_item = messages_input[-1]
        if isinstance(last_item, BaseMessage):
            query_text = str(last_item.content).strip()
        elif isinstance(last_item, dict):
            query_text = str(last_item.get("content", "")).strip()
        elif isinstance(last_item, str):
            query_text = last_item.strip()

    if not query_text:
        logger.warning("KnowledgeNode received an empty user query. Returning fallback response.")
        fallback_msg = (
            "لم يتم استلام استفسار واضح للبحث في الوثائق."
            if language == "ar"
            else "No query received to search knowledge documents."
        )
        return {
            "rag_context": "No query provided for knowledge retrieval.",
            "messages": [AIMessage(content=fallback_msg)],
        }

    logger.info("KnowledgeNode processing query (%d chars): '%s...'", len(query_text), query_text[:80])

    # 2. Retrieve document chunks via hybrid retriever
    chunks: List[Dict[str, Any]] = []
    active_retriever = retriever if retriever is not None else get_retriever()

    try:
        top_k = int(os.getenv("KNOWLEDGE_TOP_K", str(DEFAULT_TOP_K)))
        chunks = active_retriever.retrieve(query=query_text, top_k=top_k)
        logger.info(
            "Retriever returned %d chunk(s) for query. Top score: %.4f",
            len(chunks),
            chunks[0].get("score", 0.0) if chunks else 0.0,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Error executing retrieval in KnowledgeNode for query '%s': %s",
            query_text[:50],
            exc,
            exc_info=True,
        )
        chunks = []

    # 3. Format retrieved context for state and LLM prompting
    formatted_context = format_rag_context(chunks)

    # 4. Synthesize AI response citing sources
    ai_content = ""
    active_llm = llm if llm is not None else get_knowledge_llm()

    try:
        user_prompt = (
            f"User Inquiry:\n{query_text}\n\n"
            f"Retrieved Context from Zar3a Knowledge Base:\n{formatted_context}\n\n"
            f"Please synthesize an authoritative, well-structured answer strictly grounded "
            f"in the context above, citing each document or decree referenced."
        )

        prompt_messages = [
            SystemMessage(content=_KNOWLEDGE_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]

        response = active_llm.invoke(prompt_messages)
        ai_content = response.content if hasattr(response, "content") else str(response)
        logger.info("Successfully synthesized grounded knowledge response (%d chars).", len(ai_content))

    except Exception as exc:  # noqa: BLE001
        logger.error(
            "LLM synthesis failed in KnowledgeNode: %s. Using structured fallback synthesis.",
            exc,
            exc_info=True,
        )
        ai_content = _generate_fallback_synthesis(query=query_text, chunks=chunks, language=language)

    # 5. Return partial state update
    return {
        "rag_context": formatted_context,
        "messages": [AIMessage(content=ai_content)],
    }
