"""
src/api/routers/chat.py

FastAPI router exposing the Zar3a LangGraph Multi-Agent Workflow.
Executes the supervisor router, specialist pipelines (RAG and Climate), and master synthesizer.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field

from agent.state import create_initial_state
from api.dependencies import get_agent_graph

logger = logging.getLogger("Zar3a.API.Chat")

router = APIRouter()


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------

class ChatMessageRequest(BaseModel):
    """Input payload for a multi-agent chat interaction."""

    message: str = Field(
        ...,
        min_length=1,
        description="The user's query or prompt regarding urban forestry, laws, or tree selection.",
        examples=["ما هي الأشجار المناسبة لشارع ضيق في القاهرة والتي تخفض درجة الحرارة؟"],
    )
    thread_id: Optional[str] = Field(
        default=None,
        description="Optional session/thread identifier for multi-turn persistence. Generates a new UUID if omitted.",
        examples=["session-cairo-001"],
    )
    language: str = Field(
        default="ar",
        description="Preferred language for synthesis ('ar' for Arabic, 'en' for English).",
        examples=["ar"],
    )


class ChatMessageResponse(BaseModel):
    """Synthesized response returned from the LangGraph multi-agent workflow."""

    response: str = Field(..., description="Final synthesized response assembled by the synthesizer node")
    thread_id: str = Field(..., description="The active thread ID for this conversation")
    routes: List[str] = Field(..., description="Specialist pipelines triggered by the supervisor router")
    rag_context: Optional[str] = Field(None, description="Legal and regulatory context retrieved from Qdrant")
    climate_metrics: Optional[Dict[str, Any]] = Field(None, description="Quantitative microclimate cooling metrics")
    candidate_species: Optional[List[Dict[str, Any]]] = Field(None, description="Ranked recommended tree species")
    language: str = Field(..., description="Response language")


# ---------------------------------------------------------------------------
# Route Handlers
# ---------------------------------------------------------------------------

@router.post(
    "/messages",
    response_model=ChatMessageResponse,
    status_code=status.HTTP_200_OK,
    summary="Send Message to Zar3a Multi-Agent System",
    description="Invokes the compiled LangGraph workflow to process queries across environmental law and climate ML.",
)
def send_chat_message(
    payload: ChatMessageRequest,
    graph: Any = Depends(get_agent_graph),
) -> ChatMessageResponse:
    """
    Executes a turn in the Zar3a LangGraph workflow with checkpointer persistence.
    """
    thread_id = payload.thread_id.strip() if payload.thread_id else str(uuid.uuid4())
    logger.info("Executing chat message for thread '%s' (lang=%s)", thread_id, payload.language)

    # 1. Build initial state with incoming message
    user_msg = HumanMessage(content=payload.message)
    initial_state = create_initial_state(
        messages=[user_msg],
        language=payload.language,
    )

    # 2. Invoke LangGraph with thread_id checkpoint configuration
    config = {"configurable": {"thread_id": thread_id}}

    try:
        final_state = graph.invoke(initial_state, config=config)

        # 3. Extract synthesized final response
        final_response_text = final_state.get("final_response")
        if not final_response_text:
            # Fallback to last AIMessage if final_response was not populated
            messages = final_state.get("messages", [])
            for m in reversed(messages):
                if isinstance(m, AIMessage):
                    final_response_text = str(m.content)
                    break
            if not final_response_text:
                final_response_text = "تمت معالجة استفسارك بنجاح ولكن لم يتم إنشاء رد نهائي."

        return ChatMessageResponse(
            response=final_response_text,
            thread_id=thread_id,
            routes=final_state.get("routes", []),
            rag_context=final_state.get("rag_context"),
            climate_metrics=final_state.get("climate_metrics"),
            candidate_species=final_state.get("candidate_species"),
            language=final_state.get("language", payload.language),
        )

    except Exception as exc:
        logger.error("Error executing agent graph for thread '%s': %s", thread_id, exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent execution encountered an error: {exc}",
        ) from exc
