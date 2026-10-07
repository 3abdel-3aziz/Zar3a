"""
test_e2e_rag_flow.py
End-to-end test of the full LangGraph knowledge flow.
Tests that the system now returns dynamically grounded responses.
"""
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "back_end"))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

logging.basicConfig(
    level=logging.WARNING,  # Suppress verbose INFO from embedder loading
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
# Enable INFO for our key modules only
for mod in ["Zar3a.KnowledgeNode", "Zar3a.RAGRetriever", "Zar3a.RouterNode"]:
    logging.getLogger(mod).setLevel(logging.INFO)

from langchain_core.messages import HumanMessage
from src.agent.graph import compiled_graph
from src.agent.state import create_initial_state

test_queries = [
    {
        "id": "test_climate_council",
        "query": "ما هي الأهداف الرئيسية للمجلس الوطني للتغيرات المناخية؟",
        "expected_keywords": ["مجلس", "مناخ", "تغير"],
    },
    {
        "id": "test_law_147",
        "query": "ما هي أحكام قانون الموارد المائية والري رقم 147 لسنة 2021؟",
        "expected_keywords": ["147", "موارد", "مائية", "ري"],
    },
]

config_base = {"configurable": {"thread_id": "e2e-test-001"}}

for tc in test_queries:
    print(f"\n{'='*65}")
    print(f"Query: {tc['query'][:80]}")
    print("="*65)

    state = create_initial_state(
        messages=[HumanMessage(content=tc["query"])],
        language="ar",
    )
    config = {"configurable": {"thread_id": tc["id"]}}

    try:
        final = compiled_graph.invoke(state, config=config)
        response = final.get("final_response") or ""
        rag_ctx = final.get("rag_context") or ""
        routes = final.get("routes", [])

        print(f"Routes taken: {routes}")
        print(f"RAG context length: {len(rag_ctx)} chars")
        has_real_context = rag_ctx and "No relevant" not in rag_ctx and len(rag_ctx.strip()) > 100

        # Check if any expected keywords appear in the response or RAG context
        found_kws = [kw for kw in tc["expected_keywords"] if kw in response or kw in rag_ctx]

        print(f"RAG has real content: {'✅ YES' if has_real_context else '❌ NO (still empty)'}")
        print(f"Expected keywords found: {found_kws} / {tc['expected_keywords']}")
        print(f"\nResponse preview (first 400 chars):\n{response[:400]}")

    except Exception as exc:
        print(f"❌ ERROR: {exc}")
        import traceback
        traceback.print_exc()
