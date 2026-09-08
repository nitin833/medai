"""RAG tool — retrieves from Qdrant knowledge base."""

from app.graph.state import MedAIState
from app.retrieval.retrievers import build_context, retrieve


def _query_from_state(state: MedAIState) -> str:
    disease = state.get("detected_disease")
    if disease:
        return f"{disease} symptoms treatment drugs"

    for msg in reversed(state.get("messages", [])):
        content = getattr(msg, "content", None) or (msg.get("content") if isinstance(msg, dict) else None)
        if content and not content.startswith("[router]"):
            return content
    return ""


def rag_tool_node(state: MedAIState) -> dict:
    query = _query_from_state(state)
    docs = retrieve(query)
    context = build_context(docs)

    return {
        "retrieved_docs": docs,
        "rag_context": context,
    }
