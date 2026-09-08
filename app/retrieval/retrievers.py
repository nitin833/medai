"""RAG retrieval helpers."""

from app.config import RAG_TOP_K
from app.retrieval.qdrant_client import get_medical_store, get_merck_store


def retrieve(query: str, *, k: int = RAG_TOP_K) -> list:
    medical_docs = get_medical_store().as_retriever(search_kwargs={"k": k}).invoke(query)
    merck_docs = get_merck_store().as_retriever(search_kwargs={"k": k}).invoke(query)
    return medical_docs + merck_docs


def build_context(docs: list) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        parts.append(
            f"Document {i}\n\nMetadata:\n{doc.metadata}\n\nContent:\n{doc.page_content}\n"
            f"{'-' * 42}"
        )
    return "\n".join(parts)
