"""Qdrant vector store setup."""

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from app.config import (
    COLLECTION_MEDICAL,
    COLLECTION_MERCK,
    EMBEDDING_MODEL,
    QDRANT_PATH,
    QDRANT_URL,
    RAG_TOP_K,
)

_client: QdrantClient | None = None
_embeddings: HuggingFaceEmbeddings | None = None
_medical_store: QdrantVectorStore | None = None
_merck_store: QdrantVectorStore | None = None


def get_qdrant_client() -> QdrantClient:
    global _client
    if _client is None:
        if QDRANT_URL:
            _client = QdrantClient(url=QDRANT_URL)
        else:
            _client = QdrantClient(path=QDRANT_PATH)
    return _client


def get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _embeddings


def get_medical_store() -> QdrantVectorStore:
    global _medical_store
    if _medical_store is None:
        _medical_store = QdrantVectorStore(
            client=get_qdrant_client(),
            collection_name=COLLECTION_MEDICAL,
            embedding=get_embeddings(),
        )
    return _medical_store


def get_merck_store() -> QdrantVectorStore:
    global _merck_store
    if _merck_store is None:
        _merck_store = QdrantVectorStore(
            client=get_qdrant_client(),
            collection_name=COLLECTION_MERCK,
            embedding=get_embeddings(),
        )
    return _merck_store


def get_retriever_top_k() -> int:
    return RAG_TOP_K
