import pytest
from app.rag.embeddings import EmbeddingManager
from app.rag.retriever import RAGRetriever


def test_embedding_manager_builds_index():
    manager = EmbeddingManager()
    docs = [
        "The derivative of x^2 is 2x",
        "Newton's Second Law is F = ma",
        "The integral of x^3 is x^4/4",
    ]
    manager.build_index(docs)
    assert manager.index is not None
    assert len(manager.documents) == 3


def test_rag_retriever_search():
    retriever = RAGRetriever()
    docs = [
        "The derivative of x^2 is 2x using the power rule",
        "Newton's Second Law states F = ma",
        "The integral of x^3 is x^4/4 plus a constant",
    ]
    retriever.load_documents(docs)
    results = retriever.retrieve("derivative of x squared", top_k=1)
    assert len(results) >= 1
    assert "derivative" in results[0].lower()
