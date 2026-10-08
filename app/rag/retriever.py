from app.rag.embeddings import EmbeddingManager
from app.rag.document_loader import DocumentLoader


class RAGRetriever:
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.embedding_manager = EmbeddingManager(model_name=embedding_model)
        self.documents_loaded = False

    def load_documents(self, documents: list[str]):
        """Load documents into the RAG system."""
        if documents:
            self.embedding_manager.build_index(documents)
            self.documents_loaded = True

    def load_from_directory(self, directory: str, pattern: str = "*.txt"):
        """Load documents from a directory."""
        documents = DocumentLoader.load_directory(directory, pattern)
        self.load_documents(documents)

    def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        """Retrieve relevant documents for a query."""
        if not self.documents_loaded:
            return []
        results = self.embedding_manager.search(query, top_k=top_k)
        return [result["document"] for result in results]

    def retrieve_with_scores(self, query: str, top_k: int = 5) -> list[dict]:
        """Retrieve documents with similarity scores."""
        if not self.documents_loaded:
            return []
        return self.embedding_manager.search(query, top_k=top_k)
