from app.config import settings
from app.schemas import TutorRequest, TutorResponse
from app.services.tutor_service import TutorService
from app.rag.retriever import RAGRetriever


class RAGTutorService(TutorService):
    def __init__(
        self,
        provider: str | None = None,
        model_name: str | None = None,
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        super().__init__(provider=provider, model_name=model_name)
        self.rag_retriever = RAGRetriever(embedding_model=embedding_model)

    def load_rag_documents(self, documents: list[str]):
        """Load documents into the RAG retriever."""
        self.rag_retriever.load_documents(documents)

    def load_rag_from_directory(self, directory: str, pattern: str = "*.txt"):
        """Load documents from directory into the RAG retriever."""
        self.rag_retriever.load_from_directory(directory, pattern)

    def answer_question(self, payload: TutorRequest) -> TutorResponse:
        # Retrieve relevant context from RAG
        retrieved_docs = self.rag_retriever.retrieve(payload.question, top_k=3)
        context = "\n".join(retrieved_docs) if retrieved_docs else ""

        # Create enhanced payload with retrieved context
        enhanced_payload = TutorRequest(
            question=payload.question,
            subject=payload.subject,
            difficulty=payload.difficulty,
            context=context if not payload.context else f"{payload.context}\n{context}",
        )

        # Get response from parent tutor service
        response = super().answer_question(enhanced_payload)

        # Add retrieved documents info
        if retrieved_docs:
            response.steps.insert(0, f"Retrieved {len(retrieved_docs)} relevant documents from knowledge base.")

        return response
