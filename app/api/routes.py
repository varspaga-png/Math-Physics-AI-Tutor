from fastapi import APIRouter, HTTPException, UploadFile, File
from pathlib import Path

from app.config import settings
from app.schemas import TutorRequest, TutorResponse
from app.services.rag_tutor_service import RAGTutorService

router = APIRouter(prefix="/api/v1", tags=["tutor"])
service = RAGTutorService()


@router.get("/health")
def health_check() -> dict:
    return {
        "status": "ok",
        "service": settings.app_name,
        "version": settings.app_version,
        "model_provider": settings.model_provider,
        "rag_enabled": service.rag_retriever.documents_loaded,
    }


@router.post("/ask", response_model=TutorResponse)
def ask_tutor(payload: TutorRequest) -> TutorResponse:
    try:
        return service.answer_question(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/load-rag-documents")
def load_rag_documents(documents: list[str]) -> dict:
    try:
        service.load_rag_documents(documents)
        return {"status": "ok", "message": f"Loaded {len(documents)} documents"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/load-rag-directory")
def load_rag_directory(directory: str, pattern: str = "*.txt") -> dict:
    try:
        service.load_rag_from_directory(directory, pattern)
        return {"status": "ok", "message": f"Loaded documents from {directory}"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
