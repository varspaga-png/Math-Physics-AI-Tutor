from fastapi import APIRouter, HTTPException

from app.config import settings
from app.schemas import TutorRequest, TutorResponse
from app.services.tutor_service import TutorService

router = APIRouter(prefix="/api/v1", tags=["tutor"])
service = TutorService()


@router.get("/health")
def health_check() -> dict:
    return {
        "status": "ok",
        "service": settings.app_name,
        "version": settings.app_version,
        "model_provider": settings.model_provider,
    }


@router.post("/ask", response_model=TutorResponse)
def ask_tutor(payload: TutorRequest) -> TutorResponse:
    try:
        return service.answer_question(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
