from app.config import settings
from app.schemas import TutorRequest, TutorResponse
from app.services.model_manager import ModelManager


class TutorService:
    def __init__(self, provider: str | None = None, model_name: str | None = None):
        self.model_manager = ModelManager(
            provider=provider or settings.model_provider,
            model_name=model_name or settings.hf_model_name,
        )

    def answer_question(self, payload: TutorRequest) -> TutorResponse:
        backend = self.model_manager.get_backend()
        response_text = backend.generate_response(
            question=payload.question,
            subject=payload.subject,
            difficulty=payload.difficulty,
            context=payload.context,
        )

        steps = self._build_steps(payload.subject, payload.difficulty)

        return TutorResponse(
            answer=response_text,
            subject=(payload.subject or "math").lower(),
            difficulty=(payload.difficulty or "intermediate").lower(),
            model_provider=self.model_manager.provider,
            steps=steps,
        )

    @staticmethod
    def _build_steps(subject: str, difficulty: str) -> list[str]:
        subject_name = (subject or "math").lower()
        difficulty_name = (difficulty or "intermediate").lower()

        if subject_name == "physics":
            return [
                "Identify the physical quantities involved.",
                "Choose the correct law or equation.",
                "Substitute known values carefully.",
                "Check units and final interpretation.",
                f"Present the answer at a {difficulty_name} level.",
            ]

        return [
            "Identify the mathematical operation or structure.",
            "Simplify the expression or equation.",
            "Apply the correct theorem, rule, or method.",
            "Verify the final result for consistency.",
            f"Explain the answer at a {difficulty_name} level.",
        ]
