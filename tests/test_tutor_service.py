from app.config import settings
from app.schemas import TutorRequest
from app.services.tutor_service import TutorService


def test_mock_answer_generation():
    service = TutorService(provider="mock")
    response = service.answer_question(
        TutorRequest(
            question="Find the derivative of x^2",
            subject="math",
            difficulty="beginner",
        )
    )

    assert response.subject == "math"
    assert response.difficulty == "beginner"
    assert len(response.steps) >= 3
    assert "x^2" in response.answer or "derivative" in response.answer.lower()


def test_health_config_values_are_present():
    assert settings.app_name
    assert settings.app_version
