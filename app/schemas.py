from pydantic import BaseModel, Field


class TutorRequest(BaseModel):
    question: str = Field(..., min_length=3, description="Question to answer")
    subject: str = Field(default="math", description="Subject: math or physics")
    difficulty: str = Field(default="intermediate", description="Difficulty: beginner, intermediate, advanced")
    context: str | None = Field(default=None, description="Optional additional context")


class TutorResponse(BaseModel):
    answer: str
    subject: str
    difficulty: str
    model_provider: str
    steps: list[str] = []
