from __future__ import annotations

from abc import ABC, abstractmethod


class BaseModelBackend(ABC):
    @abstractmethod
    def generate_response(self, question: str, subject: str, difficulty: str, context: str | None = None) -> str:
        raise NotImplementedError


class MockModelBackend(BaseModelBackend):
    def generate_response(self, question: str, subject: str, difficulty: str, context: str | None = None) -> str:
        q = (question or "").strip()
        subject = (subject or "math").lower()
        difficulty = (difficulty or "intermediate").lower()

        if subject == "physics":
            answer = (
                f"In {subject}, the main idea is to model the system with the relevant laws of physics. "
                f"For the question '{q}', begin by identifying the known variables, then connect them using the governing equation. "
                f"For a {difficulty} level explanation, focus on the core principle, derive the relationship, and check units and signs."
            )
        else:
            answer = (
                f"For the mathematical question '{q}', start by recognizing the structure of the expression. "
                f"Then isolate the key concept: simplify, manipulate the equation, or differentiate/integrate when appropriate. "
                f"At a {difficulty} level, explain the logic step by step and state the final result clearly."
            )

        if context:
            answer += f" Context provided: {context}."

        return answer


class HuggingFaceModelBackend(BaseModelBackend):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._tokenizer = None
        self._model = None

    def _load_model(self):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("transformers is not installed. Install project requirements or switch to MODEL_PROVIDER=mock.") from exc

        if self._tokenizer is None or self._model is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForCausalLM.from_pretrained(self.model_name)

        return self._tokenizer, self._model

    def generate_response(self, question: str, subject: str, difficulty: str, context: str | None = None) -> str:
        tokenizer, model = self._load_model()
        prompt = self._build_prompt(question, subject, difficulty, context)
        inputs = tokenizer(prompt, return_tensors="pt")
        output = model.generate(**inputs, max_new_tokens=256)
        decoded = tokenizer.decode(output[0], skip_special_tokens=True)
        return decoded.split("Assistant:", 1)[-1].strip() if "Assistant:" in decoded else decoded.strip()

    @staticmethod
    def _build_prompt(question: str, subject: str, difficulty: str, context: str | None = None) -> str:
        ctx = f"Context: {context}\n" if context else ""
        return (
            "You are a helpful math and physics tutor.\n"
            f"Subject: {subject}\n"
            f"Difficulty: {difficulty}\n"
            f"{ctx}"
            f"Question: {question}\n"
            "Provide a clear, structured explanation with steps and a final answer.\n"
            "Assistant:"
        )


class ModelManager:
    def __init__(self, provider: str = "mock", model_name: str = "microsoft/Phi-3-mini-4k-instruct"):
        self.provider = provider.lower()
        self.model_name = model_name

    def get_backend(self) -> BaseModelBackend:
        if self.provider == "mock":
            return MockModelBackend()
        if self.provider == "huggingface":
            return HuggingFaceModelBackend(self.model_name)
        raise ValueError(f"Unsupported model provider: {self.provider}")
