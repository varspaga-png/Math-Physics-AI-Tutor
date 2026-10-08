# Math Physics AI Tutor

A production-ready tutoring system for mathematics and physics reasoning. This project turns the original notebook prototype into a clean, testable, API-driven AI tutoring application with a mock-first architecture for local development and support for Hugging Face model backends in production.

## Features

- Math and physics question answering
- Structured, step-by-step explanations
- Subject-aware tutor logic
- FastAPI API with health and chat endpoints
- Config-based model selection
- Offline-friendly mock backend for local development
- Simple test suite

## Quick start

1. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Start the API:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```
4. Call the endpoint:
   ```bash
   curl -X POST http://localhost:8000/api/v1/ask \
     -H "Content-Type: application/json" \
     -d '{
       "question": "Explain the derivative of x^2",
       "subject": "math",
       "difficulty": "beginner"
     }'
   ```

## Environment variables

Copy `.env.example` to `.env` and adjust values:

```bash
cp .env.example .env
```

Supported settings:

- `MODEL_PROVIDER`: `mock` or `huggingface`
- `HF_MODEL_NAME`: Hugging Face model to load when using the Hugging Face backend
- `APP_HOST`, `APP_PORT`, `DEBUG`

## Project layout

```text
.
├── app/
│   ├── api/
│   │   └── routes.py
│   ├── services/
│   │   ├── model_manager.py
│   │   └── tutor_service.py
│   ├── __init__.py
│   ├── config.py
│   ├── main.py
│   └── schemas.py
├── tests/
│   └── test_tutor_service.py
├── .env.example
├── .gitignore
├── Dockerfile
├── README.md
├── pyproject.toml
├── requirements.txt
└── run.py
```

## Notes

The original repository was notebook-only and not production-ready. This project provides a clean, deployable starting point by separating configuration, model backends, tutoring logic, and API routes.
