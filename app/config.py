from dataclasses import dataclass
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    app_name: str = "math-physics-ai-tutor"
    app_version: str = "0.1.0"
    model_provider: str = os.getenv("MODEL_PROVIDER", "mock")
    hf_model_name: str = os.getenv("HF_MODEL_NAME", "microsoft/Phi-3-mini-4k-instruct")
    app_host: str = os.getenv("APP_HOST", "0.0.0.0")
    app_port: int = int(os.getenv("APP_PORT", "8000"))
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"
    base_dir: Path = BASE_DIR


settings = Settings()
