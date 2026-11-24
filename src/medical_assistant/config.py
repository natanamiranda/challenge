
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    app_name: str = "Medical AI Assistant"
    app_env: str = "development"

    ollama_base_url: str = "http://localhost:11434"

    # Modelo responsável por gerar respostas
    ollama_model: str = "llama3.2:1b"

    # Modelo responsável por gerar embeddings
    ollama_embedding_model: str = "nomic-embed-text"

    database_url: str = "sqlite:///./data/medical_assistant.db"

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
