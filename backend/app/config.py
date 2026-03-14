"""Application settings loaded from environment variables."""

import logging
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All configuration for Lucio, read from .env file."""

    model_config = SettingsConfigDict(env_file=".env")

    # Mac Studio connection
    mac_studio_base_url: str
    mac_studio_api_key: str = "not-needed"

    # OpenRouter connection (Optional overrides for LLM only)
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_api_key: str | None = None

    # Model names (as served on Mac Studio)
    embedding_model: str
    llm_model: str

    # Embedding provider: "mac_studio" or "openrouter"
    embedding_provider: str = "mac_studio"

    # Tuning parameters
    embedding_dimensions: int = 256
    bm25_top_k: int = 50
    rerank_top_k: int = 5
    llm_max_tokens: int = 1500
    llm_temperature: float = 0.0
    embedding_batch_size: int = 100
    embedding_concurrency: int = 10

    # Runtime flags (set programmatically, not from env)
    supports_dimensions_param: bool = True
