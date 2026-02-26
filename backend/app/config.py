"""Application settings loaded from environment variables."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All configuration for Lucio, read from .env file."""

    model_config = SettingsConfigDict(env_file=".env")

    # Mac Studio connection
    mac_studio_base_url: str
    mac_studio_api_key: str = "not-needed"

    # Model names (as served on Mac Studio)
    embedding_model: str
    llm_model: str

    # Tuning parameters
    embedding_dimensions: int = 256
    bm25_top_k: int = 75
    rerank_top_k: int = 5
    llm_max_tokens: int = 250
    llm_temperature: float = 0.0
    embedding_batch_size: int = 100

    # Runtime flags (set programmatically, not from env)
    supports_dimensions_param: bool = True
