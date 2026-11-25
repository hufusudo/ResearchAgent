"""Application configuration loaded from environment variables and .env."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings for ReacherAgent."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm_api_key: str = ""
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"

    embedding_provider: Literal["openai", "local_hash"] = "local_hash"
    embedding_api_key: str = ""
    embedding_base_url: str = ""
    embedding_model: str = "text-embedding-3-small"

    max_papers_per_query: int = 10
    top_k_papers: int = 8
    max_iterations: int = 2

    academic_sources: str = "arxiv,openalex"
    openalex_email: str = ""
    semantic_scholar_api_key: str = ""

    data_dir: str = "data"
    output_dir: str = "outputs"

    @property
    def effective_embedding_api_key(self) -> str:
        """Return embedding API key, falling back to the LLM API key."""
        return self.embedding_api_key.strip() or self.llm_api_key.strip()

    @property
    def effective_embedding_base_url(self) -> str:
        """Return embedding base URL, falling back to the LLM base URL."""
        return self.embedding_base_url.strip() or self.llm_base_url.strip()

    @property
    def academic_sources_list(self) -> list[str]:
        """Return normalized list of enabled academic search sources."""
        items = [s.strip().lower() for s in self.academic_sources.split(",") if s.strip()]
        return items or ["arxiv", "openalex"]

    @property
    def data_path(self) -> Path:
        """Return resolved Path for data storage."""
        return Path(self.data_dir)

    @property
    def output_path(self) -> Path:
        """Return resolved Path for survey outputs."""
        return Path(self.output_dir)


def get_settings() -> Settings:
    """Create a fresh Settings instance from the current environment."""
    return Settings()
