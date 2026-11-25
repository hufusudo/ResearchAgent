"""LLM and Embedding abstraction layer."""

from core.llm.client import LLMClientProtocol, MockLLMClient, OpenAILLMClient, create_llm_client
from core.llm.embeddings import (
    EmbedderProtocol,
    LocalHashEmbedder,
    OpenAIEmbedder,
    create_embedder,
)

__all__ = [
    "EmbedderProtocol",
    "LLMClientProtocol",
    "LocalHashEmbedder",
    "MockLLMClient",
    "OpenAIEmbedder",
    "OpenAILLMClient",
    "create_embedder",
    "create_llm_client",
]
