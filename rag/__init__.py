"""RelevantAI - RAG package for Django backend integration."""

from .models import ArticleDocument, ArticleSummary, SearchResult
from .exceptions import (
    RAGError,
    ValidationError,
    APIError,
    StorageError,
    SummarizationError,
    SearchError,
    AnswerGenerationError,
    ConfigurationError,
    RateLimitError,
)
from .rag import RelevantAI, get_rag

__version__ = "1.0.0"

__all__ = [
    "RelevantAI",
    "get_rag",
    "ArticleDocument",
    "ArticleSummary",
    "SearchResult",
    "RAGError",
    "ValidationError",
    "APIError",
    "StorageError",
    "SummarizationError",
    "SearchError",
    "AnswerGenerationError",
    "ConfigurationError",
    "RateLimitError",
]
