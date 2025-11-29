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
from .interests import (
    FIXED_CATEGORIES,
    CATEGORY_NAMES,
    CategoryScores,
    InterestProfile,
    TopicMatcher,
    update_interest_profile,
    classify_text_to_categories,
    get_category_names,
)

__version__ = "1.0.0"

__all__ = [
    # Main RAG
    "RelevantAI",
    "get_rag",
    # Models
    "ArticleDocument",
    "ArticleSummary",
    "SearchResult",
    # Interests
    "FIXED_CATEGORIES",
    "CATEGORY_NAMES",
    "CategoryScores",
    "InterestProfile",
    "TopicMatcher",
    "update_interest_profile",
    "classify_text_to_categories",
    "get_category_names",
    # Exceptions
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
