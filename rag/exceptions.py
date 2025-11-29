"""Custom exceptions for RelevantAI."""

from typing import Optional, Dict, Any


class RAGError(Exception):
    """Base exception for RAG operations."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}
    
    def to_dict(self) -> Dict[str, Any]:
        return {"error": self.__class__.__name__, "message": self.message, "details": self.details}


class ValidationError(RAGError):
    """Invalid input data."""
    
    def __init__(self, message: str, field: Optional[str] = None, value: Any = None):
        super().__init__(message, {"field": field, "value": str(value)[:100] if value else None})
        self.field = field
        self.value = value


class APIError(RAGError):
    """External API error."""
    
    def __init__(self, message: str, service: Optional[str] = None, 
                 status_code: Optional[int] = None, response: Optional[str] = None):
        super().__init__(message, {"service": service, "status_code": status_code, "response": response[:500] if response else None})
        self.service = service
        self.status_code = status_code
        self.response = response


class StorageError(RAGError):
    """Vector store operation error."""
    
    def __init__(self, message: str, operation: Optional[str] = None, article_id: Optional[str] = None):
        super().__init__(message, {"operation": operation, "article_id": article_id})
        self.operation = operation
        self.article_id = article_id


class SummarizationError(RAGError):
    """Summarization failure."""
    
    def __init__(self, message: str, title: Optional[str] = None, content_length: Optional[int] = None):
        super().__init__(message, {"title": title, "content_length": content_length})
        self.title = title
        self.content_length = content_length


class SearchError(RAGError):
    """Search operation error."""
    
    def __init__(self, message: str, query: Optional[str] = None, filters: Optional[Dict] = None):
        super().__init__(message, {"query": query[:100] if query else None, "filters": filters})
        self.query = query
        self.filters = filters


class AnswerGenerationError(RAGError):
    """Answer generation failure."""
    
    def __init__(self, message: str, question: Optional[str] = None, sources_count: Optional[int] = None):
        super().__init__(message, {"question": question[:100] if question else None, "sources_count": sources_count})
        self.question = question
        self.sources_count = sources_count


class ConfigurationError(RAGError):
    """Missing or invalid configuration."""
    
    def __init__(self, message: str, missing_config: Optional[str] = None):
        super().__init__(message, {"missing_config": missing_config})
        self.missing_config = missing_config


class RateLimitError(APIError):
    """API rate limit exceeded."""
    
    def __init__(self, message: str, service: Optional[str] = None, retry_after: Optional[int] = None):
        super().__init__(message, service=service)
        self.retry_after = retry_after
        self.details["retry_after"] = retry_after
