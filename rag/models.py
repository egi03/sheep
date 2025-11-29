"""Data models for RelevantAI."""

from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from pydantic import BaseModel, Field


class ArticleSummary(BaseModel):
    summary: str = Field(..., min_length=10)
    key_points: List[str] = Field(default_factory=list)
    category: str = Field(default="Other")
    tags: List[str] = Field(default_factory=list)


class ArticleDocument(BaseModel):
    id: str
    title: str
    url: str
    summary: str
    category: str = "Other"
    tags: List[str] = Field(default_factory=list)
    key_points: List[str] = Field(default_factory=list)
    source: str = ""
    author: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_pinecone_metadata(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "url": self.url,
            "summary": self.summary[:1000],
            "category": self.category,
            "tags": self.tags[:10],
            "source": self.source,
            "author": self.author,
            "created_at": self.created_at.isoformat()
        }


class SearchResult(BaseModel):
    id: str
    title: str
    url: str
    summary: str
    category: str
    tags: List[str]
    similarity_score: float
    source: Optional[str] = None
    author: Optional[str] = None
    created_at: Optional[str] = None

    @classmethod
    def from_pinecone_match(cls, match: Dict[str, Any]) -> "SearchResult":
        meta = match.get("metadata", {})
        return cls(
            id=match["id"],
            title=meta.get("title", ""),
            url=meta.get("url", ""),
            summary=meta.get("summary", ""),
            category=meta.get("category", "Other"),
            tags=meta.get("tags", []),
            similarity_score=match.get("score", 0.0),
            source=meta.get("source"),
            author=meta.get("author"),
            created_at=meta.get("created_at")
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id, "title": self.title, "url": self.url,
            "summary": self.summary, "category": self.category, "tags": self.tags,
            "similarity_score": round(self.similarity_score, 4),
            "source": self.source, "author": self.author, "created_at": self.created_at
        }
