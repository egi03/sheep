"""Pinecone vector store for article embeddings."""

import os
import time
from typing import List, Optional, Dict, Any

from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings

from .models import ArticleDocument, SearchResult
from .exceptions import StorageError, SearchError, ConfigurationError, RateLimitError
from .logger import get_logger, LogTimer

logger = get_logger("relevantai.vector_store")


class VectorStore:
    def __init__(self, index_name: Optional[str] = None, pinecone_api_key: Optional[str] = None,
                 openai_api_key: Optional[str] = None, dimension: int = 1536):
        self.index_name = index_name or os.getenv("PINECONE_INDEX_NAME", "hn-articles")
        pinecone_key = pinecone_api_key or os.getenv("PINECONE_API_KEY")
        openai_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        
        if not pinecone_key:
            raise ConfigurationError("PINECONE_API_KEY required", missing_config="PINECONE_API_KEY")
        if not openai_key:
            raise ConfigurationError("OPENAI_API_KEY required", missing_config="OPENAI_API_KEY")
        
        self.pc = Pinecone(api_key=pinecone_key)
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=openai_key)
        self.dimension = dimension
        self._ensure_index()

    def _ensure_index(self):
        try:
            if self.index_name not in self.pc.list_indexes().names():
                logger.info(f"Creating index: {self.index_name}")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1")
                )
                time.sleep(5)
            self.index = self.pc.Index(self.index_name)
        except Exception as e:
            raise StorageError(f"Failed to initialize index: {e}", operation="ensure_index")

    def _embed_with_retry(self, text: str, max_retries: int = 3) -> List[float]:
        for attempt in range(max_retries):
            try:
                return self.embeddings.embed_query(text)
            except Exception as e:
                if "rate" in str(e).lower() and attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise

    def add_article(self, article: ArticleDocument) -> bool:
        try:
            with LogTimer(logger, f"embed_article_{article.id}"):
                text = f"{article.title}\n{article.summary}\n{' '.join(article.tags)}"
                vector = self._embed_with_retry(text)
            
            self.index.upsert(vectors=[{
                "id": article.id, "values": vector, "metadata": article.to_pinecone_metadata()
            }])
            return True
        except RateLimitError:
            raise
        except Exception as e:
            raise StorageError(f"Failed to add article: {e}", operation="add", article_id=article.id)

    def add_articles(self, articles: List[ArticleDocument], batch_size: int = 50) -> int:
        added = 0
        for i in range(0, len(articles), batch_size):
            batch = articles[i:i + batch_size]
            vectors = []
            for article in batch:
                try:
                    text = f"{article.title}\n{article.summary}\n{' '.join(article.tags)}"
                    vector = self._embed_with_retry(text)
                    vectors.append({"id": article.id, "values": vector, "metadata": article.to_pinecone_metadata()})
                except Exception as e:
                    logger.warning(f"Failed to embed {article.id}: {e}")
            
            if vectors:
                try:
                    self.index.upsert(vectors=vectors)
                    added += len(vectors)
                except Exception as e:
                    logger.error(f"Batch upsert failed: {e}")
        return added

    def search(self, query: str, top_k: int = 5, category_filter: Optional[str] = None) -> List[SearchResult]:
        try:
            query_vector = self._embed_with_retry(query)
            filter_dict = {"category": category_filter} if category_filter else None
            
            results = self.index.query(
                vector=query_vector, top_k=top_k, include_metadata=True, filter=filter_dict
            )
            return [SearchResult.from_pinecone_match(m) for m in results.get("matches", [])]
        except RateLimitError:
            raise
        except Exception as e:
            raise SearchError(f"Search failed: {e}", query=query, filters={"category": category_filter})

    def delete_article(self, article_id: str) -> bool:
        try:
            self.index.delete(ids=[article_id])
            return True
        except Exception as e:
            raise StorageError(f"Failed to delete: {e}", operation="delete", article_id=article_id)

    def delete_all(self) -> bool:
        try:
            self.index.delete(delete_all=True)
            return True
        except Exception as e:
            raise StorageError(f"Failed to delete all: {e}", operation="delete_all")

    def get_stats(self) -> Dict[str, Any]:
        try:
            stats = self.index.describe_index_stats()
            return {"total_vectors": stats.get("total_vector_count", 0), "index_name": self.index_name,
                   "dimension": self.dimension, "namespaces": stats.get("namespaces", {})}
        except Exception as e:
            raise StorageError(f"Failed to get stats: {e}", operation="get_stats")
