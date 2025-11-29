"""Main RAG interface for Django backend integration."""

import os
import time
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from .models import ArticleDocument, ArticleSummary, SearchResult
from .summarizer import Summarizer
from .vector_store import VectorStore
from .exceptions import (
    RAGError, ValidationError, APIError, StorageError, SearchError,
    SummarizationError, AnswerGenerationError, ConfigurationError, RateLimitError
)
from .logger import get_logger, LogTimer

logger = get_logger("relevantai")


class ExpandedQuery(BaseModel):
    expanded_query: str
    search_terms: List[str]


class GeneratedAnswer(BaseModel):
    answer: str
    confidence: str
    key_insights: List[str]


class ExtractedInterests(BaseModel):
    interests: List[str]


class RelevantAI:
    """Main RAG interface for article ingestion, search, and Q&A."""
    
    QUERY_EXPANSION_PROMPT = """Expand this query with synonyms and related terms for better search:
    
Query: {query}

Provide an expanded search query and key terms."""

    ANSWER_GENERATION_PROMPT = """Answer based on these articles. Cite sources. Say if info is insufficient.

ARTICLES:
{context}

QUESTION: {question}"""

    INTEREST_EXTRACTION_PROMPT = """Analyze these user queries and extract up to 5 technical areas of interest.
Focus on specific topics like: AI, Machine Learning, Cybersecurity, Ransomware, Python, JavaScript, 
Cloud Computing, DevOps, Zero-day vulnerabilities, Data Privacy, etc.

USER QUERIES:
{queries}

Return ONLY a comma-separated list of interests (e.g., "Ransomware, Python, Zero-day")."""

    def __init__(self, pinecone_api_key: Optional[str] = None, openai_api_key: Optional[str] = None,
                 index_name: Optional[str] = None):
        logger.info("Initializing RelevantAI...")
        
        if pinecone_api_key:
            os.environ["PINECONE_API_KEY"] = pinecone_api_key
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
        if index_name:
            os.environ["PINECONE_INDEX_NAME"] = index_name
        
        if not os.getenv("OPENAI_API_KEY"):
            raise ConfigurationError("OPENAI_API_KEY required", missing_config="OPENAI_API_KEY")
        if not os.getenv("PINECONE_API_KEY"):
            raise ConfigurationError("PINECONE_API_KEY required", missing_config="PINECONE_API_KEY")
        
        try:
            self.summarizer = Summarizer()
            self.vector_store = VectorStore()
            
            self.llm = ChatOpenAI(model="gpt-4o", temperature=0.3, api_key=os.getenv("OPENAI_API_KEY"))
            self.llm_fast = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, api_key=os.getenv("OPENAI_API_KEY"))
            
            self.query_expander = ChatPromptTemplate.from_messages([
                ("system", "You are a search query optimizer."),
                ("human", self.QUERY_EXPANSION_PROMPT)
            ]) | self.llm_fast.with_structured_output(ExpandedQuery)
            
            self.answer_generator = ChatPromptTemplate.from_messages([
                ("system", "You are an expert analyst."),
                ("human", self.ANSWER_GENERATION_PROMPT)
            ]) | self.llm.with_structured_output(GeneratedAnswer)
            
            self.interest_extractor = ChatPromptTemplate.from_messages([
                ("system", "You are a user interest analyzer. Extract technical topics from queries."),
                ("human", self.INTEREST_EXTRACTION_PROMPT)
            ]) | self.llm_fast.with_structured_output(ExtractedInterests)
            
            from .interests import TopicMatcher
            self.topic_matcher = TopicMatcher(self.vector_store.embeddings)
            
            logger.info("RelevantAI initialized")
        except ConfigurationError:
            raise
        except Exception as e:
            raise ConfigurationError(f"Failed to initialize: {e}")

    def add_article(self, article_id: str, title: str, url: str, content: str,
                   source: str = "", author: str = "") -> ArticleDocument:
        if not article_id or not article_id.strip():
            raise ValidationError("article_id cannot be empty", field="article_id")
        if not title or not title.strip():
            raise ValidationError("title cannot be empty", field="title")
        if not url or not url.strip():
            raise ValidationError("url cannot be empty", field="url")
        if not content or len(content.strip()) < 50:
            raise ValidationError("content must be at least 50 characters", field="content")
        
        logger.info(f"Adding article: {article_id}")
        
        try:
            with LogTimer(logger, "add_article_full"):
                summary = self.summarizer.summarize(title=title, content=content)
                doc = ArticleDocument(
                    id=article_id, title=title, url=url, summary=summary.summary,
                    category=summary.category, tags=summary.tags, key_points=summary.key_points,
                    source=source, author=author
                )
                self.vector_store.add_article(doc)
            logger.info(f"Article added: {article_id}")
            return doc
        except (ValidationError, SummarizationError, StorageError, RateLimitError):
            raise
        except Exception as e:
            raise StorageError(f"Failed to add article: {e}", operation="add_article", article_id=article_id)

    def add_article_with_summary(self, article_id: str, title: str, url: str, summary: str,
                                  category: str = "Other", tags: Optional[List[str]] = None,
                                  source: str = "", author: str = "") -> ArticleDocument:
        doc = ArticleDocument(
            id=article_id, title=title, url=url, summary=summary,
            category=category, tags=tags or [], source=source, author=author
        )
        self.vector_store.add_article(doc)
        return doc

    def add_articles_batch(self, articles: List[Dict[str, Any]]) -> int:
        docs = []
        for article in articles:
            try:
                summary = self.summarizer.summarize(title=article["title"], content=article["content"])
                docs.append(ArticleDocument(
                    id=article["id"], title=article["title"], url=article["url"],
                    summary=summary.summary, category=summary.category, tags=summary.tags,
                    key_points=summary.key_points, source=article.get("source", ""),
                    author=article.get("author", "")
                ))
            except Exception as e:
                logger.warning(f"Failed to process article {article.get('id')}: {e}")
        return self.vector_store.add_articles(docs) if docs else 0

    def search(self, query: str, top_k: int = 5, category: Optional[str] = None) -> List[Dict[str, Any]]:
        top_k = min(max(1, top_k), 20)
        results = self.vector_store.search(query=query, top_k=top_k, category_filter=category)
        return [r.to_dict() for r in results]

    def search_results(self, query: str, top_k: int = 5, category: Optional[str] = None) -> List[SearchResult]:
        return self.vector_store.search(query=query, top_k=min(max(1, top_k), 20), category_filter=category)

    def delete_article(self, article_id: str) -> bool:
        return self.vector_store.delete_article(article_id)

    def delete_all(self) -> bool:
        return self.vector_store.delete_all()

    def get_stats(self) -> Dict[str, Any]:
        return self.vector_store.get_stats()

    def summarize_only(self, title: str, content: str) -> ArticleSummary:
        return self.summarizer.summarize(title=title, content=content)

    def expand_query(self, query: str) -> str:
        try:
            result: ExpandedQuery = self.query_expander.invoke({"query": query})
            return result.expanded_query
        except Exception as e:
            logger.warning(f"Query expansion failed: {e}")
            return query

    def extract_user_interests(self, chat_history: List[str]) -> List[str]:
        """Extract user interests from chat history using semantic matching."""
        if not chat_history:
            return []
        
        try:
            from .interests import InterestProfile, update_interest_profile
            
            profile = InterestProfile()
            for query in chat_history:
                profile = update_interest_profile(profile, query, self.topic_matcher)
            
            interests = profile.to_list()
            logger.info(f"Extracted interests (semantic): {interests}")
            return interests
        except Exception as e:
            logger.warning(f"Semantic interest extraction failed, falling back to LLM: {e}")
            try:
                queries_text = "\n".join(f"- {q}" for q in chat_history)
                result: ExtractedInterests = self.interest_extractor.invoke({"queries": queries_text})
                interests = [i.strip() for i in result.interests if i.strip()]
                return interests[:5]
            except Exception as e2:
                logger.warning(f"LLM interest extraction also failed: {e2}")
                return []

    def update_interest_profile(self, profile_dict: Dict, query: str) -> Dict:
        """Update an interest profile with a new query. Returns updated profile dict."""
        from .interests import InterestProfile, update_interest_profile
        
        profile = InterestProfile(**profile_dict) if profile_dict else InterestProfile()
        updated = update_interest_profile(profile, query, self.topic_matcher)
        return updated.model_dump()

    def compute_article_match(self, article_data: Dict, interest_profile_dict: Dict) -> Dict[str, Any]:
        """
        Compute match score between an article and user interest profile.
        Returns dict with score, matching_topics, and should_notify.
        """
        from .interests import InterestProfile, ArticleTopics
        
        profile = InterestProfile(**interest_profile_dict) if interest_profile_dict else InterestProfile()
        
        article_topics = self.topic_matcher.extract_article_topics(
            title=article_data.get('title', ''),
            summary=article_data.get('summary', ''),
            tags=article_data.get('tags', [])
        )
        
        should_notify, score, matching = self.topic_matcher.should_notify_user(
            article_topics, profile, threshold=0.5
        )
        
        return {
            'match_score': score,
            'matching_topics': matching,
            'should_notify': should_notify,
            'article_topics': article_topics.model_dump()
        }

    def search_enhanced(self, query: str, top_k: int = 5, category: Optional[str] = None,
                        use_query_expansion: bool = True, 
                        user_interests: Optional[List[str]] = None,
                        interest_profile: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Enhanced search with semantic interest matching.
        
        Args:
            query: Search query
            top_k: Number of results
            category: Category filter
            use_query_expansion: Whether to expand the query
            user_interests: Simple list of interests (backward compatible)
            interest_profile: Full interest profile dict (preferred)
        """
        search_query = self.expand_query(query) if use_query_expansion else query
        top_k = min(max(1, top_k), 20)
        results = self.vector_store.search(query=search_query, top_k=top_k * 2, category_filter=category)
        
        from .interests import InterestProfile, UserInterest
        
        profile = None
        if interest_profile:
            # Convert dict format {"topic": confidence} to InterestProfile
            interests_dict = {}
            for topic, conf in interest_profile.items():
                if isinstance(conf, (int, float)):
                    interests_dict[topic] = UserInterest(topic=topic, confidence=float(conf))
                elif isinstance(conf, dict):
                    interests_dict[topic] = UserInterest(**conf)
            profile = InterestProfile(interests=interests_dict)
        elif user_interests:
            # Simple list of interests
            interests_dict = {}
            for interest in user_interests:
                interests_dict[interest] = UserInterest(topic=interest, confidence=0.5)
            profile = InterestProfile(interests=interests_dict)
        
        scored_results = []
        query_terms = query.lower().split()
        
        for r in results:
            base_score = r.similarity_score
            title_lower = r.title.lower()
            matching = sum(1 for t in query_terms if t in title_lower)
            title_boost = 0.1 * (matching / len(query_terms)) if matching else 0.0
            category_boost = 0.05 if category and r.category == category else 0.0
            
            interest_boost = 0.0
            matching_interests = []
            
            if profile and profile.interests:
                article_topics = self.topic_matcher.extract_article_topics(
                    title=r.title,
                    summary=r.summary,
                    tags=r.tags or []
                )
                interest_boost, matching_interests = self.topic_matcher.compute_interest_match_score(
                    article_topics, profile
                )
                interest_boost = min(0.25, interest_boost)
            
            result_dict = r.to_dict()
            total_boost = base_score + title_boost + category_boost + interest_boost
            result_dict["relevance_score"] = round(min(1.0, total_boost), 4)
            result_dict["original_similarity"] = round(base_score, 4)
            result_dict["personalized"] = interest_boost > 0
            result_dict["matching_interests"] = matching_interests
            scored_results.append(result_dict)
        
        scored_results.sort(key=lambda x: x["relevance_score"], reverse=True)
        return scored_results[:top_k]

    def _format_context(self, articles: List[Dict[str, Any]]) -> str:
        parts = []
        for i, a in enumerate(articles, 1):
            parts.append(f"ARTICLE {i}: {a['title']}\nURL: {a['url']}\nSummary: {a['summary']}\n---")
        return "\n".join(parts)

    def ask(self, question: str, top_k: int = 5, category: Optional[str] = None,
            user_interests: Optional[List[str]] = None) -> Dict[str, Any]:
        if not question or not question.strip():
            raise ValidationError("Question cannot be empty", field="question")
        
        logger.info(f"Processing question: {question[:50]}...")
        expanded_query = ""
        
        try:
            with LogTimer(logger, "ask"):
                try:
                    expanded_query = self.expand_query(question)
                except:
                    expanded_query = question
                
                try:
                    articles = self.search_enhanced(
                        query=question, top_k=top_k, category=category,
                        user_interests=user_interests
                    )
                except SearchError as e:
                    return {"answer": "Search unavailable. Try again.", "confidence": "low",
                           "key_insights": [], "sources": [], "expanded_query": expanded_query, "error": str(e)}
                
                if not articles:
                    return {"answer": "No relevant articles found. Try rephrasing.", "confidence": "low",
                           "key_insights": [], "sources": [], "expanded_query": expanded_query}
                
                try:
                    result: GeneratedAnswer = self.answer_generator.invoke({
                        "context": self._format_context(articles), "question": question
                    })
                    logger.info(f"Answer generated with {len(articles)} sources")
                    return {
                        "answer": result.answer, "confidence": result.confidence,
                        "key_insights": result.key_insights,
                        "sources": [{"title": a["title"], "url": a["url"], "summary": a["summary"],
                                    "relevance_score": a["relevance_score"]} for a in articles],
                        "expanded_query": expanded_query
                    }
                except Exception as e:
                    if "rate" in str(e).lower():
                        return {"answer": "High demand. Try again shortly.", "confidence": "low",
                               "key_insights": [], "sources": [{"title": a["title"], "url": a["url"]} for a in articles],
                               "expanded_query": expanded_query, "error": "rate_limit"}
                    return {"answer": "Couldn't generate answer. See sources:", "confidence": "low",
                           "key_insights": [], "sources": [{"title": a["title"], "url": a["url"]} for a in articles],
                           "expanded_query": expanded_query, "error": str(e)}
        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Error in ask(): {e}")
            return {"answer": "Unexpected error. Try again.", "confidence": "low",
                   "key_insights": [], "sources": [], "expanded_query": expanded_query, "error": str(e)}

    def chat(self, question: str, top_k: int = 5) -> str:
        return self.ask(question, top_k=top_k)["answer"]


_instance: Optional[RelevantAI] = None

def get_rag() -> RelevantAI:
    global _instance
    if _instance is None:
        _instance = RelevantAI()
    return _instance
