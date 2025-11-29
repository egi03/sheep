"""
Comprehensive tests for the RAG package.
Tests all modules: exceptions, logger, models, interests, vector_store, summarizer, rag.
"""

import os
import sys
import time
import logging
from datetime import datetime, timezone
from typing import Dict, Any

# Add parent directory for imports
try:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
except NameError:
    # Running via exec(), add rag directory
    sys.path.insert(0, os.path.join(os.getcwd(), 'rag'))


class TestResults:
    """Track test results."""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.errors = []
    
    def add_pass(self, name: str):
        self.passed += 1
        print(f"  ✅ {name}")
    
    def add_fail(self, name: str, error: str):
        self.failed += 1
        self.errors.append((name, error))
        print(f"  ❌ {name}: {error}")
    
    def add_skip(self, name: str, reason: str):
        self.skipped += 1
        print(f"  ⏭️  {name}: {reason}")
    
    def summary(self):
        total = self.passed + self.failed + self.skipped
        print(f"\n{'='*60}")
        print(f"TEST SUMMARY: {self.passed}/{total} passed, {self.failed} failed, {self.skipped} skipped")
        if self.errors:
            print("\nFailed tests:")
            for name, error in self.errors:
                print(f"  - {name}: {error}")
        print(f"{'='*60}")
        return self.failed == 0


results = TestResults()


# ============================================================
# TEST EXCEPTIONS MODULE
# ============================================================
def test_exceptions():
    print("\n📦 Testing exceptions.py...")
    
    from rag.exceptions import (
        RAGError, ValidationError, APIError, StorageError,
        SummarizationError, SearchError, AnswerGenerationError,
        ConfigurationError, RateLimitError
    )
    
    # Test RAGError
    try:
        e = RAGError("Test error", {"key": "value"})
        assert e.message == "Test error"
        assert e.details == {"key": "value"}
        assert e.to_dict()["error"] == "RAGError"
        results.add_pass("RAGError basic functionality")
    except Exception as ex:
        results.add_fail("RAGError basic functionality", str(ex))
    
    # Test ValidationError
    try:
        e = ValidationError("Invalid input", field="email", value="bad@")
        assert e.field == "email"
        assert e.value == "bad@"
        assert "field" in e.details
        results.add_pass("ValidationError with field/value")
    except Exception as ex:
        results.add_fail("ValidationError with field/value", str(ex))
    
    # Test APIError
    try:
        e = APIError("API failed", service="openai", status_code=500, response="Server error")
        assert e.service == "openai"
        assert e.status_code == 500
        assert e.response == "Server error"
        results.add_pass("APIError with service/status")
    except Exception as ex:
        results.add_fail("APIError with service/status", str(ex))
    
    # Test StorageError
    try:
        e = StorageError("Storage failed", operation="upsert", article_id="123")
        assert e.operation == "upsert"
        assert e.article_id == "123"
        results.add_pass("StorageError with operation/article_id")
    except Exception as ex:
        results.add_fail("StorageError with operation/article_id", str(ex))
    
    # Test SummarizationError
    try:
        e = SummarizationError("Summary failed", title="Test Article", content_length=500)
        assert e.title == "Test Article"
        assert e.content_length == 500
        results.add_pass("SummarizationError with title/length")
    except Exception as ex:
        results.add_fail("SummarizationError with title/length", str(ex))
    
    # Test SearchError
    try:
        e = SearchError("Search failed", query="test query", filters={"category": "AI"})
        assert e.query == "test query"
        assert e.filters == {"category": "AI"}
        results.add_pass("SearchError with query/filters")
    except Exception as ex:
        results.add_fail("SearchError with query/filters", str(ex))
    
    # Test AnswerGenerationError
    try:
        e = AnswerGenerationError("Generation failed", question="What is AI?", sources_count=3)
        assert e.question == "What is AI?"
        assert e.sources_count == 3
        results.add_pass("AnswerGenerationError with question/sources")
    except Exception as ex:
        results.add_fail("AnswerGenerationError with question/sources", str(ex))
    
    # Test ConfigurationError
    try:
        e = ConfigurationError("Missing config", missing_config="API_KEY")
        assert e.missing_config == "API_KEY"
        results.add_pass("ConfigurationError with missing_config")
    except Exception as ex:
        results.add_fail("ConfigurationError with missing_config", str(ex))
    
    # Test RateLimitError
    try:
        e = RateLimitError("Rate limited", service="openai", retry_after=60)
        assert e.retry_after == 60
        assert e.service == "openai"
        assert e.details["retry_after"] == 60
        results.add_pass("RateLimitError with retry_after")
    except Exception as ex:
        results.add_fail("RateLimitError with retry_after", str(ex))
    
    # Test exception inheritance
    try:
        assert issubclass(ValidationError, RAGError)
        assert issubclass(APIError, RAGError)
        assert issubclass(RateLimitError, APIError)
        results.add_pass("Exception inheritance hierarchy")
    except Exception as ex:
        results.add_fail("Exception inheritance hierarchy", str(ex))


# ============================================================
# TEST LOGGER MODULE
# ============================================================
def test_logger():
    print("\n📦 Testing logger.py...")
    
    from rag.logger import get_logger, RAGFormatter, LogTimer
    
    # Test get_logger
    try:
        logger = get_logger("test_logger", level=logging.DEBUG)
        assert logger.name == "test_logger"
        assert logger.level == logging.DEBUG
        assert len(logger.handlers) > 0
        results.add_pass("get_logger creates logger with handler")
    except Exception as ex:
        results.add_fail("get_logger creates logger with handler", str(ex))
    
    # Test RAGFormatter
    try:
        formatter = RAGFormatter(use_colors=True, json_output=False)
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="Test message", args=(), exc_info=None
        )
        formatted = formatter.format(record)
        assert "Test message" in formatted
        assert "INFO" in formatted
        results.add_pass("RAGFormatter formats log records")
    except Exception as ex:
        results.add_fail("RAGFormatter formats log records", str(ex))
    
    # Test RAGFormatter JSON output
    try:
        formatter = RAGFormatter(use_colors=False, json_output=True)
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="Test message", args=(), exc_info=None
        )
        formatted = formatter.format(record)
        import json
        parsed = json.loads(formatted)
        assert parsed["level"] == "INFO"
        assert parsed["message"] == "Test message"
        results.add_pass("RAGFormatter JSON output")
    except Exception as ex:
        results.add_fail("RAGFormatter JSON output", str(ex))
    
    # Test LogTimer
    try:
        logger = get_logger("timer_test", level=logging.DEBUG)
        with LogTimer(logger, "test_operation"):
            time.sleep(0.01)
        results.add_pass("LogTimer context manager")
    except Exception as ex:
        results.add_fail("LogTimer context manager", str(ex))


# ============================================================
# TEST MODELS MODULE
# ============================================================
def test_models():
    print("\n📦 Testing models.py...")
    
    from rag.models import ArticleSummary, ArticleDocument, SearchResult
    
    # Test ArticleSummary
    try:
        summary = ArticleSummary(
            summary="This is a test summary that is long enough.",
            key_points=["Point 1", "Point 2"],
            category="AI/ML",
            tags=["ai", "test"]
        )
        assert summary.summary == "This is a test summary that is long enough."
        assert len(summary.key_points) == 2
        assert summary.category == "AI/ML"
        results.add_pass("ArticleSummary creation and fields")
    except Exception as ex:
        results.add_fail("ArticleSummary creation and fields", str(ex))
    
    # Test ArticleSummary validation
    try:
        # Should fail - summary too short
        from pydantic import ValidationError as PydanticValidationError
        try:
            ArticleSummary(summary="Short")
            results.add_fail("ArticleSummary validation", "Should reject short summary")
        except PydanticValidationError:
            results.add_pass("ArticleSummary validates min_length")
    except Exception as ex:
        results.add_fail("ArticleSummary validates min_length", str(ex))
    
    # Test ArticleDocument
    try:
        doc = ArticleDocument(
            id="test-123",
            title="Test Article",
            url="https://example.com/article",
            summary="A long enough summary for testing purposes.",
            category="Security",
            tags=["security", "test"],
            key_points=["Key point 1"],
            source="HackerNews",
            author="Test Author"
        )
        assert doc.id == "test-123"
        assert doc.title == "Test Article"
        assert doc.category == "Security"
        results.add_pass("ArticleDocument creation")
    except Exception as ex:
        results.add_fail("ArticleDocument creation", str(ex))
    
    # Test ArticleDocument.to_pinecone_metadata
    try:
        doc = ArticleDocument(
            id="test-456",
            title="Metadata Test",
            url="https://example.com",
            summary="Test summary for metadata conversion.",
            tags=["tag1", "tag2"]
        )
        metadata = doc.to_pinecone_metadata()
        assert "title" in metadata
        assert "url" in metadata
        assert "summary" in metadata
        assert "category" in metadata
        assert "tags" in metadata
        assert "created_at" in metadata
        results.add_pass("ArticleDocument.to_pinecone_metadata")
    except Exception as ex:
        results.add_fail("ArticleDocument.to_pinecone_metadata", str(ex))
    
    # Test SearchResult
    try:
        result = SearchResult(
            id="search-1",
            title="Search Result",
            url="https://example.com",
            summary="Search result summary",
            category="Other",
            tags=["test"],
            similarity_score=0.95
        )
        assert result.similarity_score == 0.95
        results.add_pass("SearchResult creation")
    except Exception as ex:
        results.add_fail("SearchResult creation", str(ex))
    
    # Test SearchResult.from_pinecone_match
    try:
        match = {
            "id": "match-1",
            "score": 0.87,
            "metadata": {
                "title": "Matched Article",
                "url": "https://example.com",
                "summary": "Matched summary",
                "category": "AI/ML",
                "tags": ["ai"]
            }
        }
        result = SearchResult.from_pinecone_match(match)
        assert result.id == "match-1"
        assert result.similarity_score == 0.87
        assert result.title == "Matched Article"
        results.add_pass("SearchResult.from_pinecone_match")
    except Exception as ex:
        results.add_fail("SearchResult.from_pinecone_match", str(ex))
    
    # Test SearchResult.to_dict
    try:
        result = SearchResult(
            id="dict-test",
            title="Dict Test",
            url="https://example.com",
            summary="Summary",
            category="Other",
            tags=[],
            similarity_score=0.5678
        )
        d = result.to_dict()
        assert d["id"] == "dict-test"
        assert d["similarity_score"] == 0.5678
        results.add_pass("SearchResult.to_dict")
    except Exception as ex:
        results.add_fail("SearchResult.to_dict", str(ex))


# ============================================================
# TEST INTERESTS MODULE
# ============================================================
def test_interests():
    print("\n📦 Testing interests.py...")
    
    from rag.interests import (
        UserInterest, InterestProfile, TopicMatcher,
        ArticleTopics, TOPIC_TAXONOMY, update_interest_profile
    )
    
    # Test TOPIC_TAXONOMY exists and has structure
    try:
        assert "Security" in TOPIC_TAXONOMY
        assert "AI/ML" in TOPIC_TAXONOMY
        assert "subtopics" in TOPIC_TAXONOMY["Security"]
        assert "keywords" in TOPIC_TAXONOMY["Security"]
        results.add_pass("TOPIC_TAXONOMY structure")
    except Exception as ex:
        results.add_fail("TOPIC_TAXONOMY structure", str(ex))
    
    # Test UserInterest
    try:
        interest = UserInterest(topic="Security", confidence=0.7)
        assert interest.topic == "Security"
        assert interest.confidence == 0.7
        assert interest.query_count == 1
        results.add_pass("UserInterest creation")
    except Exception as ex:
        results.add_fail("UserInterest creation", str(ex))
    
    # Test UserInterest.boost_confidence
    try:
        interest = UserInterest(topic="AI", confidence=0.5)
        interest.boost_confidence(0.2)
        assert interest.confidence == 0.7
        assert interest.query_count == 2
        
        # Test capping at 1.0
        interest.boost_confidence(0.5)
        assert interest.confidence == 1.0
        results.add_pass("UserInterest.boost_confidence")
    except Exception as ex:
        results.add_fail("UserInterest.boost_confidence", str(ex))
    
    # Test UserInterest.decay_confidence
    try:
        interest = UserInterest(topic="Test", confidence=1.0)
        interest.decay_confidence(0.9)
        assert interest.confidence == 0.9
        results.add_pass("UserInterest.decay_confidence")
    except Exception as ex:
        results.add_fail("UserInterest.decay_confidence", str(ex))
    
    # Test InterestProfile
    try:
        profile = InterestProfile()
        profile.interests["Security"] = UserInterest(topic="Security", confidence=0.8)
        profile.interests["AI"] = UserInterest(topic="AI", confidence=0.6)
        assert len(profile.interests) == 2
        results.add_pass("InterestProfile creation and adding interests")
    except Exception as ex:
        results.add_fail("InterestProfile creation and adding interests", str(ex))
    
    # Test InterestProfile.get_top_interests
    try:
        profile = InterestProfile()
        profile.interests["Low"] = UserInterest(topic="Low", confidence=0.3)
        profile.interests["High"] = UserInterest(topic="High", confidence=0.9)
        profile.interests["Mid"] = UserInterest(topic="Mid", confidence=0.6)
        
        top = profile.get_top_interests(2)
        assert len(top) == 2
        assert top[0][0] == "High"
        assert top[0][1] == 0.9
        results.add_pass("InterestProfile.get_top_interests")
    except Exception as ex:
        results.add_fail("InterestProfile.get_top_interests", str(ex))
    
    # Test InterestProfile.to_list
    try:
        profile = InterestProfile()
        profile.interests["High"] = UserInterest(topic="High", confidence=0.9)
        profile.interests["Low"] = UserInterest(topic="Low", confidence=0.2)
        
        interests_list = profile.to_list()
        assert "High" in interests_list
        assert "Low" not in interests_list  # Below 0.3 threshold
        results.add_pass("InterestProfile.to_list filters low confidence")
    except Exception as ex:
        results.add_fail("InterestProfile.to_list filters low confidence", str(ex))
    
    # Test ArticleTopics
    try:
        topics = ArticleTopics(
            primary_topic="Security",
            subtopics=["vulnerabilities", "zero-day"],
            topic_scores={"Security": 0.9, "vulnerabilities": 0.7},
            keywords=["security", "hack"]
        )
        assert topics.primary_topic == "Security"
        assert len(topics.subtopics) == 2
        results.add_pass("ArticleTopics creation")
    except Exception as ex:
        results.add_fail("ArticleTopics creation", str(ex))
    
    # Test TopicMatcher initialization (without embeddings)
    try:
        matcher = TopicMatcher()
        assert matcher.embeddings_model is None
        results.add_pass("TopicMatcher init without embeddings")
    except Exception as ex:
        results.add_fail("TopicMatcher init without embeddings", str(ex))
    
    # Test TopicMatcher._fallback_keyword_match
    try:
        matcher = TopicMatcher()
        scores = matcher._fallback_keyword_match("zero-day vulnerability exploit")
        assert isinstance(scores, dict)
        # Should find Security-related matches
        assert any(score > 0 for score in scores.values())
        results.add_pass("TopicMatcher._fallback_keyword_match")
    except Exception as ex:
        results.add_fail("TopicMatcher._fallback_keyword_match", str(ex))


# ============================================================
# TEST INTERESTS WITH EMBEDDINGS (requires API key)
# ============================================================
def test_interests_with_embeddings():
    print("\n📦 Testing interests.py with embeddings...")
    
    if not os.getenv("OPENAI_API_KEY"):
        results.add_skip("TopicMatcher with embeddings", "OPENAI_API_KEY not set")
        return
    
    from langchain_openai import OpenAIEmbeddings
    from rag.interests import TopicMatcher, InterestProfile, UserInterest, update_interest_profile
    
    # Test TopicMatcher with embeddings
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        matcher = TopicMatcher(embeddings_model=embeddings)
        assert matcher.embeddings_model is not None
        results.add_pass("TopicMatcher init with embeddings")
    except Exception as ex:
        results.add_fail("TopicMatcher init with embeddings", str(ex))
        return  # Can't continue without embeddings
    
    # Test match_text_to_topics
    try:
        scores = matcher.match_text_to_topics("ransomware attack on healthcare", top_k=3)
        assert isinstance(scores, dict)
        assert len(scores) <= 3
        assert all(isinstance(v, float) for v in scores.values())
        results.add_pass("TopicMatcher.match_text_to_topics")
    except Exception as ex:
        results.add_fail("TopicMatcher.match_text_to_topics", str(ex))
    
    # Test extract_article_topics
    try:
        topics = matcher.extract_article_topics(
            title="Critical Zero-Day in Linux Kernel",
            summary="Researchers found a severe vulnerability in Linux that allows RCE.",
            tags=["security", "linux", "cve"]
        )
        assert topics.primary_topic
        assert len(topics.subtopics) > 0
        results.add_pass("TopicMatcher.extract_article_topics")
    except Exception as ex:
        results.add_fail("TopicMatcher.extract_article_topics", str(ex))
    
    # Test compute_interest_match_score
    try:
        from rag.interests import ArticleTopics
        
        profile = InterestProfile()
        profile.interests["Security"] = UserInterest(topic="Security", confidence=0.9)
        profile.interests["zero-day"] = UserInterest(topic="zero-day", confidence=0.8)
        
        article_topics = ArticleTopics(
            primary_topic="Security",
            subtopics=["zero-day", "vulnerabilities"],
            topic_scores={"Security": 0.9},
            keywords=[]
        )
        
        score, matching = matcher.compute_interest_match_score(article_topics, profile)
        assert score > 0
        assert len(matching) > 0
        results.add_pass("TopicMatcher.compute_interest_match_score")
    except Exception as ex:
        results.add_fail("TopicMatcher.compute_interest_match_score", str(ex))
    
    # Test update_interest_profile
    try:
        profile = InterestProfile()
        updated = update_interest_profile(
            profile, 
            "What are the latest ransomware threats?",
            matcher
        )
        assert len(updated.interests) > 0
        assert updated.last_updated is not None
        results.add_pass("update_interest_profile")
    except Exception as ex:
        results.add_fail("update_interest_profile", str(ex))
    
    # Test should_notify_user
    try:
        profile = InterestProfile()
        profile.interests["ransomware"] = UserInterest(topic="ransomware", confidence=0.9)
        
        article_topics = matcher.extract_article_topics(
            title="New Ransomware Strain Hits Hospitals",
            summary="A new ransomware variant is targeting healthcare organizations.",
            tags=["ransomware", "healthcare", "security"]
        )
        
        should_notify, score, matching = matcher.should_notify_user(
            article_topics, profile, threshold=0.3
        )
        assert isinstance(should_notify, bool)
        assert isinstance(score, float)
        results.add_pass("TopicMatcher.should_notify_user")
    except Exception as ex:
        results.add_fail("TopicMatcher.should_notify_user", str(ex))


# ============================================================
# TEST VECTOR STORE (requires API keys)
# ============================================================
def test_vector_store():
    print("\n📦 Testing vector_store.py...")
    
    if not os.getenv("OPENAI_API_KEY") or not os.getenv("PINECONE_API_KEY"):
        results.add_skip("VectorStore tests", "API keys not set")
        return
    
    from rag.vector_store import VectorStore
    from rag.models import ArticleDocument
    
    # Test VectorStore initialization
    try:
        store = VectorStore()
        assert store.index is not None
        results.add_pass("VectorStore initialization")
    except Exception as ex:
        results.add_fail("VectorStore initialization", str(ex))
        return
    
    # Test get_stats
    try:
        stats = store.get_stats()
        assert "total_vectors" in stats
        assert "index_name" in stats
        results.add_pass("VectorStore.get_stats")
    except Exception as ex:
        results.add_fail("VectorStore.get_stats", str(ex))
    
    # Test search
    try:
        search_results = store.search("security vulnerabilities", top_k=3)
        assert isinstance(search_results, list)
        results.add_pass("VectorStore.search")
    except Exception as ex:
        results.add_fail("VectorStore.search", str(ex))
    
    # Test search with category filter
    try:
        search_results = store.search("AI news", top_k=3, category_filter="AI/ML")
        assert isinstance(search_results, list)
        results.add_pass("VectorStore.search with category filter")
    except Exception as ex:
        results.add_fail("VectorStore.search with category filter", str(ex))


# ============================================================
# TEST SUMMARIZER (requires API key)
# ============================================================
def test_summarizer():
    print("\n📦 Testing summarizer.py...")
    
    if not os.getenv("OPENAI_API_KEY"):
        results.add_skip("Summarizer tests", "OPENAI_API_KEY not set")
        return
    
    from rag.summarizer import Summarizer
    from rag.exceptions import SummarizationError
    
    # Test Summarizer initialization
    try:
        summarizer = Summarizer()
        assert summarizer.llm is not None
        assert summarizer.chain is not None
        results.add_pass("Summarizer initialization")
    except Exception as ex:
        results.add_fail("Summarizer initialization", str(ex))
        return
    
    # Test content truncation
    try:
        summarizer = Summarizer(max_content_length=100)
        long_content = "A" * 200
        truncated = summarizer._truncate_content(long_content)
        assert len(truncated) < 200
        assert "[...truncated...]" in truncated
        results.add_pass("Summarizer._truncate_content")
    except Exception as ex:
        results.add_fail("Summarizer._truncate_content", str(ex))
    
    # Test summarization error on short content
    try:
        summarizer = Summarizer()
        try:
            summarizer.summarize("Title", "Short")
            results.add_fail("Summarizer rejects short content", "Should raise error")
        except SummarizationError:
            results.add_pass("Summarizer rejects short content")
    except Exception as ex:
        results.add_fail("Summarizer rejects short content", str(ex))
    
    # Test actual summarization (costs API calls)
    try:
        summarizer = Summarizer()
        content = """
        Artificial intelligence continues to transform how we work and live.
        Machine learning models are becoming more sophisticated and accessible.
        Companies are investing heavily in AI research and development.
        The technology sector sees AI as the next major computing paradigm.
        From healthcare to finance, AI applications are spreading rapidly.
        """
        summary = summarizer.summarize("AI Revolution Article", content)
        assert summary.summary
        assert len(summary.summary) > 10
        assert summary.category in ["AI/ML", "Business", "Programming", "Security", "DevOps", "Other"]
        results.add_pass("Summarizer.summarize produces valid output")
    except Exception as ex:
        results.add_fail("Summarizer.summarize produces valid output", str(ex))


# ============================================================
# TEST RAG MODULE (requires API keys)
# ============================================================
def test_rag():
    print("\n📦 Testing rag.py...")
    
    if not os.getenv("OPENAI_API_KEY") or not os.getenv("PINECONE_API_KEY"):
        results.add_skip("RAG tests", "API keys not set")
        return
    
    from rag import RelevantAI, get_rag
    
    # Test get_rag singleton
    try:
        rag1 = get_rag()
        rag2 = get_rag()
        assert rag1 is rag2  # Same instance
        results.add_pass("get_rag returns singleton")
    except Exception as ex:
        results.add_fail("get_rag returns singleton", str(ex))
        return
    
    rag = get_rag()
    
    # Test expand_query
    try:
        expanded = rag.expand_query("security news")
        assert isinstance(expanded, str)
        assert len(expanded) > 0
        results.add_pass("RelevantAI.expand_query")
    except Exception as ex:
        results.add_fail("RelevantAI.expand_query", str(ex))
    
    # Test extract_user_interests
    try:
        interests = rag.extract_user_interests([
            "Tell me about zero-day vulnerabilities",
            "Latest ransomware attacks"
        ])
        assert isinstance(interests, list)
        results.add_pass("RelevantAI.extract_user_interests")
    except Exception as ex:
        results.add_fail("RelevantAI.extract_user_interests", str(ex))
    
    # Test search_enhanced
    try:
        results_search = rag.search_enhanced("security vulnerabilities", top_k=3)
        assert isinstance(results_search, list)
        if results_search:
            assert "title" in results_search[0]
            assert "relevance_score" in results_search[0]
        results.add_pass("RelevantAI.search_enhanced")
    except Exception as ex:
        results.add_fail("RelevantAI.search_enhanced", str(ex))
    
    # Test search_enhanced with interests
    try:
        results_search = rag.search_enhanced(
            "latest news",
            top_k=3,
            user_interests=["Security", "zero-day"]
        )
        assert isinstance(results_search, list)
        if results_search:
            # Check personalization fields exist
            assert "personalized" in results_search[0]
            assert "matching_interests" in results_search[0]
        results.add_pass("RelevantAI.search_enhanced with interests")
    except Exception as ex:
        results.add_fail("RelevantAI.search_enhanced with interests", str(ex))
    
    # Test search_enhanced with interest_profile dict
    try:
        results_search = rag.search_enhanced(
            "latest news",
            top_k=3,
            interest_profile={"Security": 0.9, "ransomware": 0.7}
        )
        assert isinstance(results_search, list)
        results.add_pass("RelevantAI.search_enhanced with interest_profile dict")
    except Exception as ex:
        results.add_fail("RelevantAI.search_enhanced with interest_profile dict", str(ex))
    
    # Test ask
    try:
        answer = rag.ask("What are the latest security threats?", top_k=3)
        assert "answer" in answer
        assert "confidence" in answer
        assert "sources" in answer
        results.add_pass("RelevantAI.ask")
    except Exception as ex:
        results.add_fail("RelevantAI.ask", str(ex))
    
    # Test ask with personalization
    try:
        answer = rag.ask(
            "What's new in tech?",
            top_k=3,
            user_interests=["Security", "AI/ML"]
        )
        assert "answer" in answer
        results.add_pass("RelevantAI.ask with user_interests")
    except Exception as ex:
        results.add_fail("RelevantAI.ask with user_interests", str(ex))
    
    # Test compute_article_match
    try:
        match_result = rag.compute_article_match(
            article_data={
                "title": "New Ransomware Variant Discovered",
                "summary": "Security researchers found a new ransomware strain.",
                "tags": ["security", "ransomware"]
            },
            interest_profile_dict={"ransomware": 0.9, "Security": 0.8}
        )
        assert "should_notify" in match_result
        assert "score" in match_result
        results.add_pass("RelevantAI.compute_article_match")
    except Exception as ex:
        results.add_fail("RelevantAI.compute_article_match", str(ex))
    
    # Test ask with empty question
    try:
        from rag.exceptions import ValidationError
        try:
            rag.ask("")
            results.add_fail("RelevantAI.ask validates empty question", "Should raise error")
        except ValidationError:
            results.add_pass("RelevantAI.ask validates empty question")
    except Exception as ex:
        results.add_fail("RelevantAI.ask validates empty question", str(ex))


# ============================================================
# RUN ALL TESTS
# ============================================================
def run_all_tests():
    print("=" * 60)
    print("RAG PACKAGE COMPREHENSIVE TEST SUITE")
    print("=" * 60)
    
    # Set up Django environment if available
    try:
        import django
        os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'relevantai.settings')
        django.setup()
        print("✓ Django environment loaded")
    except Exception as e:
        print(f"⚠ Django not available: {e}")
    
    # Run tests in order
    test_exceptions()
    test_logger()
    test_models()
    test_interests()
    test_interests_with_embeddings()
    test_vector_store()
    test_summarizer()
    test_rag()
    
    # Print summary
    return results.summary()


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
