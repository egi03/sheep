"""
Topic and Interest Matching System for RelevantAI.

This module provides semantic matching between user interests and article topics
using embeddings for accurate similarity scoring.

Key Design Decisions:
- Uses ONLY fixed top-level categories for scoring (not unbounded topics)
- Articles get classified with scores for each category at indexing time
- User interests are tracked per-category with confidence scores
- Simple, predictable personalization based on category overlap
"""

from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field
import numpy as np


# Fixed categories - these are the ONLY topics we track for personalization
# Each category has semantic descriptors for better embedding matching
FIXED_CATEGORIES = {
    "Security": {
        "description": "Cybersecurity, hacking, vulnerabilities, ransomware, malware, data breaches, penetration testing, encryption, authentication, threat intelligence, zero-day exploits, CVE patches",
        "keywords": ["security", "hack", "attack", "exploit", "CVE", "patch", "cyber", "ransomware", 
                     "malware", "phishing", "vulnerability", "breach", "encryption", "authentication",
                     "penetration", "threat", "zero-day", "firewall", "intrusion"]
    },
    "AI/ML": {
        "description": "Artificial intelligence, machine learning, deep learning, neural networks, NLP, LLMs, GPT, transformers, generative AI, computer vision, MLOps, AI ethics and safety",
        "keywords": ["ai", "ml", "machine learning", "deep learning", "neural", "gpt", "llm", 
                     "transformer", "model", "training", "inference", "nlp", "computer vision",
                     "generative", "chatgpt", "openai", "anthropic", "claude"]
    },
    "Programming": {
        "description": "Software development, programming languages, Python, JavaScript, TypeScript, Rust, Go, web development, APIs, frameworks, testing, code quality, design patterns",
        "keywords": ["code", "programming", "developer", "software", "framework", "library",
                     "python", "javascript", "typescript", "rust", "go", "java", "api",
                     "testing", "refactor", "frontend", "backend", "web development"]
    },
    "DevOps": {
        "description": "DevOps practices, Kubernetes, Docker, CI/CD pipelines, infrastructure as code, monitoring, observability, GitOps, Terraform, Ansible, SRE, deployment, scaling",
        "keywords": ["devops", "kubernetes", "docker", "container", "ci/cd", "pipeline",
                     "infrastructure", "terraform", "ansible", "monitoring", "observability",
                     "deployment", "scaling", "sre", "gitops", "helm"]
    },
    "Cloud": {
        "description": "Cloud computing, AWS, Azure, GCP, serverless, Lambda, cloud native, multi-cloud, cloud migration, cloud architecture, PaaS, SaaS, IaaS",
        "keywords": ["cloud", "aws", "azure", "gcp", "serverless", "lambda", "ec2", "s3",
                     "cloud native", "saas", "paas", "iaas", "migration", "multi-cloud"]
    },
    "Data": {
        "description": "Data engineering, data science, analytics, big data, ETL pipelines, data warehouses, data lakes, SQL, NoSQL, streaming, real-time analytics, databases",
        "keywords": ["data", "database", "analytics", "warehouse", "pipeline", "etl",
                     "sql", "nosql", "streaming", "bigquery", "snowflake", "spark",
                     "kafka", "redis", "postgresql", "mongodb"]
    },
    "Privacy": {
        "description": "Data privacy, GDPR, CCPA, compliance, PII protection, anonymization, consent management, data protection regulations, privacy engineering",
        "keywords": ["privacy", "gdpr", "ccpa", "compliance", "pii", "consent", 
                     "anonymization", "data protection", "regulation", "personal data"]
    }
}

# List of category names for easy iteration
CATEGORY_NAMES = list(FIXED_CATEGORIES.keys())


# Legacy taxonomy for backward compatibility (deprecated - use FIXED_CATEGORIES)
TOPIC_TAXONOMY = {
    cat: {
        "subtopics": [],  # Not used in new system
        "keywords": data["keywords"]
    }
    for cat, data in FIXED_CATEGORIES.items()
}


class UserInterest(BaseModel):
    """Represents a user's interest with confidence score."""
    topic: str
    subtopics: List[str] = Field(default_factory=list)  # Deprecated - kept for compatibility
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    query_count: int = Field(default=1)
    last_seen: Optional[str] = None
    
    def boost_confidence(self, amount: float = 0.1) -> None:
        """Increase confidence, capped at 1.0."""
        self.confidence = min(1.0, self.confidence + amount)
        self.query_count += 1
    
    def decay_confidence(self, factor: float = 0.95) -> None:
        """Decay confidence over time."""
        self.confidence *= factor


class CategoryScores(BaseModel):
    """
    Scores for each fixed category (0.0 to 1.0).
    Used for both articles and user interests.
    """
    Security: float = Field(default=0.0, ge=0.0, le=1.0)
    AI_ML: float = Field(default=0.0, ge=0.0, le=1.0, alias="AI/ML")
    Programming: float = Field(default=0.0, ge=0.0, le=1.0)
    DevOps: float = Field(default=0.0, ge=0.0, le=1.0)
    Cloud: float = Field(default=0.0, ge=0.0, le=1.0)
    Data: float = Field(default=0.0, ge=0.0, le=1.0)
    Privacy: float = Field(default=0.0, ge=0.0, le=1.0)
    
    class Config:
        populate_by_name = True
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dict with proper category names."""
        return {
            "Security": self.Security,
            "AI/ML": self.AI_ML,
            "Programming": self.Programming,
            "DevOps": self.DevOps,
            "Cloud": self.Cloud,
            "Data": self.Data,
            "Privacy": self.Privacy,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> "CategoryScores":
        """Create from dict with category names."""
        return cls(
            Security=d.get("Security", 0.0),
            AI_ML=d.get("AI/ML", 0.0),
            Programming=d.get("Programming", 0.0),
            DevOps=d.get("DevOps", 0.0),
            Cloud=d.get("Cloud", 0.0),
            Data=d.get("Data", 0.0),
            Privacy=d.get("Privacy", 0.0),
        )
    
    def get_top_categories(self, n: int = 3, min_score: float = 0.1) -> List[Tuple[str, float]]:
        """Get top N categories above min_score."""
        scores = [(k, v) for k, v in self.to_dict().items() if v >= min_score]
        return sorted(scores, key=lambda x: x[1], reverse=True)[:n]
    
    def primary_category(self) -> Optional[str]:
        """Get the primary (highest scoring) category."""
        top = self.get_top_categories(n=1, min_score=0.0)
        return top[0][0] if top and top[0][1] > 0 else None


class ArticleTopics(BaseModel):
    """Represents extracted topics from an article."""
    primary_topic: str
    subtopics: List[str] = Field(default_factory=list)  # Deprecated
    topic_scores: Dict[str, float] = Field(default_factory=dict)  # Legacy format
    category_scores: Optional[CategoryScores] = None  # New: scores per fixed category
    keywords: List[str] = Field(default_factory=list)
    
    def get_category_scores(self) -> CategoryScores:
        """Get category scores, computing from topic_scores if needed."""
        if self.category_scores:
            return self.category_scores
        # Convert legacy topic_scores to CategoryScores
        return CategoryScores.from_dict(self.topic_scores)


class InterestProfile(BaseModel):
    """
    Complete interest profile for a user.
    Uses fixed categories for consistent scoring.
    """
    # New: Fixed category scores
    category_scores: CategoryScores = Field(default_factory=CategoryScores)
    
    # Legacy: Dict of interest name -> UserInterest (deprecated, kept for migration)
    interests: Dict[str, UserInterest] = Field(default_factory=dict)
    topic_embeddings: Dict[str, List[float]] = Field(default_factory=dict)
    last_updated: Optional[str] = None
    
    def get_top_interests(self, n: int = 5) -> List[Tuple[str, float]]:
        """Get top N interests sorted by confidence (uses fixed categories)."""
        # Prefer new category_scores
        if self.category_scores:
            return self.category_scores.get_top_categories(n=n, min_score=0.1)
        
        # Fallback to legacy interests dict
        sorted_interests = sorted(
            [(k, v.confidence if isinstance(v, UserInterest) else float(v)) 
             for k, v in self.interests.items()],
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_interests[:n]
    
    def to_list(self) -> List[str]:
        """Convert to simple list of interest names for backward compatibility."""
        return [k for k, confidence in self.get_top_interests(10) if confidence >= 0.3]
    
    def to_simple_dict(self) -> Dict[str, float]:
        """Convert to simple dict of category -> score for storage."""
        return self.category_scores.to_dict()
    
    @classmethod
    def from_simple_dict(cls, d: Dict[str, float]) -> "InterestProfile":
        """Create from simple dict of category -> score."""
        from datetime import datetime, timezone
        return cls(
            category_scores=CategoryScores.from_dict(d),
            last_updated=datetime.now(timezone.utc).isoformat()
        )


class TopicMatcher:
    """
    Semantic topic matching using embeddings.
    
    NEW APPROACH: Only matches against fixed categories, not arbitrary topics.
    This ensures consistent, predictable personalization.
    """
    
    def __init__(self, embeddings_model=None):
        self.embeddings_model = embeddings_model
        self._category_embeddings_cache: Dict[str, List[float]] = {}
        # Legacy cache for backward compatibility
        self._topic_embeddings_cache: Dict[str, List[float]] = {}
    
    def set_embeddings_model(self, model) -> None:
        """Set the embeddings model (OpenAIEmbeddings instance)."""
        self.embeddings_model = model
        # Clear caches when model changes
        self._category_embeddings_cache.clear()
        self._topic_embeddings_cache.clear()
    
    def _get_all_topics(self) -> List[str]:
        """Get flat list of all topics and subtopics. (Legacy - deprecated)"""
        return list(FIXED_CATEGORIES.keys())
    
    def _compute_embedding(self, text: str) -> List[float]:
        """Compute embedding for text."""
        if not self.embeddings_model:
            raise ValueError("Embeddings model not set")
        return self.embeddings_model.embed_query(text)
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        a = np.array(vec1)
        b = np.array(vec2)
        norm_product = np.linalg.norm(a) * np.linalg.norm(b)
        if norm_product == 0:
            return 0.0
        return float(np.dot(a, b) / norm_product)
    
    def _cache_category_embeddings(self) -> None:
        """Pre-compute embeddings for all fixed categories using their descriptions."""
        if self._category_embeddings_cache:
            return
        
        for category, data in FIXED_CATEGORIES.items():
            # Use the rich description for better semantic matching
            category_text = f"{category}: {data['description']}"
            self._category_embeddings_cache[category] = self._compute_embedding(category_text)
    
    def _cache_topic_embeddings(self) -> None:
        """Legacy: Pre-compute embeddings for taxonomy topics."""
        if self._topic_embeddings_cache:
            return
        # Just cache the category names for backward compatibility
        for category in FIXED_CATEGORIES.keys():
            self._topic_embeddings_cache[category.lower()] = self._compute_embedding(category)
    
    def classify_text(self, text: str) -> CategoryScores:
        """
        Classify text into fixed categories with scores.
        Returns scores for ALL categories (not just matches).
        
        This is the PRIMARY method for topic extraction.
        """
        if not self.embeddings_model:
            return self._fallback_keyword_classify(text)
        
        try:
            self._cache_category_embeddings()
            text_embedding = self._compute_embedding(text)
            
            scores = {}
            for category, category_emb in self._category_embeddings_cache.items():
                similarity = self._cosine_similarity(text_embedding, category_emb)
                # Normalize: embeddings give ~0.7-0.9 for good matches, ~0.5-0.7 for weak
                # Map to 0-1 range where 0.7 sim -> 0.5 score, 0.9 sim -> 1.0 score
                normalized = max(0.0, min(1.0, (similarity - 0.5) * 2.5))
                scores[category] = normalized
            
            return CategoryScores.from_dict(scores)
        except Exception:
            return self._fallback_keyword_classify(text)
    
    def _fallback_keyword_classify(self, text: str) -> CategoryScores:
        """Fallback to keyword matching when embeddings unavailable."""
        text_lower = text.lower()
        scores = {}
        
        for category, data in FIXED_CATEGORIES.items():
            score = 0.0
            keywords = data["keywords"]
            
            # Count keyword matches
            matches = sum(1 for kw in keywords if kw.lower() in text_lower)
            if matches > 0:
                # Normalize: more matches = higher score, max out at 5+ matches
                score = min(1.0, matches * 0.2)
            
            scores[category] = score
        
        return CategoryScores.from_dict(scores)
    
    def match_text_to_topics(self, text: str, top_k: int = 3) -> Dict[str, float]:
        """
        Legacy method: Match text to taxonomy topics using semantic similarity.
        Returns dict of topic -> similarity score.
        
        DEPRECATED: Use classify_text() instead for fixed category scores.
        """
        category_scores = self.classify_text(text)
        # Return top categories as dict
        top = category_scores.get_top_categories(n=top_k, min_score=0.0)
        return dict(top)
    
    def _fallback_keyword_match(self, text: str) -> Dict[str, float]:
        """Fallback to keyword matching when embeddings unavailable."""
        return self._fallback_keyword_classify(text).to_dict()
    
    def extract_article_topics(self, title: str, summary: str, 
                                tags: List[str] = None) -> ArticleTopics:
        """Extract topics from article metadata using fixed categories."""
        combined_text = f"{title} {summary} {' '.join(tags or [])}"
        category_scores = self.classify_text(combined_text)
        
        # Determine primary category
        top_categories = category_scores.get_top_categories(n=1, min_score=0.0)
        primary_topic = top_categories[0][0] if top_categories else "Other"
        
        return ArticleTopics(
            primary_topic=primary_topic,
            subtopics=[],  # No longer used
            topic_scores=category_scores.to_dict(),  # Legacy format
            category_scores=category_scores,  # New format
            keywords=tags or []
        )
    
    def compute_interest_match_score(
        self, 
        article_topics: ArticleTopics,
        user_interests: InterestProfile
    ) -> Tuple[float, List[str]]:
        """
        Compute match score between article and user interests.
        Uses fixed category overlap for consistent scoring.
        
        Returns (score, list of matching category names).
        """
        article_scores = article_topics.get_category_scores()
        user_scores = user_interests.category_scores
        
        matching_categories = []
        total_score = 0.0
        
        # Compare scores across all fixed categories
        for category in CATEGORY_NAMES:
            article_score = article_scores.to_dict().get(category, 0.0)
            user_score = user_scores.to_dict().get(category, 0.0)
            
            # Both article and user must have meaningful scores for a match
            if article_score >= 0.2 and user_score >= 0.2:
                # Score is product of both relevances
                match_contribution = article_score * user_score
                total_score += match_contribution
                
                if match_contribution >= 0.1:
                    matching_categories.append(category)
        
        # Normalize score (max theoretical is 7.0 if all categories match perfectly)
        normalized_score = min(1.0, total_score / 2.0)
        
        return normalized_score, matching_categories
    
    def should_notify_user(
        self, 
        article_topics: ArticleTopics,
        user_interests: InterestProfile,
        threshold: float = 0.5
    ) -> Tuple[bool, float, List[str]]:
        """
        Determine if user should be notified about an article.
        Returns (should_notify, match_score, matching_topics).
        """
        score, matching_topics = self.compute_interest_match_score(
            article_topics, user_interests
        )
        return score >= threshold, score, matching_topics


def update_interest_profile(
    profile: InterestProfile,
    query: str,
    topic_matcher: TopicMatcher
) -> InterestProfile:
    """
    Update interest profile based on a new query.
    Uses fixed categories for consistent tracking.
    
    The new approach:
    1. Classify the query into fixed categories
    2. Boost user's interest scores for matching categories
    3. Apply decay to non-matching categories
    """
    from datetime import datetime, timezone
    
    # Classify the query into fixed categories
    query_scores = topic_matcher.classify_text(query)
    
    # Get current scores as dict
    current_scores = profile.category_scores.to_dict()
    
    # Update scores based on query
    for category in CATEGORY_NAMES:
        query_score = query_scores.to_dict().get(category, 0.0)
        current_score = current_scores.get(category, 0.0)
        
        if query_score >= 0.2:
            # Boost: increase score based on query relevance
            # Use exponential moving average: new = 0.7 * old + 0.3 * query
            new_score = 0.7 * current_score + 0.3 * query_score
            current_scores[category] = min(1.0, new_score)
        else:
            # Slight decay for categories not in this query
            current_scores[category] = current_score * 0.98
    
    # Update profile with new scores
    profile.category_scores = CategoryScores.from_dict(current_scores)
    profile.last_updated = datetime.now(timezone.utc).isoformat()
    
    # Also update legacy interests dict for backward compatibility
    for category, score in current_scores.items():
        if score >= 0.2:
            if category in profile.interests and isinstance(profile.interests[category], UserInterest):
                profile.interests[category].confidence = score
                profile.interests[category].query_count += 1
            else:
                profile.interests[category] = UserInterest(
                    topic=category,
                    confidence=score,
                    query_count=1,
                    last_seen=profile.last_updated
                )
    
    # Remove low-score categories from legacy interests
    profile.interests = {
        k: v for k, v in profile.interests.items() 
        if (isinstance(v, UserInterest) and v.confidence >= 0.15) or 
           (isinstance(v, (int, float)) and v >= 0.15)
    }
    
    return profile


def classify_text_to_categories(text: str, topic_matcher: TopicMatcher) -> Dict[str, float]:
    """
    Convenience function to classify any text into fixed categories.
    Returns dict of category -> score.
    """
    return topic_matcher.classify_text(text).to_dict()


def get_category_names() -> List[str]:
    """Return the list of fixed category names."""
    return CATEGORY_NAMES.copy()
