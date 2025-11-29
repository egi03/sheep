"""
Topic and Interest Matching System for RelevantAI.

This module provides semantic matching between user interests and article topics
using embeddings for accurate similarity scoring.
"""

from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field
import numpy as np


# Standardized topic taxonomy - hierarchical structure
TOPIC_TAXONOMY = {
    "Security": {
        "subtopics": [
            "Ransomware", "Malware", "Phishing", "Zero-day", "Vulnerabilities",
            "Data Breach", "APT", "Threat Intelligence", "Penetration Testing",
            "Encryption", "Authentication", "Access Control", "Network Security",
            "Application Security", "Cloud Security", "IoT Security", "Mobile Security"
        ],
        "keywords": ["hack", "attack", "exploit", "CVE", "patch", "security", "cyber"]
    },
    "AI/ML": {
        "subtopics": [
            "Machine Learning", "Deep Learning", "Neural Networks", "NLP",
            "Computer Vision", "LLM", "GPT", "Transformers", "Generative AI",
            "Reinforcement Learning", "MLOps", "AI Ethics", "AI Safety"
        ],
        "keywords": ["ai", "ml", "model", "training", "inference", "neural", "gpt", "llm"]
    },
    "Programming": {
        "subtopics": [
            "Python", "JavaScript", "TypeScript", "Rust", "Go", "Java", "C++",
            "Web Development", "Backend", "Frontend", "API", "Microservices",
            "Testing", "Code Review", "Refactoring", "Design Patterns"
        ],
        "keywords": ["code", "programming", "developer", "software", "framework", "library"]
    },
    "DevOps": {
        "subtopics": [
            "Kubernetes", "Docker", "CI/CD", "Infrastructure", "Monitoring",
            "Observability", "GitOps", "Terraform", "Ansible", "SRE",
            "Incident Response", "On-call", "Deployment", "Scaling"
        ],
        "keywords": ["deploy", "container", "cluster", "pipeline", "infrastructure"]
    },
    "Cloud": {
        "subtopics": [
            "AWS", "Azure", "GCP", "Serverless", "Lambda", "Cloud Native",
            "Multi-cloud", "Hybrid Cloud", "Cloud Migration", "Cloud Cost",
            "Cloud Architecture", "PaaS", "SaaS", "IaaS"
        ],
        "keywords": ["cloud", "aws", "azure", "gcp", "serverless", "saas"]
    },
    "Data": {
        "subtopics": [
            "Data Engineering", "Data Science", "Analytics", "Big Data",
            "Data Pipeline", "ETL", "Data Warehouse", "Data Lake",
            "SQL", "NoSQL", "Streaming", "Real-time Analytics"
        ],
        "keywords": ["data", "database", "analytics", "warehouse", "pipeline"]
    },
    "Privacy": {
        "subtopics": [
            "GDPR", "Data Privacy", "Compliance", "PII", "Anonymization",
            "Consent", "Data Protection", "Privacy Engineering", "CCPA"
        ],
        "keywords": ["privacy", "gdpr", "compliance", "personal data", "consent"]
    }
}


class UserInterest(BaseModel):
    """Represents a user's interest with confidence score."""
    topic: str
    subtopics: List[str] = Field(default_factory=list)
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


class ArticleTopics(BaseModel):
    """Represents extracted topics from an article."""
    primary_topic: str
    subtopics: List[str] = Field(default_factory=list)
    topic_scores: Dict[str, float] = Field(default_factory=dict)
    keywords: List[str] = Field(default_factory=list)


class InterestProfile(BaseModel):
    """Complete interest profile for a user."""
    interests: Dict[str, UserInterest] = Field(default_factory=dict)
    topic_embeddings: Dict[str, List[float]] = Field(default_factory=dict)
    last_updated: Optional[str] = None
    
    def get_top_interests(self, n: int = 5) -> List[Tuple[str, float]]:
        """Get top N interests sorted by confidence."""
        sorted_interests = sorted(
            [(k, v.confidence if isinstance(v, UserInterest) else float(v)) 
             for k, v in self.interests.items()],
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_interests[:n]
    
    def to_list(self) -> List[str]:
        """Convert to simple list of interest names for backward compatibility."""
        # get_top_interests returns List[Tuple[str, float]] where float is confidence
        return [k for k, confidence in self.get_top_interests(10) if confidence >= 0.3]


class TopicMatcher:
    """
    Semantic topic matching using embeddings.
    Provides accurate matching between articles and user interests.
    """
    
    def __init__(self, embeddings_model=None):
        self.embeddings_model = embeddings_model
        self._topic_embeddings_cache: Dict[str, List[float]] = {}
    
    def set_embeddings_model(self, model) -> None:
        """Set the embeddings model (OpenAIEmbeddings instance)."""
        self.embeddings_model = model
    
    def _get_all_topics(self) -> List[str]:
        """Get flat list of all topics and subtopics."""
        topics = []
        for main_topic, data in TOPIC_TAXONOMY.items():
            topics.append(main_topic)
            topics.extend(data["subtopics"])
        return topics
    
    def _compute_embedding(self, text: str) -> List[float]:
        """Compute embedding for text."""
        if not self.embeddings_model:
            raise ValueError("Embeddings model not set")
        return self.embeddings_model.embed_query(text)
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        a = np.array(vec1)
        b = np.array(vec2)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    
    def _cache_topic_embeddings(self) -> None:
        """Pre-compute embeddings for all taxonomy topics."""
        if self._topic_embeddings_cache:
            return
        
        all_topics = self._get_all_topics()
        for topic in all_topics:
            self._topic_embeddings_cache[topic.lower()] = self._compute_embedding(topic)
    
    def match_text_to_topics(self, text: str, top_k: int = 3) -> Dict[str, float]:
        """
        Match text to taxonomy topics using semantic similarity.
        Returns dict of topic -> similarity score.
        """
        if not self.embeddings_model:
            return self._fallback_keyword_match(text)
        
        try:
            self._cache_topic_embeddings()
            text_embedding = self._compute_embedding(text)
            
            scores = {}
            for topic, topic_emb in self._topic_embeddings_cache.items():
                scores[topic] = self._cosine_similarity(text_embedding, topic_emb)
            
            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            return dict(sorted_scores[:top_k])
        except Exception:
            return self._fallback_keyword_match(text)
    
    def _fallback_keyword_match(self, text: str) -> Dict[str, float]:
        """Fallback to keyword matching when embeddings unavailable."""
        text_lower = text.lower()
        scores = {}
        
        for main_topic, data in TOPIC_TAXONOMY.items():
            score = 0.0
            
            if main_topic.lower() in text_lower:
                score += 0.5
            
            for subtopic in data["subtopics"]:
                if subtopic.lower() in text_lower:
                    score += 0.3
            
            for keyword in data["keywords"]:
                if keyword.lower() in text_lower:
                    score += 0.1
            
            if score > 0:
                scores[main_topic] = min(1.0, score)
        
        return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3])
    
    def extract_article_topics(self, title: str, summary: str, 
                                tags: List[str] = None) -> ArticleTopics:
        """Extract topics from article metadata."""
        combined_text = f"{title} {summary} {' '.join(tags or [])}"
        topic_scores = self.match_text_to_topics(combined_text, top_k=5)
        
        primary_topic = "Other"
        subtopics = []
        
        if topic_scores:
            sorted_topics = sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)
            
            for topic_name, score in sorted_topics:
                for main_topic, data in TOPIC_TAXONOMY.items():
                    if topic_name.lower() == main_topic.lower():
                        primary_topic = main_topic
                        break
                    if topic_name.lower() in [s.lower() for s in data["subtopics"]]:
                        subtopics.append(topic_name)
                        if primary_topic == "Other":
                            primary_topic = main_topic
        
        return ArticleTopics(
            primary_topic=primary_topic,
            subtopics=subtopics[:5],
            topic_scores=topic_scores,
            keywords=tags or []
        )
    
    def compute_interest_match_score(
        self, 
        article_topics: ArticleTopics,
        user_interests: InterestProfile
    ) -> Tuple[float, List[str]]:
        """
        Compute match score between article and user interests.
        Returns (score, list of matching topics).
        """
        if not user_interests.interests:
            return 0.0, []
        
        matching_topics = []
        total_score = 0.0
        
        for interest_name, interest in user_interests.interests.items():
            interest_lower = interest_name.lower()
            
            if interest_lower == article_topics.primary_topic.lower():
                match_score = 0.4 * interest.confidence
                total_score += match_score
                matching_topics.append(interest_name)
            
            for subtopic in article_topics.subtopics:
                if subtopic.lower() == interest_lower:
                    match_score = 0.3 * interest.confidence
                    total_score += match_score
                    if interest_name not in matching_topics:
                        matching_topics.append(interest_name)
            
            for interest_subtopic in interest.subtopics:
                if interest_subtopic.lower() in [s.lower() for s in article_topics.subtopics]:
                    match_score = 0.2 * interest.confidence
                    total_score += match_score
            
            if interest_lower in article_topics.topic_scores:
                embedding_score = article_topics.topic_scores[interest_lower]
                total_score += 0.1 * embedding_score * interest.confidence
        
        return min(1.0, total_score), matching_topics
    
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
    Uses semantic matching to identify topics in the query.
    """
    from datetime import datetime, timezone
    
    topic_scores = topic_matcher.match_text_to_topics(query, top_k=3)
    
    for topic_name, score in topic_scores.items():
        if score < 0.2:
            continue
        
        if topic_name in profile.interests:
            interest = profile.interests[topic_name]
            # Ensure we have a UserInterest object
            if isinstance(interest, UserInterest):
                interest.boost_confidence(score * 0.15)
            else:
                # Replace with proper UserInterest if somehow got corrupted
                profile.interests[topic_name] = UserInterest(
                    topic=topic_name,
                    confidence=min(0.7, float(interest) + score * 0.15) if isinstance(interest, (int, float)) else 0.5,
                    query_count=1,
                    last_seen=datetime.now(timezone.utc).isoformat()
                )
        else:
            profile.interests[topic_name] = UserInterest(
                topic=topic_name,
                confidence=min(0.5, score),
                query_count=1,
                last_seen=datetime.now(timezone.utc).isoformat()
            )
    
    # Decay interests not matched in this query
    for topic_name, interest in list(profile.interests.items()):
        if topic_name not in topic_scores:
            if isinstance(interest, UserInterest):
                interest.decay_confidence(0.98)
            # If not UserInterest, leave as-is or remove if too small
    
    # Remove low-confidence interests
    profile.interests = {
        k: v for k, v in profile.interests.items() 
        if (isinstance(v, UserInterest) and v.confidence >= 0.1) or 
           (isinstance(v, (int, float)) and v >= 0.1)
    }
    
    profile.last_updated = datetime.now(timezone.utc).isoformat()
    return profile
