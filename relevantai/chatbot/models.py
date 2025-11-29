from django.db import models
from django.utils import timezone


class Article(models.Model):
    """Model to store scraped articles for tracking and caching."""
    
    article_id = models.CharField(max_length=255, unique=True, db_index=True)
    title = models.CharField(max_length=500)
    url = models.URLField(max_length=2000)
    author = models.CharField(max_length=255, blank=True, default='')
    
    # Content
    content = models.TextField(blank=True, default='')
    summary = models.TextField(blank=True, default='')
    
    # Categorization
    category = models.CharField(max_length=100, blank=True, default='Other')
    tags = models.JSONField(default=list, blank=True)
    key_points = models.JSONField(default=list, blank=True)
    
    # NEW: Category scores for personalization (stores scores for each fixed category)
    # Format: {"Security": 0.8, "AI/ML": 0.2, "Programming": 0.1, ...}
    category_scores = models.JSONField(default=dict, blank=True)
    
    # Metadata
    source = models.CharField(max_length=255, default='thehackernews.com')
    topics = models.JSONField(default=list, blank=True)
    published_date = models.CharField(max_length=100, blank=True, default='')
    
    # Status tracking
    is_indexed = models.BooleanField(default=False, db_index=True)
    indexed_at = models.DateTimeField(null=True, blank=True)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['category']),
            models.Index(fields=['source']),
            models.Index(fields=['is_indexed']),
        ]
    
    def __str__(self):
        return f"{self.title[:50]}..." if len(self.title) > 50 else self.title
    
    def mark_indexed(self):
        """Mark this article as indexed in the vector store."""
        self.is_indexed = True
        self.indexed_at = timezone.now()
        self.save(update_fields=['is_indexed', 'indexed_at'])
    
    def get_category_scores_dict(self) -> dict:
        """Get category scores as dict, computing if not present."""
        if self.category_scores:
            return self.category_scores
        # Fallback: return empty or compute from category field
        from rag.interests import FIXED_CATEGORIES
        scores = {cat: 0.0 for cat in FIXED_CATEGORIES.keys()}
        if self.category in scores:
            scores[self.category] = 0.8
        return scores


class ChatSession(models.Model):
    """Model to track chat sessions for analytics."""
    
    session_id = models.CharField(max_length=255, unique=True, db_index=True)
    user = models.ForeignKey(
        'auth.User',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='chat_sessions'
    )
    title = models.CharField(max_length=255, blank=True, default='')
    interested_topics = models.JSONField(default=list, blank=True)
    interest_profile = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"Session {self.session_id}"
    
    def get_interest_profile(self):
        """Get interest profile as Pydantic model."""
        from rag.interests import InterestProfile
        if self.interest_profile:
            return InterestProfile(**self.interest_profile)
        return InterestProfile()
    
    def set_interest_profile(self, profile):
        """Set interest profile from Pydantic model."""
        self.interest_profile = profile.model_dump()
        self.interested_topics = profile.to_list()
        self.save(update_fields=['interest_profile', 'interested_topics', 'updated_at'])


class ChatMessage(models.Model):
    """Model to store individual chat messages."""
    
    MESSAGE_TYPE_CHOICES = [
        ('user', 'User'),
        ('assistant', 'Assistant'),
    ]
    
    session = models.ForeignKey(
        ChatSession, 
        on_delete=models.CASCADE, 
        related_name='messages'
    )
    message_type = models.CharField(max_length=20, choices=MESSAGE_TYPE_CHOICES)
    content = models.TextField()
    
    # For assistant messages, store additional data
    confidence = models.CharField(max_length=50, blank=True, default='')
    key_insights = models.JSONField(default=list, blank=True)
    sources = models.JSONField(default=list, blank=True)
    expanded_query = models.CharField(max_length=500, blank=True, default='')
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['created_at']
    
    def __str__(self):
        return f"{self.message_type}: {self.content[:50]}..."


class ScrapingRun(models.Model):
    """Model to track scraping runs for monitoring."""
    
    STATUS_CHOICES = [
        ('running', 'Running'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]
    
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='running')
    pages_scraped = models.IntegerField(default=0)
    articles_found = models.IntegerField(default=0)
    articles_indexed = models.IntegerField(default=0)
    error_message = models.TextField(blank=True, default='')
    
    # Timestamps
    started_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    
    class Meta:
        ordering = ['-started_at']
    
    def __str__(self):
        return f"Scraping Run {self.id} - {self.status}"
    
    def mark_completed(self, articles_found: int, articles_indexed: int):
        """Mark this run as completed."""
        self.status = 'completed'
        self.articles_found = articles_found
        self.articles_indexed = articles_indexed
        self.completed_at = timezone.now()
        self.save()
    
    def mark_failed(self, error_message: str):
        """Mark this run as failed."""
        self.status = 'failed'
        self.error_message = error_message
        self.completed_at = timezone.now()
        self.save()


class UserProfile(models.Model):
    """
    User profile for personalization and notifications.
    Links to Django auth User when available, or works standalone with email.
    """
    
    user = models.OneToOneField(
        'auth.User',
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name='profile'
    )
    email = models.EmailField(unique=True, db_index=True)
    session = models.OneToOneField(
        ChatSession,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='user_profile'
    )
    
    # Topics the user is interested in (list of category names)
    interested_topics = models.JSONField(default=list, blank=True)
    interest_profile = models.JSONField(default=dict, blank=True)
    
    email_notifications_enabled = models.BooleanField(default=False)
    notification_threshold = models.FloatField(default=0.5)
    notification_frequency = models.CharField(
        max_length=20,
        choices=[
            ('instant', 'Instant'),
            ('daily', 'Daily Digest'),
            ('weekly', 'Weekly Digest'),
        ],
        default='daily'
    )
    last_notification_sent = models.DateTimeField(null=True, blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"Profile: {self.email}"
    
    def get_interest_profile(self):
        """Get interest profile as Pydantic model."""
        from rag.interests import InterestProfile
        if self.interest_profile:
            return InterestProfile(**self.interest_profile)
        return InterestProfile()
    
    def set_interest_profile(self, profile):
        """Set interest profile from Pydantic model."""
        self.interest_profile = profile.model_dump()
        self.save(update_fields=['interest_profile', 'updated_at'])


class ArticleInterestMatch(models.Model):
    """
    Tracks matches between articles and user interests.
    Used for sending notifications about new relevant articles.
    """
    
    article = models.ForeignKey(
        Article,
        on_delete=models.CASCADE,
        related_name='interest_matches'
    )
    user_profile = models.ForeignKey(
        UserProfile,
        on_delete=models.CASCADE,
        related_name='article_matches'
    )
    
    match_score = models.FloatField()
    matching_topics = models.JSONField(default=list)
    
    notification_sent = models.BooleanField(default=False)
    notification_sent_at = models.DateTimeField(null=True, blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-match_score', '-created_at']
        unique_together = ['article', 'user_profile']
        indexes = [
            models.Index(fields=['notification_sent', 'match_score']),
            models.Index(fields=['user_profile', 'notification_sent']),
        ]
    
    def __str__(self):
        return f"{self.user_profile.email} <- {self.article.title[:30]} ({self.match_score:.2f})"

