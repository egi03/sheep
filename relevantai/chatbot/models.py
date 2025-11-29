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


class ChatSession(models.Model):
    """Model to track chat sessions for analytics."""
    
    session_id = models.CharField(max_length=255, unique=True, db_index=True)
    interested_topics = models.JSONField(default=list, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"Session {self.session_id}"


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
