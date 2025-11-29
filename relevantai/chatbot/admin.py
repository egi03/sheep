from django.contrib import admin
from .models import Article, ChatSession, ChatMessage, ScrapingRun, UserProfile, ArticleInterestMatch


@admin.register(Article)
class ArticleAdmin(admin.ModelAdmin):
    list_display = ['title', 'category', 'source', 'is_indexed', 'created_at']
    list_filter = ['is_indexed', 'category', 'source']
    search_fields = ['title', 'content', 'summary']
    readonly_fields = ['created_at', 'updated_at', 'indexed_at']
    ordering = ['-created_at']


@admin.register(ChatSession)
class ChatSessionAdmin(admin.ModelAdmin):
    list_display = ['session_id', 'user', 'title', 'message_count', 'created_at', 'updated_at']
    list_filter = ['created_at', 'user']
    search_fields = ['session_id', 'title', 'user__username']
    readonly_fields = ['session_id', 'created_at', 'updated_at']
    ordering = ['-updated_at']
    
    def message_count(self, obj):
        return obj.messages.count()
    message_count.short_description = 'Messages'


@admin.register(ChatMessage)
class ChatMessageAdmin(admin.ModelAdmin):
    list_display = ['id', 'session', 'message_type', 'short_content', 'created_at']
    list_filter = ['message_type', 'created_at']
    search_fields = ['content', 'session__session_id']
    readonly_fields = ['created_at']
    ordering = ['-created_at']
    
    def short_content(self, obj):
        return obj.content[:50] + '...' if len(obj.content) > 50 else obj.content
    short_content.short_description = 'Content'


@admin.register(ScrapingRun)
class ScrapingRunAdmin(admin.ModelAdmin):
    list_display = ['id', 'status', 'pages_scraped', 'articles_found', 'articles_indexed', 'started_at', 'completed_at']
    list_filter = ['status', 'started_at']
    readonly_fields = ['started_at', 'completed_at']
    ordering = ['-started_at']


@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ['email', 'user', 'email_notifications_enabled', 'notification_frequency', 'created_at']
    list_filter = ['email_notifications_enabled', 'notification_frequency', 'created_at']
    search_fields = ['email', 'user__username']
    readonly_fields = ['created_at', 'updated_at']
    ordering = ['-created_at']


@admin.register(ArticleInterestMatch)
class ArticleInterestMatchAdmin(admin.ModelAdmin):
    list_display = ['user_profile', 'article', 'match_score', 'notification_sent', 'created_at']
    list_filter = ['notification_sent', 'created_at']
    search_fields = ['user_profile__email', 'article__title']
    readonly_fields = ['created_at']
    ordering = ['-match_score', '-created_at']
