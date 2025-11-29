from django.urls import path
from . import views

app_name = 'chatbot'

urlpatterns = [
    # Main chat interface
    path('', views.chat_home, name='chat_home'),
    
    # RAG-powered API endpoints
    path('api/ask/', views.ask_question, name='ask_question'),
    path('api/search/', views.search_articles, name='search_articles'),
    
    # Legacy endpoint (for backward compatibility with existing frontend)
    path('api/articles/', views.get_articles, name='get_articles'),
    
    # Article details
    path('api/article/<str:article_id>/', views.get_article_detail, name='get_article_detail'),
    
    # System endpoints
    path('api/stats/', views.get_stats, name='get_stats'),
    path('api/categories/', views.get_categories, name='get_categories'),
    
    # Debug endpoint
    path('api/debug/', views.debug_system, name='debug_system'),
]
