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
    
    # Authentication
    path('api/login/', views.login_api, name='login_api'),
    path('api/signup/', views.signup_api, name='signup_api'),
    path('api/logout/', views.logout_api, name='logout_api'),
    
    # Chat History
    path('api/chat/sessions/', views.get_chat_sessions, name='get_chat_sessions'),
    path('api/chat/sessions/new/', views.create_chat_session, name='create_chat_session'),
    path('api/chat/sessions/<str:session_id>/', views.get_chat_messages, name='get_chat_messages'),
    path('api/chat/sessions/<str:session_id>/switch/', views.switch_chat_session, name='switch_chat_session'),
    path('api/chat/sessions/<str:session_id>/delete/', views.delete_chat_session, name='delete_chat_session'),
]
