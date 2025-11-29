from django.urls import path
from . import views

app_name = 'chatbot'

urlpatterns = [
    path('', views.chat_home, name='chat_home'),
    path('api/articles/', views.get_articles, name='get_articles'),
    path('api/article/<str:article_id>/', views.get_article_detail, name='get_article_detail'),
]
