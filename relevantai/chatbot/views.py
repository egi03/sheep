from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
import json
import requests


def chat_home(request):
    """Render the main chat interface."""
    return render(request, 'chatbot/chat.html')


@require_http_methods(["POST"])
def get_articles(request):
    """
    API endpoint to get relevant articles from Hacker News.
    For now, this returns mock data. Later will be integrated with AI model.
    """
    try:
        data = json.loads(request.body)
        query = data.get('query', '')
        
        # Fetch articles from Hacker News API
        articles = fetch_hackernews_articles(query)
        
        return JsonResponse({
            'success': True,
            'articles': articles
        })
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)


def fetch_hackernews_articles(query, limit=5):
    """
    Fetch articles from Hacker News based on query.
    Uses the Algolia HN Search API.
    """
    try:
        # Use Algolia's Hacker News Search API
        url = f"https://hn.algolia.com/api/v1/search?query={query}&tags=story&hitsPerPage={limit}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        articles = []
        
        for hit in data.get('hits', []):
            article = {
                'id': hit.get('objectID'),
                'title': hit.get('title', 'No Title'),
                'url': hit.get('url', f"https://news.ycombinator.com/item?id={hit.get('objectID')}"),
                'author': hit.get('author', 'Unknown'),
                'created_at': hit.get('created_at', ''),
                'story_text': hit.get('story_text', ''),
            }
            articles.append(article)
        
        return articles
    except requests.RequestException:
        # Return some mock data if API fails
        return get_mock_articles(query)


def get_mock_articles(query):
    """Return mock articles for testing when API is unavailable."""
    mock_articles = [
        {
            'id': '1',
            'title': f'Understanding {query} in Modern Tech',
            'url': 'https://news.ycombinator.com/',
            'author': 'techwriter',
            'created_at': '2024-01-15T10:30:00Z',
            'story_text': f'A comprehensive guide to understanding {query} and its applications in modern technology...',
        },
        {
            'id': '2',
            'title': f'The Future of {query}: What You Need to Know',
            'url': 'https://news.ycombinator.com/',
            'author': 'futurist',
            'created_at': '2024-01-14T14:20:00Z',
            'story_text': f'Exploring the future trends and developments in {query}...',
        },
        {
            'id': '3',
            'title': f'Best Practices for {query} in 2024',
            'url': 'https://news.ycombinator.com/',
            'author': 'developer',
            'created_at': '2024-01-13T09:15:00Z',
            'story_text': f'Learn the best practices and tips for working with {query}...',
        },
    ]
    return mock_articles


@require_http_methods(["GET"])
def get_article_detail(request, article_id):
    """
    Fetch full article details for the popup.
    """
    try:
        # Fetch article from HN API
        url = f"https://hn.algolia.com/api/v1/items/{article_id}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        article = {
            'id': data.get('id'),
            'title': data.get('title', 'No Title'),
            'url': data.get('url', f"https://news.ycombinator.com/item?id={data.get('id')}"),
            'author': data.get('author', 'Unknown'),
            'points': data.get('points', 0),
            'created_at': data.get('created_at', ''),
            'text': data.get('text', ''),
            'children': data.get('children', [])[:5],  # Top 5 comments
        }
        
        return JsonResponse({
            'success': True,
            'article': article
        })
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)
