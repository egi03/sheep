import json
import logging
import uuid
import hashlib
from typing import Optional

from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from django.utils import timezone
from django.db.models import Q
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.db import IntegrityError

from .models import Article, ChatSession, ChatMessage, ScrapingRun, UserProfile

# Configure logging to show DEBUG level
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('chatbot')
logger.setLevel(logging.DEBUG)

# Add console handler if not present
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(levelname)s] %(name)s: %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

# Import RAG and scraper - they are in the parent directory 
# and should be available via sys.path set in settings.py
print("=" * 60)
print("ATTEMPTING TO IMPORT RAG AND SCRAPER MODULES...")
print("=" * 60)

try:
    from rag import RelevantAI, get_rag, ConfigurationError
    from scraper_engine import HackerNewsScraper
    RAG_AVAILABLE = True
    print("[SUCCESS] RAG and scraper modules imported successfully!")
    print(f"  - RelevantAI: {RelevantAI}")
    print(f"  - get_rag: {get_rag}")
    print(f"  - HackerNewsScraper: {HackerNewsScraper}")
except ImportError as e:
    RAG_AVAILABLE = False
    import_error = str(e)
    print(f"[ERROR] Failed to import RAG/scraper: {import_error}")
    import traceback
    traceback.print_exc()

print(f"\nRAG_AVAILABLE = {RAG_AVAILABLE}")
print(f"OPENAI_API_KEY set: {bool(settings.OPENAI_API_KEY)}")
print(f"PINECONE_API_KEY set: {bool(settings.PINECONE_API_KEY)}")
print("=" * 60)


def get_or_create_session(request) -> ChatSession:
    """Get or create a chat session for the current user."""
    session_id = request.session.get('chat_session_id')
    
    if not session_id:
        session_id = str(uuid.uuid4())
        request.session['chat_session_id'] = session_id
    
    session, created = ChatSession.objects.get_or_create(session_id=session_id)
    
    if request.user.is_authenticated and not session.user:
        session.user = request.user
        session.save(update_fields=['user'])

    return session


def get_rag_instance() -> Optional[RelevantAI]:
    """Get or create the RAG instance with error handling."""
    print("\n" + "=" * 60)
    print("GET_RAG_INSTANCE CALLED")
    print("=" * 60)
    
    if not RAG_AVAILABLE:
        print(f"[ERROR] RAG module not available: {import_error}")
        logger.error(f"RAG module not available: {import_error}")
        return None
    
    print(f"[DEBUG] RAG_AVAILABLE = True")
    print(f"[DEBUG] OPENAI_API_KEY present: {bool(settings.OPENAI_API_KEY)}")
    print(f"[DEBUG] PINECONE_API_KEY present: {bool(settings.PINECONE_API_KEY)}")
    
    try:
        print("[DEBUG] Calling get_rag()...")
        rag = get_rag()
        print(f"[SUCCESS] RAG instance created: {rag}")
        return rag
    except ConfigurationError as e:
        print(f"[ERROR] RAG configuration error: {e}")
        logger.warning(f"RAG configuration error: {e}")
        import traceback
        traceback.print_exc()
        return None
    except Exception as e:
        print(f"[ERROR] Failed to initialize RAG: {e}")
        logger.error(f"Failed to initialize RAG: {e}")
        import traceback
        traceback.print_exc()
        return None


def chat_home(request):
    """Render the main chat interface."""
    # Get or create session for analytics
    get_or_create_session(request)
    
    context = {
        'rag_available': RAG_AVAILABLE and bool(settings.OPENAI_API_KEY),
    }
    return render(request, 'chatbot/chat.html', context)


@csrf_exempt
@require_http_methods(["POST"])
def ask_question(request):
    """
    API endpoint to ask a question to the RAG system.
    This is the main Q&A endpoint that returns intelligent answers with sources.
    Includes user memory & personalization based on chat history.
    Uses fixed category scoring for interest extraction and personalization.
    """
    try:
        data = json.loads(request.body)
        question = data.get('question', '').strip()
        
        if not question:
            return JsonResponse({
                'success': False,
                'error': 'Question is required'
            }, status=400)
        
        # Get session and save user message
        session = get_or_create_session(request)
        ChatMessage.objects.create(
            session=session,
            message_type='user',
            content=question
        )
        
        # Set session title from first message if not set
        if not session.title:
            session.title = question[:50] + ('...' if len(question) > 50 else '')
            session.save(update_fields=['title'])
        
        # Try RAG system first
        rag = get_rag_instance()
        
        if rag:
            try:
                # Count user messages in this session
                user_message_count = ChatMessage.objects.filter(
                    session=session,
                    message_type='user'
                ).count()
                
                # Load existing interest profile from session
                # Format: {"Security": 0.8, "AI/ML": 0.3, ...}
                interest_profile = session.interest_profile or {}
                
                # Update interests every message using the new fixed category system
                try:
                    # Use RAG's topic_matcher to classify the question
                    updated_profile = rag.update_interest_profile(interest_profile, question)
                    session.interest_profile = updated_profile
                    session.save(update_fields=['interest_profile'])
                    interest_profile = updated_profile
                    logger.debug(f"Updated interest profile: {interest_profile}")
                except Exception as e:
                    logger.warning(f"Failed to update interest profile: {e}")
                
                # Get top interests for display
                top_interests = [
                    cat for cat, score in sorted(
                        interest_profile.items(), 
                        key=lambda x: x[1], 
                        reverse=True
                    )[:5] 
                    if score >= 0.2
                ]
                
                # Use the RAG system with personalization
                result = rag.ask(
                    question=question,
                    top_k=settings.RAG_CONFIG.get('top_k', 5),
                    category=data.get('category'),
                    user_interests=top_interests
                )
                
                # Save assistant message with full context
                ChatMessage.objects.create(
                    session=session,
                    message_type='assistant',
                    content=result.get('answer', ''),
                    confidence=result.get('confidence', ''),
                    key_insights=result.get('key_insights', []),
                    sources=result.get('sources', []),
                    expanded_query=result.get('expanded_query', '')
                )
                
                return JsonResponse({
                    'success': True,
                    'answer': result.get('answer', 'No answer available'),
                    'confidence': result.get('confidence', 'low'),
                    'key_insights': result.get('key_insights', []),
                    'sources': result.get('sources', []),
                    'expanded_query': result.get('expanded_query', ''),
                    'user_interests': top_interests,
                    'interest_profile': interest_profile,
                    'mode': 'rag'
                })
                
            except Exception as e:
                logger.error(f"RAG ask failed: {e}")
                # Fall through to fallback search
        
        # Fallback: Search in local database if RAG is not available
        articles = Article.objects.filter(
            is_indexed=True
        ).filter(
            Q(title__icontains=question) |
            Q(summary__icontains=question) |
            Q(content__icontains=question)
        )[:5]
        
        if articles.exists():
            sources = [{
                'title': a.title,
                'url': a.url,
                'summary': a.summary[:300] if a.summary else '',
                'category': a.category,
            } for a in articles]
            
            return JsonResponse({
                'success': True,
                'answer': f"I found {len(sources)} relevant articles. Please review them for information about '{question}'.",
                'confidence': 'low',
                'key_insights': [],
                'sources': sources,
                'mode': 'fallback'
            })
        
        return JsonResponse({
            'success': True,
            'answer': "I couldn't find any relevant articles. Try rephrasing your question or use different keywords.",
            'confidence': 'low',
            'key_insights': [],
            'sources': [],
            'mode': 'fallback'
        })
        
    except json.JSONDecodeError:
        return JsonResponse({
            'success': False,
            'error': 'Invalid JSON'
        }, status=400)
    except Exception as e:
        logger.error(f"Error in ask_question: {e}")
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def search_articles(request):
    """
    API endpoint to search for articles using RAG semantic search.
    """
    try:
        data = json.loads(request.body)
        query = data.get('query', '').strip()
        
        if not query:
            return JsonResponse({
                'success': False,
                'error': 'Query is required'
            }, status=400)
        
        top_k = min(int(data.get('top_k', 5)), 20)
        category = data.get('category')
        
        rag = get_rag_instance()
        
        if rag:
            try:
                # Use enhanced search with query expansion
                results = rag.search_enhanced(
                    query=query,
                    top_k=top_k,
                    category=category,
                    use_query_expansion=settings.RAG_CONFIG.get('use_query_expansion', True)
                )
                
                return JsonResponse({
                    'success': True,
                    'articles': results,
                    'mode': 'rag'
                })
                
            except Exception as e:
                logger.error(f"RAG search failed: {e}")
        
        # Fallback to database search
        articles = Article.objects.filter(is_indexed=True)
        
        if category:
            articles = articles.filter(category=category)
        
        articles = articles.filter(
            Q(title__icontains=query) |
            Q(summary__icontains=query) |
            Q(tags__icontains=query)
        )[:top_k]
        
        results = [{
            'id': a.article_id,
            'title': a.title,
            'url': a.url,
            'summary': a.summary[:300] if a.summary else '',
            'category': a.category,
            'tags': a.tags,
            'author': a.author,
            'similarity_score': 0.5,  # Placeholder for fallback
        } for a in articles]
        
        return JsonResponse({
            'success': True,
            'articles': results,
            'mode': 'fallback'
        })
        
    except Exception as e:
        logger.error(f"Error in search_articles: {e}")
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def get_articles(request):
    """
    Legacy API endpoint for compatibility with existing frontend.
    Wraps the search functionality.
    """
    print("\n" + "=" * 60)
    print("GET_ARTICLES ENDPOINT CALLED")
    print("=" * 60)
    
    try:
        data = json.loads(request.body)
        query = data.get('query', '').strip()
        
        print(f"[DEBUG] Query received: '{query}'")
        
        if not query:
            print("[ERROR] Empty query received")
            return JsonResponse({
                'success': False,
                'error': 'Query is required'
            }, status=400)
        
        # Debug: Check database state
        total_articles = Article.objects.count()
        indexed_articles = Article.objects.filter(is_indexed=True).count()
        print(f"[DEBUG] Database state:")
        print(f"  - Total articles in DB: {total_articles}")
        print(f"  - Indexed articles: {indexed_articles}")
        
        # Get RAG instance
        print("[DEBUG] Getting RAG instance...")
        rag = get_rag_instance()
        
        if rag:
            print("[DEBUG] RAG instance obtained successfully")
            try:
                print(f"[DEBUG] Calling rag.ask() with question='{query}'")
                print(f"[DEBUG] top_k = {settings.RAG_CONFIG.get('top_k', 5)}")
                
                # Use the ask method for a conversational response
                result = rag.ask(
                    question=query,
                    top_k=settings.RAG_CONFIG.get('top_k', 5)
                )
                
                print(f"[DEBUG] RAG result received:")
                print(f"  - answer: {result.get('answer', '')[:100]}...")
                print(f"  - confidence: {result.get('confidence', '')}")
                print(f"  - sources count: {len(result.get('sources', []))}")
                print(f"  - key_insights count: {len(result.get('key_insights', []))}")
                print(f"  - error: {result.get('error', 'None')}")
                
                # Format sources for the frontend
                articles = []
                for i, source in enumerate(result.get('sources', [])):
                    print(f"  - Source {i+1}: {source.get('title', 'No Title')[:50]}...")
                    articles.append({
                        'id': hashlib.md5(source.get('url', '').encode()).hexdigest()[:16],
                        'title': source.get('title', 'No Title'),
                        'url': source.get('url', ''),
                        'summary': source.get('summary', ''),
                        'author': source.get('author', 'Unknown'),
                        'category': source.get('category', 'Other'),
                        'relevance_score': source.get('relevance_score', 0),
                    })
                
                print(f"[DEBUG] Formatted {len(articles)} articles for response")
                
                return JsonResponse({
                    'success': True,
                    'articles': articles,
                    'answer': result.get('answer', ''),
                    'confidence': result.get('confidence', 'low'),
                    'key_insights': result.get('key_insights', []),
                    'mode': 'rag'
                })
                
            except Exception as e:
                print(f"[ERROR] RAG failed with exception: {e}")
                import traceback
                traceback.print_exc()
                logger.error(f"RAG failed, falling back: {e}")
        else:
            print("[WARNING] RAG instance is None, falling back to database search")
        
        # Fallback: Search local database
        print("[DEBUG] Falling back to local database search")
        articles = Article.objects.filter(is_indexed=True).filter(
            Q(title__icontains=query) |
            Q(summary__icontains=query)
        )[:5]
        
        print(f"[DEBUG] Found {articles.count()} articles in fallback search")
        
        results = [{
            'id': a.article_id,
            'title': a.title,
            'url': a.url,
            'summary': a.summary[:200] if a.summary else 'No summary available.',
            'author': a.author or 'Unknown',
            'category': a.category,
        } for a in articles]
        
        return JsonResponse({
            'success': True,
            'articles': results,
            'answer': '',
            'mode': 'fallback'
        })
        
    except Exception as e:
        print(f"[ERROR] Exception in get_articles: {e}")
        import traceback
        traceback.print_exc()
        logger.error(f"Error in get_articles: {e}")
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)


@require_http_methods(["GET"])
def get_article_detail(request, article_id):
    """
    Fetch full article details for the popup.
    """
    try:
        # First check local database
        try:
            article = Article.objects.get(article_id=article_id)
            return JsonResponse({
                'success': True,
                'article': {
                    'id': article.article_id,
                    'title': article.title,
                    'url': article.url,
                    'author': article.author,
                    'summary': article.summary,
                    'content': article.content[:5000] if article.content else '',
                    'category': article.category,
                    'tags': article.tags,
                    'key_points': article.key_points,
                    'created_at': article.published_date,
                }
            })
        except Article.DoesNotExist:
            pass
        
        # If not found locally, return basic info
        return JsonResponse({
            'success': True,
            'article': {
                'id': article_id,
                'title': 'Article',
                'text': 'Article details not available. Click the link to read the full article.',
            }
        })
        
    except Exception as e:
        logger.error(f"Error in get_article_detail: {e}")
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)


@require_http_methods(["GET"])
def get_stats(request):
    """
    Get system statistics for monitoring.
    """
    try:
        stats = {
            'articles_total': Article.objects.count(),
            'articles_indexed': Article.objects.filter(is_indexed=True).count(),
            'chat_sessions': ChatSession.objects.count(),
            'chat_messages': ChatMessage.objects.count(),
            'rag_available': RAG_AVAILABLE and bool(settings.OPENAI_API_KEY),
        }
        
        rag = get_rag_instance()
        if rag:
            try:
                rag_stats = rag.get_stats()
                stats['vector_store'] = rag_stats
            except Exception as e:
                logger.warning(f"Failed to get RAG stats: {e}")
        
        # Get latest scraping run
        latest_run = ScrapingRun.objects.first()
        if latest_run:
            stats['last_scraping'] = {
                'status': latest_run.status,
                'articles_found': latest_run.articles_found,
                'articles_indexed': latest_run.articles_indexed,
                'completed_at': latest_run.completed_at.isoformat() if latest_run.completed_at else None,
            }
        
        return JsonResponse({
            'success': True,
            'stats': stats
        })
        
    except Exception as e:
        logger.error(f"Error in get_stats: {e}")
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)


@require_http_methods(["GET"])
def get_categories(request):
    """
    Get available article categories (using fixed categories).
    """
    from rag.interests import CATEGORY_NAMES
    
    # Return the fixed categories
    all_categories = CATEGORY_NAMES.copy()
    all_categories.sort()
    
    return JsonResponse({
        'success': True,
        'categories': all_categories
    })


@require_http_methods(["GET"])
def debug_system(request):
    """
    Debug endpoint to check the entire system step by step.
    Visit /api/debug/ to see the output.
    """
    debug_info = {
        'step1_imports': {},
        'step2_settings': {},
        'step3_database': {},
        'step4_rag_init': {},
        'step5_vector_store': {},
        'step6_test_search': {},
    }
    
    # Step 1: Check imports
    debug_info['step1_imports'] = {
        'rag_available': RAG_AVAILABLE,
        'import_error': import_error if not RAG_AVAILABLE else None,
    }
    
    # Step 2: Check settings
    debug_info['step2_settings'] = {
        'openai_api_key_set': bool(settings.OPENAI_API_KEY),
        'openai_api_key_length': len(settings.OPENAI_API_KEY) if settings.OPENAI_API_KEY else 0,
        'openai_api_key_prefix': settings.OPENAI_API_KEY[:10] + '...' if settings.OPENAI_API_KEY else None,
        'pinecone_api_key_set': bool(settings.PINECONE_API_KEY),
        'pinecone_api_key_length': len(settings.PINECONE_API_KEY) if settings.PINECONE_API_KEY else 0,
        'pinecone_index_name': settings.PINECONE_INDEX_NAME,
        'rag_config': settings.RAG_CONFIG,
    }
    
    # Step 3: Check database
    debug_info['step3_database'] = {
        'total_articles': Article.objects.count(),
        'indexed_articles': Article.objects.filter(is_indexed=True).count(),
        'sample_articles': list(Article.objects.values('article_id', 'title', 'is_indexed')[:3]),
    }
    
    # Step 4: Try to initialize RAG
    if RAG_AVAILABLE:
        try:
            print("[DEBUG] Attempting to initialize RAG for debug endpoint...")
            rag = get_rag()
            debug_info['step4_rag_init'] = {
                'success': True,
                'rag_instance': str(rag),
            }
            
            # Step 5: Check vector store stats
            try:
                stats = rag.get_stats()
                debug_info['step5_vector_store'] = {
                    'success': True,
                    'stats': stats,
                }
            except Exception as e:
                debug_info['step5_vector_store'] = {
                    'success': False,
                    'error': str(e),
                }
            
            # Step 6: Try a test search
            try:
                print("[DEBUG] Attempting test search...")
                results = rag.search(query="security", top_k=2)
                debug_info['step6_test_search'] = {
                    'success': True,
                    'results_count': len(results),
                    'results': results,
                }
            except Exception as e:
                debug_info['step6_test_search'] = {
                    'success': False,
                    'error': str(e),
                    'traceback': str(e.__traceback__),
                }
                import traceback
                traceback.print_exc()
                
        except Exception as e:
            debug_info['step4_rag_init'] = {
                'success': False,
                'error': str(e),
            }
            import traceback
            traceback.print_exc()
    else:
        debug_info['step4_rag_init'] = {
            'success': False,
            'error': 'RAG module not available',
        }
    
    return JsonResponse({
        'success': True,
        'debug': debug_info
    })


@csrf_exempt
@require_http_methods(["POST"])
def login_api(request):
    """API endpoint for user login."""
    try:
        data = json.loads(request.body)
        username = data.get('username')
        password = data.get('password')
        
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return JsonResponse({'success': True, 'username': user.username})
        else:
            return JsonResponse({'success': False, 'error': 'Invalid credentials'}, status=401)
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def signup_api(request):
    """API endpoint for user registration."""
    try:
        data = json.loads(request.body)
        username = data.get('username')
        password = data.get('password')
        email = data.get('email', '')
        
        if not username or not password:
            return JsonResponse({'success': False, 'error': 'Username and password are required'}, status=400)

        if User.objects.filter(username=username).exists():
            return JsonResponse({'success': False, 'error': 'Username already exists'}, status=400)
            
        user = User.objects.create_user(username=username, password=password, email=email)
        login(request, user)
        return JsonResponse({'success': True, 'username': user.username})
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def logout_api(request):
    """API endpoint for user logout."""
    logout(request)
    return JsonResponse({'success': True})


# ============================================================================
# Chat History API Endpoints
# ============================================================================

@require_http_methods(["GET"])
def get_chat_sessions(request):
    """Get all chat sessions for the current user."""
    if not request.user.is_authenticated:
        return JsonResponse({'success': False, 'error': 'Not authenticated'}, status=401)
    
    sessions = ChatSession.objects.filter(user=request.user).order_by('-updated_at')
    
    sessions_data = []
    for session in sessions:
        # Get first message as title if no title set
        title = session.title
        if not title:
            first_msg = session.messages.filter(message_type='user').first()
            if first_msg:
                title = first_msg.content[:50] + ('...' if len(first_msg.content) > 50 else '')
            else:
                title = 'New Chat'
        
        sessions_data.append({
            'id': session.session_id,
            'title': title,
            'created_at': session.created_at.isoformat(),
            'updated_at': session.updated_at.isoformat(),
            'message_count': session.messages.count()
        })
    
    return JsonResponse({'success': True, 'sessions': sessions_data})


@require_http_methods(["GET"])
def get_chat_messages(request, session_id):
    """Get all messages for a specific chat session."""
    if not request.user.is_authenticated:
        return JsonResponse({'success': False, 'error': 'Not authenticated'}, status=401)
    
    try:
        session = ChatSession.objects.get(session_id=session_id, user=request.user)
    except ChatSession.DoesNotExist:
        return JsonResponse({'success': False, 'error': 'Session not found'}, status=404)
    
    messages = session.messages.order_by('created_at')
    
    messages_data = []
    for msg in messages:
        messages_data.append({
            'id': msg.id,
            'type': msg.message_type,
            'content': msg.content,
            'confidence': msg.confidence,
            'key_insights': msg.key_insights,
            'sources': msg.sources,
            'created_at': msg.created_at.isoformat()
        })
    
    return JsonResponse({
        'success': True,
        'session_id': session_id,
        'title': session.title or 'Chat',
        'messages': messages_data
    })


@csrf_exempt
@require_http_methods(["POST"])
def create_chat_session(request):
    """Create a new chat session."""
    if not request.user.is_authenticated:
        return JsonResponse({'success': False, 'error': 'Not authenticated'}, status=401)
    
    new_session_id = str(uuid.uuid4())
    session = ChatSession.objects.create(
        session_id=new_session_id,
        user=request.user,
        title=''
    )
    
    # Update the Django session to use the new chat session
    request.session['chat_session_id'] = new_session_id
    
    return JsonResponse({
        'success': True,
        'session_id': new_session_id,
        'created_at': session.created_at.isoformat()
    })


@csrf_exempt
@require_http_methods(["POST"])
def switch_chat_session(request, session_id):
    """Switch to an existing chat session."""
    if not request.user.is_authenticated:
        return JsonResponse({'success': False, 'error': 'Not authenticated'}, status=401)
    
    try:
        session = ChatSession.objects.get(session_id=session_id, user=request.user)
    except ChatSession.DoesNotExist:
        return JsonResponse({'success': False, 'error': 'Session not found'}, status=404)
    
    # Update the Django session to use this chat session
    request.session['chat_session_id'] = session_id
    
    return JsonResponse({'success': True, 'session_id': session_id})


@csrf_exempt
@require_http_methods(["DELETE"])
def delete_chat_session(request, session_id):
    """Delete a chat session."""
    if not request.user.is_authenticated:
        return JsonResponse({'success': False, 'error': 'Not authenticated'}, status=401)
    
    try:
        session = ChatSession.objects.get(session_id=session_id, user=request.user)
        session.delete()
        
        # If this was the current session, clear it
        if request.session.get('chat_session_id') == session_id:
            del request.session['chat_session_id']
        
        return JsonResponse({'success': True})
    except ChatSession.DoesNotExist:
        return JsonResponse({'success': False, 'error': 'Session not found'}, status=404)


# ============================================================================
# User Preferences & Interests API Endpoints
# ============================================================================

def get_or_create_user_profile(user):
    """Get or create a UserProfile for the given user."""
    try:
        return UserProfile.objects.get(user=user)
    except UserProfile.DoesNotExist:
        return UserProfile.objects.create(
            user=user,
            email=user.email or f"{user.username}@placeholder.com"
        )


@require_http_methods(["GET"])
def get_user_interests(request):
    """Get the current user's interest topics and notification settings."""
    if not request.user.is_authenticated:
        return JsonResponse({'success': False, 'error': 'Not authenticated'}, status=401)
    
    profile = get_or_create_user_profile(request.user)
    
    # Get interest scores from current session if available
    session = get_or_create_session(request)
    interest_scores = session.interest_profile or {}
    
    # Get top interests (score >= 0.2)
    top_interests = [
        {'topic': topic, 'score': round(score, 2)}
        for topic, score in sorted(interest_scores.items(), key=lambda x: x[1], reverse=True)
        if score >= 0.2
    ]
    
    # Get the user's selected topics for notifications
    selected_topics = profile.interested_topics or []
    
    return JsonResponse({
        'success': True,
        'interests': top_interests,
        'selected_topics': selected_topics,
        'email': profile.email,
        'notifications_enabled': profile.email_notifications_enabled,
        'notification_frequency': profile.notification_frequency
    })


@csrf_exempt
@require_http_methods(["POST"])
def update_user_interests(request):
    """Update the user's selected topics for notifications."""
    if not request.user.is_authenticated:
        return JsonResponse({'success': False, 'error': 'Not authenticated'}, status=401)
    
    try:
        data = json.loads(request.body)
        selected_topics = data.get('selected_topics', [])
        
        # Validate topics against allowed categories
        from rag.interests import CATEGORY_NAMES
        valid_topics = [t for t in selected_topics if t in CATEGORY_NAMES]
        
        profile = get_or_create_user_profile(request.user)
        profile.interested_topics = valid_topics
        profile.save(update_fields=['interested_topics', 'updated_at'])
        
        return JsonResponse({
            'success': True,
            'selected_topics': valid_topics
        })
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def update_notification_settings(request):
    """Update the user's email notification settings."""
    if not request.user.is_authenticated:
        return JsonResponse({'success': False, 'error': 'Not authenticated'}, status=401)
    
    try:
        data = json.loads(request.body)
        
        profile = get_or_create_user_profile(request.user)
        
        if 'email' in data and data['email']:
            profile.email = data['email']
        
        if 'notifications_enabled' in data:
            profile.email_notifications_enabled = bool(data['notifications_enabled'])
        
        if 'frequency' in data:
            if data['frequency'] in ['instant', 'daily', 'weekly']:
                profile.notification_frequency = data['frequency']
        
        profile.save()
        
        return JsonResponse({
            'success': True,
            'email': profile.email,
            'notifications_enabled': profile.email_notifications_enabled,
            'notification_frequency': profile.notification_frequency
        })
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=500)
