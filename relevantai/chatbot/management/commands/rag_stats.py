"""
Django management command to show RAG system statistics.

Usage:
    python manage.py rag_stats
"""

from django.core.management.base import BaseCommand, CommandError
from django.conf import settings

from chatbot.models import Article, ChatSession, ChatMessage, ScrapingRun

# Import RAG from the external package
try:
    from rag import get_rag, ConfigurationError
    RAG_AVAILABLE = True
except ImportError as e:
    RAG_AVAILABLE = False
    import_error = str(e)


class Command(BaseCommand):
    help = 'Show statistics for the RAG system and local database'

    def handle(self, *args, **options):
        self.stdout.write(self.style.NOTICE('=== RelevantAI System Statistics ===\n'))

        # Database statistics
        self.stdout.write(self.style.NOTICE('📊 Database Statistics:'))
        self.stdout.write(f'  Total articles: {Article.objects.count()}')
        self.stdout.write(f'  Indexed articles: {Article.objects.filter(is_indexed=True).count()}')
        self.stdout.write(f'  Pending indexing: {Article.objects.filter(is_indexed=False).count()}')
        self.stdout.write(f'  Chat sessions: {ChatSession.objects.count()}')
        self.stdout.write(f'  Chat messages: {ChatMessage.objects.count()}')
        self.stdout.write('')

        # Category breakdown
        self.stdout.write(self.style.NOTICE('📁 Articles by Category:'))
        categories = Article.objects.filter(is_indexed=True).values_list('category', flat=True)
        category_counts = {}
        for cat in categories:
            category_counts[cat] = category_counts.get(cat, 0) + 1
        for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
            self.stdout.write(f'  {cat}: {count}')
        self.stdout.write('')

        # Scraping history
        self.stdout.write(self.style.NOTICE('🔄 Recent Scraping Runs:'))
        runs = ScrapingRun.objects.all()[:5]
        if runs:
            for run in runs:
                status_style = self.style.SUCCESS if run.status == 'completed' else self.style.ERROR
                self.stdout.write(
                    f'  [{run.started_at.strftime("%Y-%m-%d %H:%M")}] '
                    f'{status_style(run.status.upper())} - '
                    f'{run.articles_found} found, {run.articles_indexed} indexed'
                )
        else:
            self.stdout.write('  No scraping runs yet')
        self.stdout.write('')

        # RAG system statistics
        if not RAG_AVAILABLE:
            self.stdout.write(self.style.WARNING(f'⚠️  RAG module not available: {import_error}'))
            return

        if not settings.OPENAI_API_KEY or not settings.PINECONE_API_KEY:
            self.stdout.write(self.style.WARNING('⚠️  API keys not configured'))
            return

        try:
            self.stdout.write(self.style.NOTICE('🔍 Vector Store Statistics:'))
            rag = get_rag()
            stats = rag.get_stats()

            self.stdout.write(f'  Index name: {stats.get("index_name", "N/A")}')
            self.stdout.write(f'  Total vectors: {stats.get("total_vectors", 0)}')
            self.stdout.write(f'  Dimension: {stats.get("dimension", "N/A")}')

            namespaces = stats.get('namespaces', {})
            if namespaces:
                self.stdout.write('  Namespaces:')
                for ns, info in namespaces.items():
                    self.stdout.write(f'    {ns or "(default)"}: {info.get("vector_count", 0)} vectors')

        except ConfigurationError as e:
            self.stdout.write(self.style.ERROR(f'❌ Configuration error: {e}'))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'❌ Error fetching stats: {e}'))

        self.stdout.write('')
        self.stdout.write(self.style.SUCCESS('=== End of Statistics ==='))
