"""
Django management command to index articles into the Pinecone vector store.
This command uses the rag package to create embeddings and store them for
semantic search.

Usage:
    python manage.py index_articles
    python manage.py index_articles --new-only
    python manage.py index_articles --article-id abc123
    python manage.py index_articles --batch-size 10
"""

from django.core.management.base import BaseCommand, CommandError
from django.conf import settings

from chatbot.models import Article

# Import RAG from the external package
try:
    from rag import get_rag, ConfigurationError
    RAG_AVAILABLE = True
except ImportError as e:
    RAG_AVAILABLE = False
    import_error = str(e)


class Command(BaseCommand):
    help = 'Index articles in the Pinecone vector store using the RAG system'

    def add_arguments(self, parser):
        parser.add_argument(
            '--new-only',
            action='store_true',
            help='Only index articles that have not been indexed yet'
        )
        parser.add_argument(
            '--article-id',
            type=str,
            help='Index a specific article by ID'
        )
        parser.add_argument(
            '--batch-size',
            type=int,
            default=10,
            help='Number of articles to process in each batch (default: 10)'
        )
        parser.add_argument(
            '--reindex',
            action='store_true',
            help='Force reindex of all articles'
        )

    def handle(self, *args, **options):
        if not RAG_AVAILABLE:
            raise CommandError(f'RAG module not available: {import_error}')

        # Check API keys
        if not settings.OPENAI_API_KEY:
            raise CommandError('OPENAI_API_KEY is not set. Please set it in your .env file.')
        if not settings.PINECONE_API_KEY:
            raise CommandError('PINECONE_API_KEY is not set. Please set it in your .env file.')

        new_only = options['new_only']
        article_id = options['article_id']
        batch_size = options['batch_size']
        reindex = options['reindex']

        try:
            # Initialize RAG system
            self.stdout.write(self.style.NOTICE('Initializing RAG system...'))
            rag = get_rag()
            self.stdout.write(self.style.SUCCESS('RAG system initialized'))

            # Get articles to index
            if article_id:
                articles = Article.objects.filter(article_id=article_id)
                if not articles.exists():
                    raise CommandError(f'Article with ID {article_id} not found')
            elif new_only and not reindex:
                articles = Article.objects.filter(is_indexed=False)
            elif reindex:
                articles = Article.objects.all()
                # Reset indexed status
                articles.update(is_indexed=False)
            else:
                articles = Article.objects.filter(is_indexed=False)

            total = articles.count()
            if total == 0:
                self.stdout.write(self.style.WARNING('No articles to index'))
                return

            self.stdout.write(self.style.NOTICE(f'Indexing {total} article(s)...'))

            success_count = 0
            error_count = 0

            # Process articles in batches
            for i, article in enumerate(articles.iterator()):
                try:
                    self.stdout.write(f'  [{i+1}/{total}] Processing: {article.title[:50]}...')

                    # Check if article has enough content
                    content = article.content
                    if not content or len(content.strip()) < 50:
                        self.stdout.write(self.style.WARNING(
                            f'    Skipping - insufficient content ({len(content) if content else 0} chars)'
                        ))
                        continue

                    # Add article to RAG system
                    doc = rag.add_article(
                        article_id=article.article_id,
                        title=article.title,
                        url=article.url,
                        content=content,
                        source=article.source,
                        author=article.author
                    )

                    # Compute category scores for personalization
                    category_scores = {}
                    try:
                        article_topics = rag.topic_matcher.extract_article_topics(
                            title=article.title,
                            summary=doc.summary,
                            tags=doc.tags
                        )
                        category_scores = article_topics.get_category_scores().to_dict()
                    except Exception as e:
                        self.stdout.write(self.style.WARNING(
                            f'    Could not compute category scores: {e}'
                        ))

                    # Update local article with summary data and category scores
                    article.summary = doc.summary
                    article.category = doc.category
                    article.tags = doc.tags
                    article.key_points = doc.key_points
                    article.category_scores = category_scores
                    article.mark_indexed()

                    success_count += 1
                    self.stdout.write(self.style.SUCCESS(
                        f'    ✓ Indexed as category: {doc.category}'
                    ))

                except Exception as e:
                    error_count += 1
                    self.stdout.write(self.style.ERROR(f'    ✗ Error: {e}'))

                # Progress indicator
                if (i + 1) % batch_size == 0:
                    self.stdout.write(self.style.NOTICE(
                        f'  Progress: {i+1}/{total} ({success_count} success, {error_count} errors)'
                    ))

            # Final summary
            self.stdout.write('')
            self.stdout.write(self.style.SUCCESS(
                f'Indexing completed: {success_count} indexed, {error_count} errors'
            ))

            # Show vector store stats
            try:
                stats = rag.get_stats()
                self.stdout.write(self.style.NOTICE(
                    f'Vector store stats: {stats.get("total_vectors", 0)} total vectors'
                ))
            except Exception as e:
                self.stdout.write(self.style.WARNING(f'Could not get stats: {e}'))

        except ConfigurationError as e:
            raise CommandError(f'Configuration error: {e}')
        except Exception as e:
            raise CommandError(f'Indexing failed: {e}')
