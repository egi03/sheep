"""
Django management command to scrape articles from The Hacker News.
This command uses the scraper_engine package to fetch articles and stores them
in the local database for later indexing.

Usage:
    python manage.py scrape_articles --pages 3
    python manage.py scrape_articles --pages 5 --index
"""

import hashlib
from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone

from chatbot.models import Article, ScrapingRun

# Import scraper from the external package
try:
    from scraper_engine import HackerNewsScraper
    SCRAPER_AVAILABLE = True
except ImportError as e:
    SCRAPER_AVAILABLE = False
    import_error = str(e)


class Command(BaseCommand):
    help = 'Scrape articles from The Hacker News and store them in the database'

    def add_arguments(self, parser):
        parser.add_argument(
            '--pages',
            type=int,
            default=3,
            help='Number of pages to scrape (default: 3)'
        )
        parser.add_argument(
            '--index',
            action='store_true',
            help='Also index the articles in the vector store after scraping'
        )
        parser.add_argument(
            '--force',
            action='store_true',
            help='Force re-scrape of existing articles'
        )

    def handle(self, *args, **options):
        if not SCRAPER_AVAILABLE:
            raise CommandError(f'Scraper module not available: {import_error}')

        pages = options['pages']
        should_index = options['index']
        force = options['force']

        self.stdout.write(self.style.NOTICE(f'Starting scrape of {pages} page(s)...'))

        # Create scraping run record
        run = ScrapingRun.objects.create(
            status='running',
            pages_scraped=pages
        )

        try:
            # Initialize scraper
            scraper = HackerNewsScraper()

            # Scrape articles
            self.stdout.write('Fetching articles from thehackernews.com...')
            scraped_articles = scraper.scrape(pages=pages, save_to_file=False)

            self.stdout.write(self.style.SUCCESS(f'Scraped {len(scraped_articles)} articles'))

            # Store articles in database
            new_count = 0
            updated_count = 0

            for article_data in scraped_articles:
                # Generate unique ID from URL
                article_id = hashlib.md5(article_data['url'].encode()).hexdigest()[:16]

                # Check if article already exists
                existing = Article.objects.filter(article_id=article_id).first()

                if existing and not force:
                    self.stdout.write(f'  Skipping existing: {article_data["title"][:50]}...')
                    continue

                # Create or update article
                article, created = Article.objects.update_or_create(
                    article_id=article_id,
                    defaults={
                        'title': article_data['title'],
                        'url': article_data['url'],
                        'author': article_data.get('author', ''),
                        'content': article_data.get('content', ''),
                        'source': 'thehackernews.com',
                        'topics': article_data.get('topics', []),
                        'published_date': article_data.get('date', ''),
                        'is_indexed': False,  # Will be indexed separately
                    }
                )

                if created:
                    new_count += 1
                    self.stdout.write(f'  Added: {article_data["title"][:50]}...')
                else:
                    updated_count += 1
                    self.stdout.write(f'  Updated: {article_data["title"][:50]}...')

            self.stdout.write(self.style.SUCCESS(
                f'Database updated: {new_count} new, {updated_count} updated'
            ))

            # Index articles if requested
            indexed_count = 0
            if should_index:
                self.stdout.write(self.style.NOTICE('Indexing articles in vector store...'))
                from django.core.management import call_command
                call_command('index_articles', '--new-only')
                indexed_count = Article.objects.filter(is_indexed=True).count()

            # Mark run as completed
            run.mark_completed(
                articles_found=len(scraped_articles),
                articles_indexed=indexed_count
            )

            self.stdout.write(self.style.SUCCESS(
                f'Scraping completed successfully!'
            ))

        except Exception as e:
            run.mark_failed(str(e))
            raise CommandError(f'Scraping failed: {e}')
