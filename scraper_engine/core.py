import requests
from bs4 import BeautifulSoup
import json
import time
import os
from typing import List, Dict, Optional

class HackerNewsScraper:
    """
    A scraper for The Hacker News (https://thehackernews.com/).
    
    This class handles fetching, parsing, and extracting article data including
    titles, dates, authors, topics, and full content.
    """

    def __init__(self, base_url: str = "https://thehackernews.com/"):
        """
        Initialize the scraper.

        Args:
            base_url (str): The base URL of the website to scrape. Defaults to "https://thehackernews.com/".
        """
        self.base_url = base_url
        self.headers = {'User-Agent': 'Mozilla/5.0'}

    def _get_soup(self, url: str) -> Optional[BeautifulSoup]:
        """
        Internal helper to fetch a URL and return a BeautifulSoup object.

        Args:
            url (str): The URL to fetch.

        Returns:
            Optional[BeautifulSoup]: The parsed HTML, or None if the request failed.
        """
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return BeautifulSoup(response.content, 'html.parser')
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            return None

    @staticmethod
    def _clean_content(article_body: BeautifulSoup) -> str:
        """
        Internal static helper to clean the article body HTML and extract text.

        Removes unwanted elements (ads, scripts, share buttons), unwraps links,
        and preserves paragraph structure.

        Args:
            article_body (BeautifulSoup): The BeautifulSoup tag containing the article body.

        Returns:
            str: The cleaned, plain text content of the article.
        """
        if not article_body:
            return ""
        
        # Remove unwanted elements
        for selector in [
            'script', 'style', 
            'div.note-b', # Footer "Found this article interesting?"
            'div.sharebelow', 'div.float-share', # Share buttons
            'div.dog_two', # Ads
            'div.separator', # Main image container
            'div.stophere', # Marker often found before footer
            'div.ad_one_one', # Another potential ad container
            'div.check_two_webinar' # Webinar promos
        ]:
            for tag in article_body.select(selector):
                tag.decompose()

        # Unwrap links instead of replace_with to preserve inline text flow
        for a in article_body.find_all('a'):
            a.unwrap()

        # Insert newlines after block elements to preserve structure
        for block in article_body.find_all(['p', 'div', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'blockquote']):
            block.insert_after('\n')

        text = article_body.get_text(separator='', strip=False)
        
        # Post-processing to clean up excessive newlines and spaces
        lines = [line.strip() for line in text.split('\n')]
        clean_text = '\n'.join(line for line in lines if line)
        
        return clean_text

    def scrape(self, pages: int = 1, save_to_file: bool = False, output_file: str = 'scraped_articles.json') -> List[Dict]:
        """
        Scrape articles from The Hacker News.

        Args:
            pages (int): Number of pages to scrape. Defaults to 1.
            save_to_file (bool): Whether to save the results to a JSON file. Defaults to False.
            output_file (str): The path to the output JSON file if save_to_file is True. 
                               Defaults to 'scraped_articles.json'.

        Returns:
            List[Dict]: A list of dictionaries, where each dictionary represents an article 
                        and contains keys: 'title', 'date', 'author', 'topics', 'content', 'url'.

        Example:
            >>> scraper = HackerNewsScraper()
            >>> articles = scraper.scrape(pages=1)
            >>> print(articles[0]['title'])
        """
        current_url = self.base_url
        articles = []
        
        for page in range(1, pages + 1):
            print(f"--- Scraping Page {page}/{pages}: {current_url} ---")
            soup = self._get_soup(current_url)
            if not soup:
                break

            # Find article links
            story_links = soup.select('a.story-link')
            print(f"Found {len(story_links)} articles on page {page}.")

            for i, link in enumerate(story_links):
                article_url = link.get('href')
                if not article_url:
                    continue
                
                print(f"Scraping ({i+1}/{len(story_links)}): {article_url}")
                article_soup = self._get_soup(article_url)
                if not article_soup:
                    continue

                # Extract Metadata
                title_tag = article_soup.select_one('h1.story-title')
                title = title_tag.get_text(strip=True) if title_tag else "No Title"

                if title == "No Title":
                    print(f"Skipping article with no title: {article_url}")
                    continue

                # Date and Author are in div.postmeta span.author
                meta_spans = article_soup.select('div.postmeta span.author')
                date = meta_spans[0].get_text(strip=True) if len(meta_spans) > 0 else "No Date"
                author = meta_spans[1].get_text(strip=True) if len(meta_spans) > 1 else "No Author"

                # Topics
                topics_tag = article_soup.select_one('div.postmeta span.p-tags')
                topics_text = topics_tag.get_text(strip=True) if topics_tag else ""
                topics = [t.strip() for t in topics_text.split('/') if t.strip()]

                # Content
                article_body = article_soup.select_one('div.articlebody')
                content = self._clean_content(article_body)

                articles.append({
                    "title": title,
                    "date": date,
                    "author": author,
                    "topics": topics,
                    "content": content,
                    "url": article_url
                })
                
                # Rate limiting for articles
                time.sleep(0.5)

            # Find Next Page Link
            next_link = soup.select_one('a.blog-pager-older-link-mobile')
            if next_link and page < pages:
                current_url = next_link.get('href')
                print(f"Moving to next page: {current_url}")
                time.sleep(1) # Rate limiting for pages
            else:
                if page < pages:
                    print("No more pages found.")
                break

        # --- Output Logic ---
        if save_to_file:
            # Ensure directory exists
            directory = os.path.dirname(output_file)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
                
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(articles, f, indent=4, ensure_ascii=False)
            print(f"Saved {len(articles)} articles to {output_file}")

        return articles

if __name__ == "__main__":
    print("Running HackerNewsScraper dry run (1 page, no save)...")
    scraper = HackerNewsScraper()
    data = scraper.scrape(pages=1, save_to_file=False)
    print(f"Dry run complete. Found {len(data)} articles.")
    if len(data) > 0:
        print(f"Sample Article Title: {data[0]['title']}")
