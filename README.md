# RelevantAI - Cybersecurity Intelligence Chatbot

An AI-powered chatbot that scrapes cybersecurity news from The Hacker News, summarizes articles using GPT-4o, and answers questions using RAG (Retrieval-Augmented Generation) with Pinecone vector search.

## ✨ Features

- 🔍 **Semantic Search** - Find relevant articles using natural language queries
- 🤖 **AI-Powered Q&A** - Get intelligent answers with cited sources
- 📰 **Auto-Summarization** - Articles summarized with key points and categories
- 🔄 **Query Expansion** - Enhanced search using GPT-generated synonyms
- 📊 **Confidence Scoring** - Know how reliable each answer is

## 🏗️ Architecture

```
sheep/
├── rag/                    # RAG Package - AI & Vector Store
│   ├── rag.py             # Main RAG interface (RelevantAI class)
│   ├── summarizer.py      # GPT-4o article summarization
│   ├── vector_store.py    # Pinecone vector database
│   ├── models.py          # Pydantic data models
│   ├── exceptions.py      # Custom exceptions
│   └── logger.py          # Logging utilities
│
├── scraper_engine/         # Scraper Package
│   └── core.py            # HackerNewsScraper class
│
├── relevantai/             # Django Project
│   ├── chatbot/           # Main chatbot app
│   │   ├── views.py       # API endpoints
│   │   ├── models.py      # Django models
│   │   ├── urls.py        # URL routing
│   │   ├── templates/     # Chat interface
│   │   └── management/    # CLI commands
│   │       └── commands/
│   │           ├── scrape_articles.py
│   │           ├── index_articles.py
│   │           └── rag_stats.py
│   └── relevantai/        # Django config
│
├── requirements.txt        # Python dependencies
├── .env.example           # Environment template
└── pyproject.toml         # Package configuration
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy the environment template
cp .env.example .env

# Edit .env and add your API keys:
# - OPENAI_API_KEY (required)
# - PINECONE_API_KEY (required)
```

### 3. Initialize Database

```bash
cd relevantai
python manage.py migrate
```

### 4. Scrape & Index Articles

```bash
# Scrape articles from The Hacker News and index them
python manage.py scrape_articles --pages 3 --index

# Or separately:
python manage.py scrape_articles --pages 5
python manage.py index_articles
```

### 5. Run the Server

```bash
python manage.py runserver
```

Visit http://127.0.0.1:8000 to use the chatbot.

## 🎯 Usage

1. **Ask Questions**: Type natural language questions about cybersecurity
2. **Search Articles**: Find specific topics in the indexed news
3. **View Sources**: Click on cited articles to read the full content

Example queries:
- "What are the latest ransomware threats?"
- "Tell me about AI-powered malware"
- "Recent vulnerabilities in Microsoft products"

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Chat interface |
| `/api/articles/` | POST | Search articles (legacy) |
| `/api/ask/` | POST | Ask a question (RAG) |
| `/api/search/` | POST | Semantic search |
| `/api/article/<id>/` | GET | Article details |
| `/api/stats/` | GET | System statistics |
| `/api/categories/` | GET | Available categories |

### Example: Ask a Question

```bash
curl -X POST http://127.0.0.1:8000/api/ask/ \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the latest ransomware threats?"}'
```

Response:
```json
{
  "success": true,
  "answer": "Based on recent articles...",
  "confidence": "high",
  "key_insights": ["Insight 1", "Insight 2"],
  "sources": [
    {
      "title": "Article Title",
      "url": "https://...",
      "summary": "...",
      "relevance_score": 0.95
    }
  ],
  "mode": "rag"
}
```

## 🛠️ Management Commands

### Scrape Articles
```bash
# Scrape 3 pages
python manage.py scrape_articles --pages 3

# Scrape and immediately index
python manage.py scrape_articles --pages 5 --index

# Force re-scrape existing articles
python manage.py scrape_articles --pages 2 --force
```

### Index Articles
```bash
# Index new articles only
python manage.py index_articles --new-only

# Reindex all articles
python manage.py index_articles --reindex

# Index specific article
python manage.py index_articles --article-id abc123
```

### View Statistics
```bash
python manage.py rag_stats
```

## 🔧 Configuration

### Django Settings (`relevantai/relevantai/settings.py`)

```python
# RAG Configuration
RAG_CONFIG = {
    'top_k': 5,              # Articles to retrieve
    'use_query_expansion': True,  # Enable semantic expansion
    'scraper_pages': 3,      # Default pages to scrape
}
```

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | Yes | - | OpenAI API key |
| `PINECONE_API_KEY` | Yes | - | Pinecone API key |
| `PINECONE_INDEX_NAME` | No | `hn-articles` | Vector index name |

## 🧪 How It Works

1. **Scraping**: `HackerNewsScraper` fetches articles from thehackernews.com
2. **Summarization**: GPT-4o creates summaries, key points, and categories
3. **Indexing**: Articles are embedded and stored in Pinecone
4. **Query Expansion**: User questions are enhanced with related terms
5. **Semantic Search**: Relevant articles are retrieved via vector similarity
6. **Answer Generation**: GPT-4o synthesizes an answer from sources

## 📦 Tech Stack

- **Django 4.2+** - Web framework
- **LangChain** - AI orchestration
- **OpenAI GPT-4o** - Summarization & Q&A
- **Pinecone** - Vector database
- **BeautifulSoup4** - Web scraping
- **Pydantic** - Data validation

## 📝 License

MIT License
