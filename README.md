## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/egi03/sheep.git
cd sheep
```

### 2. Create and Activate Python Virtual Environment

#### On macOS/Linux:

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

#### On Windows:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# Alternatively, install from pyproject.toml
pip install -e .
```

### 4. Configure Environment Variables

```bash
# Copy example environment file (if available)
cp .env.example .env

# Edit .env with your API keys
nano .env
```

**Required environment variables:**

```
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_ENVIRONMENT=your-pinecone-environment
PINECONE_INDEX=your-index-name

# Optional
DEBUG=True
ALLOWED_HOSTS=localhost,127.0.0.1
```

### 5. Setup Django Database

```bash
cd relevantai

# Run migrations
python manage.py migrate

# Create superuser (optional, for admin panel)
python manage.py createsuperuser
```

### 6. Scrape and Index Articles

```bash
# Scrape articles and index them
python manage.py scrape_articles --pages 3 --index

# Alternative: Just scrape without indexing
python manage.py scrape_articles --pages 3

# Check RAG statistics
python manage.py rag_stats
```