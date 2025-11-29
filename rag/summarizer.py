"""Article summarization using OpenAI GPT-4o."""

import os
import time
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from .models import ArticleSummary
from .exceptions import SummarizationError, RateLimitError, ConfigurationError
from .logger import get_logger

logger = get_logger("relevantai.summarizer")


class Summarizer:
    SYSTEM_PROMPT = """You are an expert summarizer. Create concise summaries with key points.
Categorize as: AI/ML, Security, Programming, DevOps, Business, or Other.
Extract 3-5 relevant tags."""

    USER_PROMPT = """Summarize this article:

TITLE: {title}

CONTENT:
{content}

Provide: 1) 2-3 sentence summary, 2) 3-5 key points, 3) Category, 4) Tags"""

    def __init__(self, model: str = "gpt-4o", temperature: float = 0.3, 
                 max_content_length: int = 15000, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ConfigurationError("OPENAI_API_KEY not found", missing_config="OPENAI_API_KEY")
        
        self.model = model
        self.temperature = temperature
        self.max_content_length = max_content_length
        self._init_chain()

    def _init_chain(self):
        self.llm = ChatOpenAI(
            model=self.model, temperature=self.temperature, api_key=self.api_key
        ).with_structured_output(ArticleSummary)
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.SYSTEM_PROMPT),
            ("human", self.USER_PROMPT)
        ])
        self.chain = self.prompt | self.llm

    def _truncate_content(self, content: str) -> str:
        if len(content) <= self.max_content_length:
            return content
        keep = self.max_content_length // 2
        return f"{content[:keep]}\n\n[...truncated...]\n\n{content[-keep:]}"

    def summarize(self, title: str, content: str, max_retries: int = 3) -> ArticleSummary:
        if not content or len(content.strip()) < 50:
            raise SummarizationError("Content too short", title=title, content_length=len(content) if content else 0)
        
        truncated = self._truncate_content(content)
        
        for attempt in range(max_retries):
            try:
                return self.chain.invoke({"title": title, "content": truncated})
            except Exception as e:
                err_str = str(e).lower()
                
                if "rate" in err_str or "429" in err_str:
                    if attempt < max_retries - 1:
                        wait = (2 ** attempt) * 2
                        logger.warning(f"Rate limited, waiting {wait}s...")
                        time.sleep(wait)
                        continue
                    raise RateLimitError(f"Rate limit exceeded: {e}", service="openai")
                
                if "api" in err_str or "auth" in err_str:
                    raise SummarizationError(f"API error: {e}", title=title, content_length=len(content))
                
                if attempt < max_retries - 1:
                    wait = 2 ** attempt
                    logger.warning(f"Attempt {attempt + 1} failed: {e}, retrying in {wait}s...")
                    time.sleep(wait)
                    continue
                
                raise SummarizationError(f"Summarization failed: {e}", title=title, content_length=len(content))
        
        raise SummarizationError("Max retries exceeded", title=title, content_length=len(content))
