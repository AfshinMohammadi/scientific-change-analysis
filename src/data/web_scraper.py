"""
Web Scraper Module
==================

Provides tools for collecting non-academic scientific knowledge data
from news sites, blogs, policy documents, and other web sources.

Features:
- Multi-source scraping with unified output format
- Rate limiting and respectful crawling
- Content extraction and cleaning
- RSS feed monitoring
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Iterator
from datetime import datetime
from abc import ABC, abstractmethod
import re
import time
import hashlib
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


@dataclass
class WebDocument:
    """
    Unified document representation from web sources.
    
    Attributes:
        doc_id: Unique document identifier
        title: Document title
        content: Main text content
        url: Source URL
        source: Source website name
        published_date: Publication date
        authors: Author names if available
        keywords: Extracted keywords/tags
        categories: Content categories
        citations: Mentioned papers/references
        links: External links in document
        metadata: Additional metadata
    """
    doc_id: str
    title: str
    content: str
    url: str
    source: str
    published_date: Optional[datetime] = None
    authors: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    citations: List[str] = field(default_factory=list)
    links: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "content": self.content,
            "url": self.url,
            "source": self.source,
            "published_date": self.published_date.isoformat() if self.published_date else None,
            "authors": self.authors,
            "keywords": self.keywords,
            "categories": self.categories,
            "citations": self.citations,
            "links": self.links,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WebDocument':
        """Create from dictionary."""
        if isinstance(data.get('published_date'), str):
            data['published_date'] = datetime.fromisoformat(data['published_date'])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class BaseScraper(ABC):
    """Abstract base class for web scrapers."""
    
    def __init__(
        self,
        rate_limit: float = 1.0,
        timeout: int = 30,
        max_retries: int = 3,
        user_agent: Optional[str] = None
    ):
        """
        Initialize scraper.
        
        Args:
            rate_limit: Seconds between requests
            timeout: Request timeout
            max_retries: Maximum retry attempts
            user_agent: Custom user agent string
        """
        self.rate_limit = rate_limit
        self.timeout = timeout
        self.max_retries = max_retries
        self.user_agent = user_agent or "ScientificChangeAnalysis/1.0 (Research Project)"
        self._last_request_time = 0
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.user_agent})
    
    def _wait_for_rate_limit(self) -> None:
        """Wait to respect rate limiting."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self._last_request_time = time.time()
    
    def fetch_page(self, url: str) -> Optional[str]:
        """Fetch page HTML."""
        for attempt in range(self.max_retries):
            try:
                self._wait_for_rate_limit()
                response = self.session.get(url, timeout=self.timeout)
                
                if response.status_code == 200:
                    return response.text
                elif response.status_code == 429:
                    wait_time = int(response.headers.get('Retry-After', 60))
                    time.sleep(wait_time)
                elif response.status_code == 404:
                    return None
                    
            except requests.RequestException as e:
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise
        
        return None
    
    def extract_text(self, html: str) -> str:
        """Extract readable text from HTML."""
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove script and style elements
        for element in soup(['script', 'style', 'nav', 'footer', 'header']):
            element.decompose()
        
        # Get text
        text = soup.get_text(separator=' ', strip=True)
        
        # Clean whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text
    
    def extract_links(self, html: str, base_url: str) -> List[str]:
        """Extract all links from HTML."""
        soup = BeautifulSoup(html, 'html.parser')
        links = []
        
        for a in soup.find_all('a', href=True):
            href = a['href']
            full_url = urljoin(base_url, href)
            links.append(full_url)
        
        return list(set(links))
    
    def generate_doc_id(self, url: str) -> str:
        """Generate unique document ID from URL."""
        return hashlib.md5(url.encode()).hexdigest()[:12]
    
    @abstractmethod
    def scrape(self, url: str) -> Optional[WebDocument]:
        """Scrape a single URL."""
        pass
    
    @abstractmethod
    def scrape_source(
        self,
        source_config: Dict[str, Any],
        limit: int = 100
    ) -> Iterator[WebDocument]:
        """Scrape documents from a configured source."""
        pass


class ScienceNewsScraper(BaseScraper):
    """
    Scraper for science news websites.
    
    Supported sources:
    - Science News
    - Scientific American
    - Nature News
    - Science Daily
    - EurekAlert!
    """
    
    SOURCE_CONFIGS = {
        'science_news': {
            'base_url': 'https://www.sciencenews.org',
            'article_pattern': r'/article/',
            'title_selector': 'h1',
            'content_selector': 'div.rich-text',
            'date_selector': 'time',
        },
        'scientific_american': {
            'base_url': 'https://www.scientificamerican.com',
            'article_pattern': r'/article/',
            'title_selector': 'h1',
            'content_selector': 'article',
            'date_selector': 'time',
        },
        'nature_news': {
            'base_url': 'https://www.nature.com/news',
            'article_pattern': r'/articles/',
            'title_selector': 'h1',
            'content_selector': 'div.article__body',
            'date_selector': 'time',
        },
        'science_daily': {
            'base_url': 'https://www.sciencedaily.com',
            'article_pattern': r'/releases/',
            'title_selector': 'h1',
            'content_selector': 'div#storyText',
            'date_selector': 'span.date',
        },
    }
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def scrape(self, url: str) -> Optional[WebDocument]:
        """Scrape a science news article."""
        html = self.fetch_page(url)
        if not html:
            return None
        
        soup = BeautifulSoup(html, 'html.parser')
        
        # Determine source
        domain = urlparse(url).netloc
        source = self._identify_source(domain)
        
        # Extract title
        title = self._extract_title(soup)
        
        # Extract content
        content = self.extract_text(html)
        
        # Extract date
        published_date = self._extract_date(soup)
        
        # Extract links
        links = self.extract_links(html, url)
        
        # Extract citations (DOI patterns)
        citations = self._extract_citations(content)
        
        return WebDocument(
            doc_id=self.generate_doc_id(url),
            title=title,
            content=content,
            url=url,
            source=source,
            published_date=published_date,
            links=links,
            citations=citations,
            metadata={'scraped_at': datetime.now().isoformat()}
        )
    
    def _identify_source(self, domain: str) -> str:
        """Identify source from domain."""
        for name, config in self.SOURCE_CONFIGS.items():
            if domain in config['base_url']:
                return name.replace('_', ' ').title()
        return domain
    
    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract article title."""
        # Try common title selectors
        for selector in ['h1', 'h1.headline', 'title']:
            element = soup.select_one(selector)
            if element:
                return element.get_text(strip=True)
        return ""
    
    def _extract_date(self, soup: BeautifulSoup) -> Optional[datetime]:
        """Extract publication date."""
        # Try time element
        time_elem = soup.find('time')
        if time_elem and time_elem.get('datetime'):
            try:
                return datetime.fromisoformat(time_elem['datetime'].rstrip('Z'))
            except:
                pass
        
        # Try meta tag
        meta_date = soup.find('meta', {'property': 'article:published_time'})
        if meta_date and meta_date.get('content'):
            try:
                return datetime.fromisoformat(meta_date['content'].rstrip('Z'))
            except:
                pass
        
        return None
    
    def _extract_citations(self, text: str) -> List[str]:
        """Extract citation-like references from text."""
        citations = []
        
        # DOI pattern
        doi_pattern = r'10\.\d{4,}/[^\s]+'
        citations.extend(re.findall(doi_pattern, text))
        
        # arXiv pattern
        arxiv_pattern = r'arXiv:\d{4}\.\d{4,5}'
        citations.extend(re.findall(arxiv_pattern, text, re.IGNORECASE))
        
        return list(set(citations))
    
    def scrape_source(
        self,
        source_config: Dict[str, Any],
        limit: int = 100
    ) -> Iterator[WebDocument]:
        """Scrape articles from a news source."""
        base_url = source_config['base_url']
        html = self.fetch_page(base_url)
        
        if not html:
            return
        
        soup = BeautifulSoup(html, 'html.parser')
        
        # Find article links
        links = []
        for a in soup.find_all('a', href=True):
            href = a['href']
            if re.search(source_config.get('article_pattern', '/article/'), href):
                full_url = urljoin(base_url, href)
                links.append(full_url)
        
        # Scrape each article
        scraped = set()
        for url in links[:limit]:
            if url not in scraped:
                doc = self.scrape(url)
                if doc:
                    yield doc
                scraped.add(url)


class PolicyDocumentScraper(BaseScraper):
    """
    Scraper for science policy documents.
    
    Sources:
    - NIH News
    - NSF News
    - European Commission Research
    - WHO Publications
    """
    
    SOURCE_CONFIGS = {
        'nih': {
            'base_url': 'https://www.nih.gov/news-events/news-releases',
            'article_pattern': r'/news-events/',
        },
        'nsf': {
            'base_url': 'https://new.nsf.gov/news',
            'article_pattern': r'/news/',
        },
        'eu_research': {
            'base_url': 'https://research-and-innovation.ec.europa.eu/news',
            'article_pattern': r'/news/',
        },
    }
    
    def scrape(self, url: str) -> Optional[WebDocument]:
        """Scrape a policy document."""
        html = self.fetch_page(url)
        if not html:
            return None
        
        soup = BeautifulSoup(html, 'html.parser')
        
        # Extract content
        title = self._extract_title(soup)
        content = self.extract_text(html)
        
        # Extract metadata
        published_date = self._extract_date(soup)
        
        # Extract related research
        citations = self._extract_citations(content)
        links = self.extract_links(html, url)
        
        domain = urlparse(url).netloc
        source = self._identify_source(domain)
        
        return WebDocument(
            doc_id=self.generate_doc_id(url),
            title=title,
            content=content,
            url=url,
            source=source,
            published_date=published_date,
            citations=citations,
            links=links,
            categories=['policy', 'research_funding'],
            metadata={'scraped_at': datetime.now().isoformat()}
        )
    
    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract document title."""
        for selector in ['h1', 'h1.page-title', 'title']:
            element = soup.select_one(selector)
            if element:
                return element.get_text(strip=True)
        return ""
    
    def _extract_date(self, soup: BeautifulSoup) -> Optional[datetime]:
        """Extract publication date."""
        time_elem = soup.find('time')
        if time_elem and time_elem.get('datetime'):
            try:
                return datetime.fromisoformat(time_elem['datetime'].rstrip('Z'))
            except:
                pass
        return None
    
    def _extract_citations(self, text: str) -> List[str]:
        """Extract research references."""
        citations = []
        doi_pattern = r'10\.\d{4,}/[^\s]+'
        citations.extend(re.findall(doi_pattern, text))
        return list(set(citations))
    
    def _identify_source(self, domain: str) -> str:
        """Identify source from domain."""
        if 'nih' in domain:
            return 'NIH'
        elif 'nsf' in domain:
            return 'NSF'
        elif 'ec.europa' in domain:
            return 'EU Research'
        return domain
    
    def scrape_source(
        self,
        source_config: Dict[str, Any],
        limit: int = 100
    ) -> Iterator[WebDocument]:
        """Scrape policy documents from source."""
        base_url = source_config['base_url']
        html = self.fetch_page(base_url)
        
        if not html:
            return
        
        soup = BeautifulSoup(html, 'html.parser')
        
        links = []
        for a in soup.find_all('a', href=True):
            href = a['href']
            if re.search(source_config.get('article_pattern', '/news/'), href):
                full_url = urljoin(base_url, href)
                links.append(full_url)
        
        scraped = set()
        for url in links[:limit]:
            if url not in scraped:
                doc = self.scrape(url)
                if doc:
                    yield doc
                scraped.add(url)


class RSSFeedMonitor:
    """
    Monitor RSS feeds for new scientific content.
    """
    
    def __init__(self, rate_limit: float = 60.0):
        """
        Initialize RSS monitor.
        
        Args:
            rate_limit: Seconds between feed checks
        """
        self.rate_limit = rate_limit
        self.feeds = {}
    
    def add_feed(self, name: str, url: str) -> None:
        """Add an RSS feed to monitor."""
        self.feeds[name] = url
    
    def fetch_feed(self, url: str) -> List[Dict[str, Any]]:
        """Fetch and parse RSS feed."""
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Parse XML (simplified - use feedparser in production)
            import xml.etree.ElementTree as ET
            root = ET.fromstring(response.content)
            
            items = []
            for item in root.findall('.//item'):
                items.append({
                    'title': item.findtext('title', ''),
                    'link': item.findtext('link', ''),
                    'description': item.findtext('description', ''),
                    'pubDate': item.findtext('pubDate', ''),
                })
            
            return items
            
        except Exception as e:
            print(f"Error fetching feed: {e}")
            return []
    
    def monitor(
        self,
        callback: callable,
        interval: float = 300
    ) -> None:
        """
        Monitor feeds for new content.
        
        Args:
            callback: Function to call with new items
            interval: Check interval in seconds
        """
        seen = set()
        
        while True:
            for name, url in self.feeds.items():
                items = self.fetch_feed(url)
                
                for item in items:
                    item_id = hashlib.md5(item['link'].encode()).hexdigest()
                    
                    if item_id not in seen:
                        seen.add(item_id)
                        callback(name, item)
            
            time.sleep(interval)


class UnifiedWebScraper:
    """
    Unified interface for all web scraping operations.
    """
    
    def __init__(self, **kwargs):
        """Initialize unified scraper."""
        self.news_scraper = ScienceNewsScraper(**kwargs)
        self.policy_scraper = PolicyDocumentScraper(**kwargs)
        self.rss_monitor = RSSFeedMonitor()
    
    def scrape_all_sources(
        self,
        source_types: List[str] = ['news', 'policy'],
        limit_per_source: int = 50
    ) -> List[WebDocument]:
        """
        Scrape all configured sources.
        
        Args:
            source_types: Types of sources to scrape
            limit_per_source: Maximum documents per source
            
        Returns:
            List of scraped documents
        """
        documents = []
        
        if 'news' in source_types:
            for name, config in ScienceNewsScraper.SOURCE_CONFIGS.items():
                for doc in self.news_scraper.scrape_source(config, limit_per_source):
                    documents.append(doc)
        
        if 'policy' in source_types:
            for name, config in PolicyDocumentScraper.SOURCE_CONFIGS.items():
                for doc in self.policy_scraper.scrape_source(config, limit_per_source):
                    documents.append(doc)
        
        return documents
    
    def save_documents(
        self,
        documents: List[WebDocument],
        filepath: str
    ) -> None:
        """Save scraped documents to file."""
        import json
        
        with open(filepath, 'w', encoding='utf-8') as f:
            for doc in documents:
                f.write(json.dumps(doc.to_dict()) + '\n')
