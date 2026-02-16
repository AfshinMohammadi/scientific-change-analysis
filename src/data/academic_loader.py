"""
Academic Data Loader Module
===========================

Provides unified interface for loading academic data from multiple sources
including Semantic Scholar, OpenAlex, arXiv, and PubMed.

Features:
- Unified paper/citation format across sources
- Rate limiting and retry logic
- Incremental loading for large datasets
- Metadata extraction and normalization
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Iterator, Union
from datetime import datetime
from abc import ABC, abstractmethod
import json
import time
import re

import requests
from tqdm import tqdm


@dataclass
class Paper:
    """
    Unified paper representation across data sources.
    
    Attributes:
        paper_id: Unique identifier
        title: Paper title
        authors: List of author names
        year: Publication year
        abstract: Paper abstract
        citations: List of cited paper IDs
        references: List of reference paper IDs
        venue: Publication venue (journal/conference)
        doi: Digital Object Identifier
        fields: List of field/subject labels
        keywords: Extracted keywords
    """
    paper_id: str
    title: str
    authors: List[str] = field(default_factory=list)
    year: Optional[int] = None
    abstract: Optional[str] = None
    citations: List[str] = field(default_factory=list)
    references: List[str] = field(default_factory=list)
    venue: Optional[str] = None
    doi: Optional[str] = None
    fields: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    citation_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "authors": self.authors,
            "year": self.year,
            "abstract": self.abstract,
            "citations": self.citations,
            "references": self.references,
            "venue": self.venue,
            "doi": self.doi,
            "fields": self.fields,
            "keywords": self.keywords,
            "citation_count": self.citation_count,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Paper':
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class BaseDataLoader(ABC):
    """Abstract base class for data loaders."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        rate_limit: float = 1.0,
        max_retries: int = 3,
        timeout: int = 30
    ):
        """
        Initialize data loader.
        
        Args:
            api_key: API key for authenticated access
            rate_limit: Minimum seconds between requests
            max_retries: Maximum retry attempts
            timeout: Request timeout in seconds
        """
        self.api_key = api_key
        self.rate_limit = rate_limit
        self.max_retries = max_retries
        self.timeout = timeout
        self._last_request_time = 0
    
    def _wait_for_rate_limit(self) -> None:
        """Wait to respect rate limiting."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self._last_request_time = time.time()
    
    def _make_request(
        self,
        url: str,
        params: Optional[Dict] = None,
        headers: Optional[Dict] = None
    ) -> Optional[Dict]:
        """Make HTTP request with retry logic."""
        for attempt in range(self.max_retries):
            try:
                self._wait_for_rate_limit()
                
                default_headers = {"User-Agent": "ScientificChangeAnalysis/1.0"}
                if self.api_key:
                    default_headers["api_key"] = self.api_key
                
                if headers:
                    default_headers.update(headers)
                
                response = requests.get(
                    url,
                    params=params,
                    headers=default_headers,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:  # Rate limited
                    wait_time = int(response.headers.get('Retry-After', 60))
                    time.sleep(wait_time)
                elif response.status_code == 404:
                    return None
                else:
                    if attempt < self.max_retries - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff
                    
            except requests.RequestException as e:
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise
        
        return None
    
    @abstractmethod
    def load_paper(self, paper_id: str) -> Optional[Paper]:
        """Load a single paper by ID."""
        pass
    
    @abstractmethod
    def search(
        self,
        query: str,
        limit: int = 100,
        **kwargs
    ) -> Iterator[Paper]:
        """Search for papers matching query."""
        pass
    
    @abstractmethod
    def load_citations(
        self,
        paper_id: str,
        direction: str = "forward"
    ) -> List[str]:
        """Load citations for a paper."""
        pass


class SemanticScholarLoader(BaseDataLoader):
    """
    Loader for Semantic Scholar API.
    
    API Documentation: https://api.semanticscholar.org/
    
    Features:
        - 200M+ papers
        - Citation data
        - Author information
        - Field of study tags
    """
    
    BASE_URL = "https://api.semanticscholar.org/graph/v1"
    
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        super().__init__(api_key=api_key, **kwargs)
        self.fields = "paperId,title,authors,year,abstract,citationCount," \
                      "referenceCount,venue,publicationTypes,fieldsOfStudy," \
                      "externalIds,openAccessPdf"
    
    def load_paper(self, paper_id: str) -> Optional[Paper]:
        """Load a paper by Semantic Scholar ID or DOI."""
        # Handle DOI
        if paper_id.startswith('10.'):
            url = f"{self.BASE_URL}/paper/DOI:{paper_id}"
        else:
            url = f"{self.BASE_URL}/paper/{paper_id}"
        
        data = self._make_request(url, params={"fields": self.fields})
        
        if data:
            return self._parse_paper(data)
        return None
    
    def _parse_paper(self, data: Dict) -> Paper:
        """Parse API response into Paper object."""
        authors = [a.get('name', '') for a in data.get('authors', [])]
        fields = [f.get('s2FieldsOfStudy', []) for f in data.get('fieldsOfStudy', [])]
        fields = list(set([f for sublist in fields for f in sublist]))
        
        return Paper(
            paper_id=data.get('paperId', ''),
            title=data.get('title', ''),
            authors=authors,
            year=data.get('year'),
            abstract=data.get('abstract'),
            citation_count=data.get('citationCount', 0),
            venue=data.get('venue'),
            doi=data.get('externalIds', {}).get('DOI'),
            fields=fields,
            metadata={
                'source': 'semantic_scholar',
                'open_access_pdf': data.get('openAccessPdf'),
                'publication_types': data.get('publicationTypes', []),
            }
        )
    
    def search(
        self,
        query: str,
        limit: int = 100,
        year_range: Optional[tuple] = None,
        venue: Optional[str] = None,
        **kwargs
    ) -> Iterator[Paper]:
        """Search for papers."""
        url = f"{self.BASE_URL}/paper/search"
        
        params = {
            "query": query,
            "limit": min(limit, 100),
            "fields": self.fields,
        }
        
        if year_range:
            params["year"] = f"{year_range[0]}-{year_range[1]}"
        if venue:
            params["venue"] = venue
        
        offset = 0
        total = 0
        
        while total < limit:
            params["offset"] = offset
            data = self._make_request(url, params=params)
            
            if not data or 'data' not in data:
                break
            
            for paper_data in data['data']:
                if total >= limit:
                    break
                yield self._parse_paper(paper_data)
                total += 1
            
            if 'next' not in data:
                break
            offset = data['next']
    
    def load_citations(
        self,
        paper_id: str,
        direction: str = "forward",
        limit: int = 1000
    ) -> List[str]:
        """Load citations for a paper."""
        if direction == "forward":
            endpoint = "citations"
            field = "citingPaper"
        else:
            endpoint = "references"
            field = "citedPaper"
        
        url = f"{self.BASE_URL}/paper/{paper_id}/{endpoint}"
        
        params = {
            "fields": "paperId",
            "limit": min(limit, 1000),
        }
        
        data = self._make_request(url, params=params)
        
        if not data or 'data' not in data:
            return []
        
        return [item[field]['paperId'] for item in data['data'] if field in item]
    
    def load_author_papers(
        self,
        author_id: str,
        limit: int = 100
    ) -> List[Paper]:
        """Load papers by an author."""
        url = f"{self.BASE_URL}/author/{author_id}/papers"
        
        params = {
            "fields": self.fields,
            "limit": min(limit, 1000),
        }
        
        data = self._make_request(url, params=params)
        
        if not data or 'data' not in data:
            return []
        
        return [self._parse_paper(p) for p in data['data']]


class OpenAlexLoader(BaseDataLoader):
    """
    Loader for OpenAlex API.
    
    API Documentation: https://docs.openalex.org/
    
    Features:
        - Open access
        - Comprehensive coverage
        - Works, Authors, Sources, Concepts
    """
    
    BASE_URL = "https://api.openalex.org"
    
    def __init__(self, email: Optional[str] = None, **kwargs):
        # OpenAlex uses email for polite pool
        super().__init__(**kwargs)
        self.email = email
    
    def load_paper(self, paper_id: str) -> Optional[Paper]:
        """Load a paper by OpenAlex ID or DOI."""
        if paper_id.startswith('W'):
            url = f"{self.BASE_URL}/works/{paper_id}"
        elif paper_id.startswith('10.'):
            url = f"{self.BASE_URL}/works/doi:{paper_id}"
        else:
            url = f"{self.BASE_URL}/works/{paper_id}"
        
        headers = {}
        if self.email:
            headers["mailto"] = self.email
        
        data = self._make_request(url, headers=headers)
        
        if data:
            return self._parse_work(data)
        return None
    
    def _parse_work(self, data: Dict) -> Paper:
        """Parse OpenAlex work into Paper object."""
        authors = [
            a.get('author', {}).get('display_name', '')
            for a in data.get('authorships', [])
        ]
        
        concepts = [
            c.get('display_name', '')
            for c in data.get('concepts', [])
            if c.get('score', 0) > 0.3
        ]
        
        year = data.get('publication_year')
        
        return Paper(
            paper_id=data.get('id', '').split('/')[-1],
            title=data.get('title', ''),
            authors=authors,
            year=year,
            abstract=data.get('abstract'),
            citation_count=data.get('cited_by_count', 0),
            venue=data.get('primary_location', {}).get('source', {}).get('display_name'),
            doi=data.get('doi'),
            fields=concepts,
            metadata={
                'source': 'openalex',
                'type': data.get('type'),
                'open_access': data.get('open_access'),
                'keywords': data.get('keywords', []),
            }
        )
    
    def search(
        self,
        query: str,
        limit: int = 100,
        filter_params: Optional[Dict] = None,
        **kwargs
    ) -> Iterator[Paper]:
        """Search for works."""
        url = f"{self.BASE_URL}/works"
        
        params = {
            "search": query,
            "per_page": min(limit, 200),
        }
        
        if filter_params:
            filter_str = ",".join(f"{k}:{v}" for k, v in filter_params.items())
            params["filter"] = filter_str
        
        headers = {}
        if self.email:
            headers["mailto"] = self.email
        
        cursor = "*"
        total = 0
        
        while total < limit:
            params["cursor"] = cursor
            data = self._make_request(url, params=params, headers=headers)
            
            if not data or 'results' not in data:
                break
            
            for work_data in data['results']:
                if total >= limit:
                    break
                yield self._parse_work(work_data)
                total += 1
            
            cursor = data.get('meta', {}).get('next_cursor')
            if not cursor:
                break
    
    def load_citations(
        self,
        paper_id: str,
        direction: str = "forward",
        limit: int = 200
    ) -> List[str]:
        """Load citations using OpenAlex."""
        if direction == "forward":
            filter_key = "cites"
        else:
            filter_key = "referenced_by"
        
        url = f"{self.BASE_URL}/works"
        params = {
            "filter": f"{filter_key}:{paper_id}",
            "per_page": min(limit, 200),
            "select": "id",
        }
        
        data = self._make_request(url, params=params)
        
        if not data or 'results' not in data:
            return []
        
        return [w['id'].split('/')[-1] for w in data['results']]


class ArxivLoader(BaseDataLoader):
    """
    Loader for arXiv preprint server.
    
    API Documentation: https://arxiv.org/help/api/
    """
    
    BASE_URL = "http://export.arxiv.org/api/query"
    
    def __init__(self, **kwargs):
        super().__init__(rate_limit=3.0, **kwargs)  # arXiv has stricter limits
    
    def load_paper(self, paper_id: str) -> Optional[Paper]:
        """Load a paper by arXiv ID."""
        params = {
            "id_list": paper_id,
            "max_results": 1,
        }
        
        response = requests.get(self.BASE_URL, params=params, timeout=self.timeout)
        
        if response.status_code != 200:
            return None
        
        # Parse XML response (simplified)
        papers = self._parse_xml_response(response.text)
        return papers[0] if papers else None
    
    def _parse_xml_response(self, xml_text: str) -> List[Paper]:
        """Parse arXiv XML response."""
        import xml.etree.ElementTree as ET
        
        papers = []
        root = ET.fromstring(xml_text)
        
        ns = {'atom': 'http://www.w3.org/2005/Atom',
              'arxiv': 'http://arxiv.org/schemas/atom'}
        
        for entry in root.findall('atom:entry', ns):
            paper_id = entry.find('atom:id', ns).text.split('/')[-1]
            title = entry.find('atom:title', ns).text.strip()
            abstract = entry.find('atom:summary', ns).text.strip()
            
            authors = [
                a.find('atom:name', ns).text
                for a in entry.findall('atom:author', ns)
            ]
            
            # Extract year from published date
            published = entry.find('atom:published', ns).text
            year = int(published[:4]) if published else None
            
            categories = [
                c.get('term')
                for c in entry.findall('atom:category', ns)
            ]
            
            papers.append(Paper(
                paper_id=paper_id,
                title=title,
                authors=authors,
                year=year,
                abstract=abstract,
                fields=categories,
                metadata={'source': 'arxiv'}
            ))
        
        return papers
    
    def search(
        self,
        query: str,
        limit: int = 100,
        categories: Optional[List[str]] = None,
        **kwargs
    ) -> Iterator[Paper]:
        """Search arXiv papers."""
        if categories:
            query = f"cat:{','.join(categories)} AND {query}"
        
        params = {
            "search_query": f"all:{query}",
            "max_results": min(limit, 2000),
            "start": 0,
        }
        
        while True:
            response = requests.get(self.BASE_URL, params=params, timeout=self.timeout)
            
            if response.status_code != 200:
                break
            
            papers = self._parse_xml_response(response.text)
            
            if not papers:
                break
            
            for paper in papers:
                yield paper
            
            if len(papers) < params["max_results"]:
                break
            
            params["start"] += params["max_results"]
    
    def load_citations(self, paper_id: str, direction: str = "forward") -> List[str]:
        """Not directly supported by arXiv API."""
        raise NotImplementedError("arXiv does not provide citation data")


class AcademicDataLoader:
    """
    Unified interface for loading academic data from multiple sources.
    
    Provides automatic source selection and unified paper format.
    """
    
    SOURCE_LOADERS = {
        'semantic_scholar': SemanticScholarLoader,
        'openalex': OpenAlexLoader,
        'arxiv': ArxivLoader,
    }
    
    def __init__(
        self,
        source: str = 'semantic_scholar',
        api_key: Optional[str] = None,
        email: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize unified data loader.
        
        Args:
            source: Primary data source
            api_key: API key for authenticated access
            email: Email for OpenAlex polite pool
        """
        self.source = source
        loader_class = self.SOURCE_LOADERS.get(source, SemanticScholarLoader)
        
        loader_kwargs = kwargs.copy()
        if api_key:
            loader_kwargs['api_key'] = api_key
        if email and source == 'openalex':
            loader_kwargs['email'] = email
        
        self.loader = loader_class(**loader_kwargs)
    
    def load_paper(self, paper_id: str) -> Optional[Paper]:
        """Load a single paper."""
        return self.loader.load_paper(paper_id)
    
    def search(
        self,
        query: str,
        limit: int = 100,
        **kwargs
    ) -> List[Paper]:
        """Search for papers."""
        return list(self.loader.search(query, limit, **kwargs))
    
    def load_from_file(
        self,
        filepath: str,
        format: str = 'jsonl'
    ) -> List[Paper]:
        """
        Load papers from a file.
        
        Args:
            filepath: Path to data file
            format: File format ('jsonl', 'json', 'csv')
            
        Returns:
            List of Paper objects
        """
        papers = []
        
        if format == 'jsonl':
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in tqdm(f, desc="Loading papers"):
                    if line.strip():
                        data = json.loads(line)
                        papers.append(Paper.from_dict(data))
        
        elif format == 'json':
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in tqdm(data, desc="Loading papers"):
                    papers.append(Paper.from_dict(item))
        
        return papers
    
    def save_to_file(
        self,
        papers: List[Paper],
        filepath: str,
        format: str = 'jsonl'
    ) -> None:
        """Save papers to file."""
        if format == 'jsonl':
            with open(filepath, 'w', encoding='utf-8') as f:
                for paper in papers:
                    f.write(json.dumps(paper.to_dict()) + '\n')
        
        elif format == 'json':
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump([p.to_dict() for p in papers], f, indent=2)
