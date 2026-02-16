"""
Data Module
===========

Data loading and collection for academic and non-academic sources.
"""

from src.data.academic_loader import (
    AcademicDataLoader,
    SemanticScholarLoader,
    OpenAlexLoader,
    ArxivLoader,
    Paper,
)
from src.data.web_scraper import (
    UnifiedWebScraper,
    ScienceNewsScraper,
    PolicyDocumentScraper,
    WebDocument,
)

__all__ = [
    "AcademicDataLoader",
    "SemanticScholarLoader",
    "OpenAlexLoader",
    "ArxivLoader",
    "Paper",
    "UnifiedWebScraper",
    "ScienceNewsScraper",
    "PolicyDocumentScraper",
    "WebDocument",
]
