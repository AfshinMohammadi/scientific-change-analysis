"""
Scientific Change Analysis
==========================

A framework for modeling the evolution of scientific knowledge
through network science and machine learning.

Modules:
    - data: Data loading and web scraping
    - networks: Network construction and analysis
    - models: Paradigm detection and prediction models
    - utils: Visualization and utilities
"""

__version__ = "1.0.0"
__author__ = "Afshin Mohammadi"

from src.data import AcademicDataLoader, WebScraper
from src.networks import CitationNetwork, CoAuthorshipNetwork
from src.models import ParadigmShiftDetector, GraphEmbedder

__all__ = [
    "AcademicDataLoader",
    "WebScraper",
    "CitationNetwork",
    "CoAuthorshipNetwork",
    "ParadigmShiftDetector",
    "GraphEmbedder",
]
