"""
Citation Network Module
=======================

Construction and analysis of citation networks from academic papers.
Supports large-scale network building, temporal slicing, and
multi-layer network construction.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Set, Iterator
from collections import defaultdict
from datetime import datetime
import time

import networkx as nx
import numpy as np
from tqdm import tqdm


@dataclass
class NetworkStats:
    """Statistics for a citation network."""
    n_nodes: int = 0
    n_edges: int = 0
    n_components: int = 0
    density: float = 0.0
    avg_clustering: float = 0.0
    avg_path_length: Optional[float] = None
    modularity: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_nodes": self.n_nodes,
            "n_edges": self.n_edges,
            "n_components": self.n_components,
            "density": self.density,
            "avg_clustering": self.avg_clustering,
            "avg_path_length": self.avg_path_length,
            "modularity": self.modularity,
        }


@dataclass
class TemporalSlice:
    """A temporal slice of the citation network."""
    year: int
    graph: nx.DiGraph
    stats: NetworkStats
    papers: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "year": self.year,
            "stats": self.stats.to_dict(),
            "n_papers": len(self.papers),
        }


class CitationNetwork:
    """
    Citation network constructed from academic papers.
    
    Features:
    - Large-scale network construction
    - Temporal slicing for evolution analysis
    - Multi-layer networks (citations + co-authorship + concepts)
    - Efficient storage and retrieval
    
    The network is a directed graph where:
    - Nodes represent papers
    - Edges represent citations (A -> B means A cites B)
    
    Example:
        >>> network = CitationNetwork()
        >>> network.add_papers(papers)
        >>> network.build_citation_edges()
        >>> stats = network.get_statistics()
    """
    
    def __init__(
        self,
        directed: bool = True,
        min_citations: int = 0,
        max_nodes: Optional[int] = None
    ):
        """
        Initialize citation network.
        
        Args:
            directed: Whether to create directed graph
            min_citations: Minimum citations for paper inclusion
            max_nodes: Maximum number of nodes
        """
        self.directed = directed
        self.min_citations = min_citations
        self.max_nodes = max_nodes
        
        # Create graph
        self.graph = nx.DiGraph() if directed else nx.Graph()
        
        # Paper metadata storage
        self.papers: Dict[str, Dict] = {}
        
        # Temporal slices
        self.slices: Dict[int, TemporalSlice] = {}
        
        # Index structures
        self._year_index: Dict[int, Set[str]] = defaultdict(set)
        self._field_index: Dict[str, Set[str]] = defaultdict(set)
        self._author_index: Dict[str, Set[str]] = defaultdict(set)
    
    def add_paper(self, paper: Any) -> None:
        """
        Add a single paper to the network.
        
        Args:
            paper: Paper object with paper_id, title, year, etc.
        """
        paper_id = paper.paper_id
        
        # Skip if below citation threshold
        if hasattr(paper, 'citation_count'):
            if paper.citation_count < self.min_citations:
                return
        
        # Add node with attributes
        self.graph.add_node(
            paper_id,
            title=paper.title,
            year=paper.year,
            authors=paper.authors,
            venue=paper.venue,
            fields=paper.fields,
            citation_count=getattr(paper, 'citation_count', 0),
        )
        
        # Store full paper data
        self.papers[paper_id] = paper.to_dict() if hasattr(paper, 'to_dict') else paper.__dict__
        
        # Update indices
        if paper.year:
            self._year_index[paper.year].add(paper_id)
        
        for field in (paper.fields or []):
            self._field_index[field].add(paper_id)
        
        for author in (paper.authors or []):
            self._author_index[author].add(paper_id)
    
    def add_papers(self, papers: List[Any], show_progress: bool = True) -> None:
        """
        Add multiple papers to the network.
        
        Args:
            papers: List of Paper objects
            show_progress: Whether to show progress bar
        """
        if self.max_nodes and len(papers) > self.max_nodes:
            # Sort by citation count and take top
            papers = sorted(
                papers,
                key=lambda p: getattr(p, 'citation_count', 0),
                reverse=True
            )[:self.max_nodes]
        
        iterator = tqdm(papers, desc="Adding papers") if show_progress else papers
        
        for paper in iterator:
            self.add_paper(paper)
    
    def build_citation_edges(self, show_progress: bool = True) -> int:
        """
        Build citation edges from paper references.
        
        Returns:
            Number of edges added
        """
        n_edges = 0
        
        papers_items = self.papers.items()
        if show_progress:
            papers_items = tqdm(papers_items, desc="Building edges")
        
        for paper_id, paper_data in papers_items:
            references = paper_data.get('references', [])
            citations = paper_data.get('citations', [])
            
            # Add reference edges (paper cites these)
            for ref_id in references:
                if ref_id in self.graph:
                    self.graph.add_edge(paper_id, ref_id)
                    n_edges += 1
            
            # Add citation edges (these papers cite this)
            for cite_id in citations:
                if cite_id in self.graph:
                    self.graph.add_edge(cite_id, paper_id)
                    n_edges += 1
        
        return n_edges
    
    @classmethod
    def from_papers(
        cls,
        papers: List[Any],
        min_citations: int = 0,
        show_progress: bool = True
    ) -> 'CitationNetwork':
        """
        Create citation network from list of papers.
        
        Args:
            papers: List of Paper objects
            min_citations: Minimum citations for inclusion
            show_progress: Show progress bars
            
        Returns:
            Constructed CitationNetwork
        """
        network = cls(min_citations=min_citations)
        network.add_papers(papers, show_progress)
        network.build_citation_edges(show_progress)
        return network
    
    def temporal_slice(
        self,
        start_year: int,
        end_year: int,
        window_size: int = 1,
        cumulative: bool = False
    ) -> Dict[int, TemporalSlice]:
        """
        Create temporal slices of the network.
        
        Args:
            start_year: First year
            end_year: Last year (inclusive)
            window_size: Size of time window in years
            cumulative: Include all previous years
            
        Returns:
            Dictionary mapping year to TemporalSlice
        """
        self.slices = {}
        
        for year in range(start_year, end_year + 1, window_size):
            # Get papers for this year/window
            if cumulative:
                paper_ids = set()
                for y in range(start_year, year + 1):
                    paper_ids.update(self._year_index.get(y, set()))
            else:
                paper_ids = set()
                for y in range(year, min(year + window_size, end_year + 1)):
                    paper_ids.update(self._year_index.get(y, set()))
            
            if not paper_ids:
                continue
            
            # Create subgraph
            subgraph = self.graph.subgraph(paper_ids).copy()
            
            # Calculate statistics
            stats = self._calculate_stats(subgraph)
            
            # Create slice
            self.slices[year] = TemporalSlice(
                year=year,
                graph=subgraph,
                stats=stats,
                papers=list(paper_ids)
            )
        
        return self.slices
    
    def _calculate_stats(self, graph: nx.DiGraph) -> NetworkStats:
        """Calculate network statistics."""
        n_nodes = graph.number_of_nodes()
        n_edges = graph.number_of_edges()
        
        if n_nodes == 0:
            return NetworkStats()
        
        density = nx.density(graph)
        
        # Average clustering (for undirected version)
        if n_nodes > 1:
            avg_clustering = nx.average_clustering(graph.to_undirected())
        else:
            avg_clustering = 0.0
        
        # Number of weakly connected components
        n_components = nx.number_weakly_connected_components(graph)
        
        return NetworkStats(
            n_nodes=n_nodes,
            n_edges=n_edges,
            n_components=n_components,
            density=density,
            avg_clustering=avg_clustering,
        )
    
    def get_paper(self, paper_id: str) -> Optional[Dict]:
        """Get paper metadata."""
        return self.papers.get(paper_id)
    
    def get_papers_by_year(self, year: int) -> List[str]:
        """Get all papers from a specific year."""
        return list(self._year_index.get(year, []))
    
    def get_papers_by_field(self, field: str) -> List[str]:
        """Get all papers in a specific field."""
        return list(self._field_index.get(field, []))
    
    def get_papers_by_author(self, author: str) -> List[str]:
        """Get all papers by an author."""
        return list(self._author_index.get(author, []))
    
    def get_statistics(self) -> NetworkStats:
        """Get overall network statistics."""
        return self._calculate_stats(self.graph)
    
    def get_citation_counts(self) -> Dict[str, int]:
        """Get citation count for each paper."""
        return dict(self.graph.in_degree())
    
    def get_top_papers(self, n: int = 10, by: str = 'citations') -> List[Tuple[str, int]]:
        """
        Get top papers by metric.
        
        Args:
            n: Number of top papers
            by: Metric ('citations', 'pagerank', 'betweenness')
            
        Returns:
            List of (paper_id, score) tuples
        """
        if by == 'citations':
            scores = dict(self.graph.in_degree())
        elif by == 'pagerank':
            scores = nx.pagerank(self.graph)
        elif by == 'betweenness':
            scores = nx.betweenness_centrality(self.graph, k=min(1000, self.graph.number_of_nodes()))
        else:
            raise ValueError(f"Unknown metric: {by}")
        
        return sorted(scores.items(), key=lambda x: -x[1])[:n]
    
    def get_field_distribution(self) -> Dict[str, int]:
        """Get distribution of papers across fields."""
        return {field: len(papers) for field, papers in self._field_index.items()}
    
    def get_year_distribution(self) -> Dict[int, int]:
        """Get distribution of papers across years."""
        return {year: len(papers) for year, papers in self._year_index.items()}
    
    def to_undirected(self) -> nx.Graph:
        """Convert to undirected graph."""
        return self.graph.to_undirected()
    
    def get_subgraph_by_field(self, field: str) -> nx.DiGraph:
        """Get subgraph for a specific field."""
        paper_ids = self._field_index.get(field, set())
        return self.graph.subgraph(paper_ids).copy()
    
    def get_subgraph_by_years(
        self,
        start_year: int,
        end_year: int
    ) -> nx.DiGraph:
        """Get subgraph for a year range."""
        paper_ids = set()
        for year in range(start_year, end_year + 1):
            paper_ids.update(self._year_index.get(year, set()))
        return self.graph.subgraph(paper_ids).copy()
    
    def export_edgelist(self, filepath: str) -> None:
        """Export network as edge list."""
        nx.write_edgelist(self.graph, filepath, data=False)
    
    def export_graphml(self, filepath: str) -> None:
        """Export network as GraphML."""
        nx.write_graphml(self.graph, filepath)
    
    @classmethod
    def load_graphml(cls, filepath: str) -> 'CitationNetwork':
        """Load network from GraphML file."""
        graph = nx.read_graphml(filepath)
        
        network = cls(directed=isinstance(graph, nx.DiGraph))
        network.graph = graph
        
        # Rebuild indices
        for node, data in graph.nodes(data=True):
            network.papers[node] = data
            if data.get('year'):
                network._year_index[data['year']].add(node)
            for field in (data.get('fields') or []):
                network._field_index[field].add(node)
            for author in (data.get('authors') or []):
                network._author_index[author].add(node)
        
        return network
    
    def __len__(self) -> int:
        return self.graph.number_of_nodes()
    
    def __contains__(self, paper_id: str) -> bool:
        return paper_id in self.graph
    
    def __repr__(self) -> str:
        return f"CitationNetwork(n_papers={len(self)}, n_edges={self.graph.number_of_edges()})"


class CoAuthorshipNetwork:
    """
    Co-authorship network from academic papers.
    
    In this network:
    - Nodes represent authors
    - Edges represent collaboration (co-authored papers)
    - Edge weights represent number of joint papers
    """
    
    def __init__(self):
        """Initialize co-authorship network."""
        self.graph = nx.Graph()
        self.author_papers: Dict[str, List[str]] = defaultdict(list)
        self.author_metadata: Dict[str, Dict] = {}
    
    def add_papers(self, papers: List[Any]) -> None:
        """
        Build co-authorship network from papers.
        
        Args:
            papers: List of Paper objects with authors field
        """
        for paper in tqdm(papers, desc="Building co-authorship"):
            authors = paper.authors or []
            paper_id = paper.paper_id
            
            # Track papers per author
            for author in authors:
                self.author_papers[author].append(paper_id)
            
            # Add edges between all author pairs
            for i, author1 in enumerate(authors):
                for author2 in authors[i+1:]:
                    if self.graph.has_edge(author1, author2):
                        self.graph[author1][author2]['weight'] += 1
                        self.graph[author1][author2]['papers'].append(paper_id)
                    else:
                        self.graph.add_edge(author1, author2, weight=1, papers=[paper_id])
            
            # Update author metadata
            for author in authors:
                if author not in self.author_metadata:
                    self.author_metadata[author] = {
                        'papers': [],
                        'fields': [],
                    }
                self.author_metadata[author]['papers'].append(paper_id)
                if hasattr(paper, 'fields') and paper.fields:
                    self.author_metadata[author]['fields'].extend(paper.fields)
        
        # Deduplicate fields
        for author in self.author_metadata:
            self.author_metadata[author]['fields'] = list(
                set(self.author_metadata[author]['fields'])
            )
    
    def get_collaborators(self, author: str) -> List[Tuple[str, int]]:
        """
        Get collaborators of an author.
        
        Returns:
            List of (collaborator, n_joint_papers) tuples
        """
        if author not in self.graph:
            return []
        
        collaborators = []
        for neighbor in self.graph.neighbors(author):
            weight = self.graph[author][neighbor]['weight']
            collaborators.append((neighbor, weight))
        
        return sorted(collaborators, key=lambda x: -x[1])
    
    def get_prolific_authors(self, n: int = 10) -> List[Tuple[str, int]]:
        """Get authors with most papers."""
        return sorted(
            [(a, len(p)) for a, p in self.author_papers.items()],
            key=lambda x: -x[1]
        )[:n]
    
    def get_collaborative_authors(self, n: int = 10) -> List[Tuple[str, int]]:
        """Get authors with most collaborators."""
        return sorted(
            [(a, self.graph.degree(a)) for a in self.graph.nodes()],
            key=lambda x: -x[1]
        )[:n]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get network statistics."""
        return {
            "n_authors": self.graph.number_of_nodes(),
            "n_collaborations": self.graph.number_of_edges(),
            "avg_collaborators": sum(dict(self.graph.degree()).values()) / max(1, self.graph.number_of_nodes()),
            "density": nx.density(self.graph),
            "avg_clustering": nx.average_clustering(self.graph) if self.graph.number_of_nodes() > 0 else 0,
            "n_components": nx.number_connected_components(self.graph),
        }
