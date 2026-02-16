"""
Tests for Citation Network Module
=================================

Unit tests for citation network construction and analysis.
"""

import pytest
import numpy as np
import networkx as nx

from src.networks.citation_network import (
    CitationNetwork,
    CoAuthorshipNetwork,
    NetworkStats,
    TemporalSlice,
)


class MockPaper:
    """Mock paper for testing."""
    def __init__(
        self,
        paper_id: str,
        title: str,
        year: int,
        authors: list = None,
        references: list = None,
        citations: list = None,
        fields: list = None
    ):
        self.paper_id = paper_id
        self.title = title
        self.year = year
        self.authors = authors or []
        self.references = references or []
        self.citations = citations or []
        self.fields = fields or []
        self.citation_count = len(citations) if citations else 0
    
    def to_dict(self):
        return {
            'paper_id': self.paper_id,
            'title': self.title,
            'year': self.year,
            'authors': self.authors,
            'references': self.references,
            'citations': self.citations,
            'fields': self.fields,
            'citation_count': self.citation_count,
        }


@pytest.fixture
def sample_papers():
    """Create sample papers for testing."""
    papers = [
        MockPaper('p1', 'Paper 1', 2020, authors=['A1', 'A2'], 
                  references=['p2', 'p3'], citations=['p4'], fields=['ML']),
        MockPaper('p2', 'Paper 2', 2019, authors=['A1'], 
                  references=['p3'], citations=['p1'], fields=['ML']),
        MockPaper('p3', 'Paper 3', 2018, authors=['A2', 'A3'], 
                  references=[], citations=['p1', 'p2'], fields=['AI']),
        MockPaper('p4', 'Paper 4', 2021, authors=['A3'], 
                  references=['p1'], citations=[], fields=['ML']),
    ]
    return papers


class TestCitationNetwork:
    """Tests for CitationNetwork class."""
    
    def test_network_creation(self):
        """Test basic network creation."""
        network = CitationNetwork()
        assert network.graph.number_of_nodes() == 0
        assert network.graph.number_of_edges() == 0
    
    def test_add_paper(self):
        """Test adding a single paper."""
        network = CitationNetwork()
        paper = MockPaper('p1', 'Test Paper', 2020)
        
        network.add_paper(paper)
        
        assert network.graph.number_of_nodes() == 1
        assert 'p1' in network.graph.nodes()
    
    def test_add_multiple_papers(self, sample_papers):
        """Test adding multiple papers."""
        network = CitationNetwork()
        network.add_papers(sample_papers)
        
        assert network.graph.number_of_nodes() == 4
    
    def test_build_citation_edges(self, sample_papers):
        """Test building citation edges."""
        network = CitationNetwork()
        network.add_papers(sample_papers)
        
        n_edges = network.build_citation_edges()
        
        # Should have edges based on references
        assert n_edges > 0
        assert network.graph.number_of_edges() > 0
    
    def test_from_papers(self, sample_papers):
        """Test creating network from papers."""
        network = CitationNetwork.from_papers(sample_papers)
        
        assert network.graph.number_of_nodes() == 4
        assert network.graph.number_of_edges() > 0
    
    def test_get_paper(self, sample_papers):
        """Test retrieving paper metadata."""
        network = CitationNetwork.from_papers(sample_papers)
        
        paper = network.get_paper('p1')
        
        assert paper is not None
        assert paper['title'] == 'Paper 1'
        assert paper['year'] == 2020
    
    def test_temporal_slice(self, sample_papers):
        """Test temporal slicing."""
        network = CitationNetwork.from_papers(sample_papers)
        slices = network.temporal_slice(2018, 2021)
        
        assert len(slices) > 0
        assert 2018 in slices or 2019 in slices or 2020 in slices
    
    def test_get_statistics(self, sample_papers):
        """Test network statistics."""
        network = CitationNetwork.from_papers(sample_papers)
        stats = network.get_statistics()
        
        assert isinstance(stats, NetworkStats)
        assert stats.n_nodes == 4
        assert stats.n_edges > 0
    
    def test_get_top_papers(self, sample_papers):
        """Test getting top papers."""
        network = CitationNetwork.from_papers(sample_papers)
        top_papers = network.get_top_papers(n=2, by='citations')
        
        assert len(top_papers) == 2
        assert all(isinstance(p, tuple) for p in top_papers)
    
    def test_min_citations_filter(self):
        """Test filtering by minimum citations."""
        papers = [
            MockPaper('p1', 'High Cited', 2020, citations=['x']*100),
            MockPaper('p2', 'Low Cited', 2020, citations=['x']*2),
        ]
        
        network = CitationNetwork(min_citations=10)
        network.add_papers(papers)
        
        assert network.graph.number_of_nodes() == 1
        assert 'p1' in network.graph.nodes()
        assert 'p2' not in network.graph.nodes()
    
    def test_year_index(self, sample_papers):
        """Test year index building."""
        network = CitationNetwork.from_papers(sample_papers)
        
        papers_2020 = network.get_papers_by_year(2020)
        
        assert 'p1' in papers_2020
    
    def test_field_index(self, sample_papers):
        """Test field index building."""
        network = CitationNetwork.from_papers(sample_papers)
        
        ml_papers = network.get_papers_by_field('ML')
        
        assert len(ml_papers) >= 1


class TestCoAuthorshipNetwork:
    """Tests for CoAuthorshipNetwork class."""
    
    def test_network_creation(self):
        """Test basic co-authorship network creation."""
        network = CoAuthorshipNetwork()
        assert network.graph.number_of_nodes() == 0
    
    def test_add_papers(self, sample_papers):
        """Test adding papers creates author edges."""
        network = CoAuthorshipNetwork()
        network.add_papers(sample_papers)
        
        # Should have 3 authors
        assert network.graph.number_of_nodes() >= 3
        
        # Should have edges between co-authors
        assert network.graph.number_of_edges() > 0
    
    def test_collaboration_weights(self, sample_papers):
        """Test collaboration weights."""
        network = CoAuthorshipNetwork()
        network.add_papers(sample_papers)
        
        # A1 and A2 collaborate on p1
        if network.graph.has_edge('A1', 'A2'):
            weight = network.graph['A1']['A2']['weight']
            assert weight >= 1
    
    def test_get_collaborators(self, sample_papers):
        """Test getting collaborators."""
        network = CoAuthorshipNetwork()
        network.add_papers(sample_papers)
        
        collaborators = network.get_collaborators('A1')
        
        assert isinstance(collaborators, list)
        assert all(isinstance(c, tuple) and len(c) == 2 for c in collaborators)
    
    def test_get_statistics(self, sample_papers):
        """Test co-authorship statistics."""
        network = CoAuthorshipNetwork()
        network.add_papers(sample_papers)
        
        stats = network.get_statistics()
        
        assert 'n_authors' in stats
        assert 'n_collaborations' in stats
        assert stats['n_authors'] >= 3


class TestNetworkStats:
    """Tests for NetworkStats dataclass."""
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        stats = NetworkStats(
            n_nodes=100,
            n_edges=500,
            density=0.05,
            avg_clustering=0.3
        )
        
        d = stats.to_dict()
        
        assert d['n_nodes'] == 100
        assert d['n_edges'] == 500
        assert d['density'] == 0.05
        assert d['avg_clustering'] == 0.3


class TestTemporalSlice:
    """Tests for TemporalSlice dataclass."""
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        graph = nx.DiGraph()
        graph.add_nodes_from([1, 2, 3])
        
        stats = NetworkStats(n_nodes=3)
        
        slice_obj = TemporalSlice(
            year=2020,
            graph=graph,
            stats=stats,
            papers=['p1', 'p2', 'p3']
        )
        
        d = slice_obj.to_dict()
        
        assert d['year'] == 2020
        assert d['n_papers'] == 3
