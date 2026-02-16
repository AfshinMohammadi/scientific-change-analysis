"""
Tests for Paradigm Detection Module
===================================

Unit tests for paradigm shift detection algorithms.
"""

import pytest
import numpy as np
import networkx as nx

from src.models.paradigm_detector import (
    ParadigmShiftDetector,
    KeyPaperDetector,
    ParadigmShift,
    ShiftMetrics,
)


@pytest.fixture
def sample_network():
    """Create a sample network for testing."""
    # Create a directed graph with some structure
    graph = nx.DiGraph()
    
    # Add nodes with years
    for i in range(100):
        graph.add_node(
            f'p{i}',
            year=2010 + (i % 10),
            title=f'Paper {i}'
        )
    
    # Add some edges
    for i in range(1, 100):
        # Citation to earlier papers
        for j in range(max(0, i-5), i):
            if np.random.random() > 0.5:
                graph.add_edge(f'p{i}', f'p{j}')
    
    return graph


@pytest.fixture
def sample_slices():
    """Create sample temporal slices."""
    slices = {}
    
    for year in range(2010, 2020):
        graph = nx.DiGraph()
        
        n_nodes = 50 + year - 2010  # Growing network
        for i in range(n_nodes):
            graph.add_node(f'{year}_p{i}', year=year)
        
        # Add edges
        for i in range(1, n_nodes):
            for j in range(max(0, i-3), i):
                if np.random.random() > 0.6:
                    graph.add_edge(f'{year}_p{i}', f'{year}_p{j}')
        
        # Create mock slice
        from src.networks.citation_network import NetworkStats, TemporalSlice
        stats = NetworkStats(
            n_nodes=graph.number_of_nodes(),
            n_edges=graph.number_of_edges()
        )
        slices[year] = TemporalSlice(
            year=year,
            graph=graph,
            stats=stats,
            papers=list(graph.nodes())
        )
    
    return slices


class MockNetwork:
    """Mock network for testing."""
    def __init__(self, graph, slices=None):
        self.graph = graph
        self.slices = slices or {}


class TestParadigmShiftDetector:
    """Tests for ParadigmShiftDetector class."""
    
    def test_detector_creation(self):
        """Test detector initialization."""
        detector = ParadigmShiftDetector(threshold=0.7)
        
        assert detector.threshold == 0.7
        assert detector.min_papers == 50
        assert 'structural' in detector.methods
    
    def test_detector_with_custom_params(self):
        """Test custom parameters."""
        detector = ParadigmShiftDetector(
            threshold=0.5,
            min_papers=30,
            methods=['structural']
        )
        
        assert detector.threshold == 0.5
        assert detector.min_papers == 30
        assert len(detector.methods) == 1
    
    def test_detect_structural_shift(self, sample_slices):
        """Test structural shift detection."""
        detector = ParadigmShiftDetector(
            threshold=0.3,
            min_papers=20,
            methods=['structural']
        )
        
        # Create mock network
        network = MockNetwork(nx.DiGraph(), sample_slices)
        
        # Run detection
        shifts = detector.detect(network, years=range(2012, 2020))
        
        assert isinstance(shifts, list)
    
    def test_detect_returns_paradigm_shifts(self, sample_slices):
        """Test that detection returns ParadigmShift objects."""
        detector = ParadigmShiftDetector(
            threshold=0.2,
            min_papers=10,
            methods=['structural']
        )
        
        network = MockNetwork(nx.DiGraph(), sample_slices)
        shifts = detector.detect(network, years=range(2015, 2020))
        
        for shift in shifts:
            assert isinstance(shift, ParadigmShift)
            assert hasattr(shift, 'year')
            assert hasattr(shift, 'magnitude')
            assert hasattr(shift, 'shift_type')


class TestParadigmShift:
    """Tests for ParadigmShift dataclass."""
    
    def test_shift_creation(self):
        """Test creating a paradigm shift."""
        shift = ParadigmShift(
            year=2020,
            shift_type='structural',
            magnitude=0.8,
            affected_field='ML',
            key_papers=['p1', 'p2'],
            description='Test shift'
        )
        
        assert shift.year == 2020
        assert shift.magnitude == 0.8
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        shift = ParadigmShift(
            year=2020,
            shift_type='structural',
            magnitude=0.8,
            affected_field='ML',
            key_papers=['p1'],
            description='Test'
        )
        
        d = shift.to_dict()
        
        assert d['year'] == 2020
        assert d['magnitude'] == 0.8
        assert 'key_papers' in d


class TestKeyPaperDetector:
    """Tests for KeyPaperDetector class."""
    
    def test_detector_creation(self):
        """Test detector initialization."""
        detector = KeyPaperDetector()
        
        assert detector.velocity_threshold == 2.0
        assert detector.bridging_threshold == 0.5
    
    def test_detect_key_papers(self, sample_slices):
        """Test key paper detection."""
        detector = KeyPaperDetector()
        
        # Create mock network with slices
        network = MockNetwork(nx.DiGraph(), sample_slices)
        
        # Run detection for a year with slices
        if 2015 in sample_slices:
            key_papers = detector.detect_key_papers(network, 2015)
            
            assert isinstance(key_papers, list)
            # Each result should be (paper_id, metrics)
            if key_papers:
                paper_id, metrics = key_papers[0]
                assert isinstance(metrics, dict)


class TestShiftMetrics:
    """Tests for ShiftMetrics dataclass."""
    
    def test_metrics_creation(self):
        """Test creating metrics."""
        metrics = ShiftMetrics(
            community_overlap=0.5,
            modularity_change=0.2
        )
        
        assert metrics.community_overlap == 0.5
        assert metrics.modularity_change == 0.2
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics = ShiftMetrics(
            community_overlap=0.5,
            embedding_distance=0.3
        )
        
        d = metrics.to_dict()
        
        assert 'community_overlap' in d
        assert 'embedding_distance' in d
