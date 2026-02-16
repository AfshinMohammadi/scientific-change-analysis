"""
Tests for Embedding Modules
===========================

Unit tests for graph embedding and clustering modules.
"""

import pytest
import numpy as np
import networkx as nx

from src.models.embeddings import (
    EmbeddingResult,
    GraphEmbedder,
    TopicEmbedder,
)


@pytest.fixture
def sample_graph():
    """Create a sample graph for testing."""
    graph = nx.karate_club_graph()
    
    # Add some node attributes
    for node in graph.nodes():
        graph.nodes[node]['year'] = 2020
        graph.nodes[node]['title'] = f'Node {node}'
    
    return graph


@pytest.fixture
def sample_papers():
    """Create sample papers for testing."""
    return [
        {
            'paper_id': f'p{i}',
            'title': f'Machine learning paper {i}',
            'abstract': f'This is an abstract about machine learning and neural networks. Paper {i}.',
            'year': 2020 + (i % 3)
        }
        for i in range(50)
    ]


class TestEmbeddingResult:
    """Tests for EmbeddingResult class."""
    
    def test_result_creation(self):
        """Test creating an embedding result."""
        embeddings = np.random.randn(100, 64)
        node_ids = [f'n{i}' for i in range(100)]
        
        result = EmbeddingResult(
            embeddings=embeddings,
            node_ids=node_ids,
            method='test',
            dimensions=64
        )
        
        assert result.embeddings.shape == (100, 64)
        assert len(result.node_ids) == 100
    
    def test_get_embedding(self):
        """Test retrieving a single embedding."""
        embeddings = np.random.randn(10, 32)
        node_ids = [f'n{i}' for i in range(10)]
        
        result = EmbeddingResult(
            embeddings=embeddings,
            node_ids=node_ids,
            method='test',
            dimensions=32
        )
        
        emb = result.get_embedding('n5')
        
        assert emb is not None
        assert emb.shape == (32,)
    
    def test_get_embedding_missing(self):
        """Test retrieving missing embedding."""
        embeddings = np.random.randn(10, 32)
        node_ids = [f'n{i}' for i in range(10)]
        
        result = EmbeddingResult(
            embeddings=embeddings,
            node_ids=node_ids,
            method='test',
            dimensions=32
        )
        
        emb = result.get_embedding('missing')
        
        assert emb is None
    
    def test_cluster(self):
        """Test clustering functionality."""
        embeddings = np.random.randn(100, 32)
        node_ids = [f'n{i}' for i in range(100)]
        
        result = EmbeddingResult(
            embeddings=embeddings,
            node_ids=node_ids,
            method='test',
            dimensions=32
        )
        
        clusters = result.cluster(n_clusters=5)
        
        assert len(clusters) == 100
        assert len(set(clusters.values())) == 5
    
    def test_save_load(self, tmp_path):
        """Test saving and loading embeddings."""
        embeddings = np.random.randn(10, 16)
        node_ids = [f'n{i}' for i in range(10)]
        
        result = EmbeddingResult(
            embeddings=embeddings,
            node_ids=node_ids,
            method='test',
            dimensions=16
        )
        
        filepath = str(tmp_path / 'embeddings.npz')
        result.save(filepath)
        
        loaded = EmbeddingResult.load(filepath)
        
        assert loaded.embeddings.shape == (10, 16)
        assert len(loaded.node_ids) == 10


class TestGraphEmbedder:
    """Tests for GraphEmbedder class."""
    
    def test_embedder_creation(self):
        """Test embedder initialization."""
        embedder = GraphEmbedder(
            method='spectral',
            dimensions=64
        )
        
        assert embedder.method == 'spectral'
        assert embedder.dimensions == 64
    
    def test_spectral_embedding(self, sample_graph):
        """Test spectral embedding."""
        embedder = GraphEmbedder(
            method='spectral',
            dimensions=32
        )
        
        result = embedder.fit_transform(sample_graph)
        
        assert result.embeddings.shape[0] == sample_graph.number_of_nodes()
        assert result.embeddings.shape[1] == 32
    
    def test_embedding_result_type(self, sample_graph):
        """Test that fit_transform returns EmbeddingResult."""
        embedder = GraphEmbedder(method='spectral', dimensions=16)
        
        result = embedder.fit_transform(sample_graph)
        
        assert isinstance(result, EmbeddingResult)
    
    def test_node_ids_match(self, sample_graph):
        """Test that node IDs match graph nodes."""
        embedder = GraphEmbedder(method='spectral', dimensions=16)
        
        result = embedder.fit_transform(sample_graph)
        
        assert len(result.node_ids) == sample_graph.number_of_nodes()


class TestTopicEmbedder:
    """Tests for TopicEmbedder class."""
    
    def test_embedder_creation(self):
        """Test embedder initialization."""
        embedder = TopicEmbedder(method='tfidf', n_topics=20)
        
        assert embedder.method == 'tfidf'
        assert embedder.n_topics == 20
    
    def test_tfidf_embedding(self, sample_papers):
        """Test TF-IDF embedding."""
        embedder = TopicEmbedder(method='tfidf', n_topics=10)
        
        result = embedder.fit_transform(sample_papers)
        
        assert result.embeddings.shape[0] == len(sample_papers)
    
    def test_embedding_result_type(self, sample_papers):
        """Test that fit_transform returns EmbeddingResult."""
        embedder = TopicEmbedder(method='tfidf', n_topics=10)
        
        result = embedder.fit_transform(sample_papers)
        
        assert isinstance(result, EmbeddingResult)
    
    def test_lda_embedding(self, sample_papers):
        """Test LDA embedding."""
        embedder = TopicEmbedder(method='lda', n_topics=5)
        
        result = embedder.fit_transform(sample_papers[:20])  # Smaller for speed
        
        assert result.embeddings.shape[0] == 20
        assert result.embeddings.shape[1] == 5  # n_topics dimensions
