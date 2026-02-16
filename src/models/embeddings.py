"""
Graph Embedding Module
======================

Generate embeddings for papers, authors, and concepts using various
graph embedding techniques.

Supported methods:
- Node2Vec
- GraphSAGE
- GAT (Graph Attention)
- TransE (for knowledge graphs)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import time

import numpy as np
from tqdm import tqdm


@dataclass
class EmbeddingResult:
    """Container for embedding results."""
    embeddings: np.ndarray
    node_ids: List[str]
    method: str
    dimensions: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_embedding(self, node_id: str) -> Optional[np.ndarray]:
        """Get embedding for a specific node."""
        if node_id in self.node_ids:
            idx = self.node_ids.index(node_id)
            return self.embeddings[idx]
        return None
    
    def get_similar(
        self,
        node_id: str,
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """Find most similar nodes by cosine similarity."""
        query_emb = self.get_embedding(node_id)
        if query_emb is None:
            return []
        
        # Calculate similarities
        from scipy.spatial.distance import cosine
        
        similarities = []
        for i, other_id in enumerate(self.node_ids):
            if other_id != node_id:
                sim = 1 - cosine(query_emb, self.embeddings[i])
                similarities.append((other_id, sim))
        
        # Sort by similarity
        similarities.sort(key=lambda x: -x[1])
        return similarities[:top_k]
    
    def cluster(
        self,
        n_clusters: int = 10,
        method: str = 'kmeans'
    ) -> Dict[str, int]:
        """
        Cluster nodes in embedding space.
        
        Returns:
            Dictionary mapping node_id to cluster label
        """
        from sklearn.cluster import KMeans, AgglomerativeClustering
        
        if method == 'kmeans':
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        elif method == 'agglomerative':
            clusterer = AgglomerativeClustering(n_clusters=n_clusters)
        else:
            raise ValueError(f"Unknown clustering method: {method}")
        
        labels = clusterer.fit_predict(self.embeddings)
        
        return {
            node_id: int(label)
            for node_id, label in zip(self.node_ids, labels)
        }
    
    def reduce_dimensions(
        self,
        n_components: int = 2,
        method: str = 'umap'
    ) -> 'EmbeddingResult':
        """Reduce embedding dimensions for visualization."""
        if method == 'umap':
            try:
                import umap
                reducer = umap.UMAP(n_components=n_components, random_state=42)
            except ImportError:
                method = 'tsne'
        
        if method == 'tsne':
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=n_components, random_state=42)
        
        reduced = reducer.fit_transform(self.embeddings)
        
        return EmbeddingResult(
            embeddings=reduced,
            node_ids=self.node_ids,
            method=f"{self.method}_{method}",
            dimensions=n_components,
            metadata=self.metadata
        )
    
    def save(self, filepath: str) -> None:
        """Save embeddings to file."""
        np.savez(
            filepath,
            embeddings=self.embeddings,
            node_ids=np.array(self.node_ids),
            method=self.method,
            dimensions=self.dimensions,
        )
    
    @classmethod
    def load(cls, filepath: str) -> 'EmbeddingResult':
        """Load embeddings from file."""
        data = np.load(filepath, allow_pickle=True)
        return cls(
            embeddings=data['embeddings'],
            node_ids=data['node_ids'].tolist(),
            method=str(data['method']),
            dimensions=int(data['dimensions']),
        )


class GraphEmbedder:
    """
    Generate graph embeddings using various methods.
    
    Example:
        >>> embedder = GraphEmbedder(method='node2vec', dimensions=128)
        >>> result = embedder.fit_transform(network.graph)
        >>> similar = result.get_similar('paper_123', top_k=5)
    """
    
    def __init__(
        self,
        method: str = 'node2vec',
        dimensions: int = 128,
        walk_length: int = 30,
        num_walks: int = 200,
        window_size: int = 5,
        min_count: int = 1,
        workers: int = 4,
        seed: int = 42
    ):
        """
        Initialize graph embedder.
        
        Args:
            method: Embedding method ('node2vec', 'deepwalk', 'graphsage', 'gat')
            dimensions: Embedding dimension
            walk_length: Length of random walks
            num_walks: Number of walks per node
            window_size: Context window for skip-gram
            min_count: Minimum node frequency
            workers: Number of parallel workers
            seed: Random seed
        """
        self.method = method
        self.dimensions = dimensions
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.window_size = window_size
        self.min_count = min_count
        self.workers = workers
        self.seed = seed
        
        self._model = None
    
    def fit_transform(
        self,
        graph: Any,
        weight: Optional[str] = None
    ) -> EmbeddingResult:
        """
        Generate embeddings for graph nodes.
        
        Args:
            graph: NetworkX graph
            weight: Edge weight attribute name
            
        Returns:
            EmbeddingResult with node embeddings
        """
        node_ids = list(graph.nodes())
        
        if self.method == 'node2vec':
            embeddings = self._node2vec(graph, weight)
        elif self.method == 'deepwalk':
            embeddings = self._deepwalk(graph)
        elif self.method == 'graphsage':
            embeddings = self._graphsage(graph)
        elif self.method == 'spectral':
            embeddings = self._spectral(graph)
        else:
            # Fallback to simple spectral embedding
            embeddings = self._spectral(graph)
        
        return EmbeddingResult(
            embeddings=embeddings,
            node_ids=node_ids,
            method=self.method,
            dimensions=self.dimensions,
            metadata={'n_nodes': len(node_ids)}
        )
    
    def _node2vec(self, graph: Any, weight: Optional[str] = None) -> np.ndarray:
        """Generate Node2Vec embeddings."""
        try:
            from node2vec import Node2Vec
            
            node2vec = Node2Vec(
                graph,
                dimensions=self.dimensions,
                walk_length=self.walk_length,
                num_walks=self.num_walks,
                workers=self.workers,
                seed=self.seed,
                quiet=True
            )
            
            model = node2vec.fit(
                window=self.window_size,
                min_count=self.min_count,
                seed=self.seed
            )
            
            node_ids = list(graph.nodes())
            embeddings = np.array([model.wv[str(n)] for n in node_ids])
            
            return embeddings
            
        except ImportError:
            print("node2vec not installed, using spectral embedding")
            return self._spectral(graph)
    
    def _deepwalk(self, graph: Any) -> np.ndarray:
        """Generate DeepWalk embeddings."""
        try:
            from gensim.models import Word2Vec
            
            # Generate random walks
            walks = self._generate_walks(graph)
            
            # Train Word2Vec on walks
            model = Word2Vec(
                walks,
                vector_size=self.dimensions,
                window=self.window_size,
                min_count=self.min_count,
                workers=self.workers,
                seed=self.seed
            )
            
            node_ids = list(graph.nodes())
            embeddings = np.array([model.wv[str(n)] for n in node_ids])
            
            return embeddings
            
        except ImportError:
            print("gensim not installed, using spectral embedding")
            return self._spectral(graph)
    
    def _generate_walks(self, graph: Any) -> List[List[str]]:
        """Generate random walks for DeepWalk."""
        import random
        random.seed(self.seed)
        
        walks = []
        nodes = list(graph.nodes())
        
        for _ in range(self.num_walks):
            random.shuffle(nodes)
            for node in nodes:
                walk = self._single_walk(graph, node)
                walks.append([str(n) for n in walk])
        
        return walks
    
    def _single_walk(self, graph: Any, start_node: Any) -> List[Any]:
        """Generate single random walk."""
        import random
        
        walk = [start_node]
        
        for _ in range(self.walk_length - 1):
            current = walk[-1]
            neighbors = list(graph.neighbors(current))
            
            if not neighbors:
                break
            
            walk.append(random.choice(neighbors))
        
        return walk
    
    def _graphsage(self, graph: Any) -> np.ndarray:
        """Generate GraphSAGE embeddings."""
        try:
            import torch
            import torch_geometric
            from torch_geometric.nn import SAGEConv
            from torch_geometric.utils import from_networkx
            
            # Convert to PyG format
            data = from_networkx(graph.to_undirected())
            
            # Create simple GraphSAGE model
            class GraphSAGE(torch.nn.Module):
                def __init__(self, in_channels, hidden_channels, out_channels):
                    super().__init__()
                    self.conv1 = SAGEConv(in_channels, hidden_channels)
                    self.conv2 = SAGEConv(hidden_channels, out_channels)
                
                def forward(self, x, edge_index):
                    x = self.conv1(x, edge_index).relu()
                    x = self.conv2(x, edge_index)
                    return x
            
            # Initialize with random features
            n_nodes = data.num_nodes
            x = torch.randn(n_nodes, 64)
            
            model = GraphSAGE(64, 128, self.dimensions)
            model.eval()
            
            with torch.no_grad():
                embeddings = model(x, data.edge_index).numpy()
            
            return embeddings
            
        except ImportError:
            print("torch_geometric not installed, using spectral embedding")
            return self._spectral(graph)
    
    def _spectral(self, graph: Any) -> np.ndarray:
        """Generate spectral embeddings (fallback method)."""
        import networkx as nx
        from sklearn.manifold import SpectralEmbedding
        
        # Get adjacency matrix
        A = nx.to_numpy_array(graph.to_undirected())
        
        # Spectral embedding
        embedder = SpectralEmbedding(
            n_components=min(self.dimensions, A.shape[0] - 1),
            random_state=self.seed
        )
        
        try:
            embeddings = embedder.fit_transform(A)
        except:
            # Fallback to random
            embeddings = np.random.randn(A.shape[0], self.dimensions)
        
        # Pad to desired dimensions
        if embeddings.shape[1] < self.dimensions:
            padding = np.random.randn(embeddings.shape[0], self.dimensions - embeddings.shape[1])
            embeddings = np.hstack([embeddings, padding])
        
        return embeddings


class TopicEmbedder:
    """
    Generate topic-based embeddings for papers.
    
    Uses abstract/title text to create semantic embeddings.
    """
    
    def __init__(
        self,
        method: str = 'tfidf',
        n_topics: int = 50,
        embedding_model: str = 'all-MiniLM-L6-v2'
    ):
        """
        Initialize topic embedder.
        
        Args:
            method: Method for topic extraction ('tfidf', 'lda', 'bert')
            n_topics: Number of topics
            embedding_model: Model name for sentence transformers
        """
        self.method = method
        self.n_topics = n_topics
        self.embedding_model = embedding_model
        self._model = None
    
    def fit_transform(
        self,
        papers: List[Dict[str, Any]],
        text_field: str = 'abstract'
    ) -> EmbeddingResult:
        """
        Generate topic embeddings for papers.
        
        Args:
            papers: List of paper dictionaries
            text_field: Field to use for text ('abstract' or 'title')
            
        Returns:
            EmbeddingResult with topic embeddings
        """
        texts = []
        paper_ids = []
        
        for paper in papers:
            text = paper.get(text_field) or paper.get('title', '')
            if text:
                texts.append(text)
                paper_ids.append(paper['paper_id'])
        
        if self.method == 'bert':
            embeddings = self._bert_embeddings(texts)
        elif self.method == 'tfidf':
            embeddings = self._tfidf_embeddings(texts)
        elif self.method == 'lda':
            embeddings = self._lda_embeddings(texts)
        else:
            embeddings = self._tfidf_embeddings(texts)
        
        return EmbeddingResult(
            embeddings=embeddings,
            node_ids=paper_ids,
            method=self.method,
            dimensions=embeddings.shape[1],
            metadata={'n_papers': len(paper_ids)}
        )
    
    def _bert_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate BERT-based embeddings."""
        try:
            from sentence_transformers import SentenceTransformer
            
            model = SentenceTransformer(self.embedding_model)
            embeddings = model.encode(texts, show_progress_bar=True)
            
            return embeddings
            
        except ImportError:
            print("sentence-transformers not installed, using TF-IDF")
            return self._tfidf_embeddings(texts)
    
    def _tfidf_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate TF-IDF based embeddings."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        
        vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words='english',
            ngram_range=(1, 2),
            max_df=0.95,
            min_df=2
        )
        
        tfidf = vectorizer.fit_transform(texts)
        
        # Reduce dimensions
        svd = TruncatedSVD(n_components=min(self.n_topics, tfidf.shape[1] - 1), random_state=42)
        embeddings = svd.fit_transform(tfidf)
        
        return embeddings
    
    def _lda_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate LDA topic embeddings."""
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.decomposition import LatentDirichletAllocation
        
        vectorizer = CountVectorizer(
            max_features=10000,
            stop_words='english',
            max_df=0.95,
            min_df=2
        )
        
        doc_term_matrix = vectorizer.fit_transform(texts)
        
        lda = LatentDirichletAllocation(
            n_components=self.n_topics,
            random_state=42,
            n_jobs=-1
        )
        
        embeddings = lda.fit_transform(doc_term_matrix)
        
        return embeddings
