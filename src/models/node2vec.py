"""
Advanced Graph Embedding Module
===============================

Implements Node2Vec and related random-walk based embeddings
using the node2vec library and gensim.

Features:
- Node2Vec with custom walk strategies
- DeepWalk implementation
- Random walk generation
- Biased walks (BFS/DFS)
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import time
import random
from collections import defaultdict

import numpy as np
from tqdm import tqdm

try:
    from node2vec import Node2Vec as Node2VecLib
    NODE2VEC_AVAILABLE = True
except ImportError:
    NODE2VEC_AVAILABLE = False

try:
    from gensim.models import Word2Vec, KeyedVectors
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False


@dataclass
class Node2VecConfig:
    """Configuration for Node2Vec embeddings."""
    dimensions: int = 128
    walk_length: int = 30
    num_walks: int = 200
    p: float = 1.0  # Return parameter
    q: float = 1.0  # In-out parameter
    window: int = 5
    min_count: int = 1
    workers: int = 4
    seed: int = 42
    
    # Training
    epochs: int = 5
    learning_rate: float = 0.025
    min_learning_rate: float = 0.0001


class Node2VecEmbedder:
    """
    Node2Vec embedding generator with full control over parameters.
    
    Node2Vec uses biased random walks to generate node sequences,
    then applies Skip-gram to learn embeddings.
    
    Parameters:
        p: Return parameter (1 = unbiased, <1 = BFS-like, >1 = DFS-like)
        q: In-out parameter (1 = unbiased, <1 = DFS-like, >1 = BFS-like)
    
    Example:
        >>> embedder = Node2VecEmbedder(dimensions=128, p=1, q=0.5)
        >>> model = embedder.fit(graph)
        >>> embedding = embedder.get_embedding('paper_123')
    """
    
    def __init__(self, config: Optional[Node2VecConfig] = None):
        self.config = config or Node2VecConfig()
        self.model = None
        self._node_ids = []
    
    def fit(
        self,
        graph: Any,
        weight_key: str = 'weight',
        progress: bool = True
    ) -> 'Node2VecEmbedder':
        """
        Fit Node2Vec on the graph.
        
        Args:
            graph: NetworkX graph
            weight_key: Edge weight attribute name
            progress: Show progress bar
            
        Returns:
            Self for chaining
        """
        if not NODE2VEC_AVAILABLE:
            raise ImportError("node2vec package is required. Install with: pip install node2vec")
        
        self._node_ids = list(graph.nodes())
        
        # Create Node2Vec model
        self.model = Node2VecLib(
            graph,
            dimensions=self.config.dimensions,
            walk_length=self.config.walk_length,
            num_walks=self.config.num_walks,
            p=self.config.p,
            q=self.config.q,
            weight_key=weight_key,
            workers=self.config.workers,
            seed=self.config.seed,
            quiet=not progress
        )
        
        # Train Word2Vec
        self._word2vec_model = self.model.fit(
            window=self.config.window,
            min_count=self.config.min_count,
            epochs=self.config.epochs,
            total_words=len(self._node_ids)
        )
        
        return self
    
    def transform(self, node_ids: Optional[List[str]] = None) -> np.ndarray:
        """
        Get embeddings for nodes.
        
        Args:
            node_ids: Specific nodes to get embeddings for (None = all)
            
        Returns:
            Numpy array of embeddings
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if node_ids is None:
            node_ids = self._node_ids
        
        embeddings = []
        for node_id in node_ids:
            try:
                emb = self.model.wv[str(node_id)]
            except KeyError:
                # Random embedding for unknown nodes
                emb = np.random.randn(self.config.dimensions)
            embeddings.append(emb)
        
        return np.array(embeddings)
    
    def fit_transform(
        self,
        graph: Any,
        node_ids: Optional[List[str]] = None,
        progress: bool = True
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Fit and return embeddings.
        
        Args:
            graph: NetworkX graph
            node_ids: Specific nodes (None = all)
            progress: Show progress bar
            
        Returns:
            Tuple of (embeddings, node_ids)
        """
        self.fit(graph, progress=progress)
        
        if node_ids is None:
            node_ids = self._node_ids
        
        embeddings = self.transform(node_ids)
        
        return embeddings, node_ids
    
    def get_embedding(self, node_id: str) -> np.ndarray:
        """Get embedding for a single node."""
        return self.transform([node_id])[0]
    
    def get_similar(
        self,
        node_id: str,
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Find most similar nodes.
        
        Args:
            node_id: Query node
            top_k: Number of similar nodes to return
            
        Returns:
            List of (node_id, similarity) tuples
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        try:
            similar = self.model.wv.most_similar(str(node_id), topn=top_k)
            return [(n, s) for n, s in similar]
        except KeyError:
            return []
    
    def save(self, filepath: str) -> None:
        """Save embeddings to file."""
        if self.model is not None:
            self.model.wv.save_word2vec_format(filepath)
    
    def load(self, filepath: str) -> None:
        """Load embeddings from file."""
        if GENSIM_AVAILABLE:
            self._word2vec_model = KeyedVectors.load_word2vec_format(filepath)
            self._node_ids = [str(n) for n in self._word2vec_model.index_to_key]


class DeepWalk:
    """
    DeepWalk implementation using random walks and Skip-gram.
    
    DeepWalk uses uniform random walks, unlike Node2Vec's biased walks.
    This can be preferable for certain graph structures.
    
    Example:
        >>> walker = DeepWalk(walk_length=40, num_walks=80)
        >>> walks = walker.generate_walks(graph)
        >>> model = walker.train(walks, dimensions=128)
    """
    
    def __init__(
        self,
        walk_length: int = 40,
        num_walks: int = 80,
        window: int = 5,
        dimensions: int = 128,
        workers: int = 4,
        seed: int = 42
    ):
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.window = window
        self.dimensions = dimensions
        self.workers = workers
        self.seed = seed
        
        self.model = None
        self._node_ids = []
    
    def _random_walk(
        self,
        graph: Any,
        start_node: Any,
        walk_length: int
    ) -> List[str]:
        """Generate a single random walk."""
        walk = [start_node]
        
        for _ in range(walk_length - 1):
            current = walk[-1]
            neighbors = list(graph.neighbors(current))
            
            if not neighbors:
                break
            
            walk.append(random.choice(neighbors))
        
        return [str(n) for n in walk]
    
    def generate_walks(
        self,
        graph: Any,
        progress: bool = True
    ) -> List[List[str]]:
        """
        Generate random walks for all nodes.
        
        Args:
            graph: NetworkX graph
            progress: Show progress bar
            
        Returns:
            List of walks (each walk is list of node IDs)
        """
        walks = []
        nodes = list(graph.nodes())
        self._node_ids = [str(n) for n in nodes]
        
        random.seed(self.seed)
        
        iterator = range(self.num_walks)
        if progress:
            iterator = tqdm(iterator, desc="Generating walks")
        
        for _ in iterator:
            random.shuffle(nodes)
            for node in nodes:
                walk = self._random_walk(graph, node, self.walk_length)
                walks.append(walk)
        
        return walks
    
    def train(
        self,
        walks: List[List[str]],
        epochs: int = 5,
        progress: bool = True
    ) -> 'DeepWalk':
        """
        Train Skip-gram model on walks.
        
        Args:
            walks: List of random walks
            epochs: Number of training epochs
            progress: Show progress
            
        Returns:
            Self for chaining
        """
        if not GENSIM_AVAILABLE:
            raise ImportError("gensim is required. Install with: pip install gensim")
        
        self.model = Word2Vec(
            sentences=walks,
            vector_size=self.dimensions,
            window=self.window,
            min_count=1,
            workers=self.workers,
            epochs=epochs,
            seed=self.seed
        )
        
        return self
    
    def fit(
        self,
        graph: Any,
        epochs: int = 5,
        progress: bool = True
    ) -> 'DeepWalk':
        """
        Full fit: generate walks and train.
        
        Args:
            graph: NetworkX graph
            epochs: Training epochs
            progress: Show progress
            
        Returns:
            Self for chaining
        """
        walks = self.generate_walks(graph, progress)
        self.train(walks, epochs, progress)
        return self
    
    def transform(self, node_ids: Optional[List[str]] = None) -> np.ndarray:
        """Get embeddings for nodes."""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if node_ids is None:
            node_ids = self._node_ids
        
        embeddings = []
        for node_id in node_ids:
            try:
                emb = self.model.wv[str(node_id)]
            except KeyError:
                emb = np.random.randn(self.dimensions)
            embeddings.append(emb)
        
        return np.array(embeddings)
    
    def fit_transform(
        self,
        graph: Any,
        epochs: int = 5,
        progress: bool = True
    ) -> Tuple[np.ndarray, List[str]]:
        """Fit and return embeddings."""
        self.fit(graph, epochs, progress)
        return self.transform(), self._node_ids


class BiasedRandomWalker:
    """
    Custom biased random walk generator.
    
    Allows fine-grained control over walk strategies:
    - BFS-like walks (explore local structure)
    - DFS-like walks (explore global structure)
    - Custom transition probabilities
    """
    
    def __init__(
        self,
        walk_length: int = 30,
        num_walks: int = 200,
        p: float = 1.0,
        q: float = 1.0,
        seed: int = 42
    ):
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.p = p  # Return parameter
        self.q = q  # In-out parameter
        self.seed = seed
    
    def _compute_transition_probs(
        self,
        graph: Any,
        source: Any,
        current: Any
    ) -> Dict[Any, float]:
        """
        Compute transition probabilities from current node.
        
        Based on Node2Vec's transition probability formula.
        """
        neighbors = list(graph.neighbors(current))
        if not neighbors:
            return {}
        
        # Compute unnormalized probabilities
        probs = {}
        for neighbor in neighbors:
            if neighbor == source:
                # Return to previous node
                prob = 1.0 / self.p
            elif graph.has_edge(source, neighbor):
                # Neighbor is connected to source (distance 1)
                prob = 1.0
            else:
                # Neighbor is not connected to source (distance 2)
                prob = 1.0 / self.q
            
            # Apply edge weight if available
            edge_data = graph.get_edge_data(current, neighbor)
            weight = edge_data.get('weight', 1.0) if edge_data else 1.0
            prob *= weight
            
            probs[neighbor] = prob
        
        # Normalize
        total = sum(probs.values())
        if total > 0:
            probs = {k: v/total for k, v in probs.items()}
        
        return probs
    
    def _walk(
        self,
        graph: Any,
        start_node: Any
    ) -> List[Any]:
        """Generate a single biased random walk."""
        walk = [start_node]
        
        while len(walk) < self.walk_length:
            current = walk[-1]
            neighbors = list(graph.neighbors(current))
            
            if not neighbors:
                break
            
            if len(walk) == 1:
                # First step: uniform random
                next_node = random.choice(neighbors)
            else:
                # Subsequent steps: biased by p and q
                source = walk[-2]
                probs = self._compute_transition_probs(graph, source, current)
                
                if not probs:
                    break
                
                # Sample next node
                nodes = list(probs.keys())
                weights = list(probs.values())
                next_node = random.choices(nodes, weights=weights)[0]
            
            walk.append(next_node)
        
        return walk
    
    def generate_walks(
        self,
        graph: Any,
        progress: bool = True
    ) -> List[List[str]]:
        """Generate walks for all nodes."""
        random.seed(self.seed)
        
        walks = []
        nodes = list(graph.nodes())
        
        iterator = range(self.num_walks)
        if progress:
            iterator = tqdm(iterator, desc="Generating biased walks")
        
        for _ in iterator:
            random.shuffle(nodes)
            for node in nodes:
                walk = self._walk(graph, node)
                walks.append([str(n) for n in walk])
        
        return walks


class Metapath2Vec:
    """
    Metapath2Vec for heterogeneous networks.
    
    Uses metapath-guided random walks to generate node sequences
    that follow specific relationship patterns.
    
    Example metapaths:
    - Paper-Author-Paper (PAP): Papers by same author
    - Paper-Venue-Paper (PVP): Papers in same venue
    - Paper-Field-Paper (PFP): Papers in same field
    """
    
    def __init__(
        self,
        metapaths: List[List[str]],
        walk_length: int = 100,
        num_walks: int = 50,
        dimensions: int = 128,
        window: int = 5,
        workers: int = 4,
        seed: int = 42
    ):
        self.metapaths = metapaths
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.dimensions = dimensions
        self.window = window
        self.workers = workers
        self.seed = seed
        
        self.model = None
    
    def _metapath_walk(
        self,
        graph: Any,
        start_node: Any,
        metapath: List[str]
    ) -> List[str]:
        """Generate a walk following a metapath pattern."""
        walk = [start_node]
        current_type_idx = 0
        
        for _ in range(self.walk_length - 1):
            current = walk[-1]
            current_type = metapath[current_type_idx % len(metapath)]
            next_type = metapath[(current_type_idx + 1) % len(metapath)]
            
            # Get neighbors of next type
            neighbors = []
            for neighbor in graph.neighbors(current):
                neighbor_type = graph.nodes[neighbor].get('type', '')
                if neighbor_type == next_type:
                    neighbors.append(neighbor)
            
            if not neighbors:
                break
            
            walk.append(random.choice(neighbors))
            current_type_idx += 1
        
        return [str(n) for n in walk]
    
    def generate_walks(
        self,
        graph: Any,
        progress: bool = True
    ) -> List[List[str]]:
        """Generate metapath-guided walks."""
        random.seed(self.seed)
        
        walks = []
        
        # Get nodes by type for each metapath
        for metapath in self.metapaths:
            start_type = metapath[0]
            start_nodes = [
                n for n in graph.nodes()
                if graph.nodes[n].get('type', '') == start_type
            ]
            
            iterator = range(self.num_walks)
            if progress:
                iterator = tqdm(iterator, desc=f"Metapath {'-'.join(metapath)}")
            
            for _ in iterator:
                random.shuffle(start_nodes)
                for node in start_nodes:
                    walk = self._metapath_walk(graph, node, metapath)
                    walks.append(walk)
        
        return walks
    
    def fit(
        self,
        graph: Any,
        epochs: int = 5,
        progress: bool = True
    ) -> 'Metapath2Vec':
        """Fit model on graph."""
        if not GENSIM_AVAILABLE:
            raise ImportError("gensim is required")
        
        walks = self.generate_walks(graph, progress)
        
        self.model = Word2Vec(
            sentences=walks,
            vector_size=self.dimensions,
            window=self.window,
            min_count=1,
            workers=self.workers,
            epochs=epochs,
            seed=self.seed
        )
        
        return self
    
    def transform(self, node_ids: List[str]) -> np.ndarray:
        """Get embeddings."""
        if self.model is None:
            raise ValueError("Model not fitted")
        
        embeddings = []
        for node_id in node_ids:
            try:
                emb = self.model.wv[str(node_id)]
            except KeyError:
                emb = np.random.randn(self.dimensions)
            embeddings.append(emb)
        
        return np.array(embeddings)
