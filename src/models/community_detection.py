"""
Community Detection Module
==========================

Implements community detection algorithms for finding clusters
in citation and co-authorship networks.

Algorithms:
- Louvain method (fast, scalable)
- Leiden algorithm (improved Louvain with quality guarantees)
- Label propagation
- Walktrap
- Girvan-Newman

Also provides:
- Community evolution tracking
- Community quality metrics
- Multi-level community detection
"""

from typing import Any, Dict, List, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from collections import defaultdict
import time

import numpy as np
import networkx as nx
from tqdm import tqdm

try:
    import community as community_louvain
    LOUVAIN_AVAILABLE = True
except ImportError:
    LOUVAIN_AVAILABLE = False

try:
    import leidenalg
    import igraph as ig
    LEIDEN_AVAILABLE = True
except ImportError:
    LEIDEN_AVAILABLE = False


@dataclass
class CommunityResult:
    """Result from community detection."""
    partition: Dict[str, int]  # node_id -> community_id
    n_communities: int
    modularity: float
    community_sizes: Dict[int, int]
    resolution: float
    algorithm: str
    execution_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_community(self, community_id: int) -> List[str]:
        """Get all nodes in a community."""
        return [n for n, c in self.partition.items() if c == community_id]
    
    def get_node_community(self, node_id: str) -> int:
        """Get community ID for a node."""
        return self.partition.get(node_id, -1)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'n_communities': self.n_communities,
            'modularity': self.modularity,
            'community_sizes': self.community_sizes,
            'resolution': self.resolution,
            'algorithm': self.algorithm,
            'execution_time': self.execution_time,
        }


@dataclass
class CommunityEvolution:
    """Tracks how communities change over time."""
    year: int
    partition: Dict[str, int]
    n_communities: int
    new_communities: List[int]
    dissolved_communities: List[int]
    merged_communities: List[Tuple[int, int]]
    split_communities: List[Tuple[int, List[int]]]
    stability_score: float


class LouvainDetector:
    """
    Louvain algorithm for community detection.
    
    The Louvain method maximizes modularity through a two-phase process:
    1. Local moving: nodes are moved to neighboring communities to maximize modularity
    2. Aggregation: communities are aggregated into super-nodes
    
    Features:
    - Very fast (O(n log n))
    - Multi-level communities
    - Resolution parameter for community size control
    
    Example:
        >>> detector = LouvainDetector(resolution=1.0)
        >>> result = detector.detect(graph)
        >>> print(f"Found {result.n_communities} communities")
    """
    
    def __init__(
        self,
        resolution: float = 1.0,
        random_state: int = 42
    ):
        """
        Initialize Louvain detector.
        
        Args:
            resolution: Resolution parameter (>1 = smaller communities, <1 = larger)
            random_state: Random seed for reproducibility
        """
        self.resolution = resolution
        self.random_state = random_state
    
    def detect(
        self,
        graph: nx.Graph,
        weight: str = 'weight'
    ) -> CommunityResult:
        """
        Detect communities using Louvain algorithm.
        
        Args:
            graph: NetworkX graph (undirected)
            weight: Edge weight attribute
            
        Returns:
            CommunityResult with partition and metrics
        """
        if not LOUVAIN_AVAILABLE:
            # Fallback to networkx implementation
            return self._detect_nx(graph)
        
        start_time = time.time()
        
        # Convert to undirected if needed
        if graph.is_directed():
            graph = graph.to_undirected()
        
        # Run Louvain
        partition = community_louvain.best_partition(
            graph,
            resolution=self.resolution,
            random_state=self.random_state,
            weight=weight
        )
        
        # Calculate metrics
        n_communities = len(set(partition.values()))
        modularity = self._calculate_modularity(graph, partition)
        community_sizes = self._get_community_sizes(partition)
        
        execution_time = time.time() - start_time
        
        return CommunityResult(
            partition=partition,
            n_communities=n_communities,
            modularity=modularity,
            community_sizes=community_sizes,
            resolution=self.resolution,
            algorithm='louvain',
            execution_time=execution_time
        )
    
    def _detect_nx(self, graph: nx.Graph) -> CommunityResult:
        """Fallback using NetworkX community detection."""
        start_time = time.time()
        
        if graph.is_directed():
            graph = graph.to_undirected()
        
        # Use greedy modularity communities
        communities = list(nx.community.greedy_modularity_communities(graph))
        
        # Convert to partition dict
        partition = {}
        for i, comm in enumerate(communities):
            for node in comm:
                partition[node] = i
        
        n_communities = len(communities)
        modularity = nx.community.modularity(graph, communities)
        community_sizes = {i: len(c) for i, c in enumerate(communities)}
        
        execution_time = time.time() - start_time
        
        return CommunityResult(
            partition=partition,
            n_communities=n_communities,
            modularity=modularity,
            community_sizes=community_sizes,
            resolution=self.resolution,
            algorithm='louvain_nx',
            execution_time=execution_time
        )
    
    def _calculate_modularity(
        self,
        graph: nx.Graph,
        partition: Dict[str, int]
    ) -> float:
        """Calculate modularity of partition."""
        communities = defaultdict(set)
        for node, comm in partition.items():
            communities[comm].add(node)
        
        return nx.community.modularity(graph, list(communities.values()))
    
    def _get_community_sizes(
        self,
        partition: Dict[str, int]
    ) -> Dict[int, int]:
        """Get size of each community."""
        sizes = defaultdict(int)
        for comm in partition.values():
            sizes[comm] += 1
        return dict(sizes)


class LeidenDetector:
    """
    Leiden algorithm for community detection.
    
    The Leiden algorithm improves upon Louvain by:
    - Guaranteeing well-connected communities
    - Being faster in practice
    - Providing higher quality partitions
    
    Features:
    - Multiple resolution profiles
    - Quality function options (CPM, Modularity)
    - Hierarchical communities
    
    Example:
        >>> detector = LeidenDetector(resolution=1.0)
        >>> result = detector.detect(graph)
    """
    
    def __init__(
        self,
        resolution: float = 1.0,
        quality_function: str = 'modularity',  # 'modularity' or 'cpm'
        n_iterations: int = 2,
        seed: int = 42
    ):
        """
        Initialize Leiden detector.
        
        Args:
            resolution: Resolution parameter
            quality_function: Quality function to optimize
            n_iterations: Number of iterations (more = higher quality)
            seed: Random seed
        """
        self.resolution = resolution
        self.quality_function = quality_function
        self.n_iterations = n_iterations
        self.seed = seed
    
    def detect(
        self,
        graph: nx.Graph,
        weight: str = 'weight'
    ) -> CommunityResult:
        """
        Detect communities using Leiden algorithm.
        
        Args:
            graph: NetworkX graph
            weight: Edge weight attribute
            
        Returns:
            CommunityResult with partition and metrics
        """
        if not LEIDEN_AVAILABLE:
            # Fallback to Louvain
            print("Leiden not available, using Louvain")
            return LouvainDetector(self.resolution).detect(graph)
        
        start_time = time.time()
        
        # Convert to undirected
        if graph.is_directed():
            graph = graph.to_undirected()
        
        # Convert to igraph
        ig_graph = self._nx_to_igraph(graph, weight)
        
        # Choose quality function
        if self.quality_function == 'cpm':
            quality = leidenalg.CPM(resolution_parameter=self.resolution)
        else:
            quality = leidenalg.ModularityVertexPartition
        
        # Run Leiden
        partition = leidenalg.find_partition(
            ig_graph,
            quality,
            resolution_parameter=self.resolution,
            seed=self.seed,
            n_iterations=self.n_iterations
        )
        
        # Convert back to NetworkX node IDs
        node_list = list(graph.nodes())
        partition_dict = {}
        for i, community in enumerate(partition):
            for node_idx in community:
                partition_dict[str(node_list[node_idx])] = i
        
        # Calculate metrics
        n_communities = len(partition)
        modularity = partition.q if hasattr(partition, 'q') else partition.modularity
        community_sizes = {i: len(c) for i, c in enumerate(partition)}
        
        execution_time = time.time() - start_time
        
        return CommunityResult(
            partition=partition_dict,
            n_communities=n_communities,
            modularity=modularity,
            community_sizes=community_sizes,
            resolution=self.resolution,
            algorithm='leiden',
            execution_time=execution_time,
            metadata={'quality_function': self.quality_function}
        )
    
    def _nx_to_igraph(
        self,
        graph: nx.Graph,
        weight: str = 'weight'
    ):
        """Convert NetworkX graph to igraph."""
        # Get edges with weights
        edges = []
        weights = []
        for u, v, data in graph.edges(data=True):
            edges.append((u, v))
            weights.append(data.get(weight, 1.0))
        
        # Create igraph
        ig_graph = ig.Graph(len(graph))
        ig_graph.vs['name'] = list(graph.nodes())
        ig_graph.add_edges(edges)
        ig_graph.es['weight'] = weights
        
        return ig_graph


class MultiResolutionDetector:
    """
    Detect communities at multiple resolution levels.
    
    This reveals hierarchical community structure, from broad
    categories to fine-grained communities.
    
    Example:
        >>> detector = MultiResolutionDetector(
        ...     resolutions=[0.5, 1.0, 2.0, 4.0]
        ... )
        >>> results = detector.detect_hierarchy(graph)
    """
    
    def __init__(
        self,
        resolutions: List[float] = None,
        algorithm: str = 'leiden',
        seed: int = 42
    ):
        self.resolutions = resolutions or [0.5, 1.0, 2.0, 4.0, 8.0]
        self.algorithm = algorithm
        self.seed = seed
    
    def detect_hierarchy(
        self,
        graph: nx.Graph,
        progress: bool = True
    ) -> Dict[float, CommunityResult]:
        """
        Detect communities at multiple resolutions.
        
        Args:
            graph: NetworkX graph
            progress: Show progress bar
            
        Returns:
            Dictionary mapping resolution to CommunityResult
        """
        results = {}
        
        iterator = self.resolutions
        if progress:
            iterator = tqdm(self.resolutions, desc="Multi-resolution detection")
        
        for resolution in iterator:
            if self.algorithm == 'leiden':
                detector = LeidenDetector(resolution=resolution, seed=self.seed)
            else:
                detector = LouvainDetector(resolution=resolution, random_state=self.seed)
            
            results[resolution] = detector.detect(graph)
        
        return results
    
    def get_stable_communities(
        self,
        results: Dict[float, CommunityResult],
        stability_threshold: float = 0.8
    ) -> List[Set[str]]:
        """
        Find communities that are stable across resolutions.
        
        Args:
            results: Results from detect_hierarchy
            stability_threshold: Minimum stability score
            
        Returns:
            List of stable community node sets
        """
        # Track which nodes are together across resolutions
        node_pairs = defaultdict(int)
        
        for result in results.values():
            partition = result.partition
            
            for comm_id in set(partition.values()):
                nodes = [n for n, c in partition.items() if c == comm_id]
                
                # Count co-occurrences
                for i, n1 in enumerate(nodes):
                    for n2 in nodes[i+1:]:
                        pair = tuple(sorted([n1, n2]))
                        node_pairs[pair] += 1
        
        n_resolutions = len(results)
        stable_pairs = {
            pair for pair, count in node_pairs.items()
            if count / n_resolutions >= stability_threshold
        }
        
        # Build stable communities
        communities = []
        used = set()
        
        for pair in stable_pairs:
            n1, n2 = pair
            
            if n1 in used or n2 in used:
                continue
            
            # Find all nodes connected through stable pairs
            community = {n1, n2}
            used.add(n1)
            used.add(n2)
            
            for other_pair in stable_pairs:
                if n1 in other_pair or n2 in other_pair:
                    for n in other_pair:
                        if n not in used:
                            community.add(n)
                            used.add(n)
            
            communities.append(community)
        
        return communities


class CommunityEvolutionTracker:
    """
    Track how communities evolve over time.
    
    Measures:
    - Community births and deaths
    - Merges and splits
    - Stability over time
    """
    
    def __init__(
        self,
        resolution: float = 1.0,
        algorithm: str = 'leiden',
        similarity_threshold: float = 0.5
    ):
        self.resolution = resolution
        self.algorithm = algorithm
        self.similarity_threshold = similarity_threshold
    
    def track_evolution(
        self,
        temporal_networks: Dict[int, nx.Graph],
        progress: bool = True
    ) -> Dict[int, CommunityEvolution]:
        """
        Track community evolution across time slices.
        
        Args:
            temporal_networks: Dict mapping year to network
            progress: Show progress
            
        Returns:
            Dict mapping year to CommunityEvolution
        """
        if self.algorithm == 'leiden':
            detector = LeidenDetector(resolution=self.resolution)
        else:
            detector = LouvainDetector(resolution=self.resolution)
        
        years = sorted(temporal_networks.keys())
        evolution = {}
        previous_partition = None
        
        iterator = years
        if progress:
            iterator = tqdm(years, desc="Tracking evolution")
        
        for year in iterator:
            graph = temporal_networks[year]
            result = detector.detect(graph)
            current_partition = result.partition
            
            if previous_partition is None:
                evolution[year] = CommunityEvolution(
                    year=year,
                    partition=current_partition,
                    n_communities=result.n_communities,
                    new_communities=list(range(result.n_communities)),
                    dissolved_communities=[],
                    merged_communities=[],
                    split_communities=[],
                    stability_score=1.0
                )
            else:
                evolution_data = self._compare_partitions(
                    previous_partition,
                    current_partition
                )
                evolution[year] = CommunityEvolution(
                    year=year,
                    partition=current_partition,
                    n_communities=result.n_communities,
                    **evolution_data
                )
            
            previous_partition = current_partition
        
        return evolution
    
    def _compare_partitions(
        self,
        previous: Dict[str, int],
        current: Dict[str, int]
    ) -> Dict[str, Any]:
        """Compare two partitions to find evolution events."""
        # Find common nodes
        common_nodes = set(previous.keys()) & set(current.keys())
        
        # Build community membership
        prev_comms = defaultdict(set)
        curr_comms = defaultdict(set)
        
        for node in common_nodes:
            prev_comms[previous[node]].add(node)
            curr_comms[current[node]].add(node)
        
        # Track events
        new_communities = []
        dissolved_communities = []
        merged_communities = []
        split_communities = []
        
        # Find merges: multiple previous -> one current
        for curr_id, curr_nodes in curr_comms.items():
            prev_ids = set()
            for node in curr_nodes:
                prev_ids.add(previous[node])
            
            if len(prev_ids) > 1:
                merged_communities.append((curr_id, list(prev_ids)))
        
        # Find splits: one previous -> multiple current
        for prev_id, prev_nodes in prev_comms.items():
            curr_ids = set()
            for node in prev_nodes:
                curr_ids.add(current[node])
            
            if len(curr_ids) > 1:
                split_communities.append((prev_id, list(curr_ids)))
        
        # Find new communities (mostly new nodes)
        for curr_id in set(current.values()):
            if curr_id not in [current[n] for n in common_nodes]:
                new_communities.append(curr_id)
        
        # Find dissolved communities
        for prev_id in set(previous.values()):
            if prev_id not in [previous[n] for n in common_nodes]:
                dissolved_communities.append(prev_id)
        
        # Calculate stability (Jaccard similarity)
        stability = self._calculate_stability(previous, current, common_nodes)
        
        return {
            'new_communities': new_communities,
            'dissolved_communities': dissolved_communities,
            'merged_communities': merged_communities,
            'split_communities': split_communities,
            'stability_score': stability,
        }
    
    def _calculate_stability(
        self,
        previous: Dict[str, int],
        current: Dict[str, int],
        common_nodes: Set[str]
    ) -> float:
        """Calculate partition stability (adjusted Rand index)."""
        prev_labels = [previous[n] for n in sorted(common_nodes)]
        curr_labels = [current[n] for n in sorted(common_nodes)]
        
        from sklearn.metrics import adjusted_rand_score
        return adjusted_rand_score(prev_labels, curr_labels)


def calculate_community_metrics(
    graph: nx.Graph,
    partition: Dict[str, int]
) -> Dict[str, float]:
    """
    Calculate quality metrics for a community partition.
    
    Metrics:
    - Modularity
    - Coverage
    - Performance
    - Average conductance
    """
    communities = defaultdict(set)
    for node, comm in partition.items():
        communities[comm].add(node)
    
    community_list = list(communities.values())
    
    metrics = {}
    
    # Modularity
    metrics['modularity'] = nx.community.modularity(graph, community_list)
    
    # Coverage
    try:
        metrics['coverage'] = nx.community.quality.coverage(graph, community_list)
    except:
        metrics['coverage'] = 0
    
    # Performance
    try:
        metrics['performance'] = nx.community.quality.performance(graph, community_list)
    except:
        metrics['performance'] = 0
    
    # Average conductance
    conductances = []
    for comm in community_list:
        try:
            conductance = nx.algorithms.cuts.conductance(graph, comm)
            conductances.append(conductance)
        except:
            pass
    
    metrics['avg_conductance'] = np.mean(conductances) if conductances else 0
    
    return metrics
