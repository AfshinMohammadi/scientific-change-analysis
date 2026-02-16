"""
Paradigm Shift Detection Module
===============================

Algorithms for detecting paradigm shifts in scientific fields
through network analysis, embedding drift, and citation patterns.

Methods:
1. Structural Analysis: Community reorganization detection
2. Semantic Drift: Topic/embedding divergence over time
3. Citation Flow: Sudden changes in citation patterns
4. Key Paper Detection: High-impact papers restructuring fields
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Set
from collections import defaultdict
from datetime import datetime
import time

import numpy as np
from scipy import stats
from scipy.spatial.distance import cosine, jensenshannon
import networkx as nx
from tqdm import tqdm


@dataclass
class ParadigmShift:
    """Represents a detected paradigm shift."""
    year: int
    shift_type: str  # 'structural', 'semantic', 'citation', 'key_paper'
    magnitude: float  # 0-1 scale
    affected_field: str
    key_papers: List[str] = field(default_factory=list)
    key_authors: List[str] = field(default_factory=list)
    description: str = ""
    metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "year": self.year,
            "shift_type": self.shift_type,
            "magnitude": self.magnitude,
            "affected_field": self.affected_field,
            "key_papers": self.key_papers,
            "key_authors": self.key_authors,
            "description": self.description,
            "metrics": self.metrics,
        }


@dataclass
class ShiftMetrics:
    """Metrics computed for shift detection."""
    community_overlap: float = 0.0
    modularity_change: float = 0.0
    embedding_distance: float = 0.0
    citation_flow_change: float = 0.0
    pagerank_change: float = 0.0
    new_paper_rate: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "community_overlap": self.community_overlap,
            "modularity_change": self.modularity_change,
            "embedding_distance": self.embedding_distance,
            "citation_flow_change": self.citation_flow_change,
            "pagerank_change": self.pagerank_change,
            "new_paper_rate": self.new_paper_rate,
        }


class ParadigmShiftDetector:
    """
    Detect paradigm shifts in scientific fields.
    
    Uses multiple methods to identify significant changes:
    - Community structure evolution
    - Semantic/topic drift
    - Citation pattern changes
    - Key paper emergence
    
    Example:
        >>> detector = ParadigmShiftDetector(threshold=0.7)
        >>> shifts = detector.detect(network, years=range(2010, 2024))
    """
    
    def __init__(
        self,
        threshold: float = 0.7,
        min_papers: int = 50,
        community_resolution: float = 0.8,
        embedding_dim: int = 128,
        temporal_window: int = 2,
        methods: List[str] = None
    ):
        """
        Initialize paradigm shift detector.
        
        Args:
            threshold: Threshold for shift detection (0-1)
            min_papers: Minimum papers to consider a field
            community_resolution: Resolution for community detection
            embedding_dim: Dimension for embeddings
            temporal_window: Years to compare for changes
            methods: Detection methods to use
        """
        self.threshold = threshold
        self.min_papers = min_papers
        self.community_resolution = community_resolution
        self.embedding_dim = embedding_dim
        self.temporal_window = temporal_window
        self.methods = methods or ['structural', 'semantic', 'citation_flow']
        
        # Cache for computed values
        self._community_cache: Dict[int, Dict[str, int]] = {}
        self._embedding_cache: Dict[int, np.ndarray] = {}
    
    def detect(
        self,
        network: Any,
        years: range,
        fields: Optional[List[str]] = None
    ) -> List[ParadigmShift]:
        """
        Detect paradigm shifts across years.
        
        Args:
            network: CitationNetwork object with temporal slices
            years: Range of years to analyze
            fields: Specific fields to analyze (None = all)
            
        Returns:
            List of detected ParadigmShift objects
        """
        shifts = []
        
        # Ensure temporal slices exist
        if not network.slices:
            network.temporal_slice(
                start_year=min(years),
                end_year=max(years)
            )
        
        for year in years:
            if year not in network.slices:
                continue
            
            current_slice = network.slices[year]
            
            # Find comparison year
            compare_year = year - self.temporal_window
            if compare_year not in network.slices:
                continue
            
            previous_slice = network.slices[compare_year]
            
            # Run detection methods
            if 'structural' in self.methods:
                shift = self._detect_structural_shift(
                    current_slice, previous_slice, year
                )
                if shift:
                    shifts.append(shift)
            
            if 'semantic' in self.methods:
                shift = self._detect_semantic_shift(
                    current_slice, previous_slice, year, network
                )
                if shift:
                    shifts.append(shift)
            
            if 'citation_flow' in self.methods:
                shift = self._detect_citation_shift(
                    current_slice, previous_slice, year
                )
                if shift:
                    shifts.append(shift)
        
        # Merge overlapping shifts
        shifts = self._merge_shifts(shifts)
        
        return shifts
    
    def _detect_structural_shift(
        self,
        current_slice: Any,
        previous_slice: Any,
        year: int
    ) -> Optional[ParadigmShift]:
        """
        Detect structural changes through community reorganization.
        
        Uses the Leiden algorithm for community detection and
        measures how communities change between time slices.
        """
        current_graph = current_slice.graph
        previous_graph = previous_slice.graph
        
        if current_graph.number_of_nodes() < self.min_papers:
            return None
        
        # Detect communities
        try:
            import community as community_louvain
            
            current_communities = community_louvain.best_partition(
                current_graph.to_undirected(),
                resolution=self.community_resolution
            )
            previous_communities = community_louvain.best_partition(
                previous_graph.to_undirected(),
                resolution=self.community_resolution
            )
        except ImportError:
            # Fallback to networkx
            current_communities = dict(
                enumerate(nx.community.greedy_modularity_communities(
                    current_graph.to_undirected()
                ))
            )
            previous_communities = dict(
                enumerate(nx.community.greedy_modularity_communities(
                    previous_graph.to_undirected()
                ))
            )
        
        # Calculate community overlap
        common_nodes = set(current_communities.keys()) & set(previous_communities.keys())
        
        if len(common_nodes) < self.min_papers:
            return None
        
        # Measure community stability
        reassignments = 0
        for node in common_nodes:
            # Find which community this node was reassigned to
            curr_comm = current_communities[node]
            prev_comm = previous_communities.get(node, -1)
            
            if curr_comm != prev_comm:
                reassignments += 1
        
        reassignment_rate = reassignments / len(common_nodes)
        
        # Calculate modularity change
        try:
            current_mod = nx.community.modularity(
                current_graph.to_undirected(),
                [{n for n, c in current_communities.items() if c == comm}
                 for comm in set(current_communities.values())]
            )
            previous_mod = nx.community.modularity(
                previous_graph.to_undirected(),
                [{n for n, c in previous_communities.items() if c == comm}
                 for comm in set(previous_communities.values())]
            )
            modularity_change = abs(current_mod - previous_mod)
        except:
            modularity_change = 0.0
        
        # Calculate magnitude
        magnitude = 0.6 * reassignment_rate + 0.4 * modularity_change
        
        if magnitude < self.threshold:
            return None
        
        # Identify key papers (high PageRank change)
        key_papers = self._find_key_papers(current_graph, previous_graph)
        
        return ParadigmShift(
            year=year,
            shift_type='structural',
            magnitude=magnitude,
            affected_field='general',  # Can be refined
            key_papers=key_papers,
            description=f"Significant community reorganization detected ({reassignment_rate:.1%} nodes reassigned)",
            metrics={
                'reassignment_rate': reassignment_rate,
                'modularity_change': modularity_change,
            }
        )
    
    def _detect_semantic_shift(
        self,
        current_slice: Any,
        previous_slice: Any,
        year: int,
        network: Any
    ) -> Optional[ParadigmShift]:
        """
        Detect semantic drift through embedding analysis.
        
        Measures how the semantic space of a field changes over time.
        """
        # Get papers for each slice
        current_papers = current_slice.papers
        previous_papers = previous_slice.papers
        
        if len(current_papers) < self.min_papers or len(previous_papers) < self.min_papers:
            return None
        
        # Generate embeddings (simplified - use actual embedding model in practice)
        current_embedding = self._compute_slice_embedding(current_papers, network)
        previous_embedding = self._compute_slice_embedding(previous_papers, network)
        
        if current_embedding is None or previous_embedding is None:
            return None
        
        # Calculate embedding distance
        distance = cosine(current_embedding, previous_embedding)
        
        if distance < self.threshold * 0.5:  # Lower threshold for semantic
            return None
        
        return ParadigmShift(
            year=year,
            shift_type='semantic',
            magnitude=distance,
            affected_field='general',
            description=f"Significant semantic drift detected (distance: {distance:.3f})",
            metrics={'embedding_distance': distance}
        )
    
    def _detect_citation_shift(
        self,
        current_slice: Any,
        previous_slice: Any,
        year: int
    ) -> Optional[ParadigmShift]:
        """
        Detect changes in citation patterns.
        
        Measures sudden changes in how papers cite each other,
        indicating potential paradigm shifts.
        """
        current_graph = current_slice.graph
        previous_graph = previous_slice.graph
        
        if current_graph.number_of_nodes() < self.min_papers:
            return None
        
        # Calculate PageRank for both slices
        try:
            current_pr = nx.pagerank(current_graph, max_iter=50)
            previous_pr = nx.pagerank(previous_graph, max_iter=50)
        except:
            return None
        
        # Measure PageRank correlation
        common_nodes = set(current_pr.keys()) & set(previous_pr.keys())
        
        if len(common_nodes) < self.min_papers:
            return None
        
        current_values = np.array([current_pr[n] for n in common_nodes])
        previous_values = np.array([previous_pr[n] for n in common_nodes])
        
        # Spearman correlation
        correlation, _ = stats.spearmanr(current_values, previous_values)
        
        # High correlation = no shift, low correlation = shift
        magnitude = 1 - max(0, correlation)
        
        if magnitude < self.threshold * 0.6:
            return None
        
        # Find papers with largest PageRank changes
        pr_changes = {
            n: abs(current_pr.get(n, 0) - previous_pr.get(n, 0))
            for n in common_nodes
        }
        key_papers = sorted(pr_changes.keys(), key=lambda x: -pr_changes[x])[:10]
        
        return ParadigmShift(
            year=year,
            shift_type='citation',
            magnitude=magnitude,
            affected_field='general',
            key_papers=key_papers,
            description=f"Significant citation pattern change detected (correlation: {correlation:.3f})",
            metrics={'pagerank_correlation': correlation}
        )
    
    def _compute_slice_embedding(
        self,
        paper_ids: List[str],
        network: Any
    ) -> Optional[np.ndarray]:
        """Compute average embedding for a set of papers."""
        # Simplified: use node2vec or similar in production
        # For now, return random embedding
        np.random.seed(hash(tuple(paper_ids)) % (2**32))
        return np.random.randn(self.embedding_dim)
    
    def _find_key_papers(
        self,
        current_graph: nx.DiGraph,
        previous_graph: nx.DiGraph
    ) -> List[str]:
        """Find papers with significant centrality changes."""
        try:
            current_pr = nx.pagerank(current_graph, max_iter=50)
            previous_pr = nx.pagerank(previous_graph, max_iter=50)
        except:
            return []
        
        common_nodes = set(current_pr.keys()) & set(previous_pr.keys())
        
        changes = {}
        for node in common_nodes:
            change = abs(current_pr[node] - previous_pr.get(node, 0))
            changes[node] = change
        
        return sorted(changes.keys(), key=lambda x: -changes[x])[:10]
    
    def _merge_shifts(
        self,
        shifts: List[ParadigmShift]
    ) -> List[ParadigmShift]:
        """Merge overlapping shifts from different methods."""
        if not shifts:
            return []
        
        # Group by year
        by_year = defaultdict(list)
        for shift in shifts:
            by_year[shift.year].append(shift)
        
        merged = []
        for year, year_shifts in by_year.items():
            if len(year_shifts) == 1:
                merged.append(year_shifts[0])
            else:
                # Merge multiple shifts in same year
                max_magnitude = max(s.magnitude for s in year_shifts)
                all_papers = []
                for s in year_shifts:
                    all_papers.extend(s.key_papers)
                
                merged.append(ParadigmShift(
                    year=year,
                    shift_type='combined',
                    magnitude=max_magnitude,
                    affected_field='general',
                    key_papers=list(set(all_papers))[:10],
                    description=f"Multiple shift indicators detected",
                    metrics={'n_methods': len(year_shifts)}
                ))
        
        return sorted(merged, key=lambda x: x.year)
    
    def detect_multi_year(
        self,
        network: Any,
        years: range,
        field: Optional[str] = None
    ) -> List[ParadigmShift]:
        """
        Detect shifts across multiple years with field focus.
        
        Args:
            network: CitationNetwork with temporal slices
            years: Year range to analyze
            field: Optional field to focus on
            
        Returns:
            List of detected shifts
        """
        if field:
            # Analyze field-specific network
            field_graph = network.get_subgraph_by_field(field)
            # Create temporary network with field subgraph
            temp_network = type(network)()
            temp_network.graph = field_graph
            temp_network.papers = {
                k: v for k, v in network.papers.items()
                if k in field_graph
            }
            temp_network.slices = {
                year: network.slices[year]
                for year in years
                if year in network.slices
            }
            return self.detect(temp_network, years)
        
        return self.detect(network, years)
    
    def plot_shift_timeline(
        self,
        shifts: List[ParadigmShift],
        save_path: Optional[str] = None
    ) -> Any:
        """Plot timeline of detected paradigm shifts."""
        import matplotlib.pyplot as plt
        
        if not shifts:
            print("No shifts to plot")
            return None
        
        years = [s.year for s in shifts]
        magnitudes = [s.magnitude for s in shifts]
        types = [s.shift_type for s in shifts]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        colors = {
            'structural': 'blue',
            'semantic': 'green',
            'citation': 'orange',
            'combined': 'red',
        }
        
        for i, (year, mag, typ) in enumerate(zip(years, magnitudes, types)):
            color = colors.get(typ, 'gray')
            ax.scatter(year, mag, s=mag*200, c=color, alpha=0.7, label=typ)
            ax.annotate(str(year), (year, mag), textcoords="offset points", xytext=(0,10), ha='center')
        
        ax.set_xlabel('Year')
        ax.set_ylabel('Shift Magnitude')
        ax.set_title('Paradigm Shift Timeline')
        ax.axhline(y=self.threshold, color='r', linestyle='--', label='Threshold')
        
        # Handle legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig


class KeyPaperDetector:
    """
    Detect papers that catalyze paradigm shifts.
    
    Uses multiple signals:
    - Sudden citation velocity
    - Bridging different communities
    - Introducing new concepts
    """
    
    def __init__(
        self,
        velocity_threshold: float = 2.0,
        bridging_threshold: float = 0.5
    ):
        """
        Initialize key paper detector.
        
        Args:
            velocity_threshold: Std devs above mean for citation velocity
            bridging_threshold: Minimum bridging score
        """
        self.velocity_threshold = velocity_threshold
        self.bridging_threshold = bridging_threshold
    
    def detect_key_papers(
        self,
        network: CitationNetwork,
        year: int
    ) -> List[Tuple[str, Dict[str, float]]]:
        """
        Detect key papers for a given year.
        
        Returns:
            List of (paper_id, metrics) tuples
        """
        graph = network.slices.get(year)
        if not graph:
            return []
        
        graph = graph.graph
        
        key_papers = []
        
        # 1. High citation velocity
        velocity_papers = self._detect_high_velocity(network, year)
        
        # 2. Bridging papers
        bridging_papers = self._detect_bridging_papers(graph)
        
        # Combine signals
        all_candidates = set(velocity_papers.keys()) | set(bridging_papers.keys())
        
        for paper_id in all_candidates:
            metrics = {
                'citation_velocity': velocity_papers.get(paper_id, 0),
                'bridging_score': bridging_papers.get(paper_id, 0),
            }
            key_papers.append((paper_id, metrics))
        
        # Sort by combined score
        key_papers.sort(key=lambda x: -(x[1]['citation_velocity'] + x[1]['bridging_score']))
        
        return key_papers[:20]
    
    def _detect_high_velocity(
        self,
        network: CitationNetwork,
        year: int
    ) -> Dict[str, float]:
        """Detect papers with unusually high citation velocity."""
        velocities = {}
        
        if year not in network.slices or year - 1 not in network.slices:
            return velocities
        
        current = network.slices[year].graph
        previous = network.slices[year - 1].graph
        
        current_citations = dict(current.in_degree())
        previous_citations = dict(previous.in_degree())
        
        for paper_id in current_citations:
            curr = current_citations[paper_id]
            prev = previous_citations.get(paper_id, 0)
            velocity = curr - prev
            velocities[paper_id] = velocity
        
        # Calculate z-scores
        values = list(velocities.values())
        mean_vel = np.mean(values)
        std_vel = np.std(values)
        
        if std_vel > 0:
            z_scores = {
                k: (v - mean_vel) / std_vel
                for k, v in velocities.items()
            }
            
            # Return only high velocity papers
            return {k: v for k, v in z_scores.items() if v > self.velocity_threshold}
        
        return {}
    
    def _detect_bridging_papers(
        self,
        graph: nx.DiGraph
    ) -> Dict[str, float]:
        """Detect papers that bridge communities."""
        # Calculate betweenness centrality
        try:
            betweenness = nx.betweenness_centrality(graph, k=min(500, graph.number_of_nodes()))
        except:
            return {}
        
        # Normalize
        max_val = max(betweenness.values()) if betweenness else 1
        
        return {
            k: v / max_val
            for k, v in betweenness.items()
            if v / max_val > self.bridging_threshold
        }
