"""
Clustering Module
=================

Implements advanced clustering using UMAP for dimensionality reduction
and HDBSCAN for density-based clustering.

Features:
- UMAP for visualization and preprocessing
- HDBSCAN for automatic cluster detection
- Cluster analysis and validation metrics
- Integration with embeddings pipeline
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import time

import numpy as np
from tqdm import tqdm

try:
    import umap
    import umap.plot
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False

from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
    normalized_mutual_info_score
)
from sklearn.neighbors import NearestNeighbors


@dataclass
class UMAPConfig:
    """Configuration for UMAP dimensionality reduction."""
    n_components: int = 2
    n_neighbors: int = 15
    min_dist: float = 0.1
    metric: str = 'cosine'
    spread: float = 1.0
    learning_rate: float = 1.0
    n_epochs: int = 200
    random_state: int = 42
    low_memory: bool = False
    transform_seed: int = 42
    
    # For supervised UMAP
    target_metric: str = 'categorical'
    target_weight: float = 0.5


@dataclass
class HDBSCANConfig:
    """Configuration for HDBSCAN clustering."""
    min_cluster_size: int = 10
    min_samples: int = 5
    cluster_selection_epsilon: float = 0.0
    metric: str = 'euclidean'
    alpha: float = 1.0
    cluster_selection_method: str = 'eom'  # 'eom' or 'leaf'
    prediction_data: bool = True
    min_cluster_size_ratio: float = 0.01  # As ratio of data size


@dataclass
class ClusterResult:
    """Result from clustering analysis."""
    labels: np.ndarray
    probabilities: Optional[np.ndarray]
    cluster_centers: Optional[np.ndarray]
    n_clusters: int
    n_noise: int
    metrics: Dict[str, float]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'n_clusters': self.n_clusters,
            'n_noise': self.n_noise,
            'metrics': self.metrics,
            'metadata': self.metadata,
        }


class UMAPReducer:
    """
    UMAP-based dimensionality reduction for embeddings.
    
    UMAP (Uniform Manifold Approximation and Projection) preserves
    both local and global structure better than t-SNE.
    
    Example:
        >>> reducer = UMAPReducer(n_components=2, n_neighbors=15)
        >>> embedding_2d = reducer.fit_transform(embeddings)
        >>> reducer.plot(embeddings, labels=cluster_labels)
    """
    
    def __init__(self, config: Optional[UMAPConfig] = None):
        self.config = config or UMAPConfig()
        self.model = None
        self.embedding_ = None
    
    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        progress: bool = True
    ) -> 'UMAPReducer':
        """
        Fit UMAP on data.
        
        Args:
            X: Data matrix (n_samples, n_features)
            y: Optional target labels for supervised UMAP
            progress: Show progress bar
            
        Returns:
            Self for chaining
        """
        if not UMAP_AVAILABLE:
            raise ImportError("umap-learn is required. Install with: pip install umap-learn")
        
        self.model = umap.UMAP(
            n_components=self.config.n_components,
            n_neighbors=self.config.n_neighbors,
            min_dist=self.config.min_dist,
            metric=self.config.metric,
            spread=self.config.spread,
            learning_rate=self.config.learning_rate,
            n_epochs=self.config.n_epochs if progress else None,
            random_state=self.config.random_state,
            low_memory=self.config.low_memory,
            transform_seed=self.config.transform_seed,
        )
        
        if y is not None:
            # Supervised UMAP
            self.model.fit(X, y=y)
        else:
            self.model.fit(X)
        
        self.embedding_ = self.model.embedding_
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform new data to UMAP space."""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model.transform(X)
    
    def fit_transform(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        progress: bool = True
    ) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(X, y, progress)
        return self.embedding_
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Transform from UMAP space back to original space."""
        if self.model is None:
            raise ValueError("Model not fitted.")
        return self.model.inverse_transform(X)
    
    def plot(
        self,
        X: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
        values: Optional[np.ndarray] = None,
        theme: str = 'viridis',
        save_path: Optional[str] = None
    ) -> Any:
        """
        Create interactive UMAP visualization.
        
        Args:
            X: Data to plot (uses fitted embedding if None)
            labels: Cluster/community labels
            values: Continuous values for coloring
            theme: Color theme
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        import matplotlib.pyplot as plt
        
        if X is None:
            embedding = self.embedding_
        else:
            embedding = self.transform(X)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        if labels is not None:
            scatter = ax.scatter(
                embedding[:, 0],
                embedding[:, 1],
                c=labels,
                cmap=theme,
                alpha=0.7,
                s=10
            )
            plt.colorbar(scatter, ax=ax, label='Cluster')
        elif values is not None:
            scatter = ax.scatter(
                embedding[:, 0],
                embedding[:, 1],
                c=values,
                cmap=theme,
                alpha=0.7,
                s=10
            )
            plt.colorbar(scatter, ax=ax, label='Value')
        else:
            ax.scatter(
                embedding[:, 0],
                embedding[:, 1],
                alpha=0.7,
                s=10
            )
        
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.set_title('UMAP Projection')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig


class HDBSCANClusterer:
    """
    HDBSCAN clustering for automatic cluster detection.
    
    HDBSCAN (Hierarchical Density-Based Spatial Clustering) finds
    clusters of varying densities without requiring the number
    of clusters as input.
    
    Features:
    - Automatic cluster count detection
    - Noise/outlier identification
    - Soft clustering with membership probabilities
    - Variable density cluster detection
    
    Example:
        >>> clusterer = HDBSCANClusterer(min_cluster_size=15)
        >>> result = clusterer.fit_predict(embeddings)
        >>> print(f"Found {result.n_clusters} clusters")
    """
    
    def __init__(self, config: Optional[HDBSCANConfig] = None):
        self.config = config or HDBSCANConfig()
        self.model = None
        self.labels_ = None
        self.probabilities_ = None
    
    def fit(
        self,
        X: np.ndarray,
        progress: bool = True
    ) -> 'HDBSCANClusterer':
        """
        Fit HDBSCAN on data.
        
        Args:
            X: Data matrix (n_samples, n_features)
            progress: Show progress (HDBSCAN handles internally)
            
        Returns:
            Self for chaining
        """
        if not HDBSCAN_AVAILABLE:
            raise ImportError("hdbscan is required. Install with: pip install hdbscan")
        
        # Adjust min_cluster_size if using ratio
        min_cluster_size = self.config.min_cluster_size
        if self.config.min_cluster_size_ratio > 0:
            min_cluster_size = max(
                min_cluster_size,
                int(len(X) * self.config.min_cluster_size_ratio)
            )
        
        self.model = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=self.config.min_samples,
            cluster_selection_epsilon=self.config.cluster_selection_epsilon,
            metric=self.config.metric,
            alpha=self.config.alpha,
            cluster_selection_method=self.config.cluster_selection_method,
            prediction_data=self.config.prediction_data,
        )
        
        self.model.fit(X)
        
        self.labels_ = self.model.labels_
        self.probabilities_ = self.model.probabilities_
        
        return self
    
    def predict(
        self,
        X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict cluster labels for new data.
        
        Args:
            X: New data points
            
        Returns:
            Tuple of (labels, probabilities)
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if not self.config.prediction_data:
            raise ValueError("prediction_data must be True for prediction")
        
        labels, strengths = hdbscan.approximate_predict(self.model, X)
        return labels, strengths
    
    def fit_predict(
        self,
        X: np.ndarray,
        return_metrics: bool = True
    ) -> ClusterResult:
        """
        Fit and return cluster results.
        
        Args:
            X: Data matrix
            return_metrics: Calculate validation metrics
            
        Returns:
            ClusterResult with labels and analysis
        """
        self.fit(X)
        
        labels = self.labels_
        probabilities = self.probabilities_
        
        # Calculate statistics
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        # Calculate metrics
        metrics = {}
        if return_metrics and n_clusters > 1:
            # Only calculate for non-noise points
            mask = labels != -1
            if mask.sum() > n_clusters:
                X_filtered = X[mask]
                labels_filtered = labels[mask]
                
                try:
                    metrics['silhouette'] = silhouette_score(X_filtered, labels_filtered)
                except:
                    metrics['silhouette'] = -1
                
                try:
                    metrics['calinski_harabasz'] = calinski_harabasz_score(X_filtered, labels_filtered)
                except:
                    metrics['calinski_harabasz'] = 0
        
        # Cluster sizes
        unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
        metrics['cluster_sizes'] = dict(zip(unique_labels.tolist(), counts.tolist()))
        metrics['avg_cluster_size'] = np.mean(counts) if len(counts) > 0 else 0
        metrics['noise_ratio'] = n_noise / len(labels)
        
        return ClusterResult(
            labels=labels,
            probabilities=probabilities,
            cluster_centers=None,  # HDBSCAN doesn't have explicit centers
            n_clusters=n_clusters,
            n_noise=n_noise,
            metrics=metrics,
            metadata={
                'min_cluster_size': self.config.min_cluster_size,
                'min_samples': self.config.min_samples,
            }
        )
    
    def get_cluster_exemplars(
        self,
        X: np.ndarray
    ) -> Dict[int, List[int]]:
        """
        Get exemplar points for each cluster.
        
        Exemplars are points most representative of their cluster.
        
        Args:
            X: Original data matrix
            
        Returns:
            Dictionary mapping cluster label to exemplar indices
        """
        if self.model is None:
            raise ValueError("Model not fitted.")
        
        exemplars = {}
        
        for cluster_label in set(self.labels_):
            if cluster_label == -1:
                continue
            
            # Get points in cluster
            cluster_mask = self.labels_ == cluster_label
            cluster_indices = np.where(cluster_mask)[0]
            
            if len(cluster_indices) == 0:
                continue
            
            # Get points with highest probability
            cluster_probs = self.probabilities_[cluster_mask]
            top_indices = cluster_indices[np.argsort(cluster_probs)[-5:]]
            
            exemplars[cluster_label] = top_indices.tolist()
        
        return exemplars
    
    def condensed_tree_(self):
        """Get the condensed tree for visualization."""
        if self.model is None:
            raise ValueError("Model not fitted.")
        return self.model.condensed_tree_
    
    def plot_condensed_tree(
        self,
        save_path: Optional[str] = None
    ) -> Any:
        """Plot the condensed tree hierarchy."""
        import matplotlib.pyplot as plt
        
        if self.model is None:
            raise ValueError("Model not fitted.")
        
        fig, ax = plt.subplots(figsize=(12, 8))
        self.model.condensed_tree_.plot(select_clusters=True, axis=ax)
        
        plt.title('HDBSCAN Condensed Tree')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig


class ClusteringPipeline:
    """
    Complete clustering pipeline: UMAP + HDBSCAN.
    
    Combines dimensionality reduction and clustering for
    optimal results on high-dimensional embedding data.
    
    Example:
        >>> pipeline = ClusteringPipeline(
        ...     umap_dim=50,
        ...     min_cluster_size=20
        ... )
        >>> result = pipeline.fit_predict(embeddings)
    """
    
    def __init__(
        self,
        umap_dim: int = 50,
        umap_n_neighbors: int = 15,
        min_cluster_size: int = 15,
        min_samples: int = 5,
        random_state: int = 42
    ):
        self.umap_dim = umap_dim
        self.umap_n_neighbors = umap_n_neighbors
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.random_state = random_state
        
        self.umap_model = None
        self.hdbscan_model = None
        self.reduced_data_ = None
    
    def fit(
        self,
        X: np.ndarray,
        progress: bool = True
    ) -> 'ClusteringPipeline':
        """
        Fit the complete pipeline.
        
        Args:
            X: High-dimensional data
            progress: Show progress
            
        Returns:
            Self for chaining
        """
        # Step 1: UMAP reduction
        if progress:
            print("Step 1: UMAP dimensionality reduction...")
        
        umap_config = UMAPConfig(
            n_components=self.umap_dim,
            n_neighbors=self.umap_n_neighbors,
            random_state=self.random_state
        )
        
        self.umap_model = UMAPReducer(umap_config)
        self.reduced_data_ = self.umap_model.fit_transform(X, progress=progress)
        
        # Step 2: HDBSCAN clustering
        if progress:
            print("Step 2: HDBSCAN clustering...")
        
        hdbscan_config = HDBSCANConfig(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples
        )
        
        self.hdbscan_model = HDBSCANClusterer(hdbscan_config)
        self.hdbscan_model.fit(self.reduced_data_)
        
        return self
    
    def predict(
        self,
        X: np.ndarray
    ) -> np.ndarray:
        """Predict cluster labels for new data."""
        if self.umap_model is None or self.hdbscan_model is None:
            raise ValueError("Pipeline not fitted.")
        
        reduced = self.umap_model.transform(X)
        labels, _ = self.hdbscan_model.predict(reduced)
        return labels
    
    def fit_predict(
        self,
        X: np.ndarray,
        progress: bool = True
    ) -> ClusterResult:
        """Fit and return results."""
        self.fit(X, progress)
        
        labels = self.hdbscan_model.labels_
        probabilities = self.hdbscan_model.probabilities_
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        metrics = {}
        if n_clusters > 1:
            mask = labels != -1
            if mask.sum() > n_clusters:
                try:
                    metrics['silhouette'] = silhouette_score(
                        self.reduced_data_[mask],
                        labels[mask]
                    )
                except:
                    metrics['silhouette'] = -1
        
        metrics['noise_ratio'] = n_noise / len(labels)
        
        return ClusterResult(
            labels=labels,
            probabilities=probabilities,
            cluster_centers=None,
            n_clusters=n_clusters,
            n_noise=n_noise,
            metrics=metrics,
            metadata={
                'umap_dim': self.umap_dim,
                'min_cluster_size': self.min_cluster_size,
            }
        )
    
    def visualize(
        self,
        X: Optional[np.ndarray] = None,
        save_path: Optional[str] = None
    ) -> Any:
        """Create 2D visualization of clusters."""
        import matplotlib.pyplot as plt
        
        # Reduce to 2D for visualization
        viz_config = UMAPConfig(n_components=2, random_state=self.random_state)
        viz_reducer = UMAPReducer(viz_config)
        
        if X is not None:
            embedding_2d = viz_reducer.fit_transform(X)
        else:
            embedding_2d = viz_reducer.fit_transform(self.reduced_data_)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot noise points first
        noise_mask = self.hdbscan_model.labels_ == -1
        ax.scatter(
            embedding_2d[noise_mask, 0],
            embedding_2d[noise_mask, 1],
            c='gray',
            alpha=0.3,
            s=5,
            label='Noise'
        )
        
        # Plot clusters
        scatter = ax.scatter(
            embedding_2d[~noise_mask, 0],
            embedding_2d[~noise_mask, 1],
            c=self.hdbscan_model.labels_[~noise_mask],
            cmap='tab20',
            alpha=0.7,
            s=10
        )
        
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.set_title(f'Clustering Results ({self.hdbscan_model.labels_.max() + 1} clusters)')
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig


def find_optimal_clusters(
    X: np.ndarray,
    max_clusters: int = 50,
    method: str = 'silhouette',
    progress: bool = True
) -> Dict[str, Any]:
    """
    Find optimal number of clusters using various metrics.
    
    Args:
        X: Data matrix
        max_clusters: Maximum clusters to test
        method: Optimization criterion
        progress: Show progress
        
    Returns:
        Dictionary with optimal k and scores
    """
    from sklearn.cluster import KMeans
    
    scores = {
        'silhouette': [],
        'calinski_harabasz': [],
        'davies_bouldin': [],
    }
    
    k_range = range(2, min(max_clusters + 1, len(X)))
    
    iterator = k_range
    if progress:
        iterator = tqdm(k_range, desc="Finding optimal k")
    
    for k in iterator:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        
        try:
            scores['silhouette'].append(silhouette_score(X, labels))
        except:
            scores['silhouette'].append(-1)
        
        try:
            scores['calinski_harabasz'].append(calinski_harabasz_score(X, labels))
        except:
            scores['calinski_harabasz'].append(0)
        
        try:
            scores['davies_bouldin'].append(davies_bouldin_score(X, labels))
        except:
            scores['davies_bouldin'].append(float('inf'))
    
    # Find optimal k
    if method == 'silhouette':
        optimal_k = k_range[np.argmax(scores['silhouette'])]
    elif method == 'calinski_harabasz':
        optimal_k = k_range[np.argmax(scores['calinski_harabasz'])]
    elif method == 'davies_bouldin':
        optimal_k = k_range[np.argmin(scores['davies_bouldin'])]
    else:
        optimal_k = k_range[np.argmax(scores['silhouette'])]
    
    return {
        'optimal_k': optimal_k,
        'scores': scores,
        'k_range': list(k_range),
    }
