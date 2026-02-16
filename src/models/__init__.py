"""
Models Module
=============

Paradigm detection, prediction, embedding, and GNN models.
"""

from src.models.paradigm_detector import (
    ParadigmShiftDetector,
    KeyPaperDetector,
    ParadigmShift,
    ShiftMetrics,
)
from src.models.embeddings import (
    GraphEmbedder,
    TopicEmbedder,
    EmbeddingResult,
)

# Advanced embeddings
from src.models.node2vec import (
    Node2VecEmbedder,
    Node2VecConfig,
    DeepWalk,
    BiasedRandomWalker,
    Metapath2Vec,
)

# Clustering
from src.models.clustering import (
    UMAPReducer,
    UMAPConfig,
    HDBSCANClusterer,
    HDBSCANConfig,
    ClusteringPipeline,
    ClusterResult,
    find_optimal_clusters,
)

# Community detection
from src.models.community_detection import (
    LouvainDetector,
    LeidenDetector,
    MultiResolutionDetector,
    CommunityEvolutionTracker,
    CommunityResult,
    CommunityEvolution,
    calculate_community_metrics,
    LOUVAIN_AVAILABLE,
    LEIDEN_AVAILABLE,
)

# Sentence embeddings
from src.models.sentence_embeddings import (
    PaperEmbedder,
    SentenceTransformerConfig,
    SemanticClusterer,
    CrossLingualMatcher,
    create_paper_embeddings,
    SENTENCE_TRANSFORMERS_AVAILABLE,
)

__all__ = [
    # Paradigm detection
    "ParadigmShiftDetector",
    "KeyPaperDetector",
    "ParadigmShift",
    "ShiftMetrics",
    
    # Basic embeddings
    "GraphEmbedder",
    "TopicEmbedder",
    "EmbeddingResult",
    
    # Node2Vec
    "Node2VecEmbedder",
    "Node2VecConfig",
    "DeepWalk",
    "BiasedRandomWalker",
    "Metapath2Vec",
    
    # Clustering
    "UMAPReducer",
    "UMAPConfig",
    "HDBSCANClusterer",
    "HDBSCANConfig",
    "ClusteringPipeline",
    "ClusterResult",
    "find_optimal_clusters",
    
    # Community detection
    "LouvainDetector",
    "LeidenDetector",
    "MultiResolutionDetector",
    "CommunityEvolutionTracker",
    "CommunityResult",
    "CommunityEvolution",
    "calculate_community_metrics",
    "LOUVAIN_AVAILABLE",
    "LEIDEN_AVAILABLE",
    
    # Sentence embeddings
    "PaperEmbedder",
    "SentenceTransformerConfig",
    "SemanticClusterer",
    "CrossLingualMatcher",
    "create_paper_embeddings",
    "SENTENCE_TRANSFORMERS_AVAILABLE",
]
