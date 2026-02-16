"""
Graph Neural Network Module
===========================

GNN implementations using PyTorch Geometric.
"""

from src.models.gnn.gat import (
    GCN,
    GAT,
    GraphSAGEModel,
    HeteroGNN,
    LinkPredictionGNN,
    GNNTrainer,
    GNNConfig,
    GNNEmbedder,
    networkx_to_pyg,
    TORCH_GEOMETRIC_AVAILABLE,
)

__all__ = [
    "GCN",
    "GAT",
    "GraphSAGEModel",
    "HeteroGNN",
    "LinkPredictionGNN",
    "GNNTrainer",
    "GNNConfig",
    "GNNEmbedder",
    "networkx_to_pyg",
    "TORCH_GEOMETRIC_AVAILABLE",
]
