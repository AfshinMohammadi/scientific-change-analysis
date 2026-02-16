"""
Graph Neural Network Module
===========================

Implements GNN architectures using PyTorch Geometric:
- Graph Convolutional Networks (GCN)
- Graph Attention Networks (GAT)
- GraphSAGE
- Heterogeneous GNNs

These models are used for:
- Paper classification
- Citation prediction
- Field emergence prediction
- Node embedding generation
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:
    import torch_geometric
    from torch_geometric.nn import (
        GCNConv, GATConv, SAGEConv, GINConv,
        global_mean_pool, global_max_pool,
        HeteroConv, Linear,
    )
    from torch_geometric.data import Data, HeteroData
    from torch_geometric.utils import from_networkx, negative_sampling
    from torch_geometric.loader import NeighborLoader, LinkNeighborLoader
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    print("Warning: PyTorch Geometric not installed. GNN features disabled.")


@dataclass
class GNNConfig:
    """Configuration for GNN models."""
    hidden_dim: int = 128
    num_layers: int = 3
    dropout: float = 0.3
    learning_rate: float = 0.001
    weight_decay: float = 5e-4
    epochs: int = 200
    patience: int = 20
    batch_size: int = 64
    num_neighbors: List[int] = None
    
    def __post_init__(self):
        if self.num_neighbors is None:
            self.num_neighbors = [25, 10]


class GCN(nn.Module):
    """
    Graph Convolutional Network for node classification.
    
    Architecture:
        Input -> GCN layers -> BatchNorm -> Dropout -> Output
    
    Example:
        >>> model = GCN(num_features=128, hidden_dim=64, num_classes=10)
        >>> output = model(x, edge_index)
    """
    
    def __init__(
        self,
        num_features: int,
        hidden_dim: int = 128,
        num_classes: int = 10,
        num_layers: int = 3,
        dropout: float = 0.3
    ):
        super().__init__()
        
        if not TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError("PyTorch Geometric is required for GCN")
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Build layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        # Input layer
        self.convs.append(GCNConv(num_features, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        # Output layer
        self.convs.append(GCNConv(hidden_dim, num_classes))
    
    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """Forward pass."""
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.convs[-1](x, edge_index)
        return x
    
    def get_embeddings(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """Get node embeddings before final classification."""
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            if i < len(self.convs) - 2:
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class GAT(nn.Module):
    """
    Graph Attention Network with multi-head attention.
    
    Features:
    - Multi-head attention mechanism
    - Learnable attention weights
    - Residual connections
    
    Example:
        >>> model = GAT(num_features=128, hidden_dim=64, num_classes=10, heads=4)
        >>> output = model(x, edge_index)
    """
    
    def __init__(
        self,
        num_features: int,
        hidden_dim: int = 128,
        num_classes: int = 10,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.3
    ):
        super().__init__()
        
        if not TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError("PyTorch Geometric is required for GAT")
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.heads = heads
        
        # Build layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        # Input layer
        self.convs.append(GATConv(num_features, hidden_dim // heads, heads=heads, dropout=dropout))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        # Output layer (single head for classification)
        self.convs.append(GATConv(hidden_dim, num_classes, heads=1, concat=False, dropout=dropout))
    
    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """Forward pass with attention."""
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.convs[-1](x, edge_index)
        return x
    
    def get_attention_weights(self, x: Tensor, edge_index: Tensor) -> List[Tensor]:
        """Extract attention weights for interpretability."""
        attention_weights = []
        
        for conv in self.convs[:-1]:
            x, (edge_idx, attn) = conv(x, edge_index, return_attention_weights=True)
            attention_weights.append(attn)
            x = F.elu(x)
        
        return attention_weights


class GraphSAGEModel(nn.Module):
    """
    GraphSAGE for inductive learning on large graphs.
    
    Features:
    - Neighbor sampling for scalability
    - Inductive learning capability
    - Multiple aggregation methods
    
    Example:
        >>> model = GraphSAGEModel(num_features=128, hidden_dim=64, num_classes=10)
        >>> output = model(x, edge_index)
    """
    
    def __init__(
        self,
        num_features: int,
        hidden_dim: int = 128,
        num_classes: int = 10,
        num_layers: int = 3,
        dropout: float = 0.3,
        aggr: str = 'mean'
    ):
        super().__init__()
        
        if not TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError("PyTorch Geometric is required for GraphSAGE")
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Build layers
        self.convs = nn.ModuleList()
        
        # Input layer
        self.convs.append(SAGEConv(num_features, hidden_dim, aggr=aggr))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim, aggr=aggr))
        
        # Output layer
        self.convs.append(SAGEConv(hidden_dim, num_classes, aggr=aggr))
    
    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """Forward pass."""
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.convs[-1](x, edge_index)
        return x


class HeteroGNN(nn.Module):
    """
    Heterogeneous Graph Neural Network for multi-relational data.
    
    Handles different node types (papers, authors, venues) and
    edge types (cites, writes, publishes).
    """
    
    def __init__(
        self,
        metadata: Tuple[List[str], List[Tuple[str, str, str]]],
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        
        if not TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError("PyTorch Geometric is required for HeteroGNN")
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        node_types, edge_types = metadata
        
        # Create heterogeneous convolution layers
        self.convs = nn.ModuleList()
        
        for i in range(num_layers):
            conv_dict = {}
            for edge_type in edge_types:
                src, rel, dst = edge_type
                in_dim = hidden_dim if i > 0 else -1  # Lazy initialization
                conv_dict[edge_type] = SAGEConv((-1, -1), hidden_dim)
            
            self.convs.append(HeteroConv(conv_dict, aggr='mean'))
    
    def forward(self, x_dict: Dict[str, Tensor], edge_index_dict: Dict[Tuple[str, str, str], Tensor]) -> Dict[str, Tensor]:
        """Forward pass on heterogeneous graph."""
        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.relu(x) for key, x in x_dict.items()}
            if i < self.num_layers - 1:
                x_dict = {key: F.dropout(x, p=self.dropout, training=self.training) 
                         for key, x in x_dict.items()}
        
        return x_dict


class LinkPredictionGNN(nn.Module):
    """
    GNN for link prediction (citation prediction).
    
    Uses encoder-decoder architecture:
    - Encoder: GNN to generate node embeddings
    - Decoder: Dot product for link prediction
    """
    
    def __init__(
        self,
        num_features: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.encoder = GraphSAGEModel(
            num_features=num_features,
            hidden_dim=hidden_dim,
            num_classes=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )
    
    def encode(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """Encode nodes to embeddings."""
        return self.encoder(x, edge_index)
    
    def decode(self, z: Tensor, edge_index: Tensor) -> Tensor:
        """Decode edge probabilities from embeddings."""
        # Dot product decoder
        return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
    
    def decode_all(self, z: Tensor) -> Tensor:
        """Decode all possible edges."""
        prob_adj = torch.matmul(z, z.t())
        return prob_adj
    
    def forward(self, x: Tensor, edge_index: Tensor, pos_edge: Tensor, neg_edge: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward pass with positive and negative edges."""
        z = self.encode(x, edge_index)
        pos_score = self.decode(z, pos_edge)
        neg_score = self.decode(z, neg_edge)
        return pos_score, neg_score


class GNNTrainer:
    """
    Trainer for GNN models with early stopping and logging.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: GNNConfig,
        device: str = 'auto'
    ):
        self.model = model
        self.config = config
        
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    def train_epoch(
        self,
        data: Data,
        train_mask: Optional[Tensor] = None
    ) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        self.optimizer.zero_grad()
        
        out = self.model(data.x, data.edge_index)
        
        if train_mask is not None:
            loss = F.cross_entropy(out[train_mask], data.y[train_mask])
            pred = out[train_mask].argmax(dim=1)
            correct = (pred == data.y[train_mask]).sum().item()
            acc = correct / train_mask.sum().item()
        else:
            loss = F.cross_entropy(out, data.y)
            pred = out.argmax(dim=1)
            acc = (pred == data.y).sum().item() / data.y.size(0)
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item(), acc
    
    @torch.no_grad()
    def evaluate(
        self,
        data: Data,
        mask: Optional[Tensor] = None
    ) -> Tuple[float, float]:
        """Evaluate the model."""
        self.model.eval()
        
        out = self.model(data.x, data.edge_index)
        
        if mask is not None:
            loss = F.cross_entropy(out[mask], data.y[mask])
            pred = out[mask].argmax(dim=1)
            correct = (pred == data.y[mask]).sum().item()
            acc = correct / mask.sum().item()
        else:
            loss = F.cross_entropy(out, data.y)
            pred = out.argmax(dim=1)
            acc = (pred == data.y).sum().item() / data.y.size(0)
        
        return loss.item(), acc
    
    def train(
        self,
        data: Data,
        train_mask: Tensor,
        val_mask: Tensor,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Train the model with early stopping.
        
        Args:
            data: PyG Data object
            train_mask: Boolean mask for training nodes
            val_mask: Boolean mask for validation nodes
            verbose: Whether to print progress
            
        Returns:
            Training history
        """
        data = data.to(self.device)
        train_mask = train_mask.to(self.device)
        val_mask = val_mask.to(self.device)
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(self.config.epochs):
            train_loss, train_acc = self.train_epoch(data, train_mask)
            val_loss, val_acc = self.evaluate(data, val_mask)
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            if verbose and epoch % 10 == 0:
                print(f'Epoch {epoch:3d}: Train Loss={train_loss:.4f}, '
                      f'Train Acc={train_acc:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}')
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    if verbose:
                        print(f'Early stopping at epoch {epoch}')
                    break
        
        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        return self.history


def networkx_to_pyg(
    graph,
    node_features: Optional[Dict[str, np.ndarray]] = None,
    labels: Optional[Dict[str, int]] = None
) -> Data:
    """
    Convert NetworkX graph to PyTorch Geometric Data object.
    
    Args:
        graph: NetworkX graph
        node_features: Dictionary mapping node_id to feature vector
        labels: Dictionary mapping node_id to label
        
    Returns:
        PyG Data object
    """
    if not TORCH_GEOMETRIC_AVAILABLE:
        raise ImportError("PyTorch Geometric is required")
    
    # Convert graph
    data = from_networkx(graph)
    
    # Add features
    if node_features:
        node_list = list(graph.nodes())
        features = []
        for node in node_list:
            if node in node_features:
                features.append(node_features[node])
            else:
                # Use node attributes or random
                if hasattr(graph.nodes[node], '__getitem__'):
                    features.append(np.random.randn(64))
                else:
                    features.append(np.random.randn(64))
        data.x = torch.tensor(np.array(features), dtype=torch.float)
    elif data.x is None:
        # Create random features
        data.x = torch.randn(data.num_nodes, 64)
    
    # Add labels
    if labels:
        node_list = list(graph.nodes())
        label_list = [labels.get(node, 0) for node in node_list]
        data.y = torch.tensor(label_list, dtype=torch.long)
    
    return data


class GNNEmbedder:
    """
    Generate node embeddings using trained GNN models.
    """
    
    def __init__(
        self,
        model_type: str = 'graphsage',
        hidden_dim: int = 128,
        num_layers: int = 3,
        device: str = 'auto'
    ):
        self.model_type = model_type
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model = None
    
    def fit_transform(
        self,
        graph,
        node_features: Optional[Dict] = None,
        epochs: int = 50,
        verbose: bool = False
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Generate embeddings for graph nodes.
        
        Args:
            graph: NetworkX graph
            node_features: Optional node features
            epochs: Training epochs
            verbose: Print progress
            
        Returns:
            Tuple of (embeddings array, node id list)
        """
        # Convert to PyG
        data = networkx_to_pyg(graph, node_features)
        data = data.to(self.device)
        
        num_features = data.x.size(1)
        
        # Create model
        if self.model_type == 'gcn':
            self.model = GCN(
                num_features=num_features,
                hidden_dim=self.hidden_dim,
                num_classes=self.hidden_dim,
                num_layers=self.num_layers
            )
        elif self.model_type == 'gat':
            self.model = GAT(
                num_features=num_features,
                hidden_dim=self.hidden_dim,
                num_classes=self.hidden_dim,
                num_layers=self.num_layers
            )
        else:  # graphsage
            self.model = GraphSAGEModel(
                num_features=num_features,
                hidden_dim=self.hidden_dim,
                num_classes=self.hidden_dim,
                num_layers=self.num_layers
            )
        
        self.model.to(self.device)
        
        # Unsupervised training via reconstruction
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Get embeddings
            out = self.model(data.x, data.edge_index)
            
            # Reconstruction loss (adjacency)
            adj = torch.zeros(data.num_nodes, data.num_nodes, device=self.device)
            adj[data.edge_index[0], data.edge_index[1]] = 1
            
            # Inner product decoder
            recon = torch.sigmoid(torch.mm(out, out.t()))
            
            # Binary cross entropy
            loss = F.binary_cross_entropy(recon.view(-1), adj.view(-1))
            
            loss.backward()
            optimizer.step()
            
            if verbose and epoch % 10 == 0:
                print(f'Epoch {epoch}: Loss={loss.item():.4f}')
        
        # Extract embeddings
        self.model.eval()
        with torch.no_grad():
            embeddings = self.model(data.x, data.edge_index).cpu().numpy()
        
        node_ids = list(graph.nodes())
        
        return embeddings, node_ids
