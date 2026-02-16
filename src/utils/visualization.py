"""
Visualization Module
====================

Tools for visualizing networks, paradigm shifts, and analysis results.
"""

from typing import Any, Dict, List, Optional, Tuple
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection


class NetworkVisualizer:
    """Visualization toolkit for scientific networks."""
    
    def __init__(self, network: Any, output_dir: str = "results/figures"):
        """
        Initialize visualizer.
        
        Args:
            network: CitationNetwork or CoAuthorshipNetwork
            output_dir: Directory for saving figures
        """
        self.network = network
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def plot_network_overview(
        self,
        max_nodes: int = 1000,
        layout: str = 'spring',
        node_size: str = 'degree',
        title: str = "Network Overview",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot overall network structure.
        
        Args:
            max_nodes: Maximum nodes to display
            layout: Layout algorithm ('spring', 'kamada_kawai', 'circular')
            node_size: Size metric ('degree', 'pagerank', 'uniform')
            title: Plot title
            save_path: Path to save figure
        """
        graph = self.network.graph
        
        if graph.number_of_nodes() > max_nodes:
            # Sample nodes
            import random
            nodes = random.sample(list(graph.nodes()), max_nodes)
            graph = graph.subgraph(nodes)
        
        fig, ax = plt.subplots(figsize=(14, 12))
        
        # Calculate layout
        if layout == 'spring':
            pos = plt.nx.spring_layout(graph, seed=42, k=2/np.sqrt(graph.number_of_nodes()))
        elif layout == 'kamada_kawai':
            pos = plt.nx.kamada_kawai_layout(graph)
        else:
            pos = plt.nx.circular_layout(graph)
        
        # Calculate node sizes
        if node_size == 'degree':
            sizes = [100 + 20 * graph.degree(n) for n in graph.nodes()]
        elif node_size == 'pagerank':
            import networkx as nx
            pr = nx.pagerank(graph)
            sizes = [100 + 10000 * pr[n] for n in graph.nodes()]
        else:
            sizes = 100
        
        # Color by year if available
        node_colors = []
        for node in graph.nodes():
            node_data = graph.nodes[node]
            year = node_data.get('year', 0)
            node_colors.append(year)
        
        # Draw edges
        plt.nx.draw_networkx_edges(graph, pos, alpha=0.1, ax=ax)
        
        # Draw nodes
        scatter = plt.nx.draw_networkx_nodes(
            graph, pos,
            node_size=sizes,
            node_color=node_colors,
            cmap=plt.cm.viridis,
            alpha=0.7,
            ax=ax
        )
        
        ax.set_title(title, fontsize=14)
        ax.axis('off')
        
        # Add colorbar
        if any(node_colors):
            cbar = plt.colorbar(scatter, ax=ax, label='Publication Year')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_temporal_evolution(
        self,
        slices: Dict[int, Any],
        metrics: List[str] = None,
        title: str = "Network Evolution",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot network metrics over time.
        
        Args:
            slices: Dictionary of year -> network slice
            metrics: Metrics to plot
            title: Plot title
            save_path: Path to save figure
        """
        metrics = metrics or ['n_nodes', 'n_edges', 'density', 'avg_clustering']
        
        years = sorted(slices.keys())
        data = {m: [] for m in metrics}
        
        for year in years:
            slice_obj = slices[year]
            stats = slice_obj.stats if hasattr(slice_obj, 'stats') else slice_obj
            
            for m in metrics:
                if hasattr(stats, m):
                    data[m].append(getattr(stats, m))
                elif isinstance(stats, dict):
                    data[m].append(stats.get(m, 0))
                else:
                    data[m].append(0)
        
        n_metrics = len(metrics)
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for i, (metric, values) in enumerate(data.items()):
            if i >= len(axes):
                break
            
            ax = axes[i]
            ax.plot(years, values, 'o-', linewidth=2, markersize=6)
            ax.fill_between(years, values, alpha=0.3)
            ax.set_xlabel('Year')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(metric.replace('_', ' ').title())
            ax.grid(True, alpha=0.3)
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_communities(
        self,
        communities: Dict[str, int],
        max_nodes: int = 1000,
        title: str = "Community Structure",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot network with community coloring.
        
        Args:
            communities: Dictionary mapping node_id to community label
            max_nodes: Maximum nodes to display
            title: Plot title
            save_path: Path to save figure
        """
        import networkx as nx
        
        graph = self.network.graph
        
        # Filter to nodes with community assignments
        nodes_with_comm = [n for n in graph.nodes() if n in communities]
        if len(nodes_with_comm) > max_nodes:
            nodes_with_comm = list(np.random.choice(nodes_with_comm, max_nodes, replace=False))
        
        subgraph = graph.subgraph(nodes_with_comm)
        
        fig, ax = plt.subplots(figsize=(14, 12))
        
        pos = nx.spring_layout(subgraph, seed=42)
        
        # Get community colors
        unique_comms = list(set(communities.values()))
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_comms)))
        comm_to_color = {c: colors[i] for i, c in enumerate(unique_comms)}
        
        node_colors = [comm_to_color[communities[n]] for n in subgraph.nodes()]
        
        # Draw
        nx.draw_networkx_edges(subgraph, pos, alpha=0.1, ax=ax)
        nx.draw_networkx_nodes(
            subgraph, pos,
            node_color=node_colors,
            node_size=50,
            alpha=0.7,
            ax=ax
        )
        
        ax.set_title(title, fontsize=14)
        ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_field_distribution(
        self,
        top_n: int = 20,
        title: str = "Papers by Field",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot distribution of papers across fields."""
        field_dist = self.network.get_field_distribution()
        
        # Sort and take top N
        sorted_fields = sorted(field_dist.items(), key=lambda x: -x[1])[:top_n]
        fields = [f[0] for f in sorted_fields]
        counts = [f[1] for f in sorted_fields]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        bars = ax.barh(range(len(fields)), counts, color='steelblue')
        ax.set_yticks(range(len(fields)))
        ax.set_yticklabels(fields)
        ax.set_xlabel('Number of Papers')
        ax.set_title(title)
        ax.invert_yaxis()
        
        # Add value labels
        for bar, count in zip(bars, counts):
            ax.text(bar.get_width() + 100, bar.get_y() + bar.get_height()/2,
                   f'{count:,}', va='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig


class ShiftVisualizer:
    """Visualization tools for paradigm shift analysis."""
    
    def __init__(self, output_dir: str = "results/figures"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def plot_shift_timeline(
        self,
        shifts: List[Any],
        title: str = "Paradigm Shift Timeline",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot timeline of detected paradigm shifts."""
        if not shifts:
            print("No shifts to visualize")
            return None
        
        years = [s.year for s in shifts]
        magnitudes = [s.magnitude for s in shifts]
        types = [s.shift_type for s in shifts]
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Color by type
        type_colors = {
            'structural': '#3498db',
            'semantic': '#2ecc71',
            'citation': '#e74c3c',
            'combined': '#9b59b6',
        }
        
        for year, mag, typ in zip(years, magnitudes, types):
            color = type_colors.get(typ, 'gray')
            ax.scatter(year, mag, s=mag*300, c=color, alpha=0.7, edgecolors='black')
            ax.annotate(str(year), (year, mag), textcoords="offset points",
                       xytext=(0, 15), ha='center', fontsize=10)
        
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Shift Magnitude', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add threshold line
        ax.axhline(y=0.7, color='red', linestyle='--', alpha=0.5, label='Threshold')
        
        # Legend
        handles = [mpatches.Patch(color=c, label=t) for t, c in type_colors.items()]
        ax.legend(handles=handles, loc='upper right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_shift_comparison(
        self,
        shifts: List[Any],
        title: str = "Shift Characteristics Comparison",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Compare characteristics of detected shifts."""
        if not shifts:
            return None
        
        # Extract metrics
        shift_data = []
        for s in shifts:
            row = {
                'year': s.year,
                'magnitude': s.magnitude,
                'type': s.shift_type,
            }
            row.update(s.metrics)
            shift_data.append(row)
        
        import pandas as pd
        df = pd.DataFrame(shift_data)
        
        # Plot
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Magnitude over time
        ax = axes[0, 0]
        ax.bar(df['year'], df['magnitude'], color='steelblue', alpha=0.7)
        ax.set_xlabel('Year')
        ax.set_ylabel('Magnitude')
        ax.set_title('Shift Magnitude by Year')
        
        # Type distribution
        ax = axes[0, 1]
        type_counts = df['type'].value_counts()
        ax.pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%')
        ax.set_title('Shift Types Distribution')
        
        # Key papers count
        ax = axes[1, 0]
        key_paper_counts = [len(s.key_papers) for s in shifts]
        ax.bar(df['year'], key_paper_counts, color='coral', alpha=0.7)
        ax.set_xlabel('Year')
        ax.set_ylabel('Number of Key Papers')
        ax.set_title('Key Papers per Shift')
        
        # Magnitude distribution
        ax = axes[1, 1]
        ax.hist(df['magnitude'], bins=10, color='green', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Magnitude')
        ax.set_ylabel('Count')
        ax.set_title('Magnitude Distribution')
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_embedding_evolution(
        self,
        embeddings_by_year: Dict[int, Any],
        labels: Optional[Dict[str, int]] = None,
        title: str = "Semantic Space Evolution",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot how embeddings evolve over time."""
        years = sorted(embeddings_by_year.keys())
        n_years = len(years)
        
        fig, axes = plt.subplots(1, min(n_years, 4), figsize=(5*min(n_years, 4), 5))
        if n_years == 1:
            axes = [axes]
        
        for i, year in enumerate(years[:4]):
            ax = axes[i]
            emb = embeddings_by_year[year]
            
            if emb.embeddings.shape[1] > 2:
                emb = emb.reduce_dimensions(n_components=2)
            
            x = emb.embeddings[:, 0]
            y = emb.embeddings[:, 1]
            
            if labels:
                colors = [labels.get(n, 0) for n in emb.node_ids]
                scatter = ax.scatter(x, y, c=colors, alpha=0.5, s=20, cmap='tab10')
            else:
                ax.scatter(x, y, alpha=0.5, s=20)
            
            ax.set_title(f'Year {year}')
            ax.set_xlabel('Dim 1')
            ax.set_ylabel('Dim 2')
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig


def create_summary_report(
    network: Any,
    shifts: List[Any],
    output_path: str = "results/analysis_report.txt"
) -> str:
    """Create text summary of analysis results."""
    report = []
    report.append("=" * 60)
    report.append("SCIENTIFIC CHANGE ANALYSIS REPORT")
    report.append("=" * 60)
    report.append("")
    
    # Network statistics
    stats = network.get_statistics()
    report.append("NETWORK STATISTICS")
    report.append("-" * 40)
    report.append(f"Total Papers: {stats.n_nodes:,}")
    report.append(f"Total Citations: {stats.n_edges:,}")
    report.append(f"Network Density: {stats.density:.6f}")
    report.append(f"Avg Clustering: {stats.avg_clustering:.4f}")
    report.append("")
    
    # Paradigm shifts
    report.append("PARADIGM SHIFTS DETECTED")
    report.append("-" * 40)
    report.append(f"Total Shifts: {len(shifts)}")
    report.append("")
    
    for shift in shifts:
        report.append(f"Year {shift.year}:")
        report.append(f"  Type: {shift.shift_type}")
        report.append(f"  Magnitude: {shift.magnitude:.3f}")
        report.append(f"  Key Papers: {len(shift.key_papers)}")
        report.append("")
    
    report_content = "\n".join(report)
    
    with open(output_path, 'w') as f:
        f.write(report_content)
    
    return report_content
