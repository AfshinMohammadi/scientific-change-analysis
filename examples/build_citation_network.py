"""
Build Citation Network Example
==============================

Demonstrates how to load academic data and construct citation networks.
"""

import sys
sys.path.insert(0, '..')

from src.data import AcademicDataLoader
from src.networks import CitationNetwork
from src.utils import NetworkVisualizer


def main():
    """Build and analyze a citation network."""
    
    print("=" * 60)
    print("Citation Network Construction")
    print("=" * 60)
    print()
    
    # Initialize data loader
    print("Initializing data loader...")
    loader = AcademicDataLoader(source='semantic_scholar')
    
    # Load papers from file (or search)
    # Option 1: Load from file
    # papers = loader.load_from_file('../data/raw/papers.jsonl')
    
    # Option 2: Search for papers (for demo)
    print("Searching for papers on 'machine learning'...")
    papers = list(loader.search(
        query='machine learning',
        limit=500,
        year_range=(2018, 2023)
    ))
    
    print(f"  Found {len(papers)} papers")
    print()
    
    # Build citation network
    print("Building citation network...")
    network = CitationNetwork.from_papers(papers, show_progress=True)
    
    print(f"  Nodes: {network.graph.number_of_nodes()}")
    print(f"  Edges: {network.graph.number_of_edges()}")
    print()
    
    # Create temporal slices
    print("Creating temporal slices...")
    slices = network.temporal_slice(
        start_year=2018,
        end_year=2023,
        window_size=1
    )
    
    print(f"  Created {len(slices)} temporal slices")
    for year, slice_obj in slices.items():
        print(f"    {year}: {slice_obj.graph.number_of_nodes()} papers, "
              f"{slice_obj.graph.number_of_edges()} citations")
    print()
    
    # Calculate network statistics
    print("Network Statistics:")
    print("-" * 40)
    stats = network.get_statistics()
    print(f"  Total papers: {stats.n_nodes}")
    print(f"  Total citations: {stats.n_edges}")
    print(f"  Density: {stats.density:.6f}")
    print(f"  Avg clustering: {stats.avg_clustering:.4f}")
    print()
    
    # Get top papers
    print("Top papers by citations:")
    top_papers = network.get_top_papers(n=5, by='citations')
    for i, (paper_id, count) in enumerate(top_papers, 1):
        paper = network.get_paper(paper_id)
        title = paper.get('title', 'Unknown')[:50] if paper else 'Unknown'
        print(f"  {i}. {title}... ({count} citations)")
    print()
    
    # Field distribution
    print("Field distribution:")
    field_dist = network.get_field_distribution()
    for field, count in sorted(field_dist.items(), key=lambda x: -x[1])[:5]:
        print(f"  {field}: {count}")
    print()
    
    # Visualizations
    print("Creating visualizations...")
    viz = NetworkVisualizer(network)
    
    # Plot field distribution
    viz.plot_field_distribution(
        top_n=10,
        save_path="../results/figures/field_distribution.png"
    )
    print("  Saved: field_distribution.png")
    
    # Plot temporal evolution
    viz.plot_temporal_evolution(
        slices,
        save_path="../results/figures/temporal_evolution.png"
    )
    print("  Saved: temporal_evolution.png")
    
    print()
    print("Done!")


if __name__ == "__main__":
    main()
