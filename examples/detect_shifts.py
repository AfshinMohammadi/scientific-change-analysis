"""
Detect Paradigm Shifts Example
==============================

Demonstrates paradigm shift detection in scientific fields.
"""

import sys
sys.path.insert(0, '..')

from src.data import AcademicDataLoader
from src.networks import CitationNetwork
from src.models import ParadigmShiftDetector, GraphEmbedder
from src.utils import NetworkVisualizer, ShiftVisualizer, create_summary_report
import numpy as np


def main():
    """Detect and analyze paradigm shifts."""
    
    print("=" * 60)
    print("Paradigm Shift Detection")
    print("=" * 60)
    print()
    
    # Load or create network (use pre-built for demo)
    print("Loading citation network...")
    # In practice, load from saved file:
    # network = CitationNetwork.load_graphml('../data/processed/network.graphml')
    
    # For demo, create synthetic data
    print("  Creating synthetic network for demonstration...")
    loader = AcademicDataLoader(source='semantic_scholar')
    papers = list(loader.search(query='deep learning', limit=1000, year_range=(2015, 2023)))
    
    network = CitationNetwork.from_papers(papers, show_progress=True)
    network.temporal_slice(start_year=2015, end_year=2023)
    
    print(f"  Network: {network.graph.number_of_nodes()} nodes, "
          f"{network.graph.number_of_edges()} edges")
    print()
    
    # Initialize detector
    print("Initializing paradigm shift detector...")
    detector = ParadigmShiftDetector(
        threshold=0.5,  # Lower threshold for demo
        min_papers=30,
        methods=['structural', 'citation_flow']
    )
    print()
    
    # Detect shifts
    print("Detecting paradigm shifts...")
    shifts = detector.detect(
        network,
        years=range(2017, 2023)
    )
    
    print(f"  Detected {len(shifts)} paradigm shifts")
    print()
    
    # Display results
    if shifts:
        print("Detected Paradigm Shifts:")
        print("-" * 40)
        
        for shift in shifts:
            print(f"\nYear: {shift.year}")
            print(f"  Type: {shift.shift_type}")
            print(f"  Magnitude: {shift.magnitude:.3f}")
            print(f"  Description: {shift.description}")
            
            if shift.key_papers:
                print(f"  Key papers: {len(shift.key_papers)}")
                for i, paper_id in enumerate(shift.key_papers[:3], 1):
                    paper = network.get_paper(paper_id)
                    title = paper.get('title', 'Unknown')[:40] if paper else paper_id
                    print(f"    {i}. {title}...")
        
        print()
        
        # Create visualizations
        print("Creating visualizations...")
        shift_viz = ShiftVisualizer()
        
        # Shift timeline
        shift_viz.plot_shift_timeline(
            shifts,
            save_path="../results/figures/shift_timeline.png"
        )
        print("  Saved: shift_timeline.png")
        
        # Shift comparison
        shift_viz.plot_shift_comparison(
            shifts,
            save_path="../results/figures/shift_comparison.png"
        )
        print("  Saved: shift_comparison.png")
        
        # Generate embeddings for analysis
        print("\nGenerating embeddings...")
        embedder = GraphEmbedder(method='spectral', dimensions=64)
        embeddings = embedder.fit_transform(network.graph)
        
        print(f"  Generated {embeddings.dimensions}-dim embeddings for {len(embeddings.node_ids)} nodes")
        
        # Cluster papers
        clusters = embeddings.cluster(n_clusters=20)
        print(f"  Identified {len(set(clusters.values()))} clusters")
        
        # Create report
        print("\nGenerating analysis report...")
        report = create_summary_report(
            network,
            shifts,
            output_path="../results/analysis_report.txt"
        )
        print("  Saved: analysis_report.txt")
        
    else:
        print("No significant paradigm shifts detected.")
        print("Try adjusting threshold or analyzing more data.")
    
    print()
    print("Done!")


if __name__ == "__main__":
    main()
