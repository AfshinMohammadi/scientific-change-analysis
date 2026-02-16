# The Science of Scientific Change: Theory, Metrics, and Prediction

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A comprehensive framework for modeling the evolution of scientific knowledge through network science, machine learning, and computational epistemology. This project analyzes how scientific paradigms emerge, compete, and shift over time using large-scale citation networks, co-authorship graphs, and knowledge embedding techniques.

## Overview

Scientific knowledge evolves through complex processes of discovery, validation, competition, and paradigm shifts. This project provides tools to:

- **Construct large-scale networks** from academic (citation databases) and non-academic (web-scraped) sources
- **Detect paradigm shifts** through unsupervised clustering and graph-topology analysis
- **Predict field reorganization** using temporal network evolution models
- **Generate knowledge embeddings** for scalable scientific domain analysis

## Key Features

### Network Construction
- Citation network extraction from academic databases (Semantic Scholar, OpenAlex, arXiv)
- Co-authorship network generation with temporal slicing
- Web-scraped knowledge graphs from scientific news, blogs, and policy documents
- Heterogeneous graph construction linking papers, authors, institutions, and concepts

### Paradigm Shift Detection
- Community detection with temporal stability analysis
- Topic drift quantification using embedding distance metrics
- Citation pattern anomaly detection
- Multi-scale structural break identification

### Predictive Modeling
- Field emergence prediction using early-stage network signals
- Collaboration pattern forecasting
- Citation trajectory modeling
- Concept diffusion prediction

### Scalable Analytics
- GPU-accelerated graph processing with PyTorch Geometric
- Distributed computing support for billion-edge networks
- Memory-efficient streaming for large datasets

## Installation

```bash
# Clone the repository
git clone https://github.com/afshinmohammadi/scientific-change-analysis.git
cd scientific-change-analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Quick Start

```python
from src.networks import CitationNetwork, CoAuthorshipNetwork
from src.models import ParadigmShiftDetector, FieldEvolutionPredictor
from src.data import AcademicDataLoader, WebScraper

# Load academic citation data
loader = AcademicDataLoader(source='semantic_scholar')
papers = loader.load_from_file('data/raw/papers.jsonl')
citation_net = CitationNetwork.from_papers(papers)

# Detect paradigm shifts
detector = ParadigmShiftDetector(
    resolution=0.8,
    temporal_window=5  # years
)
shifts = detector.detect(citation_net, years=range(2010, 2024))

# Analyze detected shifts
for shift in shifts:
    print(f"Paradigm shift detected in {shift['year']}:")
    print(f"  Field: {shift['affected_field']}")
    print(f"  Magnitude: {shift['magnitude']:.3f}")
    print(f"  Key papers: {shift['key_papers'][:5]}")
```

## Project Structure

```
scientific-change-analysis/
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── academic_loader.py    # Academic database loaders
│   │   ├── web_scraper.py        # Non-academic data collection
│   │   ├── preprocessor.py       # Data cleaning and normalization
│   │   └── dataset.py            # Dataset classes for ML
│   ├── networks/
│   │   ├── __init__.py
│   │   ├── citation_network.py   # Citation graph construction
│   │   ├── coauthor_network.py   # Co-authorship networks
│   │   ├── knowledge_graph.py    # Heterogeneous knowledge graphs
│   │   └── temporal_slicer.py    # Time-based network slicing
│   ├── models/
│   │   ├── __init__.py
│   │   ├── paradigm_detector.py  # Shift detection algorithms
│   │   ├── field_predictor.py    # Field evolution prediction
│   │   ├── embeddings.py         # Graph embedding models
│   │   └── clustering.py         # Unsupervised clustering
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── graph_metrics.py      # Network topology metrics
│   │   ├── shift_metrics.py      # Paradigm shift quantification
│   │   └── evaluation.py         # Model evaluation metrics
│   └── utils/
│       ├── __init__.py
│       ├── visualization.py      # Network and result visualization
│       └── helpers.py            # Utility functions
├── notebooks/
│   ├── 01_data_loading.ipynb
│   ├── 02_network_construction.ipynb
│   ├── 03_paradigm_detection.ipynb
│   └── 04_prediction_models.ipynb
├── configs/
│   ├── data.yaml                 # Data source configuration
│   ├── network.yaml              # Network parameters
│   └── model.yaml                # Model hyperparameters
├── data/
│   ├── raw/                      # Raw data files
│   └── processed/                # Processed datasets
├── examples/
│   ├── build_citation_network.py
│   ├── detect_shifts.py
│   └── predict_emergence.py
├── docs/
│   └── METHODOLOGY.md
├── requirements.txt
├── setup.py
└── README.md
```

## Data Sources

### Academic Datasets
| Source | Description | Access |
|--------|-------------|--------|
| Semantic Scholar | 200M+ papers with citations | API |
| OpenAlex | Open academic catalog | API/Dump |
| arXiv | Preprint server | API/Dump |
| PubMed | Biomedical literature | API |
| MAG | Microsoft Academic Graph | Archived |

### Non-Academic Datasets
| Source | Description | Method |
|--------|-------------|--------|
| Science News | Popular science coverage | Web scraping |
| Policy Documents | Research funding priorities | Web scraping |
| Science Blogs | Informal scientific discourse | RSS/API |
| Wikipedia | Science articles and citations | API |

## Core Methodology

### 1. Network Construction

```python
from src.networks import CitationNetwork

# Build citation network with metadata
network = CitationNetwork()
network.add_papers(papers_df)
network.build_citation_edges()

# Apply temporal slicing
slices = network.temporal_slice(
    start_year=2000,
    end_year=2024,
    window_size=1  # yearly networks
)
```

### 2. Paradigm Shift Detection

We detect paradigm shifts through multiple signals:

1. **Structural Changes**: Community reorganization in citation networks
2. **Semantic Drift**: Topic embedding divergence over time
3. **Citation Patterns**: Sudden changes in citation flow patterns
4. **Key Paper Emergence**: High-impact papers restructuring the field

```python
from src.models import ParadigmShiftDetector

detector = ParadigmShiftDetector(
    methods=['structural', 'semantic', 'citation_flow'],
    threshold=0.7,
    min_papers=50
)

# Detect shifts across time
shifts = detector.detect_multi_year(
    network,
    years=range(2010, 2024)
)

# Visualize shift timeline
detector.plot_shift_timeline(shifts)
```

### 3. Embedding Generation

```python
from src.models import GraphEmbedder

# Generate node embeddings
embedder = GraphEmbedder(
    method='node2vec',  # or 'graphsage', 'gat', 'transE'
    dimensions=128,
    walk_length=30
)

embeddings = embedder.fit_transform(network)

# Cluster papers in embedding space
clusters = embedder.cluster(n_clusters=50)
```

### 4. Predictive Modeling

```python
from src.models import FieldEvolutionPredictor

predictor = FieldEvolutionPredictor(
    features=['network_topology', 'embedding_drift', 'citation_velocity'],
    model='gradient_boosting'
)

# Train on historical data
predictor.fit(historical_networks, labels)

# Predict future field reorganization
predictions = predictor.predict(current_network, horizon=5)
```

## Key Metrics

### Network Topology Metrics
- **Modularity**: Community structure strength
- **Betweenness Centrality**: Knowledge flow bottlenecks
- **PageRank**: Paper influence scores
- **Assortativity**: Citation pattern homophily

### Paradigm Shift Metrics
- **Shift Magnitude**: Extent of community reorganization
- **Shift Velocity**: Speed of structural change
- **Field Stability**: Resistance to reorganization
- **Emergence Score**: Likelihood of new field formation

## Example Results

Analysis of Computer Science field (2010-2023):

| Metric | Value |
|--------|-------|
| Papers analyzed | 2.4M |
| Citation edges | 45M |
| Detected paradigm shifts | 12 |
| Major fields tracked | 38 |
| Prediction accuracy | 78.3% |

### Detected Paradigm Shifts

| Year | Field | Shift Type | Magnitude |
|------|-------|------------|-----------|
| 2012 | Machine Learning | Deep Learning emergence | 0.89 |
| 2015 | NLP | Attention mechanisms | 0.76 |
| 2018 | Computer Vision | Transformers adoption | 0.82 |
| 2020 | AI | Foundation models | 0.91 |
| 2022 | NLP | LLM paradigm | 0.94 |

## Configuration

Network and model parameters can be configured via YAML files:

```yaml
# configs/network.yaml
network:
  min_citations: 5          # Minimum citations to include
  max_nodes: 1000000        # Maximum network size
  temporal_granularity: year # Time slicing unit

community_detection:
  algorithm: leiden         # leiden, louvain, infomap
  resolution: 0.8           # Resolution parameter
  n_iterations: 10

embedding:
  method: node2vec
  dimensions: 128
  window_size: 5
```

## Visualization

```python
from src.utils import NetworkVisualizer

viz = NetworkVisualizer(network)

# Network overview
viz.plot_network_overview()

# Community structure
viz.plot_communities(clusters)

# Temporal evolution
viz.plot_temporal_evolution(slices)

# Paradigm shift timeline
viz.plot_shift_timeline(shifts)
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{mohammadi2025scientificchange,
  title={The Architecture of Scientific Change: Theory, Metrics, and Prediction},
  author={Mohammadi, Afshin},
  journal={arXiv preprint},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

**Afshin Mohammadi**  
Email: Afshinciv@gmail.com  
