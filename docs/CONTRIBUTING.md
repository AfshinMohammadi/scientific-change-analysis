# Contributing Guide

Thank you for your interest in contributing to this project!

## Getting Started

1. Fork the repository
2. Clone your fork locally
   ```bash
   git clone https://github.com/yourusername/scientific-change-analysis.git
   cd scientific-change-analysis
   ```
3. Create a virtual environment
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
4. Install development dependencies
   ```bash
   pip install -e ".[dev]"
   ```

## Development Workflow

1. Create a feature branch
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes
   - Follow PEP 8 guidelines
   - Add docstrings
   - Include type hints

3. Run tests
   ```bash
   pytest tests/
   ```

4. Run linting
   ```bash
   black src/
   flake8 src/
   ```

5. Commit and push
   ```bash
   git commit -m "feat: description"
   git push origin feature/your-feature-name
   ```

6. Create a pull request

## Project Structure

```
scientific-change-analysis/
├── src/
│   ├── data/        # Data loading and scraping
│   ├── networks/    # Network construction
│   ├── models/      # Detection and prediction models
│   └── utils/       # Visualization and utilities
├── notebooks/       # Jupyter notebooks
├── configs/         # Configuration files
└── examples/        # Example scripts
```

## Code Style

- Use meaningful variable names
- Write docstrings for public functions
- Keep functions focused (< 50 lines)
- Add type hints

## Testing

- Write tests for new functionality
- Use pytest fixtures for common setup
- Aim for >80% coverage

## Questions?

Open an issue for questions or suggestions.

Thank you for contributing!
