"""
Setup script for scientific-change-analysis package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="scientific_change_analysis",
    version="1.0.0",
    author="Afshin Mohammadi",
    author_email="Afshinciv@gmail.com",
    description="Framework for modeling scientific knowledge evolution through network science",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/afshinmohammadi/scientific-change-analysis",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
        "networkx>=2.6.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
        "requests>=2.26.0",
        "beautifulsoup4>=4.10.0",
        "pyyaml>=5.4.0",
        "tqdm>=4.62.0",
    ],
    extras_require={
        "embeddings": [
            "gensim>=4.0.0",
            "node2vec>=0.4.0",
        ],
        "deep_learning": [
            "torch>=1.9.0",
            "torch-geometric>=2.0.0",
            "sentence-transformers>=2.0.0",
        ],
        "dev": [
            "pytest>=6.2.0",
            "black>=21.0",
            "flake8>=3.9.0",
        ],
    },
)
