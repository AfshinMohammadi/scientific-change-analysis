"""
Sentence Transformer Module
===========================

Implements semantic embeddings for papers using sentence transformers.
Provides high-quality text embeddings for:
- Paper abstract/title similarity
- Semantic clustering
- Topic modeling
- Cross-lingual paper matching
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import time

import numpy as np
from tqdm import tqdm

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


@dataclass
class SentenceTransformerConfig:
    """Configuration for sentence transformer."""
    model_name: str = 'all-MiniLM-L6-v2'  # Fast and good quality
    batch_size: int = 32
    device: str = 'auto'  # 'auto', 'cuda', 'cpu'
    normalize_embeddings: bool = True
    max_seq_length: int = 512
    
    # For multi-GPU
    parallelize: bool = False
    
    # Model alternatives by use case:
    # - 'all-MiniLM-L6-v2': Fast, good quality (384 dim)
    # - 'all-mpnet-base-v2': Best quality, slower (768 dim)
    # - 'multi-qa-mpnet-base-dot-v1': Optimized for semantic search
    # - 'all-roberta-large-v1': Best quality, largest
    # - 'paraphrase-multilingual-mpnet-base-v2': Multi-lingual


class PaperEmbedder:
    """
    Generate semantic embeddings for papers using sentence transformers.
    
    Uses paper titles and abstracts to create high-quality embeddings
    that capture semantic meaning.
    
    Example:
        >>> embedder = PaperEmbedder(model='all-mpnet-base-v2')
        >>> embeddings = embedder.embed_papers(papers)
        >>> similar = embedder.find_similar('paper_123', top_k=10)
    """
    
    def __init__(
        self,
        config: Optional[SentenceTransformerConfig] = None
    ):
        self.config = config or SentenceTransformerConfig()
        self.model = None
        self._embeddings = None
        self._paper_ids = []
    
    def _load_model(self) -> None:
        """Load the sentence transformer model."""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers is required. "
                "Install with: pip install sentence-transformers"
            )
        
        if self.model is None:
            self.model = SentenceTransformer(self.config.model_name)
            
            # Set device
            if self.config.device == 'auto':
                import torch
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            else:
                device = self.config.device
            
            self.model = self.model.to(device)
            
            # Set max sequence length
            self.model.max_seq_length = self.config.max_seq_length
    
    def embed_texts(
        self,
        texts: List[str],
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings
            show_progress: Show progress bar
            
        Returns:
            Numpy array of embeddings (n_texts, embedding_dim)
        """
        self._load_model()
        
        embeddings = self.model.encode(
            texts,
            batch_size=self.config.batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=self.config.normalize_embeddings,
            convert_to_numpy=True
        )
        
        return embeddings
    
    def embed_papers(
        self,
        papers: List[Dict[str, Any]],
        text_fields: List[str] = None,
        show_progress: bool = True
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Generate embeddings for papers.
        
        Args:
            papers: List of paper dictionaries
            text_fields: Fields to use for text (default: title + abstract)
            show_progress: Show progress bar
            
        Returns:
            Tuple of (embeddings, paper_ids)
        """
        if text_fields is None:
            text_fields = ['title', 'abstract']
        
        texts = []
        paper_ids = []
        
        for paper in tqdm(papers, desc="Preparing texts", disable=not show_progress):
            # Combine specified fields
            text_parts = []
            for field in text_fields:
                value = paper.get(field, '')
                if value:
                    text_parts.append(str(value))
            
            text = ' '.join(text_parts)
            
            if text:
                texts.append(text)
                paper_ids.append(paper.get('paper_id', str(len(paper_ids))))
        
        # Generate embeddings
        print(f"Generating embeddings for {len(texts)} papers...")
        embeddings = self.embed_texts(texts, show_progress)
        
        self._embeddings = embeddings
        self._paper_ids = paper_ids
        
        return embeddings, paper_ids
    
    def embed_single(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        self._load_model()
        
        embedding = self.model.encode(
            text,
            normalize_embeddings=self.config.normalize_embeddings,
            convert_to_numpy=True
        )
        
        return embedding
    
    def find_similar(
        self,
        query: Union[str, np.ndarray],
        top_k: int = 10,
        papers: Optional[List[Dict]] = None,
        embeddings: Optional[np.ndarray] = None,
        paper_ids: Optional[List[str]] = None
    ) -> List[Tuple[str, float]]:
        """
        Find most similar papers to a query.
        
        Args:
            query: Query text or embedding vector
            top_k: Number of results to return
            papers: List of papers (if not already embedded)
            embeddings: Pre-computed embeddings
            paper_ids: Paper IDs corresponding to embeddings
            
        Returns:
            List of (paper_id, similarity_score) tuples
        """
        # Get embeddings if not provided
        if embeddings is None:
            if self._embeddings is None:
                if papers is None:
                    raise ValueError("Must provide either embeddings or papers")
                embeddings, paper_ids = self.embed_papers(papers)
            else:
                embeddings = self._embeddings
                paper_ids = self._paper_ids
        
        # Get query embedding
        if isinstance(query, str):
            query_embedding = self.embed_single(query)
        else:
            query_embedding = query
        
        # Ensure 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Calculate similarities
        from sklearn.metrics.pairwise import cosine_similarity
        
        similarities = cosine_similarity(query_embedding, embeddings)[0]
        
        # Get top-k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = [
            (paper_ids[i], float(similarities[i]))
            for i in top_indices
        ]
        
        return results
    
    def semantic_search(
        self,
        queries: List[str],
        corpus_embeddings: np.ndarray,
        corpus_ids: List[str],
        top_k: int = 10,
        show_progress: bool = True
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Perform semantic search for multiple queries.
        
        Args:
            queries: List of query texts
            corpus_embeddings: Pre-computed corpus embeddings
            corpus_ids: IDs corresponding to corpus embeddings
            top_k: Results per query
            show_progress: Show progress bar
            
        Returns:
            Dict mapping query to list of (doc_id, score) tuples
        """
        # Embed queries
        query_embeddings = self.embed_texts(queries, show_progress)
        
        # Calculate all similarities
        from sklearn.metrics.pairwise import cosine_similarity
        
        similarity_matrix = cosine_similarity(query_embeddings, corpus_embeddings)
        
        # Get top-k for each query
        results = {}
        for i, query in enumerate(queries):
            top_indices = np.argsort(similarity_matrix[i])[::-1][:top_k]
            results[query] = [
                (corpus_ids[j], float(similarity_matrix[i][j]))
                for j in top_indices
            ]
        
        return results


class SemanticClusterer:
    """
    Cluster papers by semantic similarity.
    
    Uses sentence embeddings + clustering algorithms to
    group semantically similar papers.
    """
    
    def __init__(
        self,
        embedding_model: str = 'all-MiniLM-L6-v2',
        n_clusters: int = 50,
        clustering_method: str = 'agglomerative'
    ):
        self.embedding_model = embedding_model
        self.n_clusters = n_clusters
        self.clustering_method = clustering_method
        
        self.embedder = PaperEmbedder(
            SentenceTransformerConfig(model_name=embedding_model)
        )
        self.cluster_model = None
    
    def fit_predict(
        self,
        papers: List[Dict[str, Any]],
        show_progress: bool = True
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Cluster papers by semantic content.
        
        Args:
            papers: List of paper dictionaries
            show_progress: Show progress bars
            
        Returns:
            Tuple of (cluster_labels, paper_ids)
        """
        # Generate embeddings
        embeddings, paper_ids = self.embedder.embed_papers(
            papers,
            show_progress=show_progress
        )
        
        # Cluster
        if self.clustering_method == 'agglomerative':
            from sklearn.cluster import AgglomerativeClustering
            self.cluster_model = AgglomerativeClustering(
                n_clusters=self.n_clusters,
                metric='cosine',
                linkage='average'
            )
        elif self.clustering_method == 'kmeans':
            from sklearn.cluster import KMeans
            self.cluster_model = KMeans(
                n_clusters=self.n_clusters,
                random_state=42,
                n_init=10
            )
        else:
            from sklearn.cluster import KMeans
            self.cluster_model = KMeans(
                n_clusters=self.n_clusters,
                random_state=42,
                n_init=10
            )
        
        labels = self.cluster_model.fit_predict(embeddings)
        
        return labels, paper_ids
    
    def get_cluster_keywords(
        self,
        papers: List[Dict[str, Any]],
        labels: np.ndarray,
        n_keywords: int = 10
    ) -> Dict[int, List[str]]:
        """
        Extract keywords for each cluster.
        
        Args:
            papers: Original papers
            labels: Cluster labels
            n_keywords: Keywords per cluster
            
        Returns:
            Dict mapping cluster_id to keyword list
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        # Group papers by cluster
        cluster_texts = {}
        for paper, label in zip(papers, labels):
            if label not in cluster_texts:
                cluster_texts[label] = []
            
            text = f"{paper.get('title', '')} {paper.get('abstract', '')}"
            cluster_texts[label].append(text)
        
        # Extract keywords per cluster
        keywords = {}
        
        for cluster_id, texts in cluster_texts.items():
            if len(texts) < 2:
                keywords[cluster_id] = []
                continue
            
            vectorizer = TfidfVectorizer(
                max_features=100,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            try:
                tfidf = vectorizer.fit_transform(texts)
                feature_names = vectorizer.get_feature_names_out()
                
                # Get top keywords by mean TF-IDF
                mean_tfidf = np.array(tfidf.mean(axis=0)).flatten()
                top_indices = np.argsort(mean_tfidf)[::-1][:n_keywords]
                
                keywords[cluster_id] = [
                    feature_names[i] for i in top_indices
                ]
            except:
                keywords[cluster_id] = []
        
        return keywords


class CrossLingualMatcher:
    """
    Match papers across languages using multilingual embeddings.
    
    Uses multilingual sentence transformers to find similar
    papers regardless of language.
    """
    
    def __init__(self):
        # Use multilingual model
        config = SentenceTransformerConfig(
            model_name='paraphrase-multilingual-mpnet-base-v2'
        )
        self.embedder = PaperEmbedder(config)
    
    def find_cross_lingual_matches(
        self,
        papers_lang1: List[Dict[str, Any]],
        papers_lang2: List[Dict[str, Any]],
        threshold: float = 0.8,
        show_progress: bool = True
    ) -> List[Tuple[str, str, float]]:
        """
        Find matching papers across languages.
        
        Args:
            papers_lang1: Papers in first language
            papers_lang2: Papers in second language
            threshold: Minimum similarity threshold
            show_progress: Show progress
            
        Returns:
            List of (paper_id_1, paper_id_2, similarity) tuples
        """
        # Embed both sets
        emb1, ids1 = self.embedder.embed_papers(papers_lang1, show_progress=show_progress)
        emb2, ids2 = self.embedder.embed_papers(papers_lang2, show_progress=show_progress)
        
        # Calculate cross-similarity
        from sklearn.metrics.pairwise import cosine_similarity
        
        similarity_matrix = cosine_similarity(emb1, emb2)
        
        # Find matches above threshold
        matches = []
        for i in range(len(ids1)):
            for j in range(len(ids2)):
                sim = similarity_matrix[i, j]
                if sim >= threshold:
                    matches.append((ids1[i], ids2[j], float(sim)))
        
        # Sort by similarity
        matches.sort(key=lambda x: -x[2])
        
        return matches


def create_paper_embeddings(
    papers: List[Dict[str, Any]],
    model_name: str = 'all-MiniLM-L6-v2',
    batch_size: int = 32,
    show_progress: bool = True
) -> Tuple[np.ndarray, List[str]]:
    """
    Convenience function to create paper embeddings.
    
    Args:
        papers: List of paper dictionaries
        model_name: Sentence transformer model name
        batch_size: Batch size for encoding
        show_progress: Show progress bar
        
    Returns:
        Tuple of (embeddings, paper_ids)
    """
    config = SentenceTransformerConfig(
        model_name=model_name,
        batch_size=batch_size
    )
    
    embedder = PaperEmbedder(config)
    return embedder.embed_papers(papers, show_progress=show_progress)
