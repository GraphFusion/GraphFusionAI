import torch
import torch.nn.functional as F
from typing import List, Union, Optional, Dict
from sentence_transformers import SentenceTransformer
import numpy as np

class EmbeddingModel:
    """
    Model for generating dense vector embeddings from text using sentence transformers.
    Provides encoding and similarity calculation functionality with caching and batch processing.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", 
                 cache_size: int = 10000,
                 batch_size: int = 32,
                 device: Optional[str] = None):
        """
        Initialize the embedding model.

        Args:
            model_name: Name of the sentence transformer model to use
            cache_size: Maximum number of embeddings to cache
            batch_size: Batch size for encoding multiple texts
            device: Device to run model on ('cuda', 'cpu', or None for auto)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SentenceTransformer(model_name).to(self.device)
        self.batch_size = batch_size
        self.cache: Dict[str, torch.Tensor] = {}
        self.cache_size = cache_size

    def encode(self, texts: Union[str, List[str]], use_cache: bool = True) -> torch.Tensor:
        """
        Generate embeddings for input text(s) with caching and batching.

        Args:
            texts: Input text or list of texts to encode
            use_cache: Whether to use embedding cache

        Returns:
            Tensor containing text embeddings
        """
        if isinstance(texts, str):
            texts = [texts]

        # Check cache for existing embeddings
        if use_cache:
            cached_embeddings = []
            texts_to_encode = []
            for text in texts:
                if text in self.cache:
                    cached_embeddings.append(self.cache[text])
                else:
                    texts_to_encode.append(text)
            
            if not texts_to_encode:
                return torch.stack(cached_embeddings)
        else:
            texts_to_encode = texts

        # Generate embeddings in batches
        all_embeddings = []
        for i in range(0, len(texts_to_encode), self.batch_size):
            batch = texts_to_encode[i:i + self.batch_size]
            embeddings = self.model.encode(batch, convert_to_tensor=True)
            embeddings = F.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings)

            # Update cache
            if use_cache:
                for text, emb in zip(batch, embeddings):
                    if len(self.cache) >= self.cache_size:
                        self.cache.pop(next(iter(self.cache)))
                    self.cache[text] = emb

        # Combine cached and new embeddings
        if use_cache and cached_embeddings:
            all_embeddings.append(torch.stack(cached_embeddings))

        return torch.cat(all_embeddings)

    def similarity(self, query_embedding: torch.Tensor, memory_embeddings: torch.Tensor,
                  method: str = 'cosine') -> torch.Tensor:
        """
        Calculate similarities between query and memory embeddings.

        Args:
            query_embedding: Query text embedding
            memory_embeddings: Memory text embeddings to compare against
            method: Similarity method ('cosine' or 'dot')

        Returns:
            Tensor of similarity scores
        """
        if method == 'cosine':
            return torch.matmul(query_embedding, memory_embeddings.T).squeeze()
        elif method == 'dot':
            return torch.sum(query_embedding * memory_embeddings, dim=1)
        else:
            raise ValueError(f"Unsupported similarity method: {method}")
