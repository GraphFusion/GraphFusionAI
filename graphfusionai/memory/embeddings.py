import torch
from sentence_transformers import SentenceTransformer
from typing import List, Union

class EmbeddingModel:
    """
    Converts textual knowledge into dense vector representations for storage and retrieval.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        """
        Initializes the embedding model.

        Args:
            model_name (str): The SentenceTransformer model to use.
            device (str): The device to run the model on (e.g., "cpu" or "cuda").
        """
        self.device = device
        self.model = SentenceTransformer(model_name).to(device)

    def encode(self, texts: Union[str, List[str]]) -> torch.Tensor:
        """
        Converts text or a list of texts into embeddings.

        Args:
            texts (str | List[str]): The text(s) to be converted.

        Returns:
            torch.Tensor: A tensor containing the embeddings.
        """
        if isinstance(texts, str):
            texts = [texts]

        embeddings = self.model.encode(texts, convert_to_tensor=True)
        return embeddings

    def similarity(self, query_embedding: torch.Tensor, memory_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Computes cosine similarity between a query embedding and stored memory embeddings.

        Args:
            query_embedding (torch.Tensor): The embedding of the query.
            memory_embeddings (torch.Tensor): The stored memory embeddings.

        Returns:
            torch.Tensor: Similarity scores.
        """
        return torch.nn.functional.cosine_similarity(query_embedding, memory_embeddings, dim=-1)
