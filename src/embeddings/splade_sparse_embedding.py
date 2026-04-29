import asyncio
from typing import List

from src.strategies.sparse_embeddings import SparseEmbeddingStrategy
from sentence_transformers import SparseEncoder
import torch


class SentenceTransformerSparseEmbedding(SparseEmbeddingStrategy):
    def __init__(self):
        self.model = SparseEncoder("naver/splade-cocondenser-ensembledistil")

    def _sparse_tensor_to_pinecone_dict(self, sparse_tensor: torch.Tensor) -> dict:
        dense = sparse_tensor.to_dense().cpu()
        non_zero = torch.nonzero(dense).squeeze(1)
        return {
            "indices": non_zero.tolist(),
            "values": dense[non_zero].tolist()
        }

    async def embed_query(self, query: str) -> dict:
        embeddings = await asyncio.to_thread(self.model.encode, [query])
        return self._sparse_tensor_to_pinecone_dict(embeddings[0])
    
    async def embed_documents(self, documents : List[str]) -> List[dict]:
        embeddings = await asyncio.to_thread(self.model.encode, documents)
        return [self._sparse_tensor_to_pinecone_dict(embedding) for embedding in embeddings]
