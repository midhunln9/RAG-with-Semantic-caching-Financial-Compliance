from abc import ABC, abstractmethod
from typing import List


class SparseEmbeddingStrategy(ABC):
    @abstractmethod
    def embed_query(self, query: str) -> dict: ...
    @abstractmethod
    def embed_documents(self, documents: List[str]) -> List[dict]: ...
