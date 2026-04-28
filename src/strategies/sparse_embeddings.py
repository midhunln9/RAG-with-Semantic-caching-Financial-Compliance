from abc import ABC, abstractmethod
from typing import List

from langchain_core.documents import Document

class SparseEmbeddingStrategy(ABC):
    @abstractmethod
    def embed_query(self, query : str) -> dict : ...
 