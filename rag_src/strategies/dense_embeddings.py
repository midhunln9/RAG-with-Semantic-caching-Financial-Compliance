from abc import ABC, abstractmethod
from typing import List

from langchain_core.documents import Document


class DenseEmbeddingStrategy(ABC):
    @abstractmethod
    def get_sentence_embedding_dimension(self) -> int: ...
    @abstractmethod
    def embed_query(self, query: str) -> List[float]: ...
    @abstractmethod
    def embed_documents(self, documents: List[Document]) -> List[List[float]]: ...
