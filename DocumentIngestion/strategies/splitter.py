from abc import ABC, abstractmethod

from langchain_core.documents import Document


class Splitter(ABC):
    @abstractmethod
    def split_documents(self, documents: list[Document]) -> list[Document]:
        pass
