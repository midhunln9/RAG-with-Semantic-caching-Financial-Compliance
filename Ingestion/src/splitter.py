from Ingestion.strategies.splitter import Splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

class RecursiveCharacterTextSplitterMethod(Splitter):
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_documents(self, documents: list[Document]) -> list[Document]:
        return self.splitter.split_documents(documents)