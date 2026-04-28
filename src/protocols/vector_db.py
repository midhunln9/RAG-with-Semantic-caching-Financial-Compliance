from typing import Protocol, List
from langchain_core.documents import Document

class VectorDBProtocol(Protocol):
    async def query_vector_store_for_rankx(self, query: str) -> List[dict]:
        ...
    async def hybrid_search_pinecone(self, query: str) -> List[Document]:
        ...