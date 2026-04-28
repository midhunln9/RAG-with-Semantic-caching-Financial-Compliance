from typing import List

from langchain_core.documents import Document
from pydantic import BaseModel

from src.protocols.vector_db import VectorDBProtocol
from src.strategies.llm_strategy import LLMStrategy


class RagWorkflowService:
    def __init__(self, llm_strategy: LLMStrategy, vector_db: VectorDBProtocol):
        self.llm_strategy = llm_strategy
        self.vector_db = vector_db

    async def rewrite_query_service(
        self, prompt: str, response_class: type[BaseModel]
    ) -> str:
        response = await self.llm_strategy.generate_response(prompt, response_class)
        return response.rewritten_query

    async def retrieve_documents_service(
        self, rewritten_query: str
    ) -> List[Document]:
        return await self.vector_db.hybrid_search_pinecone(rewritten_query)

    async def generate_answer_service(self, prompt: str) -> str:
        response = await self.llm_strategy.generate_response(prompt)
        return response.content if hasattr(response, "content") else str(response)
