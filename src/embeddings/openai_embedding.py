from typing import List
from src.strategies.dense_embeddings import DenseEmbeddingStrategy
from openai import AsyncOpenAI
import os


class OpenAIEmbedding(DenseEmbeddingStrategy):
    def __init__(self):
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model_name = "text-embedding-3-small"

    async def get_sentence_embedding_dimension(self) -> int:
        response = await self.client.embeddings.create(
            model=self.model_name,
            input="This is a test query to get embedding dimension"
        )
        embedding_dim = len(response.data[0].embedding)
        return embedding_dim

    async def embed_query(self, query: str) -> List[float]:
        embedding_query = await self.client.embeddings.create(
            model=self.model_name,
            input=query
        )

        return embedding_query.data[0].embedding