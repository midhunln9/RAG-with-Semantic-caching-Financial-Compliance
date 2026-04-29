import asyncio

from langchain_core.documents import Document

from src.embeddings.openai_embedding import OpenAIEmbedding


class DenseEmbeddingService:
    def __init__(self, dense_embedding: OpenAIEmbedding):
        self.dense_embedding = dense_embedding

    def get_embedding_dimension(self) -> int:
        return asyncio.run(self.dense_embedding.get_sentence_embedding_dimension())

    def get_dense_embeddings(self, chunk_documents: list[Document]) -> list[list[float]]:
        chunk_texts = [chunk.page_content for chunk in chunk_documents]
        embedding_response = asyncio.run(
            self.dense_embedding.client.embeddings.create(
                model=self.dense_embedding.model_name,
                input=chunk_texts,
            )
        )
        return [item.embedding for item in embedding_response.data]
