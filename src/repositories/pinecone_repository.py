from typing import List
from langchain_core.documents import Document
from pinecone import Pinecone, ServerlessSpec
from src.strategies.dense_embeddings import DenseEmbeddingStrategy
from src.strategies.sparse_embeddings import SparseEmbeddingStrategy
from src.configs.pinecone_config import PineconeConfig
from src.protocols.vector_db import VectorDBProtocol
import pandas as pd
from typing import List, Dict
import os


class PineconeRepository(VectorDBProtocol):
    def __init__(self,dense_embedding_strategy: DenseEmbeddingStrategy,
    sparse_embedding_strategy: SparseEmbeddingStrategy, pinecone_config: PineconeConfig):
        self.client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.dense_embedding_strategy = dense_embedding_strategy
        self.sparse_embedding_strategy = sparse_embedding_strategy
        self.pinecone_config = pinecone_config
    
    async def hybrid_search_pinecone(self, query: str) -> List[Document]:
        index = self.client.Index(self.pinecone_config.index_name)

        query_vector = await self.dense_embedding_strategy.embed_query(query)
        sparse_embedding = await self.sparse_embedding_strategy.embed_query(query)

        results = index.query(
            vector=query_vector,
            sparse_vector=sparse_embedding,
            top_k=10,
            include_metadata=True
        )

        # Convert to Document format
        documents = []
        for match in results.matches:
            metadata = match.metadata or {}
            doc = Document(
                page_content=metadata.get("text", ""),
                metadata={
                    "id": match.id,
                    "score": float(match.score),
                    **metadata
                }
            )
            documents.append(doc)

        return documents

    async def query_vector_store_for_rankx(self, query: str) -> List[Dict]:
        index = self.client.Index(self.pinecone_config.index_name)

        query_vector = await self.dense_embedding_strategy.embed_query(query)
        sparse_embedding = await self.sparse_embedding_strategy.embed_query(query)

        results = index.query(
            vector=query_vector,
            sparse_vector=sparse_embedding,
            top_k=10,
            include_metadata=False
        )

        # Return RankX-compatible format
        return [
            {
                "id": match.id,
                "score": float(match.score)  # ensure JSON serializable
            }
            for match in results.matches
        ]
                
            
            
            

            








