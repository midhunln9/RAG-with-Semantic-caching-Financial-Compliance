import math
import os
from typing import Dict, List

from langchain_core.documents import Document
from loguru import logger
from pinecone import Pinecone, ServerlessSpec

from rag_src.configs.pinecone_config import PineconeConfig
from rag_src.protocols.vector_db import VectorDBProtocol
from rag_src.strategies.dense_embeddings import DenseEmbeddingStrategy
from rag_src.strategies.sparse_embeddings import SparseEmbeddingStrategy


class PineconeRepository(VectorDBProtocol):
    def __init__(
        self,
        dense_embedding_strategy: DenseEmbeddingStrategy,
        sparse_embedding_strategy: SparseEmbeddingStrategy,
        pinecone_config: PineconeConfig,
    ):
        self.client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.dense_embedding_strategy = dense_embedding_strategy
        self.sparse_embedding_strategy = sparse_embedding_strategy
        self.pinecone_config = pinecone_config

    async def hybrid_search_pinecone(self, query: str) -> List[Document]:
        index = self.client.Index(self.pinecone_config.index_name)

        query_vector = await self.dense_embedding_strategy.embed_query(query)
        sparse_embedding = await self.sparse_embedding_strategy.embed_query(query)

        results = index.query(
            vector=query_vector, sparse_vector=sparse_embedding, top_k=10, include_metadata=True
        )

        # Convert to Document format
        documents = []
        for match in results.matches:
            metadata = match.metadata or {}
            doc = Document(
                page_content=metadata.get("text", ""),
                metadata={"id": match.id, "score": float(match.score), **metadata},
            )
            documents.append(doc)

        return documents

    async def query_vector_store_for_rankx(self, query: str) -> List[Dict]:
        index = self.client.Index(self.pinecone_config.index_name)

        query_vector = await self.dense_embedding_strategy.embed_query(query)
        sparse_embedding = await self.sparse_embedding_strategy.embed_query(query)

        results = index.query(
            vector=query_vector, sparse_vector=sparse_embedding, top_k=10, include_metadata=False
        )

        # Return RankX-compatible format
        return [
            {
                "id": match.id,
                "score": float(match.score),  # ensure JSON serializable
            }
            for match in results.matches
        ]

    async def upsert_chunks(self, chunks: List[Document]):
        """Ingest document chunks into Pinecone index in batches."""
        batch_size = self.pinecone_config.batch_size
        total_batches = math.ceil(len(chunks) / batch_size) if chunks else 0
        logger.info(
            f"PineconeRepository.upsert_chunks: received {len(chunks)} chunk(s) "
            f"-> {total_batches} batch(es) of up to {batch_size}"
        )

        if not self.client.has_index(self.pinecone_config.index_name):
            logger.info(f"Index '{self.pinecone_config.index_name}' not found -- creating")
            dimension = await self.dense_embedding_strategy.get_sentence_embedding_dimension()
            self.client.create_index(
                name=self.pinecone_config.index_name,
                dimension=dimension,
                metric=self.pinecone_config.metric,
                spec=ServerlessSpec(
                    cloud=self.pinecone_config.cloud,
                    region=self.pinecone_config.region,
                ),
            )
            logger.success(
                f"Index '{self.pinecone_config.index_name}' created (dimension={dimension})"
            )
        else:
            logger.info(f"Index '{self.pinecone_config.index_name}' already exists")

        index = self.client.Index(self.pinecone_config.index_name)

        for i in range(0, len(chunks), batch_size):
            batch_number = i // batch_size + 1
            chunk_batch = chunks[i : i + batch_size]
            logger.info(
                f"Batch {batch_number}/{total_batches}: embedding {len(chunk_batch)} chunk(s)"
            )

            ids = [
                f"{chunk.metadata['source']}_{chunk.metadata['page']}_{j}"
                for j, chunk in enumerate(chunk_batch)
            ]

            dense_embeddings = await self.dense_embedding_strategy.embed_documents(chunk_batch)
            sparse_embeddings = await self.sparse_embedding_strategy.embed_documents(
                [chunk.page_content for chunk in chunk_batch]
            )

            metadata = [
                {
                    "source": chunk.metadata["source"],
                    "page": chunk.metadata["page"],
                    "text": chunk.page_content,
                }
                for chunk in chunk_batch
            ]

            dict_vector_chunks = [
                {
                    "id": vec_id,
                    "values": dense_embedding,
                    "sparse_values": sparse_embedding,
                    "metadata": metadata_chunk,
                }
                for vec_id, dense_embedding, sparse_embedding, metadata_chunk in zip(
                    ids, dense_embeddings, sparse_embeddings, metadata
                )
            ]

            index.upsert(vectors=dict_vector_chunks)
            logger.success(
                f"Batch {batch_number}/{total_batches}: upserted {len(dict_vector_chunks)} vector(s) to Pinecone"
            )

        logger.success(
            f"PineconeRepository.upsert_chunks: all {total_batches} batch(es) complete "
            f"({len(chunks)} chunk(s) total)"
        )
