import hashlib
from pathlib import Path

from langchain_core.documents import Document

from DocumentIngestion.repository.pinecone_repository import PineconeRepository
from DocumentIngestion.services.dense_embedding_service import DenseEmbeddingService
from DocumentIngestion.services.sparse_embedding_service import SparseEmbeddingService


class PrepareAndUpsertService:
    def __init__(
        self,
        pinecone_repository: PineconeRepository,
        dense_embedding_service: DenseEmbeddingService,
        sparse_embedding_service: SparseEmbeddingService,
    ):
        self.pinecone_repository = pinecone_repository
        self.dense_embedding_service = dense_embedding_service
        self.sparse_embedding_service = sparse_embedding_service

    def prepare_and_upsert(
        self, chunked_documents: list[Document], batch_size: int = 200
    ) -> int:
        dense_embedding_dimension = self.dense_embedding_service.get_embedding_dimension()
        self.pinecone_repository.ensure_index_exists(dense_embedding_dimension)

        total_upserted = 0

        for start_index in range(0, len(chunked_documents), batch_size):
            chunk_batch = chunked_documents[start_index : start_index + batch_size]

            chunk_ids = self._build_chunk_ids(chunk_batch)
            dense_embeddings = self.dense_embedding_service.get_dense_embeddings(
                chunk_batch
            )
            sparse_embeddings = self.sparse_embedding_service.get_sparse_embeddings(
                chunk_batch
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
                    "id": chunk_id,
                    "values": dense_embedding,
                    "sparse_values": sparse_embedding,
                    "metadata": metadata_chunk,
                }
                for chunk_id, dense_embedding, sparse_embedding, metadata_chunk in zip(
                    chunk_ids,
                    dense_embeddings,
                    sparse_embeddings,
                    metadata,
                    strict=True,
                )
            ]

            total_upserted += self.pinecone_repository.upsert_chunks(
                dict_vector_chunks
            )

        return total_upserted

    def _build_chunk_ids(self, chunk_batch: list[Document]) -> list[str]:
        chunk_ids: list[str] = []

        for chunk in chunk_batch:
            document_name = Path(str(chunk.metadata["source"])).stem.replace(" ", "_")
            page_number = chunk.metadata["page"]
            chunk_hash = hashlib.sha256(
                chunk.page_content.encode("utf-8")
            ).hexdigest()[:16]
            chunk_ids.append(f"{document_name}_{page_number}_{chunk_hash}")

        return chunk_ids
