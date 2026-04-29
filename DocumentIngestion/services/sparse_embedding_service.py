from langchain_core.documents import Document

from src.embeddings.splade_sparse_embedding import SentenceTransformerSparseEmbedding


class SparseEmbeddingService:
    def __init__(self, sparse_embedding: SentenceTransformerSparseEmbedding):
        self.sparse_embedding = sparse_embedding

    def get_sparse_embeddings(self, chunk_documents: list[Document]) -> list[dict]:
        chunk_texts = [chunk.page_content for chunk in chunk_documents]
        sparse_tensors = self.sparse_embedding.model.encode_document(chunk_texts)
        return [
            self.sparse_embedding._sparse_tensor_to_pinecone_dict(sparse_tensor)
            for sparse_tensor in sparse_tensors
        ]
