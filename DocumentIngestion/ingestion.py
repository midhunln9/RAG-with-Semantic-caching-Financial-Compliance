import os
import sys
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from loguru import logger


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from DocumentIngestion.configs.pinecone_config import PineconeConfig
from DocumentIngestion.recursive_character_text_splitter import (
    RecursiveCharacterTextSplitter,
)
from DocumentIngestion.repository.drive_repo import GoogleDriveRepository
from DocumentIngestion.repository.pinecone_repository import PineconeRepository
from DocumentIngestion.services.chunker_service import ChunkerService
from DocumentIngestion.services.dense_embedding_service import DenseEmbeddingService
from DocumentIngestion.services.ingestor_service import IngestorService
from DocumentIngestion.services.prepare_and_upsert_service import (
    PrepareAndUpsertService,
)
from DocumentIngestion.services.sparse_embedding_service import SparseEmbeddingService
from src.embeddings.openai_embedding import OpenAIEmbedding
from src.embeddings.splade_sparse_embedding import SentenceTransformerSparseEmbedding


def main() -> None:
    load_dotenv(find_dotenv())

    google_drive_folder_link = os.getenv("GOOGLE_DRIVE_FOLDER_LINK")
    if not google_drive_folder_link:
        raise RuntimeError("GOOGLE_DRIVE_FOLDER_LINK is not set in the environment.")

    google_drive_repository = GoogleDriveRepository(google_drive_folder_link)

    ingestor_service = IngestorService(google_drive_repository)
    chunker_service = ChunkerService(
        RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    )

    pdf_documents = ingestor_service.ingest()
    logger.info(f"Fetched {len(pdf_documents)} PDF documents from Google Drive.")

    chunked_documents = chunker_service.start_chunking(pdf_documents)
    logger.info(f"Prepared {len(chunked_documents)} chunked documents.")

    pinecone_repository = PineconeRepository(PineconeConfig())
    dense_embedding_service = DenseEmbeddingService(OpenAIEmbedding())
    sparse_embedding_service = SparseEmbeddingService(
        SentenceTransformerSparseEmbedding()
    )
    prepare_and_upsert_service = PrepareAndUpsertService(
        pinecone_repository=pinecone_repository,
        dense_embedding_service=dense_embedding_service,
        sparse_embedding_service=sparse_embedding_service,
    )

    upserted_chunks = prepare_and_upsert_service.prepare_and_upsert(chunked_documents)
    logger.info(f"Upserted {upserted_chunks} chunks to Pinecone.")


if __name__ == "__main__":
    main()
