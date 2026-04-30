import asyncio
from datetime import datetime

from dotenv import load_dotenv
from loguru import logger

from DocumentIngestion.pipeline import Pipeline
from DocumentIngestion.repositories.file_repo import FileRepo
from DocumentIngestion.src.ingestion_and_chunk import ChunkerService
from DocumentIngestion.src.splitter import RecursiveCharacterTextSplitterMethod
from DocumentIngestion.src.upsert_service import UpsertService
from rag_src.configs.pinecone_config import PineconeConfig
from rag_src.embeddings.openai_embedding import OpenAIEmbedding
from rag_src.embeddings.splade_sparse_embedding import SentenceTransformerSparseEmbedding
from rag_src.repositories.pinecone_repository import PineconeRepository

load_dotenv()


async def main():
    log_file_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logger.add(
        f"logs/ingestion_{log_file_name}.log",
        level="INFO",
        retention="7 days",
    )

    logger.info("Initialising ingestion pipeline")

    document_source = FileRepo(folder_path="documents")
    splitter = RecursiveCharacterTextSplitterMethod(chunk_size=1000, chunk_overlap=200)
    logger.info(
        f"Source folder: {document_source.folder_path} | "
        f"chunk_size={splitter.chunk_size}, chunk_overlap={splitter.chunk_overlap}"
    )

    dense_embedding_model = OpenAIEmbedding()
    sparse_embedding_model = SentenceTransformerSparseEmbedding()
    logger.info(f"Dense embedder: {dense_embedding_model.model_name} | Sparse embedder: SPLADE")

    pinecone_config = PineconeConfig()
    repository = PineconeRepository(
        dense_embedding_strategy=dense_embedding_model,
        sparse_embedding_strategy=sparse_embedding_model,
        pinecone_config=pinecone_config,
    )
    logger.info(
        f"Pinecone index: {pinecone_config.index_name} (batch_size={pinecone_config.batch_size})"
    )

    chunker_service = ChunkerService(document_source=document_source, splitter=splitter)
    upsert_service = UpsertService(repo=repository)

    pipeline = Pipeline(chunker_service=chunker_service, upsert_service=upsert_service)
    await pipeline.run_pipeline()

    logger.success("Ingestion run complete")


if __name__ == "__main__":
    asyncio.run(main())
