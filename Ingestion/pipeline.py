from loguru import logger

from Ingestion.src.ingestion_and_chunk import ChunkerService
from Ingestion.src.upsert_service import UpsertService


class Pipeline:
    def __init__(self, chunker_service: ChunkerService, upsert_service: UpsertService):
        self.chunker_service = chunker_service
        self.upsert_service = upsert_service

    async def run_pipeline(self):
        logger.info("Pipeline starting")

        logger.info("Step 1/2: ingest + chunk")
        chunks = self.chunker_service.ingest_and_chunk()

        logger.info("Step 2/2: upsert to vector store")
        await self.upsert_service.upsert(chunks)

        logger.success("Pipeline finished")
