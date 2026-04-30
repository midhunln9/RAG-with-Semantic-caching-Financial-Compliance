from typing import List

from langchain_core.documents import Document
from loguru import logger

from rag_src.protocols.vector_db import VectorDBProtocol


class UpsertService:
    def __init__(self, repo: VectorDBProtocol):
        self.repo = repo

    async def upsert(self, chunks: List[Document]):
        if not chunks:
            logger.warning("UpsertService.upsert called with 0 chunks -- skipping")
            return

        logger.info(f"Upserting {len(chunks)} chunk(s) to vector store")
        await self.repo.upsert_chunks(chunks)
        logger.success(f"Upsert finished -- {len(chunks)} chunk(s) sent")
