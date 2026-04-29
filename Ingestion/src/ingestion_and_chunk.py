import os

from Ingestion.protocols.document_source import DocumentSource
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from loguru import logger
from Ingestion.strategies.splitter import Splitter


class ChunkerService:
    def __init__(self, document_source: DocumentSource, splitter: Splitter):
        self.document_source = document_source
        self.splitter = splitter

    def ingest_and_chunk(self) -> list[Document]:
        documents = self.document_source.get_documents()
        total = len(documents)
        logger.info(f"Found {total} document(s) to process")

        chunks: list[Document] = []
        for i, doc in enumerate(documents, start=1):
            name = os.path.basename(doc)
            logger.info(f"[{i}/{total}] Loading {name}")
            pages = PyPDFLoader(doc, mode="page").load()
            new_chunks = self.splitter.split_documents(pages)
            chunks.extend(new_chunks)
            logger.info(
                f"[{i}/{total}] {name} -> {len(pages)} page(s), "
                f"{len(new_chunks)} chunk(s) (running total: {len(chunks)})"
            )

        logger.success(f"Chunking finished -- produced {len(chunks)} total chunk(s)")
        return chunks




