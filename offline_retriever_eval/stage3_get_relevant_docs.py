import asyncio
import json
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from loguru import logger

from rag_src.configs.pinecone_config import PineconeConfig
from rag_src.embeddings.openai_embedding import OpenAIEmbedding
from rag_src.embeddings.splade_sparse_embedding import SentenceTransformerSparseEmbedding
from rag_src.repositories.pinecone_repository import PineconeRepository

load_dotenv()

INPUT_PATH = Path("offline_retriever_eval/GoldenDataset/hundred_chunks_queries.csv")
OUTPUT_PATH = Path(
    "offline_retriever_eval/GoldenDataset/final_hundred_chunks_queries_relevant_docs.csv"
)


def configure_logger():
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <7}</level> | {message}"
        ),
    )


async def run():
    configure_logger()
    logger.info("Stage 3: retrieving relevant docs from {}", INPUT_PATH)

    df = pd.read_csv(INPUT_PATH)
    repo = PineconeRepository(
        dense_embedding_strategy=OpenAIEmbedding(),
        sparse_embedding_strategy=SentenceTransformerSparseEmbedding(),
        pinecone_config=PineconeConfig(),
    )

    all_ids: list[str] = []
    all_scores: list[str] = []
    for i, query in enumerate(df["generated_query"], start=1):
        results = await repo.query_vector_store_for_rankx(query)
        all_ids.append(json.dumps([m["id"] for m in results]))
        all_scores.append(json.dumps([m["score"] for m in results]))
        if i % 10 == 0:
            logger.info("Progress: {}/{}", i, len(df))

    df["relevant_doc_ids"] = all_ids
    df["relevant_doc_scores"] = all_scores
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False, header=True)
    logger.success("Stage 3 complete: wrote {} rows to {}", len(df), OUTPUT_PATH)


def main():
    asyncio.run(run())


if __name__ == "__main__":
    main()
