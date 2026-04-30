import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from pinecone import Pinecone

load_dotenv()

INDEX_NAME = "ingestion-open-ai-embedding-small"
NAMESPACE = "__default__"
NUM_CHUNKS = 100
OUTPUT_PATH = Path("offline_retriever_eval/GoldenDataset/hundred_chunks.csv")


def configure_logger():
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <7}</level> | {message}"
        ),
    )


def main():
    configure_logger()
    logger.info("Stage 1: fetching {} chunks from Pinecone index '{}'", NUM_CHUNKS, INDEX_NAME)

    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(INDEX_NAME)

    selected_ids = []
    for ids_page in index.list(namespace=NAMESPACE):
        selected_ids.extend(ids_page)
        if len(selected_ids) >= NUM_CHUNKS:
            selected_ids = selected_ids[:NUM_CHUNKS]
            break

    response = index.fetch(ids=selected_ids, namespace=NAMESPACE)
    rows = [
        {"id": vid, "text": v.metadata.get("text")}
        for vid, v in response.vectors.items()
    ]

    df = pd.DataFrame(rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False, header=True)
    logger.success("Stage 1 complete: wrote {} rows to {}", len(df), OUTPUT_PATH)


if __name__ == "__main__":
    main()
