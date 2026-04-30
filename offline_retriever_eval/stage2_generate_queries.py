import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from openai import OpenAI

load_dotenv()

INPUT_PATH = Path("offline_retriever_eval/GoldenDataset/hundred_chunks.csv")
OUTPUT_PATH = Path("offline_retriever_eval/GoldenDataset/hundred_chunks_queries.csv")
MODEL = "gpt-4o-mini"

QUERY_GENERATOR_PROMPT = """You are generating a query for a financial compliance retrieval system.

Given a document chunk, write the most appropriate user query for the chunk.

Instructions:
- Use natural, realistic language as a compliance, legal, audit, or risk professional would.
- Do NOT copy sentences verbatim from the chunk.

Output rules:
- Return ONLY the query.
- Do NOT include explanations, labels, JSON, or any extra text.

Chunk:
{{CHUNK_TEXT}}
"""


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
    logger.info("Stage 2: generating queries from {}", INPUT_PATH)

    df = pd.read_csv(INPUT_PATH)
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    queries = []
    for i, chunk in enumerate(df["text"], start=1):
        prompt = QUERY_GENERATOR_PROMPT.replace("{{CHUNK_TEXT}}", chunk)
        response = client.responses.create(model=MODEL, input=prompt)
        queries.append(response.output_text)
        if i % 10 == 0:
            logger.info("Progress: {}/{}", i, len(df))

    df["generated_query"] = queries
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False, header=True)
    logger.success("Stage 2 complete: wrote {} rows to {}", len(df), OUTPUT_PATH)


if __name__ == "__main__":
    main()
