import json
import sys
from pathlib import Path

import pandas as pd
from loguru import logger
from ranx import Qrels, Run, evaluate

INPUT_PATH = Path(
    "offline_retriever_eval/GoldenDataset/final_hundred_chunks_queries_relevant_docs.csv"
)
OUTPUT_PATH = Path("offline_retriever_eval/GoldenDataset/retrieval_metrics.csv")
METRICS = [
    "hit_rate@1",
    "hit_rate@3",
    "hit_rate@5",
    "hit_rate@10",
    "recall@10",
    "precision@10",
    "mrr@10",
    "map@10",
    "ndcg@10",
]


def configure_logger():
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <7}</level> | {message}"
        ),
    )


class RanxDataPrep:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def build_qrels(self) -> Qrels:
        qrels: dict[str, dict[str, int]] = {}
        for _, row in self.df.iterrows():
            qid = str(row["id"])
            qrels[qid] = {qid: 1}
        return Qrels(qrels)

    def build_run(self) -> Run:
        run: dict[str, dict[str, float]] = {}
        for _, row in self.df.iterrows():
            qid = str(row["id"])
            ids = json.loads(row["relevant_doc_ids"])
            scores = json.loads(row["relevant_doc_scores"])
            run[qid] = {str(d): float(s) for d, s in zip(ids, scores)}
        return Run(run)


def main():
    configure_logger()
    logger.info("Stage 4: running ranx evaluation on {}", INPUT_PATH)

    df = pd.read_csv(INPUT_PATH)
    prep = RanxDataPrep(df)
    qrels = prep.build_qrels()
    run = prep.build_run()

    results = evaluate(qrels, run, metrics=METRICS)

    metrics_df = pd.DataFrame([{"metric": m, "value": v} for m, v in results.items()])
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(OUTPUT_PATH, index=False, header=True)

    logger.success("Stage 4 complete: wrote metrics to {}", OUTPUT_PATH)
    for m, v in results.items():
        logger.info("  {} = {:.4f}", m, v)


if __name__ == "__main__":
    main()
