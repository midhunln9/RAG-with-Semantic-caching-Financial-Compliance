import os
import time
from typing import Any

from pinecone import Pinecone, ServerlessSpec

from DocumentIngestion.configs.pinecone_config import PineconeConfig


class PineconeRepository:
    def __init__(self, pinecone_config: PineconeConfig):
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        if not pinecone_api_key:
            raise RuntimeError("PINECONE_API_KEY is not set in the environment.")

        self.client = Pinecone(api_key=pinecone_api_key)
        self.pinecone_config = pinecone_config
        self.namespace = os.getenv("PINECONE_NAMESPACE")
        self.index = None

    def ensure_index_exists(self, dense_embedding_dimension: int) -> None:
        if not self.client.has_index(self.pinecone_config.index_name):
            self.client.create_index(
                name=self.pinecone_config.index_name,
                dimension=dense_embedding_dimension,
                metric=self.pinecone_config.metric,
                spec=ServerlessSpec(
                    cloud=self.pinecone_config.cloud,
                    region=self.pinecone_config.region,
                ),
            )

        index_description = self.client.describe_index(self.pinecone_config.index_name)
        while not self._is_index_ready(index_description):
            time.sleep(2)
            index_description = self.client.describe_index(
                self.pinecone_config.index_name
            )

        existing_dimension = getattr(index_description, "dimension", None)
        if existing_dimension and existing_dimension != dense_embedding_dimension:
            raise RuntimeError(
                "The existing Pinecone index dimension does not match the dense embedding dimension."
            )

        self.index = self.client.Index(host=index_description.host)

    def upsert_chunks(self, dict_vector_chunks: list[dict[str, Any]]) -> int:
        if self.index is None:
            raise RuntimeError("Pinecone index is not ready for upsert.")

        upsert_arguments = {
            "vectors": dict_vector_chunks,
            "show_progress": False,
        }
        if self.namespace:
            upsert_arguments["namespace"] = self.namespace

        upsert_response = self.index.upsert(**upsert_arguments)
        return upsert_response.upserted_count

    def _is_index_ready(self, index_description: Any) -> bool:
        index_status = getattr(index_description, "status", None)
        if index_status is None:
            return True

        if isinstance(index_status, dict):
            return bool(index_status.get("ready"))

        return bool(getattr(index_status, "ready", False))
