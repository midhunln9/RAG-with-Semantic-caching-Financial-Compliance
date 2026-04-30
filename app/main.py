import os
from contextlib import asynccontextmanager
from datetime import datetime

from dotenv import find_dotenv, load_dotenv
from fastapi import FastAPI
from loguru import logger

from app.routes.chat import router as chat_router
from rag_src.configs.pinecone_config import PineconeConfig
from rag_src.embeddings.openai_embedding import OpenAIEmbedding
from rag_src.embeddings.splade_sparse_embedding import (
    SentenceTransformerSparseEmbedding,
)
from rag_src.graph import Graph
from rag_src.llm.llm_factory import get_llm_strategy
from rag_src.nodes import Nodes
from rag_src.repositories.conversation_db import ConversationDB
from rag_src.repositories.pinecone_repository import PineconeRepository
from rag_src.services import RagWorkflowService

load_dotenv(find_dotenv())

log_file_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


@asynccontextmanager
async def start_shut(app):
    logger.add(
        f"logs/log_{log_file_name}.log",
        rotation="1 day",
        retention="7 days",
        level="INFO",
    )
    logger.info("Starting application initialization")

    # Embeddings
    dense_embedding = OpenAIEmbedding()
    sparse_embedding = SentenceTransformerSparseEmbedding()

    # Vector DB (Pinecone)
    pinecone_config = PineconeConfig()
    vector_store = PineconeRepository(
        dense_embedding_strategy=dense_embedding,
        sparse_embedding_strategy=sparse_embedding,
        pinecone_config=pinecone_config,
    )

    # Conversation DB (Postgres)
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is not set in the environment")
    conversation_db = ConversationDB(database_url)
    await conversation_db.connect()

    # LLM workflows
    app.state.graphs = {}
    for llm_name in ("nvidia", "openai"):
        llm = get_llm_strategy(llm_name)
        rag_service = RagWorkflowService(llm_strategy=llm, vector_db=vector_store)
        nodes = Nodes(rag_service=rag_service, conversation_db=conversation_db)
        graph = Graph(nodes=nodes)
        app.state.graphs[llm_name] = graph.build_graph()

    logger.info("Application initialization complete")

    try:
        yield
    finally:
        logger.info("Shutting down application")
        await conversation_db.close()


app = FastAPI(title="GenAIRag", version="0.1.0", lifespan=start_shut)
app.include_router(chat_router)
