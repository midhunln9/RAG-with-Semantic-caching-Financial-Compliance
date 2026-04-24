from fastapi import FastAPI

from contextlib import asynccontextmanager

from src.state import AgentState
from src.services import RagWorkflowService
from src.nodes import Nodes
from src.graph import Graph
from src.llm.llm_factory import get_llm_strategy

from app.routes.chat import router as chat_router

from loguru import logger
from datetime import datetime

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

log_file_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

@asynccontextmanager
async def start_shut(app):
    logger.add(f"logs/log_{log_file_name}.log", rotation="1 day", retention="7 days", level="INFO")
    # llm
    llm = get_llm_strategy("openai")
    # service
    service = RagWorkflowService(llm_strategy = llm)
    # nodes
    nodes = Nodes(service = service)
    # graph
    graph = Graph(nodes = nodes)
    # build workflow
    app.state.graph = graph.build_graph()
    yield
    
app = FastAPI(title="GenAIRag", version="0.1.0", lifespan=start_shut)

app.include_router(chat_router)