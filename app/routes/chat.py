from typing import Literal

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel

router = APIRouter()


class ChatRequest(BaseModel):
    payload: str
    session_id: str
    llm: Literal["openai", "nvidia"]


@router.post("/chat")
async def chat(chat_request: ChatRequest, request: Request):
    workflow = request.app.state.graphs[chat_request.llm]
    logger.info(
        f"/chat received llm={chat_request.llm} session_id={chat_request.session_id} "
        f"payload={chat_request.payload!r}"
    )

    async def generate():
        rag_answer_streamed = False
        try:
            async for mode, data in workflow.astream(
                {
                    "query": chat_request.payload,
                    "session_id": chat_request.session_id,
                },
                stream_mode=["messages", "updates"],
            ):
                if mode == "messages":
                    chunk, metadata = data
                    if metadata.get("langgraph_node") == "rag_answer":
                        rag_answer_streamed = True
                        yield chunk.content
                    continue

                if mode != "updates":
                    continue

                for node_name, node_update in data.items():
                    if node_name == "rag_answer" and rag_answer_streamed:
                        continue
                    if not isinstance(node_update, dict):
                        continue
                    final_answer = node_update.get("final_answer")
                    if final_answer:
                        yield final_answer
                        break
        except Exception as e:
            logger.exception(
                f"Workflow stream failed for llm={chat_request.llm} "
                f"session_id={chat_request.session_id}: {e}"
            )
            raise

    return StreamingResponse(generate(), media_type="text/plain")
