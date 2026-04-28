from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel

router = APIRouter()


class ChatRequest(BaseModel):
    payload: str
    session_id: str


@router.post("/chat")
async def chat(chat_request: ChatRequest, request: Request):
    workflow = request.app.state.graph
    logger.info(
        f"/chat received session_id={chat_request.session_id} "
        f"payload={chat_request.payload!r}"
    )

    async def generate():
        try:
            async for chunk, metadata in workflow.astream(
                {
                    "query": chat_request.payload,
                    "session_id": chat_request.session_id,
                },
                stream_mode="messages",
            ):
                if metadata.get("langgraph_node") == "rag_answer":
                    yield chunk.content
        except Exception as e:
            logger.exception(
                f"Workflow stream failed for session_id={chat_request.session_id}: {e}"
            )
            raise

    return StreamingResponse(generate(), media_type="text/plain")
