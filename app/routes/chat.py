from fastapi import APIRouter, Request
from pydantic import BaseModel

from fastapi.responses import StreamingResponse

router = APIRouter()

class ChatRequest(BaseModel):
    payload: str

@router.post("/chat")
async def chat(chat_request: ChatRequest, request: Request):
    workflow = request.app.state.graph
    async def generate():
        async for chunk, metadata in workflow.astream(
            {"query": chat_request.payload},
            stream_mode="messages",
        ):
            if metadata.get("langgraph_node") == "rewrite_query":
                yield chunk.content

    return StreamingResponse(generate(), media_type="text/plain")
