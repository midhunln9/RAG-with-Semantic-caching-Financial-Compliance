from typing import TypedDict

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    query: str
    session_id: str
    rewritten_query: str
    retrieved_docs: list[Document]
    past_conversations: list[BaseMessage]
    final_answer: str
