from typing import NotRequired, Required, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    query: Required[str]
    session_id: Required[str]
    is_on_topic: NotRequired[bool]
    rewritten_query: NotRequired[str]
    cache_key: NotRequired[str]
    cache_hit: NotRequired[bool]
    cached_answer: NotRequired[str]
    retrieved_docs: NotRequired[list[Document]]
    past_conversations: NotRequired[list[BaseMessage]]
    final_answer: NotRequired[str]
