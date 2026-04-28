from langchain_core.messages import HumanMessage
from loguru import logger

from src.prompts.final_answer_prompt import FINAL_ANSWER_PROMPT
from src.prompts.rewrite_prompt import QUERY_REWRITE_PROMPT
from src.repositories.conversation_db import ConversationDB
from src.schemas.rewrite_query_json_call import RewriteQueryJsonCall
from src.services import RagWorkflowService
from src.state import AgentState


class Nodes:
    def __init__(
        self,
        rag_service: RagWorkflowService,
        conversation_db: ConversationDB,
    ):
        self.rag_service = rag_service
        self.conversation_db = conversation_db

    async def rewrite_query(self, state: AgentState) -> dict:
        query = state["query"]
        session_id = state["session_id"]
        logger.info(f"[rewrite_query] session_id={session_id} query={query!r}")

        # Save the user's message before rewriting so it's stored even if
        # downstream steps fail.
        await self.conversation_db.save_human_message(session_id, query)

        prompt = QUERY_REWRITE_PROMPT.format(user_query=query)
        rewritten_query = await self.rag_service.rewrite_query_service(
            prompt=prompt, response_class=RewriteQueryJsonCall
        )
        logger.info(f"[rewrite_query] rewritten_query={rewritten_query!r}")
        return {"rewritten_query": rewritten_query}

    async def retrieve_documents(self, state: AgentState) -> dict:
        rewritten_query = state["rewritten_query"]
        logger.info(f"[retrieve_documents] querying for {rewritten_query!r}")
        retrieved_docs = await self.rag_service.retrieve_documents_service(
            rewritten_query
        )
        logger.info(f"[retrieve_documents] retrieved {len(retrieved_docs)} documents")
        return {"retrieved_docs": retrieved_docs}

    async def get_conversations(self, state: AgentState) -> dict:
        session_id = state["session_id"]
        logger.info(f"[get_conversations] session_id={session_id}")
        past_conversations = await self.conversation_db.get_last_messages(
            session_id, limit=10
        )
        logger.info(
            f"[get_conversations] fetched {len(past_conversations)} past messages"
        )
        return {"past_conversations": past_conversations}

    async def rag_answer(self, state: AgentState) -> dict:
        session_id = state["session_id"]
        rewritten_query = state["rewritten_query"]
        retrieved_docs = state["retrieved_docs"]
        past_conversations = state["past_conversations"]

        retrieved_context = (
            "\n\n".join(
                f"[doc {i + 1}] {doc.page_content}"
                for i, doc in enumerate(retrieved_docs)
            )
            or "(no documents retrieved)"
        )

        past_conversation_text = (
            "\n".join(
                f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
                for m in past_conversations
            )
            or "(no prior conversation)"
        )

        prompt = FINAL_ANSWER_PROMPT.format(
            rewritten_query=rewritten_query,
            retrieved_context=retrieved_context,
            past_conversation=past_conversation_text,
        )

        logger.info(f"[rag_answer] generating final answer for session_id={session_id}")
        final_answer = await self.rag_service.generate_answer_service(prompt)
        logger.info(
            f"[rag_answer] generated answer ({len(final_answer)} chars) for session_id={session_id}"
        )

        # Save the assistant's reply so it's part of history on the next turn.
        await self.conversation_db.save_ai_message(session_id, final_answer)

        return {"final_answer": final_answer}
