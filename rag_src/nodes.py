from langchain_core.messages import HumanMessage
from loguru import logger

from rag_src.guardrails import (
    FINANCIAL_COMPLIANCE_OFF_TOPIC_RESPONSE,
    FinancialComplianceGuardrail,
)
from rag_src.prompts.final_answer_prompt import FINAL_ANSWER_PROMPT
from rag_src.prompts.rewrite_prompt import QUERY_REWRITE_PROMPT
from rag_src.protocols.cache import CacheProtocol
from rag_src.repositories.conversation_db import ConversationDB
from rag_src.schemas.rewrite_query_json_call import RewriteQueryJsonCall
from rag_src.services import RagWorkflowService
from rag_src.state import AgentState


class Nodes:
    def __init__(
        self,
        rag_service: RagWorkflowService,
        conversation_db: ConversationDB,
        topic_guardrail: FinancialComplianceGuardrail,
        cache: CacheProtocol | None = None,
    ):
        self.rag_service = rag_service
        self.conversation_db = conversation_db
        self.topic_guardrail = topic_guardrail
        self.cache = cache

    async def guard_prompt(self, state: AgentState) -> dict:
        query = state["query"]
        session_id = state["session_id"]
        logger.info(f"[guard_prompt] session_id={session_id} query={query!r}")

        decision = await self.topic_guardrail.evaluate(query)
        return {"is_on_topic": decision.is_on_topic}

    async def return_off_topic_response(self, state: AgentState) -> dict:
        session_id = state["session_id"]
        logger.info(f"[return_off_topic_response] blocked session_id={session_id}")
        return {"final_answer": FINANCIAL_COMPLIANCE_OFF_TOPIC_RESPONSE}

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

    async def check_cache(self, state: AgentState) -> dict:
        cache_key = state["rewritten_query"]
        session_id = state["session_id"]

        if self.cache is None:
            logger.info(
                f"[check_cache] cache disabled; treating as miss for session_id={session_id}"
            )
            return {"cache_key": cache_key, "cache_hit": False}

        try:
            cached_answer = self.cache.get(cache_key)
        except Exception as e:
            if hasattr(self.cache, "mark_backend_unavailable"):
                self.cache.mark_backend_unavailable("lookup", e)
            logger.warning(
                "[check_cache] cache lookup failed for "
                f"session_id={session_id} key={cache_key!r}: {e}"
            )
            return {"cache_key": cache_key, "cache_hit": False}

        if cached_answer is None:
            logger.info(f"[check_cache] miss for session_id={session_id} key={cache_key!r}")
            return {"cache_key": cache_key, "cache_hit": False}

        logger.info(f"[check_cache] hit for session_id={session_id} key={cache_key!r}")
        return {
            "cache_key": cache_key,
            "cache_hit": True,
            "cached_answer": cached_answer,
        }

    async def cache_miss(self, state: AgentState) -> dict:
        logger.info(f"[cache_miss] continuing RAG workflow for session_id={state['session_id']}")
        return {}

    async def retrieve_documents(self, state: AgentState) -> dict:
        rewritten_query = state["rewritten_query"]
        logger.info(f"[retrieve_documents] querying for {rewritten_query!r}")
        retrieved_docs = await self.rag_service.retrieve_documents_service(rewritten_query)
        logger.info(f"[retrieve_documents] retrieved {len(retrieved_docs)} documents")
        return {"retrieved_docs": retrieved_docs}

    async def get_conversations(self, state: AgentState) -> dict:
        session_id = state["session_id"]
        logger.info(f"[get_conversations] session_id={session_id}")
        past_conversations = await self.conversation_db.get_last_messages(session_id, limit=10)
        logger.info(f"[get_conversations] fetched {len(past_conversations)} past messages")
        return {"past_conversations": past_conversations}

    async def return_cached_answer(self, state: AgentState) -> dict:
        session_id = state["session_id"]
        cached_answer = state["cached_answer"]
        logger.info(f"[return_cached_answer] returning cached answer for session_id={session_id}")

        # Save the assistant's reply so follow-up turns still see full history.
        await self.conversation_db.save_ai_message(session_id, cached_answer)

        return {"final_answer": cached_answer}

    async def rag_answer(self, state: AgentState) -> dict:
        session_id = state["session_id"]
        rewritten_query = state["rewritten_query"]
        retrieved_docs = state["retrieved_docs"]
        past_conversations = state["past_conversations"]

        retrieved_context = (
            "\n\n".join(f"[doc {i + 1}] {doc.page_content}" for i, doc in enumerate(retrieved_docs))
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

    async def store_answer_in_cache(self, state: AgentState) -> dict:
        if self.cache is None:
            return {}

        cache_key = state.get("cache_key", state["rewritten_query"])
        final_answer = state["final_answer"]
        session_id = state["session_id"]

        try:
            self.cache.set(cache_key, final_answer)
            logger.info(
                "[store_answer_in_cache] cached answer for "
                f"session_id={session_id} key={cache_key!r}"
            )
        except Exception as e:
            if hasattr(self.cache, "mark_backend_unavailable"):
                self.cache.mark_backend_unavailable("write", e)
            logger.warning(
                f"[store_answer_in_cache] cache write failed for session_id={session_id} "
                f"key={cache_key!r}: {e}"
            )

        return {}
