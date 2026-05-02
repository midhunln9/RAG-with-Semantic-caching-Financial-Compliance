from guardrails import AsyncGuard
from loguru import logger

from rag_src.prompts.topic_guard_prompt import TOPIC_GUARD_PROMPT
from rag_src.schemas.topic_guard_decision import TopicGuardDecision
from rag_src.strategies.llm_strategy import LLMStrategy

FINANCIAL_COMPLIANCE_OFF_TOPIC_RESPONSE = (
    "Kindly stick to the concept of financial compliance"
)


class FinancialComplianceGuardrail:
    """Runs a topicality check before the main RAG workflow starts."""

    def __init__(self, llm_strategy: LLMStrategy):
        self.llm_strategy = llm_strategy
        self.guard = AsyncGuard.for_pydantic(
            TopicGuardDecision,
            name="financial_compliance_topic_guard",
            description="Checks whether an incoming prompt is about financial compliance.",
        )

    async def evaluate(self, user_prompt: str) -> TopicGuardDecision:
        prompt = TOPIC_GUARD_PROMPT.format(user_prompt=user_prompt)
        validation_outcome = await self.guard(
            self._call_structured_llm,
            messages=[{"role": "user", "content": prompt}],
            num_reasks=0,
        )

        validated_output = validation_outcome.validated_output
        if validated_output is None:
            raise ValueError(
                "Financial compliance guardrail returned no validated output. "
                f"Guardrails error: {validation_outcome.error}"
            )

        if isinstance(validated_output, TopicGuardDecision):
            decision = validated_output
        elif isinstance(validated_output, str):
            decision = TopicGuardDecision.model_validate_json(validated_output)
        else:
            decision = TopicGuardDecision.model_validate(validated_output)

        logger.info(
            f"[financial_compliance_guard] is_on_topic={decision.is_on_topic} "
            f"for query={user_prompt!r}"
        )
        return decision

    async def _call_structured_llm(self, *, messages: list[dict] | None = None, **kwargs) -> str:
        prompt = self._flatten_messages(messages or [])
        response = await self.llm_strategy.generate_response(
            prompt,
            response_class=TopicGuardDecision,
        )
        return response.model_dump_json()

    @staticmethod
    def _flatten_messages(messages: list[dict]) -> str:
        return "\n\n".join(
            message.get("content", "")
            for message in messages
            if isinstance(message, dict) and message.get("content")
        )
