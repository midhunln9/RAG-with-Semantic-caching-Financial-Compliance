import os

from langchain_core.messages import AIMessage
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from pydantic import BaseModel

from rag_src.strategies.llm_strategy import LLMStrategy


class NVIDIALLM(LLMStrategy):
    def __init__(self):
        self.model_name = "meta/llama-3.3-70b-instruct"
        self.client = ChatNVIDIA(
            api_key=os.getenv("NVIDIA_API_KEY"),
            model=self.model_name,
            temperature=0.2,
            top_p=0.7,
            max_tokens=1024,
        )

    @property
    def cache_llm_string(self) -> str:
        return f"rag_pipeline:v1:nvidia:{self.model_name}"

    async def generate_response(
        self, prompt: str, response_class: type[BaseModel] | None = None
    ) -> AIMessage | BaseModel:
        if not response_class:
            response = await self.client.ainvoke(prompt)
            return response

        llm = self.client.with_structured_output(response_class, method="json_schema")
        response = await llm.ainvoke(prompt)
        return response
