from langchain_openai import ChatOpenAI
from src.strategies.llm_strategy import LLMStrategy
import os
from pydantic import BaseModel
from langchain_core.messages import AIMessage

class OpenAILLM(LLMStrategy):
    def __init__(self):
        self.client = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o-mini", streaming = True)

    async def generate_response(self, prompt: str, response_class: type[BaseModel] | None = None) -> AIMessage | BaseModel:
        if not response_class:
            response = await self.client.ainvoke(prompt)
            return response
        else:
            llm = self.client.with_structured_output(response_class, method = "json_schema")
            response = await llm.ainvoke(prompt)
            return response