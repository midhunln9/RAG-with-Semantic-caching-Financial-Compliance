from pydantic import BaseModel

from src.strategies.llm_strategy import LLMStrategy
from src.schemas.rewrite_query_json_call import RewriteQueryJsonCall

class RagWorkflowService:
  def __init__(self, llm_strategy: LLMStrategy):
    self.llm_strategy = llm_strategy
  
  async def rewrite_query_service(self, prompt: str, response_class : type[BaseModel]) -> str:
    response = await self.llm_strategy.generate_response(prompt, response_class)
    return response.rewritten_query